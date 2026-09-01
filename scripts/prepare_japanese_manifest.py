# Copyright 2026 OPPO and Fudan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""P1d: 日本語manifestの生成・検証・split付与。

生成するもの:

* ``data/manifests/gol.jsonl``  … 手元にtarがあるgameの発話（P1eが実際に読める範囲）
* ``data/manifests/moe.jsonl``  … 手元にzipがある話者の発話
* ``data/manifests/all.jsonl``  … 上記の結合（splitを付与済み）

あわせて ``metadata.tsv`` 全7,405,094行に対して除外条件だけを適用し、
**raw hours と accepted hours** を集計する（08章 P1a のゴール）。
これはmanifestを作らずストリーミングで数えるため、tarが手元に無くても実行できる。

sample_rate は game / 話者ごとに実ファイルから読む。gol-datasetは44.1 kHzと48 kHzが
混在しており、dataset単位で仮定してはならない（data-inventory.md 参照）。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import soundfile as sf

from cutetts.training import artifacts, text_rules
from cutetts.training.manifest import (
    Utterance,
    load_manifest,
    manifest_checksum,
    summarize,
    validate,
    write_manifest,
)
from cutetts.training.pairing import PairSampler, assert_no_leakage

csv.field_size_limit(10**9)

MOE_BAK_RE = re.compile(r"\.\d{8,}\.bak\.json$|\.bak\.json$")
"""moe-speech-plusのzipには ``<name>.<timestamp>.bak.json`` が本体と同数含まれる。
除外しないとmanifestが二重計上する（data-inventory.md で確認済み）。"""


# --------------------------------------------------------------------- 共通

def _probe_sample_rate(open_bytes) -> int | None:
    try:
        return int(sf.info(io.BytesIO(open_bytes)).samplerate)
    except Exception:
        return None


def _accepted(u: Utterance, generic_ids: frozenset[str]) -> bool:
    """除外条件（D-016）に1つも当たらなければ True。"""
    if not u.text_raw.strip():
        return False
    if text_rules.is_punctuation_only(u.text_raw):
        return False
    if text_rules.has_name_placeholder(u.text_raw):
        return False
    if u.speaker_id in generic_ids:
        return False
    return True


# ------------------------------------------------------------------ gol

def gol_full_accounting(metadata_tsv: Path, generic_ids: frozenset[str],
                        *, min_duration: float, max_duration: float) -> dict:
    """metadata.tsv 全行に除外条件を適用し raw/accepted を数える（manifestは作らない）。"""
    raw_n = raw_s = 0
    acc_n = acc_s = 0
    reasons = Counter()
    reason_seconds = Counter()
    speakers_raw: set[str] = set()
    speakers_acc: set[str] = set()
    with metadata_tsv.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            speaker, text = row[1], row[2]
            try:
                duration = float(row[4])
            except ValueError:
                continue
            raw_n += 1
            raw_s += duration
            speakers_raw.add(speaker)

            hit = []
            if not text.strip():
                hit.append("empty_text")
            elif text_rules.is_punctuation_only(text):
                hit.append("punctuation_only")
            if text_rules.has_name_placeholder(text):
                hit.append("name_placeholder")
            elif text_rules.contains_markup(text):
                hit.append("markup")
            if speaker in generic_ids:
                hit.append("generic_speaker")
            if duration < min_duration:
                hit.append("too_short")
            elif duration > max_duration:
                hit.append("too_long")

            if hit:
                for code in hit:
                    reasons[code] += 1
                reason_seconds[hit[0]] += duration
            else:
                acc_n += 1
                acc_s += duration
                speakers_acc.add(speaker)
    return {
        "raw_utterances": raw_n,
        "raw_hours": raw_s / 3600.0,
        "raw_speakers": len(speakers_raw),
        "accepted_utterances": acc_n,
        "accepted_hours": acc_s / 3600.0,
        "accepted_speakers": len(speakers_acc),
        "excluded_utterances": raw_n - acc_n,
        "excluded_hours": (raw_s - acc_s) / 3600.0,
        "exclusion_counts": dict(reasons),
        "exclusion_hours_primary_reason": {k: v / 3600.0 for k, v in reason_seconds.items()},
    }


_PART_SUFFIX = re.compile(r"_part\d+$")


def group_tars_by_game(tar_dir: Path) -> dict[str, list[Path]]:
    """tarを game_id 単位にまとめる。

    大きいgameは `<game>_part1.tar` / `<game>_part2.tar` に分割されている
    （602 tar に対し metadata 上の game は 596）。ファイル名をそのまま
    game_id として扱うと **分割されたgameが丸ごと落ちる**。
    """
    groups: dict[str, list[Path]] = {}
    for path in sorted(tar_dir.glob("*.tar")):
        groups.setdefault(_PART_SUFFIX.sub("", path.stem), []).append(path)
    return groups


def _index_members(paths: list[Path]) -> dict[str, Path]:
    """分割gameについて、wavのbasename → それが入っているtar を作る。"""
    index: dict[str, Path] = {}
    for path in paths:
        try:
            with tarfile.open(path) as archive:
                for member in archive:
                    if member.name.endswith(".wav"):
                        index[Path(member.name).name] = path
        except Exception as error:
            print(f"  [warn] tar走査失敗 {path.name}: {type(error).__name__}", file=sys.stderr)
    return index


def gol_records(metadata_tsv: Path, tar_dir: Path, generic_ids: frozenset[str]) -> Iterator[Utterance]:
    """手元にtarがあるgameだけmanifest化する。sample_rateは実ファイルから読む。"""
    groups = group_tars_by_game(tar_dir)
    if not groups:
        return
    rate_by_game: dict[str, int] = {}
    member_index: dict[str, dict[str, Path]] = {}
    for game, paths in groups.items():
        try:
            with tarfile.open(paths[0]) as archive:
                for member in archive:
                    if not member.name.endswith(".wav"):
                        continue
                    payload = archive.extractfile(member)
                    if payload is None:
                        continue
                    rate = _probe_sample_rate(payload.read())
                    if rate:
                        rate_by_game[game] = rate
                    break
        except Exception as error:  # 途中まで落ちたtarなどは飛ばす
            print(f"  [warn] tar読み取り失敗 {game}: {type(error).__name__}", file=sys.stderr)
            continue
        if len(paths) > 1:
            member_index[game] = _index_members(paths)
            print(f"  分割game {game[:12]}…: {len(paths)}本 / "
                  f"wav {len(member_index[game]):,}件")
    with metadata_tsv.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 5 or row[0] not in rate_by_game:
                continue
            game, speaker, text, file_path = row[0], row[1], row[2], row[3]
            try:
                duration = float(row[4])
            except ValueError:
                continue
            member = file_path.split("/", 1)[1] if "/" in file_path else file_path
            if game in member_index:
                container = member_index[game].get(Path(file_path).name)
                if container is None:      # どのpartにも無いwav
                    continue
            else:
                container = groups[game][0]
            yield Utterance(
                utterance_id=f"gol:{game}:{Path(file_path).name}",
                dataset_id="gol",
                audio_ref=f"{container}::{member}",
                text_raw=text,
                speaker_id=speaker,
                duration=duration,
                sample_rate=rate_by_game[game],
                game_id=game,
                text_normalized=text_rules.strip_markup(text) or None,
            )


# ------------------------------------------------------------------ moe

def moe_records(zip_dir: Path) -> Iterator[Utterance]:
    for path in sorted(zip_dir.glob("*.zip")):
        try:
            archive = zipfile.ZipFile(path)
        except Exception as error:
            print(f"  [warn] zip読み取り失敗 {path.name}: {type(error).__name__}", file=sys.stderr)
            continue
        with archive:
            names = archive.namelist()
            wavs = sorted(n for n in names if n.endswith(".wav"))
            rate: int | None = None
            for name in wavs:
                stem = name[:-4]
                meta_name = f"{stem}.json"
                if meta_name not in names or MOE_BAK_RE.search(meta_name):
                    continue
                try:
                    meta = json.loads(archive.read(meta_name).decode("utf-8"))
                except Exception:
                    continue
                if rate is None:
                    rate = _probe_sample_rate(archive.read(name))
                text = (meta.get("anime_whisper_transcription") or "").strip()
                duration = meta.get("duration")
                if duration is None:
                    continue
                speaker = Path(name).parts[1] if len(Path(name).parts) > 1 else path.stem
                yield Utterance(
                    utterance_id=f"moe:{speaker}:{Path(name).stem}",
                    dataset_id="moe",
                    audio_ref=f"{path}::{name}",
                    text_raw=text,
                    speaker_id=speaker,
                    duration=float(duration),
                    sample_rate=int(rate or 44100),
                    quality_score=meta.get("speechMOS"),
                    text_normalized=(meta.get("parakeet_jp_transcription") or None),
                )


# ------------------------------------------------------------------ split

def assign_splits(records: list[Utterance], *, seed: int,
                  zero_shot_fraction: float = 0.12,
                  dev_fraction: float = 0.05) -> dict[str, int]:
    """speaker_id単位の暫定split。

    voice_cluster_id が未付与のため **これは暫定**であり、P1eのspeaker embeddingが
    揃ってvoiceクラスタが確定したら作り直す必要がある（D-015、R-004）。
    """
    import hashlib

    by_speaker: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_speaker[record.speaker_id].append(index)

    def bucket(speaker: str) -> float:
        digest = hashlib.sha256(f"{seed}:{speaker}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    counts: Counter[str] = Counter()
    for speaker, indices in by_speaker.items():
        value = bucket(speaker)
        if value < zero_shot_fraction / 2:
            speaker_split = "test-zero-shot"
        elif value < zero_shot_fraction:
            speaker_split = "dev-zero-shot"
        else:
            speaker_split = None
        for position, index in enumerate(sorted(indices)):
            if speaker_split is not None:
                split = speaker_split
            else:
                inner = bucket(f"{speaker}#{position}")
                if inner < dev_fraction / 2:
                    split = "test-seen"
                elif inner < dev_fraction:
                    split = "dev-seen"
                else:
                    split = "train"
            records[index] = Utterance(**{**records[index].__dict__, "split": split})
            counts[split] += 1
    return dict(counts)


# ------------------------------------------------------------------ main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P1d: 日本語manifestの生成と検証")
    parser.add_argument("--gol-metadata", default="data/raw/gol/metadata.tsv")
    parser.add_argument("--gol-tars", default="data/raw/gol/tars")
    parser.add_argument("--moe-zips", default="data/raw/moe")
    parser.add_argument("--out-dir", default="data/manifests")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--pair-samples", type=int, default=100)
    parser.add_argument("--skip-full-accounting", action="store_true",
                        help="metadata.tsv 全件のraw/accepted集計を省略する（7.4M行で数分かかる）")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generic_ids = text_rules.generic_speaker_ids()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = artifacts.new_run_dir("p1d", args.artifact_root, timestamp=args.timestamp)
    metrics: dict = {"phase": "p1d", "seed": args.seed}

    print("[1/6] gol 全件の raw/accepted 集計")
    if args.skip_full_accounting:
        metrics["gol_full_accounting"] = {"skipped": True}
        print("      スキップ")
    else:
        metrics["gol_full_accounting"] = gol_full_accounting(
            Path(args.gol_metadata), generic_ids,
            min_duration=args.min_duration, max_duration=args.max_duration,
        )
        acc = metrics["gol_full_accounting"]
        print(f"      raw {acc['raw_utterances']:,} 発話 / {acc['raw_hours']:.1f} h")
        print(f"      accepted {acc['accepted_utterances']:,} 発話 / {acc['accepted_hours']:.1f} h")

    print("[2/6] gol manifest（手元のtarのみ）")
    gol = list(gol_records(Path(args.gol_metadata), Path(args.gol_tars), generic_ids))
    print(f"      {len(gol):,} 発話")

    print("[3/6] moe manifest（手元のzipのみ）")
    moe = list(moe_records(Path(args.moe_zips)))
    print(f"      {len(moe):,} 発話")

    print("[4/6] validate")
    all_records = gol + moe
    issues = validate(all_records, min_duration=args.min_duration,
                      max_duration=args.max_duration, generic_speaker_ids=generic_ids)
    issue_counts = Counter(issue.code for issue in issues)
    bad_ids = {issue.utterance_id for issue in issues}
    accepted = [r for r in all_records if r.utterance_id not in bad_ids]
    metrics["local_manifest"] = {
        "raw": summarize(all_records),
        "accepted": summarize(accepted),
        "validation_counts": dict(issue_counts),
    }
    print(f"      issue {len(issues):,} 件 / accepted {len(accepted):,} 発話")

    print("[5/6] split付与（speaker_id単位・暫定）")
    split_counts = assign_splits(accepted, seed=args.seed)
    metrics["splits"] = split_counts
    metrics["split_note"] = (
        "voice_cluster_id 未付与のため speaker_id 単位の暫定split。"
        "P1eのspeaker embeddingでvoiceクラスタが確定したら作り直すこと（D-015 / R-004）。"
    )
    train_speakers = {r.speaker_id for r in accepted if r.split == "train"}
    zero_shot_speakers = {r.speaker_id for r in accepted
                          if r.split in ("dev-zero-shot", "test-zero-shot")}
    overlap = train_speakers & zero_shot_speakers
    metrics["zero_shot_speaker_overlap"] = len(overlap)
    if overlap:
        raise SystemExit(f"zero-shot splitがtrainと話者を共有している: {len(overlap)} 件")
    print(f"      {split_counts} / zero-shot話者重複 {len(overlap)} 件")

    for name, subset in (("gol", [r for r in accepted if r.dataset_id == "gol"]),
                         ("moe", [r for r in accepted if r.dataset_id == "moe"]),
                         ("all", accepted)):
        path = out_dir / f"{name}.jsonl"
        written = write_manifest(path, subset)
        metrics.setdefault("outputs", {})[name] = {
            "path": str(path), "records": written, "sha256": manifest_checksum(path),
        }

    print("[6/6] pairing の leakage 検査")
    sampler = PairSampler(accepted, seed=args.seed, min_utterances_per_group=2)
    pairs = sampler.sample(args.pair_samples)
    assert_no_leakage(pairs)
    metrics["pairing"] = {
        "eligible_groups": len(sampler.eligible_groups()),
        "sampled_pairs": len(pairs),
        "mean_reference_seconds": (
            sum(p.reference_seconds for p in pairs) / len(pairs) if pairs else 0.0
        ),
        "leakage": False,
    }
    print(f"      {len(pairs)} ペア / leakage なし / "
          f"平均reference {metrics['pairing']['mean_reference_seconds']:.2f} 秒")

    # 片方のdatasetだけを処理することがある（moeだけ足す等）。
    # 入力が無いときは記録をnullにして、artifact書き出しで落とさない。
    gol_metadata = Path(args.gol_metadata)
    artifacts.write_run_metadata(
        run_dir, phase="p1d", command=[Path(sys.argv[0]).name] + sys.argv[1:],
        seed=args.seed,
        inputs={
            "gol_metadata": str(args.gol_metadata),
            "gol_metadata_bytes": (gol_metadata.stat().st_size
                                   if gol_metadata.is_file() else None),
            "gol_tars": sorted(p.name for p in Path(args.gol_tars).glob("*.tar")),
            "moe_zips": sorted(p.name for p in Path(args.moe_zips).glob("*.zip")),
        },
    )
    artifacts.write_metrics(run_dir, metrics)
    print(f"\n完了: {run_dir}")


if __name__ == "__main__":
    main()
