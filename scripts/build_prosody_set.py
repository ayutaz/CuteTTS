# Copyright 2026 ayutaz
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

"""抑揚・アクセント評価（M1）のsetを作る。

**人間の実音声と同一文・同一話者で比べる**ためのペアを作る。
CERと同じく、この指標にも人間を基準にした床が要る。

## n を先に決める

初回の n=67 では**検出できる最小差が 0.084** で、観測した「床との差
+0.027」はその下にあった（[R-032](../docs/japanese-training/risks-and-decisions.md)）。
**0.05 の差を検出するには n=189 が要る。** 既定は 240（余裕を持たせる）。

## 規約

* **話者は `metadata.tsv` から引く。** ファイル名から推測すると
  `z0102#00253.wav` 形式で失敗し、game_id で代用して**別話者を同一人物
  として扱う**（実測で27人が14人に潰れた）。
* **referenceは話者ごとに1つ固定**（その話者の最長発話）。対象ごとに
  替えると条件が増える。**対象発話そのものは絶対に使わない**
  （モデルが対象の抑揚を直接聞くことになる）。
* **学習manifestと重複させない。** 暗記した発話で抑揚を測ると値が膨らむ。
  ローカルの6 gameは学習の8 gameと重複していないが、明示的に確認する。
* **測定の前に凍結する。** 結果を見てから選び直すと、その操作だけで基準線が動く。

    python scripts/build_prosody_set.py --count 240
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutetts.training import artifacts, text_rules  # noqa: E402
from cutetts.training.manifest import load_manifest  # noqa: E402

#: referenceに使う音声の最短の長さ（秒）。短すぎるとspeaker条件が安定しない。
MIN_REFERENCE_SECONDS = 3.0

#: 対象発話の長さの範囲（秒）。短いとF0のフレームが足りず、
#: 長すぎると生成に時間がかかる。
MIN_TARGET_SECONDS = 2.0
MAX_TARGET_SECONDS = 15.0

#: 1話者から採る対象の上限。少数の話者に偏らせない。
MAX_PER_SPEAKER = 6

_STRIP = re.compile(r"[\s、。「」『』・…‥！？!?,.\-―ー~〜\"'()（）]")


def has_lexical_content(text: str) -> bool:
    """語彙的内容を持つか。`build_eval_set` と同じ規則。

    `ふあぁぁぁ` のような感情表現は抑揚の比較に向かない
    （読みも抑揚も規則が効かない）。
    """
    stripped = _STRIP.sub("", unicodedata.normalize("NFKC", text))
    if len(stripped) < 10:
        return False
    if re.search(r"(.)\1{3,}", stripped):
        return False
    if len(set(stripped)) / len(stripped) < 0.45:
        return False
    return bool(re.search(r"[ァ-ヶ一-龥]", stripped))


def local_name(game_id: str, speaker_id: str, file_path: str) -> str:
    """抽出後のファイル名。game と話者が名前から分かるようにする。"""
    return f"{game_id[:8]}_{speaker_id[:8]}_{Path(file_path).name}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抑揚評価のsetを作る（M1）")
    parser.add_argument("--gol-metadata", default="data/raw/gol/metadata.tsv")
    parser.add_argument("--tar-dir", default="data/raw/gol/tars")
    parser.add_argument("--train-manifest",
                        default="data/manifests_s1v2/all_clustered.jsonl")
    parser.add_argument("--audio-dir", default="data/eval/prosody_audio")
    parser.add_argument("--out", default="data/eval/prosody_eval_set_v2.json")
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("prosody-evalset", args.artifact_root,
                                    timestamp=args.timestamp)

    tar_dir = Path(args.tar_dir)
    games = {path.stem for path in tar_dir.glob("*.tar")}
    if not games:
        raise SystemExit(f"tar が無い: {tar_dir}")
    print(f"ローカルtar {len(games)} game")

    excluded = {record.utterance_id for record in load_manifest(args.train_manifest)}
    print(f"学習manifest {len(excluded):,} 発話を除外対象として読み込み")

    by_speaker: dict[tuple[str, str], list[dict]] = defaultdict(list)
    overlap = 0
    with Path(args.gol_metadata).open(encoding="utf-8", newline="") as handle:
        handle.readline()
        for line in handle:
            if line[:32] not in games:
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 5:
                continue
            game_id, speaker_id, text, file_path, duration = fields[:5]
            utterance_id = f"gol:{game_id}:{Path(file_path).name}"
            if utterance_id in excluded:
                overlap += 1
                continue
            try:
                seconds = float(duration)
            except ValueError:
                continue
            # **話者は (game, speaker) で括る。** golの話者IDは表示名の
            # SHA-256 なので、同名キャラが別作品にいると同じIDになる
            # （別の声優かもしれない）。game をまたいで reference を選ぶと
            # **別の声を基準にして抑揚を測る**ことになる。
            by_speaker[(game_id, speaker_id)].append({
                "game_id": game_id, "speaker_id": speaker_id, "text": text.strip(),
                "file_path": file_path, "seconds": seconds,
            })

    print(f"{sum(len(v) for v in by_speaker.values()):,} 発話 / {len(by_speaker)} 話者"
          f"（学習と重複して除外 {overlap}）")

    def usable_target(row: dict) -> bool:
        if not (MIN_TARGET_SECONDS <= row["seconds"] <= MAX_TARGET_SECONDS):
            return False
        text = row["text"]
        if text_rules.is_punctuation_only(text) or text_rules.contains_markup(text):
            return False
        if text_rules.has_name_placeholder(text):
            return False
        return has_lexical_content(text)

    items: list[dict] = []
    for (game_id, speaker_id), rows in sorted(by_speaker.items()):
        if len(rows) < 2:
            continue
        # referenceは話者ごとに1つ固定（最長発話）。対象からは必ず外す。
        reference = max(rows, key=lambda r: r["seconds"])
        if reference["seconds"] < MIN_REFERENCE_SECONDS:
            continue
        candidates = [r for r in rows
                      if r["file_path"] != reference["file_path"] and usable_target(r)]

        def key(row: dict) -> str:
            return hashlib.sha256(
                f"{args.seed}:{row['file_path']}".encode("utf-8")).hexdigest()

        for target in sorted(candidates, key=key)[:MAX_PER_SPEAKER]:
            items.append({
                "text": target["text"],
                "speaker": speaker_id,
                # **話者の同一性は (game, speaker) で決まる。** 話者IDだけで
                # 括ると、同名キャラの別作品を同一人物として数えてしまう
                # （実測で 77 → 85 に増えた）。集計はこちらで行う。
                "speaker_key": f"{game_id[:8]}:{speaker_id[:8]}",
                "game_id": target["game_id"],
                "human_wav": local_name(target["game_id"], speaker_id,
                                        target["file_path"]),
                "reference_wav": local_name(reference["game_id"], speaker_id,
                                            reference["file_path"]),
                "human_seconds": target["seconds"],
                "reference_seconds": reference["seconds"],
                "_target_path": target["file_path"],
                "_reference_path": reference["file_path"],
            })

    def item_key(item: dict) -> str:
        return hashlib.sha256(
            f"{args.seed}:{item['_target_path']}".encode("utf-8")).hexdigest()

    items.sort(key=item_key)
    picked = items[:args.count]
    if len(picked) < args.count:
        print(f"**候補が {len(picked)} 件しかない**（要求 {args.count}）")

    wanted: dict[str, set[str]] = defaultdict(set)
    for item in picked:
        # **抽出元のtarは path 自身の game で決める。** item の game を
        # 使うと、別gameの reference を対象のtarから探して取りこぼす。
        wanted[item["_target_path"][:32]].add(item["_target_path"])
        wanted[item["_reference_path"][:32]].add(item["_reference_path"])
    print(f"{len(picked)} 文 / {len({i['speaker_key'] for i in picked})} 話者 → "
          f"音声 {sum(len(v) for v in wanted.values())} 本を抽出")

    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for game_id, paths in sorted(wanted.items()):
        # tar内は game_id を除いた相対path
        inside = {path.split("/", 1)[1] if "/" in path else path: path
                  for path in paths}
        with tarfile.open(tar_dir / f"{game_id}.tar") as archive:
            for member in archive:
                if member.name not in inside:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                speaker = Path(inside[member.name]).parent.name
                target = audio_dir / local_name(game_id, speaker, member.name)
                target.write_bytes(source.read())
                extracted += 1
        print(f"  {game_id[:8]} … {extracted} 本")

    missing = [item for item in picked
               if not (audio_dir / item["human_wav"]).is_file()
               or not (audio_dir / item["reference_wav"]).is_file()]
    if missing:
        raise SystemExit(f"抽出できなかった音声が {len(missing)} 件ある")

    for item in picked:                     # 内部用のpathは残さない
        item.pop("_target_path", None)
        item.pop("_reference_path", None)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": 2,
        "seed": args.seed,
        "created_for": "M1 / 抑揚とアクセントの測定",
        # **`/` で書く。** Windowsで `\` を書くとLinuxで1つのファイル名に
        # なり、`data\eval\prosody_audio` というディレクトリが作られる
        "audio_dir": audio_dir.as_posix(),
        "note": (
            "人間の実音声と同一文・同一話者で比べるためのset。"
            "**referenceは話者ごとに1つ固定した別発話**（対象発話を渡すと抑揚が漏れる）。"
            "学習manifestとは game 単位で重複しない。"
            "n は検出力から決めた（n=67 では検出限界 0.084 で判定できなかった。R-032）。"
            "**結果を見てから変更しないこと。**"
        ),
        "items": picked,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    speakers = {item["speaker_key"] for item in picked}
    print(f"\n{len(picked)} 文 / {len(speakers)} 話者")
    counts = defaultdict(int)
    for item in picked:
        counts[item["speaker_key"]] += 1
    print(f"1話者あたり {min(counts.values())}〜{max(counts.values())} 文")
    for item in picked[:3]:
        print(f"  {item['speaker'][:8]} {item['human_seconds']:5.2f}s  {item['text'][:32]}")
        print(f"    ref {item['reference_wav']}  ({item['reference_seconds']:.1f}s)")

    artifacts.write_run_metadata(
        run_dir, phase="prosody-evalset",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"gol_metadata": args.gol_metadata, "tar_dir": str(tar_dir),
                "train_manifest": args.train_manifest},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "prosody-evalset", "count": len(picked),
        "speakers": len(speakers), "extracted_audio": extracted,
        "train_overlap_excluded": overlap, "output": str(out),
        "sha256": artifacts.file_checksum(out),
    })
    print(f"\n完了: {out}  sha256 {artifacts.file_checksum(out)[:16]}...")


if __name__ == "__main__":
    main()
