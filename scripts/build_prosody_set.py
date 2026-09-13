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
CERと同じく、この指標にも**人間を基準にした床**が要る。

素材は `measure_asr_floor` が抽出済みの gol 原音声（`data/eval/asr_floor/`）。
80発話 / 11話者がローカルにあり、テキストも揃っている。

**referenceには同一話者の別発話を使う。** 対象発話そのものを渡すと、
モデルが対象の抑揚を直接聞くことになり、測りたいものが漏れる。

    python scripts/build_prosody_set.py --count 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutetts.training import artifacts  # noqa: E402

#: referenceに使う音声の最短の長さ（秒）。短すぎるとspeaker条件が安定しない。
MIN_REFERENCE_SECONDS = 2.0


def original_name(wav: str) -> str:
    """`0926C280_as02_shizuru_0004.wav` → `as02_shizuru_0004.wav`。

    `measure_asr_floor` が game_id の先頭8文字を前置して保存している。
    """
    name = Path(wav).name
    return name.split("_", 1)[1] if "_" in name else name


def load_speakers(metadata_path: str, games: set[str]) -> dict[tuple[str, str], str]:
    """`(game_id, 元のwav名)` → `speaker_id` を引く。

    **ファイル名から話者を推測してはいけない。** `as02_shizuru_0004.wav` は
    読めるが `z0102#00253.wav` は読めず、game_id で代用すると
    **別の話者を同一人物として扱う**（実測で27人が14人に潰れた）。
    referenceが別話者になると、測りたいものが測れない。

    metadata.tsv は数百万行ある。game_id は先頭32文字に固定長で入るので、
    **分割する前にそこだけ見て捨てる**。
    """
    found: dict[tuple[str, str], str] = {}
    with Path(metadata_path).open(encoding="utf-8", newline="") as handle:
        handle.readline()                       # header
        for line in handle:
            if line[:32] not in games:
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 5:
                continue
            found[(fields[0], Path(fields[3]).name)] = fields[1]
    return found


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抑揚評価のsetを作る（M1）")
    parser.add_argument("--floor-metrics",
                        default="artifacts/asr-floor/2026-08-31T15-49-53/metrics.json")
    parser.add_argument("--audio-dir", default="data/eval/asr_floor")
    parser.add_argument("--gol-metadata", default="data/raw/gol/metadata.tsv")
    parser.add_argument("--out", default="data/eval/prosody_eval_set.json")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("prosody-evalset", args.artifact_root,
                                    timestamp=args.timestamp)

    payload = json.loads(Path(args.floor_metrics).read_text(encoding="utf-8"))
    audio_dir = Path(args.audio_dir)

    games = {str(row.get("game_id")) for row in payload["rows"]}
    print(f"metadata.tsv から {len(games)} game の話者を引く...")
    speakers = load_speakers(args.gol_metadata, games)

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    unresolved = 0
    for row in payload["rows"]:
        wav = Path(row["wav"]).name
        if not (audio_dir / wav).is_file():
            continue
        speaker = speakers.get((str(row.get("game_id")), original_name(wav)))
        if speaker is None:
            unresolved += 1
            continue
        by_speaker[speaker].append({
            "wav": wav, "text": row["text"], "seconds": float(row["seconds"]),
            "group": row.get("group", "plain"), "game_id": row.get("game_id"),
        })

    print(f"{sum(len(v) for v in by_speaker.values())} 発話 / {len(by_speaker)} 話者"
          + (f"（話者を引けなかった発話 {unresolved}）" if unresolved else ""))

    items: list[dict] = []
    skipped_single = 0
    for speaker, rows in sorted(by_speaker.items()):
        if len(rows) < 2:
            # referenceに使える別発話が無い話者は使えない
            skipped_single += 1
            continue
        usable = [r for r in rows if r["seconds"] >= MIN_REFERENCE_SECONDS]
        for target in rows:
            reference = next(
                (r for r in usable if r["wav"] != target["wav"]), None)
            if reference is None:
                continue
            items.append({
                "text": target["text"],
                "speaker": speaker,
                "human_wav": target["wav"],
                "reference_wav": reference["wav"],
                "human_seconds": target["seconds"],
                "reference_seconds": reference["seconds"],
                "group": target["group"],
            })

    def key(item: dict) -> str:
        return hashlib.sha256(
            f"{args.seed}:{item['human_wav']}".encode("utf-8")).hexdigest()

    items.sort(key=key)
    picked = items[:args.count]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": 1,
        "seed": args.seed,
        "created_for": "M1 / 抑揚とアクセントの測定",
        "audio_dir": str(audio_dir),
        "note": (
            "人間の実音声と同一文・同一話者で比べるためのset。"
            "**referenceは同一話者の別発話**（対象発話を渡すと抑揚が漏れる）。"
            "結果を見てから変更しないこと。"
        ),
        "items": picked,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    speakers = {i["speaker"] for i in picked}
    print(f"{len(picked)} 文 / {len(speakers)} 話者")
    if skipped_single:
        print(f"発話が1件しかなく使えなかった話者: {skipped_single}")
    for item in picked[:4]:
        print(f"  {item['speaker']:10s} {item['text'][:30]}")
        print(f"    対象 {item['human_wav']}  ref {item['reference_wav']}")

    artifacts.write_run_metadata(
        run_dir, phase="prosody-evalset",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"floor_metrics": args.floor_metrics, "audio_dir": str(audio_dir)},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "prosody-evalset", "count": len(picked),
        "speakers": len(speakers), "output": str(out),
        "sha256": artifacts.file_checksum(out),
    })
    print(f"\n完了: {out}  sha256 {artifacts.file_checksum(out)[:16]}...")


if __name__ == "__main__":
    main()
