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

"""韻律転写の診断set（M3b）を、天井set（M2）から作り替える。

**問いは「モデルは韻律の情報を与えれば使えるのか」。**

M1/M2 の測り方では、参照音声は**同一話者の別の文**だった（韻律の手がかりを
与えない条件）。ここでは**同じ台詞の別テイク**を参照に渡す。

    参照 = テイクA（同じ台詞）
    比較対象 = テイクB（同じ台詞の別テイク）

* モデルの輪郭が **天井（+0.38）に近づく** → 情報を与えれば使える。
  **足りないのは入力の情報**であって、容量や学習量ではない
* **+0.12 から動かない** → 参照から韻律を読んでいない。
  条件づけの経路そのものを作る必要がある（学習の変更）

**通常の評価と混ぜてはいけない。** 参照が同じ台詞なのは診断のための反則で、
実運用では使えない条件（対象文の音声が既にあるなら合成は要らない）。

    python scripts/build_prosody_transfer_set.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutetts.training import artifacts  # noqa: E402


def to_transfer(item: dict) -> dict:
    """天井setの1件を「韻律転写」の1件へ。

    天井setは `human_wav`=テイクA / `take_b_wav`=テイクB /
    `reference_wav`=同一話者の別の文（床）。

    転写setでは **参照をテイクA**、**比較対象をテイクB** にする。
    床（別の文）は `floor_wav` として残す（集計で使う）。
    """
    return {
        "text": item["text"],
        "speaker": item["speaker"],
        "speaker_key": item["speaker_key"],
        "game_id": item["game_id"],
        # 比較対象（人間）= テイクB
        "human_wav": item["take_b_wav"],
        "human_seconds": item["take_b_seconds"],
        # 参照 = テイクA（**同じ台詞**。ここが診断の要）
        "reference_wav": item["human_wav"],
        "reference_seconds": item["human_seconds"],
        # 元の床（同一話者・別の文）。比較のために残す
        "floor_wav": item["reference_wav"],
        "floor_seconds": item["reference_seconds"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="韻律転写の診断setを作る（M3b）")
    parser.add_argument("--ceiling-set", default="data/eval/prosody_ceiling_set_v1.json")
    parser.add_argument("--out", default="data/eval/prosody_transfer_set_v1.json")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("prosody-transfer-set", args.artifact_root,
                                    timestamp=args.timestamp)

    payload = json.loads(Path(args.ceiling_set).read_text(encoding="utf-8"))
    items = [to_transfer(item) for item in payload["items"]]
    speakers = {item["speaker_key"] for item in items}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": 1,
        "seed": payload.get("seed"),
        "created_for": "M3b / 韻律転写の診断（参照＝同じ台詞の別テイク）",
        "audio_dir": payload.get("audio_dir"),
        "derived_from": str(args.ceiling_set),
        "derived_from_sha256": artifacts.file_checksum(args.ceiling_set),
        "note": (
            "**参照音声が対象と同じ台詞**の診断set。"
            "モデルが韻律の情報を与えられたら使えるのかを見る（M3b）。"
            "天井（人間 対 人間）は +0.38、通常条件のモデルは +0.12。"
            "**実運用では使えない条件**なので、通常の評価と混ぜないこと。"
        ),
        "items": items,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"  {len(items)} 件 / {len(speakers)} 話者")
    for item in items[:3]:
        print(f"    {item['speaker_key']} 参照={item['reference_seconds']:.2f}s "
              f"対象={item['human_seconds']:.2f}s  {item['text'][:28]}")
    artifacts.write_run_metadata(
        run_dir, phase="prosody-transfer-set",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=payload.get("seed"),
        inputs={"ceiling_set": args.ceiling_set},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "prosody-transfer-set", "count": len(items),
        "speakers": len(speakers), "output": str(out),
        "sha256": artifacts.file_checksum(out),
    })
    print(f"\n完了: {out}  sha256 {artifacts.file_checksum(out)[:16]}...")


if __name__ == "__main__":
    main()
