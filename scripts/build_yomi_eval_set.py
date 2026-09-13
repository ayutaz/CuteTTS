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

"""J3（読み付与）の効果を測る評価setを作る。

**J3 が置換する語を含む文だけ**を集める。置換が起きない文を混ぜると
効果が薄まり、n を増やした意味が無くなる（J2 のときは 200文中193文が
置換対象で、それでも -11.80pt の検出に n=200 を要した）。

gol の実テキストから、次を満たす文を採る。

* J3 の判定（byte-fallback を含む語）に掛かる語を **1つ以上** 含む
* 学習manifestに **utterance_id でもテキストでも含まれない**
* `has_lexical_content`（語彙的内容を持つ）を通る
* 正規化後16〜50字

**測定の前に凍結する。** 結果を見てから選び直すと、その操作だけで基準線が動く。

    python scripts/build_yomi_eval_set.py --count 300
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutetts.training import artifacts, text_rules  # noqa: E402
from cutetts.training.manifest import load_manifest  # noqa: E402
from cutetts.training.yomi import ReadingAssigner  # noqa: E402

csv.field_size_limit(10**9)

_STRIP = re.compile(r"[\s、。「」『』・…‥！？!?,.\-―ー~〜\"'()（）]")


def _normalized(text: str) -> str:
    import unicodedata

    return _STRIP.sub("", unicodedata.normalize("NFKC", text))


def _has_lexical_content(text: str) -> bool:
    """`build_eval_set.has_lexical_content` と同じ規則。

    語彙的内容を持たない感情表現（`ふあぁぁぁ`）を除く。
    """
    stripped = _STRIP.sub("", text)
    if len(stripped) < 8:
        return False
    if re.search(r"(.)\1{3,}", stripped):
        return False
    if len(set(stripped)) / len(stripped) < 0.45:
        return False
    return bool(re.search(r"[ァ-ヶ一-龥]", stripped))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="J3（読み付与）の評価setを作る")
    parser.add_argument("--gol-metadata", default="data/raw/gol/metadata.tsv")
    parser.add_argument("--train-manifest",
                        default="data/manifests_s1v2/all_clustered.jsonl")
    parser.add_argument("--model-dir", default="model/CuteTTS")
    parser.add_argument("--out", default="data/eval/yomi_eval_set.json")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--scan-limit", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("yomi-evalset", args.artifact_root,
                                    timestamp=args.timestamp)

    assigner = ReadingAssigner.from_model_dir(args.model_dir)

    excluded_ids: set[str] = set()
    excluded_texts: set[str] = set()
    for record in load_manifest(args.train_manifest):
        excluded_ids.add(record.utterance_id)
        excluded_texts.add(_normalized(record.text_raw or ""))
    print(f"学習manifest: {len(excluded_ids):,} 発話 / "
          f"{len(excluded_texts):,} 異なりテキストを除外")

    generic = text_rules.generic_speaker_ids()
    # **話者ごとに1文だけ保持しながら走査する。**
    # 候補を貯めてから絞ると、metadata.tsv の先頭付近の少数話者に偏る
    # （実際に候補1,800件から53話者しか取れなかった）。
    by_speaker: dict[str, dict] = {}
    with Path(args.gol_metadata).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for index, row in enumerate(reader):
            if index >= args.scan_limit:
                break
            if len(row) < 5:
                continue
            game, speaker, text, wav, _ = row[0], row[1], row[2], row[3], row[4]
            text = text.strip()
            utterance_id = f"gol:{game}:{Path(wav).name}"
            if utterance_id in excluded_ids or speaker in generic:
                continue
            if speaker in by_speaker:        # 1話者1文。重複は即座に捨てる
                continue
            if not (16 <= len(text) <= 50):
                continue
            if text_rules.is_punctuation_only(text) or text_rules.contains_markup(text):
                continue
            if text_rules.has_name_placeholder(text):
                continue
            if not _has_lexical_content(text):
                continue
            if _normalized(text) in excluded_texts:
                continue
            spoken = assigner.apply(text)
            if spoken == text:                      # J3 が何もしない文は採らない
                continue
            by_speaker[speaker] = {
                "utterance_id": utterance_id, "text": text, "spoken": spoken,
                "speaker_id": speaker, "replaced": assigner.replaced,
            }

    print(f"候補 {len(by_speaker):,} 話者（J3 が置換する語を含む文を持つ）")
    if len(by_speaker) < args.count:
        raise SystemExit(f"候補が足りない（{len(by_speaker)} / {args.count}）。"
                         "--scan-limit を増やすこと")

    # seed と utterance_id のハッシュで決定的に選ぶ
    def key(item: dict) -> str:
        return hashlib.sha256(
            f"{args.seed}:{item['utterance_id']}".encode("utf-8")).hexdigest()

    picked = sorted(by_speaker.values(), key=key)[:args.count]
    seen = {item["speaker_id"] for item in picked}

    payload = {
        "version": 1,
        "seed": args.seed,
        "created_for": "J3 / D-034",
        "note": (
            "J3（読み付与）の効果を測る評価set。**結果を見てから変更しないこと。**"
            "J3が置換する語を含む文だけを集めてある（置換が起きない文を混ぜると"
            "効果が薄まる）。学習manifestとは utterance_id とテキストの両方で分離済み。"
        ),
        "subsets": {"yomi": [
            {k: v for k, v in item.items() if k != "replaced"} for item in picked]},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")

    total_words = sum(len(item["replaced"]) for item in picked)
    print(f"{len(picked)} 文 / {len(seen)} 話者 / 置換語 {total_words} 件")
    print("例:")
    for item in picked[:5]:
        print(f"  {item['text'][:44]}")
        print(f"    → {'  '.join(f'{s}→{r}' for s, r in item['replaced'])}")

    artifacts.write_run_metadata(
        run_dir, phase="yomi-evalset",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"gol_metadata": args.gol_metadata,
                "train_manifest": args.train_manifest},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "yomi-evalset", "count": len(picked), "speakers": len(seen),
        "replaced_words": total_words, "output": str(out),
        "sha256": artifacts.file_checksum(out),
    })
    print(f"\n完了: {out}  sha256 {artifacts.file_checksum(out)[:16]}...")


if __name__ == "__main__":
    main()
