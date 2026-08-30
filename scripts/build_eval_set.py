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

"""S0 の固定評価set を作る。

**ゲート値を先に固定するため、学習を始める前に必ず作る。**
結果を見てから評価文を選び直せてしまうと、ゲートが意味を失う。

3つのsubsetを作る:

* ``in_domain``      … gol の実テキスト（会話文）。S0の主ゲート
* ``out_of_domain``  … 数字・日付・単位・固有名詞。R-010 の分布外性能
* ``phonetic``       … 促音・撥音・長音・無声化を含む固定文。P1c の積み残し

in_domain は **学習manifestに含まれない発話** から選ぶ（leakage防止）。
選定は seed 固定で決定的。出力は `data/eval/s0_eval_set.json` に固定し、
以降のStageでも同じものを使う。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

from cutetts.training import artifacts, text_rules
from cutetts.training.manifest import load_manifest

csv.field_size_limit(10**9)

# 数字・日付・単位・固有名詞。R-010 で「学習分布の外」と分かっている領域。
OUT_OF_DOMAIN = [
    "今日は八月三十一日、気温は摂氏三十二度です。",
    "会議は午後二時十五分から第三会議室で行います。",
    "東京都渋谷区の人口はおよそ二十三万人です。",
    "価格は千二百八十円、消費税込みです。",
    "新幹線のぞみ百三十七号は九時四十二分に発車します。",
    "この製品の型番はエーピー三千五百二十です。",
    "売上高は前年比百十五パーセントに達しました。",
    "北緯三十五度、東経百三十九度の地点です。",
    "彼は千九百八十七年に大阪府堺市で生まれました。",
    "全長二十三点五メートル、重量は四トンあります。",
    "電話番号は零三の千二百三十四の五千六百七十八です。",
    "第七十二回全国大会は名古屋で開催されます。",
]

# 促音・撥音・長音・無声化。P1c で「未実施」とした音韻条件。
PHONETIC = [
    "切符を買って、はっきりと言った。",           # 促音
    "みんなで新聞を読んだ関心事。",               # 撥音
    "コーヒーとケーキをどうぞ、ゆっくり。",       # 長音
    "ессь",                                       # 置き換え対象（下で除去）
    "白紙の設計図を素直に見つめた。",             # 無声化（し・す・く）
    "ちょっと待って、そこはまっすぐだよ。",       # 促音+長音
    "三年間、九分九厘まで来ていた。",             # 連濁・数詞
    "パンダとペンギンが並んでいる。",             # 撥音+濁音
    "father",                                     # 置き換え対象（下で除去）
    "教育を受けた父は、京都へ行った。",           # 拗音+長音
    "ざあざあ降る雨の中を歩いた。",               # 長音+濁音
    "きっかけは些細な一言だった。",               # 促音
]
PHONETIC = [t for t in PHONETIC if re.search(r"[ぁ-んァ-ヶ一-龥]", t)]


def has_lexical_content(text: str) -> bool:
    """語彙的内容を持つ文か。**CERを一切参照せず、テキストの性質だけで判定する。**

    visual novel のテキストには、言語内容を持たない感情表現が混ざる
    （例: `ふあぁぁぁぁっ、あぁぁ、ああぁぁぁ！`）。
    これを評価文に入れると、ASR転写を比べても TTS の品質を測れない。
    ゲートを歪めるので除外する。

    除外条件（すべてテキストのみから決まる）:

    * 同一文字が4回以上連続する（`ぁぁぁぁ`、`ああああ`）
    * 異なり文字の比率が 0.45 未満（語彙的多様性が低い）
    * 漢字・カタカナが1文字も無い（ひらがなの嘆声だけ）
    """
    stripped = re.sub(r"[\s、。「」『』・…‥！？!?,.\-―ー~〜\"'()（）]", "", text)
    if len(stripped) < 8:
        return False
    if re.search(r"(.){3,}", stripped):
        return False
    if len(set(stripped)) / len(stripped) < 0.45:
        return False
    if not re.search(r"[ァ-ヶ一-龥]", stripped):
        return False
    return True


def pick_in_domain(metadata_tsv: Path, exclude_ids: set[str], *, count: int, seed: int,
                   scan_limit: int = 2_000_000) -> list[dict]:
    """gol の metadata.tsv から、学習に使わない発話を決定的に選ぶ。

    手元のmanifestは取得済みtarに限られるため、評価setは
    **gol全体のmetadata**から選ぶ。こうすると学習データが増えても
    評価setを作り直さずに済む（除外IDで弾く）。
    """
    generic = text_rules.generic_speaker_ids()
    candidates: list[dict] = []
    with metadata_tsv.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="	")
        next(reader, None)
        for i, row in enumerate(reader):
            if i >= scan_limit:
                break
            if len(row) < 5:
                continue
            game, speaker, text, file_path, duration = row[0], row[1], row[2], row[3], row[4]
            utterance_id = f"gol:{game}:{Path(file_path).name}"
            if utterance_id in exclude_ids or speaker in generic:
                continue
            text = text.strip()
            if not (16 <= len(text) <= 40):
                continue
            if text_rules.is_punctuation_only(text) or text_rules.contains_markup(text):
                continue
            if text_rules.has_name_placeholder(text):
                continue
            if not re.search(r"[ぁ-んァ-ヶ一-龥]", text):
                continue
            if not has_lexical_content(text):
                continue
            try:
                seconds = float(duration)
            except ValueError:
                continue
            candidates.append({
                "utterance_id": utterance_id,
                "text": text,
                "speaker_id": speaker,
                "duration": seconds,
            })
    # seed と utterance_id のハッシュで決定的に並べる
    def key(item: dict) -> str:
        return hashlib.sha256(f"{seed}:{item['utterance_id']}".encode("utf-8")).hexdigest()
    candidates.sort(key=key)
    # 話者が偏らないよう、1話者1文に制限してから採る
    seen: set[str] = set()
    picked: list[dict] = []
    for item in candidates:
        if item["speaker_id"] in seen:
            continue
        seen.add(item["speaker_id"])
        picked.append(item)
        if len(picked) >= count:
            break
    return picked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S0 の固定評価setを作る")
    parser.add_argument("--gol-metadata", default="data/raw/gol/metadata.tsv")
    parser.add_argument("--train-manifest", default="data/manifests/all_clustered.jsonl",
                        help="ここに含まれる utterance は in_domain から除外する")
    parser.add_argument("--out", default="data/eval/s0_eval_set.json")
    parser.add_argument("--in-domain-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("s0-evalset", args.artifact_root, timestamp=args.timestamp)

    train_ids = {r.utterance_id for r in load_manifest(args.train_manifest)}
    print(f"学習manifest: {len(train_ids):,} 発話（in_domain から除外する）")

    in_domain = pick_in_domain(Path(args.gol_metadata), train_ids,
                               count=args.in_domain_count, seed=args.seed)
    print(f"in_domain: {len(in_domain)} 文（{len({x['speaker_id'] for x in in_domain})} 話者）")

    payload = {
        "version": 2,
        "seed": args.seed,
        "created_for": "S0",
        "note": (
            "S0のゲート値を固定するための評価set。**結果を見てから変更しないこと。**"
            "in_domain は学習manifestに含まれない発話から、1話者1文で選んでいる。"
            "v2: 語彙的内容を持たない感情表現（`ふあぁぁぁ`等）を除外した。"
            "除外はテキストの性質のみで判定し、CERは一切参照していない。"
        ),
        "subsets": {
            "in_domain": in_domain,
            "out_of_domain": [{"text": t} for t in OUT_OF_DOMAIN],
            "phonetic": [{"text": t} for t in PHONETIC],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts = {k: len(v) for k, v in payload["subsets"].items()}
    print("subset:", counts, "合計", sum(counts.values()), "文")

    artifacts.write_run_metadata(
        run_dir, phase="s0-evalset",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"gol_metadata": args.gol_metadata, "train_manifest": args.train_manifest},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "s0-evalset",
        "counts": counts,
        "total": sum(counts.values()),
        "in_domain_speakers": len({x["speaker_id"] for x in in_domain}),
        "output": str(out),
        "sha256": artifacts.file_checksum(out),
    })
    print(f"\n完了: {out}\n  checksum: {artifacts.file_checksum(out)[:16]}...")


if __name__ == "__main__":
    main()
