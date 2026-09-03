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

"""数詞の読みを測る評価setを作る（J2 / D-008）。

既存の `out_of_domain` は **12文** しかなく、J2（漢数字の読み展開）の効果
（-18.76pt）が 95%CI [-39.81, +1.70] で有意にならなかった。
効果量は大きいのに文数が足りない。in_domain を 600文にしたのと同じ拡張が要る。

templateと数値を直積して決定的に生成する。**測定の前に凍結し、
結果を見てから変更しない**（評価setをCERを見た後に差し替えると、
その操作だけで基準線が動く。実測で5.2pt動いた）。

読みは `cutetts.training.reading` が生成するので、参照読みも同時に記録する。
CERは `to_arabic_numerals` で両側を正規化してから測る（素のCERは
「1280円」と「千二百八十円」を不一致と数えるため、正しく読めるほど悪化する）。

    python scripts/build_numeral_eval_set.py --count 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutetts.training import artifacts  # noqa: E402
from cutetts.training.reading import expand_kanji_numerals, read_number  # noqa: E402

#: (template, 数値の範囲, 種別)。`{n}` に漢数字が入る。
TEMPLATES: list[tuple[str, tuple[int, int], str]] = [
    ("価格は{n}円、消費税込みです。", (100, 99_999), "金額"),
    ("この本は{n}円で買えます。", (500, 9_999), "金額"),
    ("会議室の定員は{n}名です。", (10, 999), "人数"),
    ("会場にはおよそ{n}人が集まりました。", (100, 99_999), "人数"),
    ("この町の人口は{n}人ほどです。", (1_000, 999_999), "人数"),
    ("気温は摂氏{n}度まで上がりました。", (1, 45), "温度"),
    ("湿度は{n}パーセントを超えています。", (10, 99), "割合"),
    ("売上は前年比{n}パーセントに達しました。", (50, 300), "割合"),
    ("全長は{n}メートルあります。", (2, 9_999), "長さ"),
    ("駅までおよそ{n}キロメートルです。", (1, 500), "長さ"),
    ("重さは{n}キログラムほどでした。", (2, 9_999), "重量"),
    ("箱の中には{n}個入っています。", (2, 999), "個数"),
    ("図書室には{n}冊の本があります。", (100, 99_999), "個数"),
    ("彼は{n}年に生まれました。", (1900, 2025), "年"),
    ("創業から{n}年が経ちました。", (2, 300), "年"),
    ("第{n}回の大会が開かれます。", (2, 999), "順序"),
    ("次の列車は{n}号車です。", (2, 20), "順序"),
    ("受付番号は{n}番になります。", (2, 9_999), "順序"),
    ("この道路の制限速度は{n}キロです。", (20, 120), "速度"),
    ("会員は{n}人を超えました。", (1_000, 9_999_999), "人数"),
]

#: 桁ごとの読みを網羅するための数値。範囲サンプルだけだと桁が偏る。
ANCHORS = [11, 13, 15, 20, 23, 30, 31, 45, 60, 72, 80, 99,
           100, 101, 115, 137, 139, 300, 320, 600, 608, 800, 999,
           1000, 1280, 1987, 3520, 5678, 8000,
           10_000, 15_000, 23_000, 230_000, 1_000_000]


def build_items(count: int, seed: int) -> list[dict]:
    """templateと数値を組み合わせて決定的に生成する。"""
    rng = random.Random(seed)
    items: list[dict] = []
    seen: set[str] = set()

    # まず桁を網羅するアンカーを置く。**値域が合うtemplateだけを使う**
    # （`彼は百万年に生まれました` のような不自然な文を作らない）。
    for value in ANCHORS:
        fits = [t for t in TEMPLATES if t[1][0] <= value <= t[1][1]]
        if not fits:
            continue
        template, _, kind = fits[rng.randrange(len(fits))]
        items.append(_make(template, value, kind))

    # 残りは範囲から引く
    while len(items) < count:
        template, (low, high), kind = TEMPLATES[rng.randrange(len(TEMPLATES))]
        value = rng.randint(low, high)
        item = _make(template, value, kind)
        if item["text"] in seen:
            continue
        seen.add(item["text"])
        items.append(item)

    for item in items:
        seen.add(item["text"])
    return items[:count]


def _make(template: str, value: int, kind: str) -> dict:
    kanji = _to_kanji(value)
    text = template.format(n=kanji)
    spoken = expand_kanji_numerals(text)
    return {
        "text": text,
        "value": value,
        "kind": kind,
        "spoken": spoken,
        "reading": read_number(value),
        # 単独の位（`百` `千`）は `十分` `百姓` `千葉` を壊さないため展開しない。
        # 展開が起きない文はJ2の効果がゼロなので、集計で分けられるようにする。
        "expanded": spoken != text,
    }


_DIGITS = "〇一二三四五六七八九"
_SMALL = [(1000, "千"), (100, "百"), (10, "十")]


def _to_kanji(value: int) -> str:
    """整数を漢数字にする。読みの生成と逆向きの変換。"""
    if value == 0:
        return "〇"
    parts: list[str] = []
    for unit, name in ((10**8, "億"), (10**4, "万")):
        section, value = divmod(value, unit)
        if section:
            parts.append(_below_10000(section) + name)
    if value:
        parts.append(_below_10000(value))
    return "".join(parts)


def _below_10000(value: int) -> str:
    parts: list[str] = []
    for unit, name in _SMALL:
        digit, value = divmod(value, unit)
        if not digit:
            continue
        parts.append(name if digit == 1 else _DIGITS[digit] + name)
    if value:
        parts.append(_DIGITS[value])
    return "".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="数詞の読みを測る評価setを作る")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--out", default="data/eval/numeral_eval_set.json")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("numeral-evalset", args.artifact_root,
                                    timestamp=args.timestamp)
    items = build_items(args.count, args.seed)

    payload = {
        "version": 1,
        "seed": args.seed,
        "created_for": "J2 / D-008",
        "note": (
            "漢数字の読みを測る評価set。**結果を見てから変更しないこと。**"
            "既存の out_of_domain は12文しかなく、J2の効果 -18.76pt が"
            "95%CI [-39.81,+1.70] で有意にならなかった。"
            "CERは to_arabic_numerals で両側を正規化してから測る。"
        ),
        "subsets": {"numerals": items},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")

    kinds: dict[str, int] = {}
    for item in items:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    print(f"{len(items)} 文  種別: {kinds}")
    expanded = sum(1 for i in items if i["expanded"])
    print(f"読み展開が起きる文: {expanded}/{len(items)}"
          f"（残りは `百` `千` の単独で、語を壊さないため展開しない）")
    print(f"桁の分布: " + ", ".join(
        f"{d}桁 {sum(1 for i in items if len(str(i['value'])) == d)}"
        for d in range(1, 9) if any(len(str(i["value"])) == d for i in items)))
    for item in items[:5]:
        print(f"  {item['text']}")
        print(f"    → {item['spoken']}")

    artifacts.write_run_metadata(
        run_dir, phase="numeral-evalset",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"count": str(args.count)},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "numeral-evalset", "count": len(items), "kinds": kinds,
        "expanded": expanded,
        "output": str(out), "sha256": artifacts.file_checksum(out),
    })
    print(f"\n完了: {out}  sha256 {artifacts.file_checksum(out)[:16]}...")


if __name__ == "__main__":
    main()
