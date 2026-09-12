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

"""`artifacts/**/metrics.json` のCER実行を横断で集計し、差に信頼区間を付ける。

S1では平均CERの順位だけで19回の実験を進めたが、その順位は2つの理由で
読めていなかった（どちらもGPU不要で、既存artifactの再集計だけで分かる）。

* in_domain 30文では **6.9pt 未満の差は検出できない**。2〜3ptの比較は
  すべてこの下にあった。
* `max_decode_length` に張り付いた行が発音誤りとして平均に入っていた。
  S0系は打ち切り0件、S1系は1〜7件で、除外すると差はほぼ消える。

    python scripts/summarize_eval_runs.py
    python scripts/summarize_eval_runs.py --compare s0-trained s1v2-3000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutetts.training.evalstats import (  # noqa: E402
    align,
    paired_compare,
    required_n,
    summarize,
)


def load_runs(artifact_root: Path, subset: str) -> dict[str, list[dict]]:
    """`label` -> 行 の対応を作る。同じlabelが複数あれば新しい方を採る。"""
    found: dict[str, tuple[float, list[dict]]] = {}
    for path in sorted(artifact_root.glob("**/metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        label = payload.get("label")
        rows = payload.get("rows")
        if not label or not isinstance(rows, list):
            continue
        picked = [r for r in rows if isinstance(r, dict) and r.get("subset") == subset]
        if not picked:
            continue
        stamp = path.stat().st_mtime
        if label not in found or stamp > found[label][0]:
            found[label] = (stamp, picked)
    return {label: rows for label, (_, rows) in found.items()}


def print_table(runs: dict[str, list[dict]], *, max_decode_length: int) -> None:
    summaries = [
        summarize(rows, label=label, max_decode_length=max_decode_length)
        for label, rows in runs.items()
    ]
    summaries.sort(key=lambda s: s.mean_excluding_truncated)
    print(f"{len(summaries)}実行 / 打ち切り除外後の昇順\n")
    print(f"  {'label':24s} {'n':>3s} {'打切':>5s} {'>1.0':>5s} "
          f"{'mean':>7s} {'median':>7s} {'micro':>7s} {'打切除外':>9s}")
    for s in summaries:
        mark = "  ←" if abs(s.mean_excluding_truncated - s.mean) > 0.015 else ""
        print(f"  {s.label:24s} {s.n:3d} {s.n_truncated:3d}/{s.n:<2d} {s.n_over_one:4d}  "
              f"{s.mean*100:6.2f}% {s.median*100:6.2f}% {s.micro*100:6.2f}% "
              f"{s.mean_excluding_truncated*100:7.2f}%{mark}")
    print("\n  打切 = max_decode_length に張り付いた行（停止の失敗であって発音誤りではない）")
    print("  >1.0 = 挿入が参照長を超えた行（暴走生成）")


def print_comparison(runs: dict[str, list[dict]], a: str, b: str) -> None:
    for label in (a, b):
        if label not in runs:
            raise SystemExit(f"label が見つからない: {label}\n候補: {', '.join(sorted(runs))}")
    left, right = align(runs[a], runs[b])
    if not left:
        raise SystemExit("共通するテキストが無い（評価setが違う）")
    result = paired_compare(left, right)
    print(f"=== {a}  →  {b} ===")
    print(f"  共通 {result.n} 文（indexではなくテキストで対応付け）")
    print(f"  平均 {sum(left)/len(left)*100:.2f}%  →  {sum(right)/len(right)*100:.2f}%")
    print(f"  差   {result.difference*100:+.2f}pt   sd {result.sd*100:.2f}pt")
    print(f"  95%信頼区間 [{result.low*100:+.2f}, {result.high*100:+.2f}] pt")
    print(f"  判定 {'有意' if result.significant else '**有意差なし**'}")
    print(f"  改善 {result.better}文 / 悪化 {result.worse}文 / 同じ {result.unchanged}文")
    print(f"  この n で検出できる最小差 {result.mde*100:.2f}pt")
    print(f"  上位3文が差に占める割合 {result.top_contribution*100:.0f}%")
    print(f"\n  2.0pt を検出するのに必要な文数: {required_n(result.sd*100, 2.0)}")
    print(f"  3.0pt を検出するのに必要な文数: {required_n(result.sd*100, 3.0)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CER実行の横断集計と検定")
    parser.add_argument("--artifact-root", default="artifacts", type=Path)
    parser.add_argument("--subset", default="in_domain")
    parser.add_argument("--max-decode-length", type=int, default=400)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="2実行を対応のある形で比較する")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = load_runs(args.artifact_root, args.subset)
    if not runs:
        raise SystemExit(f"{args.artifact_root} に {args.subset} の行が無い")
    if args.compare:
        print_comparison(runs, *args.compare)
    else:
        print_table(runs, max_decode_length=args.max_decode_length)


if __name__ == "__main__":
    main()
