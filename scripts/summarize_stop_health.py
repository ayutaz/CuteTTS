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

"""停止の健全性を既存の評価runから集計する（G1）。**GPU不要。**

R-021 は `max_decode_length` への張り付き（打ち切り）だけを別勘定にした。
しかし試聴では**打ち切りに至らない「喋り続け」が9文中4文**で出た。
これは発音の誤りではなく停止の失敗だが、CERには誤りとして入る。

**保存済みの転写を読むだけなので、生成をやり直さない。**

    python scripts/summarize_stop_health.py
    python scripts/summarize_stop_health.py --root artifacts/t2 --subset in_domain

`--compare A B` で2つのrunを**対応のある検定**で比べる（テキストで対応付ける）。
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutetts.training.evalstats import paired_compare  # noqa: E402
from cutetts.training.stopping import (  # noqa: E402
    excess_chars,
    has_extra_tail,
    has_self_repeat,
    stop_health,
)


def load_runs(root: str) -> dict[str, dict]:
    """`<root>/**/s0-cer/*/metrics.json` を読んで label → payload にする。"""
    runs: dict[str, dict] = {}
    for path in sorted(glob.glob(f"{root}/**/s0-cer/*/metrics.json", recursive=True)):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        label = str(payload.get("label") or "")
        if not label or not payload.get("rows"):
            continue
        # shard は結合済みのものだけを見る
        if payload.get("shard"):
            continue
        runs[label] = payload
    return runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="停止の健全性を集計する（G1）")
    parser.add_argument("--root", default="artifacts")
    parser.add_argument("--subset", default="in_domain")
    parser.add_argument("--min-n", type=int, default=100,
                        help="この件数未満のrunは表示しない（率が安定しない）")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="2つのlabelを対応のある検定で比べる")
    parser.add_argument("--examples", type=int, default=0,
                        help="余計な尾の実例をこの件数だけ表示する")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"run が無い: {args.root}/**/s0-cer/*/metrics.json")

    subsets = (args.subset,) if args.subset else None
    rows = []
    for label, payload in runs.items():
        health = stop_health(payload["rows"], label=label, subsets=subsets)
        if health.n >= args.min_n:
            rows.append(health)
    rows.sort(key=lambda h: h.extra_tail_rate)

    print(f"  subset={args.subset}  （**喋り続け率** が主指標。打切は R-021 の別勘定）\n")
    print(f"  {'label':26s} {'n':>4s} {'喋り続け':>8s} {'自己反復':>8s} "
          f"{'打切':>6s} {'超過(平均)':>10s} {'超過(最大)':>10s}")
    for health in rows:
        print(f"  {health.label:26s} {health.n:4d} "
              f"{health.extra_tail_rate:7.1%} {health.self_repeat_rate:7.1%} "
              f"{health.truncated_rate:5.1%} {health.mean_excess_chars:9.1f}字 "
              f"{health.max_excess_chars:9d}字")

    if args.compare:
        a, b = args.compare
        for label in (a, b):
            if label not in runs:
                raise SystemExit(f"label が無い: {label}")

        def flags(label: str) -> dict[str, float]:
            out = {}
            for row in runs[label]["rows"]:
                if row.get("status") != "ok" or row.get("hypothesis") is None:
                    continue
                if subsets and row.get("subset") not in subsets:
                    continue
                out[row["text"]] = float(
                    has_extra_tail(row["text"], row["hypothesis"]))
            return out

        table_a, table_b = flags(a), flags(b)
        shared = sorted(set(table_a) & set(table_b))
        result = paired_compare([table_a[t] for t in shared],
                                [table_b[t] for t in shared])
        verdict = ("**有意に良い**" if result.significant and result.difference < 0
                   else "**有意に悪い**" if result.significant else "有意差なし")
        print(f"\n  {a} → {b}: {result.difference*100:+.2f}pt "
              f"CI[{result.low*100:+.2f},{result.high*100:+.2f}]  {verdict}  "
              f"（検出限界 {result.mde*100:.2f}pt, n={result.n}）")

    if args.examples:
        print(f"\n  === 余計な尾の実例（{args.examples}件まで）===")
        shown = 0
        for label, payload in runs.items():
            for row in payload["rows"]:
                if shown >= args.examples:
                    return
                if row.get("status") != "ok" or row.get("hypothesis") is None:
                    continue
                if subsets and row.get("subset") not in subsets:
                    continue
                if not has_extra_tail(row["text"], row["hypothesis"]):
                    continue
                mark = " 自己反復" if has_self_repeat(row["hypothesis"]) else ""
                print(f"  [{label}] +{excess_chars(row['text'], row['hypothesis'])}字{mark}")
                print(f"    参照: {row['text']}")
                print(f"    転写: {row['hypothesis']}")
                shown += 1


if __name__ == "__main__":
    main()
