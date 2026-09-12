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

"""CER実行の集計と、2実行の差の検定。

S1では19回の学習の順位を **平均CERだけ** で読み、次の実験を決めていた。
その読み方は2つの点で成立していなかった。

1. **検出力**。in_domain は30文で、対応のある差の標準偏差は約13〜16pt。
   検出できる最小差は約6.9ptで、比較してきた2〜3ptはすべてその下にある。
   `paired_compare` は差と一緒に信頼区間と MDE を返す。

2. **打ち切り生成の混入**。`max_decode_length` に張り付いた生成は停止に
   失敗しているだけで、発音を誤っているわけではない。それがCERとして
   平均に入っていた。S0系は打ち切り0件、S1系は1〜7件あり、除外すると
   両者の差はほぼ消える。`summarize` は打ち切りを別勘定にする。

どちらも「モデルが悪い」と「測り方が悪い」を取り違えさせる。
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

DEFAULT_PATCH_RATE = 6.25
"""LMのtoken rate（12.5 Hz ÷ patch 2）。秒数からpatch数を戻すのに使う。"""

Z80 = 2.802
"""検出力80%・両側5%の係数（1.96 + 0.84）。MDE = Z80 * sd / sqrt(n)。"""


def truncation_seconds(max_decode_length: int, *, patch_rate: float = DEFAULT_PATCH_RATE) -> float:
    """`max_decode_length` に張り付いたときの生成長（秒）。"""
    return max_decode_length / patch_rate


@dataclass(frozen=True)
class RunSummary:
    """1実行のCER集計。割合はすべて 0〜1 で、百分率にしない。"""

    label: str
    n: int
    mean: float
    median: float
    micro: float
    """corpus CER（誤り文字の総数 ÷ 参照文字の総数）。長文の重みが正しく効く。"""
    n_truncated: int
    """`max_decode_length` に張り付いた行。停止の失敗であって発音誤りではない。"""
    n_over_one: int
    """CER が 1.0 を超えた行。挿入が参照長を上回る＝暴走生成。"""
    mean_excluding_truncated: float
    """打ち切り行を除いた平均。発音品質の指標はこちら。"""


def summarize(
    rows: Sequence[dict],
    *,
    label: str = "",
    max_decode_length: int = 400,
    patch_rate: float = DEFAULT_PATCH_RATE,
    tolerance: float = 0.5,
) -> RunSummary:
    """1実行の行から集計を作る。

    ``rows`` は `evaluate_japanese_cer.py` が書く形（`cer` / `seconds` /
    `text` を持つ dict）。`text` があれば micro CER も出す。
    """
    if not rows:
        raise ValueError("rows が空")
    cap = truncation_seconds(max_decode_length, patch_rate=patch_rate) - tolerance
    cers = [float(r["cer"]) for r in rows]
    truncated = [r for r in rows if float(r.get("seconds", 0.0)) >= cap]
    kept = [r for r in rows if float(r.get("seconds", 0.0)) < cap]

    lengths = [len(str(r.get("text", ""))) for r in rows]
    total = sum(lengths)
    micro = (sum(c * n for c, n in zip(cers, lengths)) / total) if total else float("nan")

    kept_cers = [float(r["cer"]) for r in kept]
    return RunSummary(
        label=label,
        n=len(rows),
        mean=statistics.fmean(cers),
        median=statistics.median(cers),
        micro=micro,
        n_truncated=len(truncated),
        n_over_one=sum(1 for c in cers if c > 1.0),
        mean_excluding_truncated=statistics.fmean(kept_cers) if kept_cers else float("nan"),
    )


@dataclass(frozen=True)
class PairedComparison:
    """2実行の対応のある比較。``b`` から ``a`` を引いた向き。"""

    n: int
    difference: float
    sd: float
    low: float
    high: float
    """差の95%信頼区間（対応のあるbootstrap）。"""
    worse: int
    better: int
    unchanged: int
    mde: float
    """この n と sd で検出できる最小の差（検出力80%）。"""
    top_contribution: float
    """差の大きい上位3文が、平均の差に占める割合。1に近いほど少数の文で決まっている。"""

    @property
    def significant(self) -> bool:
        """信頼区間が0を跨がないこと。"""
        return not (self.low <= 0.0 <= self.high)


def paired_compare(
    a: Sequence[float],
    b: Sequence[float],
    *,
    seed: int = 42,
    resamples: int = 10_000,
) -> PairedComparison:
    """同じ文集合で測った2実行を対応のある形で比べる。

    文の難易度は文の主効果として相殺されるので、対応のある比較でなければ
    ならない。独立2標本として扱うと検出力を大きく損なう。
    """
    if len(a) != len(b):
        raise ValueError(f"長さが違う: {len(a)} vs {len(b)}")
    if len(a) < 2:
        raise ValueError("2文以上必要")
    diff = [float(y) - float(x) for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(diff, k=len(diff))) for _ in range(resamples)
    )
    lo = means[int(resamples * 0.025)]
    hi = means[int(resamples * 0.975)]
    sd = statistics.stdev(diff)
    ranked = sorted(diff, key=abs, reverse=True)
    total = statistics.fmean(diff) * len(diff)
    top = (sum(ranked[:3]) / total) if total else float("nan")
    return PairedComparison(
        n=len(diff),
        difference=statistics.fmean(diff),
        sd=sd,
        low=lo,
        high=hi,
        worse=sum(1 for d in diff if d > 0),
        better=sum(1 for d in diff if d < 0),
        unchanged=sum(1 for d in diff if d == 0),
        mde=Z80 * sd / len(diff) ** 0.5,
        top_contribution=top,
    )


def required_n(sd: float, difference: float) -> int:
    """``difference`` を検出力80%で検出するのに要る文数。"""
    if difference <= 0:
        raise ValueError("difference は正の値")
    return int(math.ceil((Z80 * sd / difference) ** 2))


def align(rows_a: Iterable[dict], rows_b: Iterable[dict]) -> tuple[list[float], list[float]]:
    """2実行の行を **テキストで** 対応付ける。

    index で対応付けると、評価setが差し替わったときに別の文どうしを
    比べてしまう（v1→v2 で実際に起きた。index 5以降が1つずれていた）。
    """
    map_a = {str(r["text"]): float(r["cer"]) for r in rows_a}
    map_b = {str(r["text"]): float(r["cer"]) for r in rows_b}
    shared = [t for t in map_a if t in map_b]
    return [map_a[t] for t in shared], [map_b[t] for t in shared]
