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

"""CER集計と検定のテスト。

S1の判断を誤らせた2点を、ここで固定する。

* 打ち切り生成（`max_decode_length` 張り付き）が発音誤りとして平均に入る
* 検出力を確かめずに 2〜3pt の差を「効果あり」と読む
"""

from __future__ import annotations

import pytest

from cutetts.training.evalstats import (
    align,
    paired_compare,
    required_n,
    summarize,
    truncation_seconds,
)


def _row(cer: float, seconds: float, text: str = "あいうえおかきくけこ") -> dict:
    return {"cer": cer, "seconds": seconds, "text": text, "subset": "in_domain"}


def test_truncation_seconds_matches_the_patch_rate():
    """400 patch ÷ 6.25 patch/s = 64秒。評価の既定値。"""
    assert truncation_seconds(400) == 64.0


def test_summarize_separates_truncated_rows():
    """打ち切り行は別勘定にし、発音品質の平均からは外す。"""
    rows = [_row(0.1, 3.0), _row(0.2, 4.0), _row(0.9, 64.0)]

    result = summarize(rows, label="t")

    assert result.n == 3
    assert result.n_truncated == 1
    assert result.mean == pytest.approx((0.1 + 0.2 + 0.9) / 3)
    assert result.mean_excluding_truncated == pytest.approx(0.15)


def test_summarize_counts_runaway_rows():
    """CER > 1.0 は挿入が参照長を超えた行。数を残す。"""
    result = summarize([_row(0.1, 3.0), _row(1.25, 5.0)], label="t")

    assert result.n_over_one == 1


def test_summarize_micro_weights_by_reference_length():
    """micro CER は長い文の重みが大きい。macro との違いが出ること。"""
    rows = [_row(0.0, 3.0, "あい"), _row(0.5, 3.0, "あいうえおかきくけこ")]

    result = summarize(rows, label="t")

    assert result.mean == pytest.approx(0.25)
    # 誤り 0*2 + 0.5*10 = 5 文字 / 参照 12 文字
    assert result.micro == pytest.approx(5 / 12)


def test_paired_compare_reports_no_significance_for_a_small_difference():
    """ばらつきの大きい少数標本では、2pt程度の差は有意にならない。"""
    a = [0.10, 0.50, 0.20, 0.60, 0.30, 0.45, 0.25, 0.55]
    b = [0.30, 0.30, 0.40, 0.40, 0.50, 0.25, 0.45, 0.35]

    result = paired_compare(a, b)

    assert result.n == 8
    assert not result.significant
    assert result.low <= 0.0 <= result.high


def test_paired_compare_detects_a_consistent_shift():
    """全文が同じ向きに動けば、少数でも有意になる。"""
    a = [0.30, 0.40, 0.50, 0.20, 0.35, 0.45]
    b = [x - 0.10 for x in a]

    result = paired_compare(a, b)

    assert result.significant
    assert result.difference == pytest.approx(-0.10)
    assert result.better == 6
    assert result.worse == 0


def test_paired_compare_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_compare([0.1, 0.2], [0.1])


def test_required_n_grows_with_the_square_of_the_ratio():
    """半分の差を検出するには4倍の文数が要る。"""
    assert required_n(13.5, 4.0) * 4 == pytest.approx(required_n(13.5, 2.0), rel=0.02)


def test_required_n_matches_the_measured_case():
    """実測 sd=13.5pt で 2pt を検出するには 350文以上。"""
    assert required_n(13.5, 2.0) == 358


def test_align_matches_on_text_not_index():
    """評価setが差し替わっても、同じ文どうしを比べること。

    v1→v2 で index 5 以降が1つずれ、別の文を比較していた。
    """
    a = [_row(0.1, 3.0, "いち"), _row(0.2, 3.0, "に"), _row(0.3, 3.0, "さん")]
    b = [_row(0.9, 3.0, "に"), _row(0.8, 3.0, "さん")]

    left, right = align(a, b)

    assert left == [0.2, 0.3]
    assert right == [0.9, 0.8]


def test_align_returns_empty_for_disjoint_sets():
    a = [_row(0.1, 3.0, "いち")]
    b = [_row(0.2, 3.0, "に")]

    assert align(a, b) == ([], [])
