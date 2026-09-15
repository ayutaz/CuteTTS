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

"""評価runの集計のテスト。

**集計をライブラリに置いた理由は `--shard` で分割して測るため。**
分割した行を結合して集計し直すと一括と一致しなければならない。
集計をscript側に書いたままだと、結合用にもう一度書くことになり必ずずれる。
"""

from __future__ import annotations

import math

import pytest

from cutetts.training.evalstats import summarize_subsets
from cutetts.training.prosody import summarize_run


def cer_row(subset, index, cer, *, reading=None, numeric=None, status="ok"):
    return {"subset": subset, "index": index, "cer": cer,
            "cer_reading": reading, "cer_numeric": numeric, "status": status}


def prosody_row(index, human_range, model_range, similarity, floor, *,
                accent=None, status="ok"):
    return {
        "index": index, "status": status,
        "human": {"semitone_range": human_range, "semitone_sd": 1.0,
                  "seconds": 2.0},
        "model": {"semitone_range": model_range, "semitone_sd": 1.0,
                  "seconds": 2.0},
        "contour_similarity": similarity, "floor_similarity": floor,
        "accent": accent,
    }


# ---- CER ----


def test_subset_summary_counts_only_successful_rows():
    rows = [cer_row("a", 0, 0.1), cer_row("a", 1, 0.3),
            cer_row("a", 2, None, status="error")]

    summary = summarize_subsets(rows, ["a"])

    assert summary["a"]["n"] == 2
    assert summary["a"]["cer_mean"] == pytest.approx(0.2)
    assert summary["a"]["cer_min"] == pytest.approx(0.1)
    assert summary["a"]["cer_max"] == pytest.approx(0.3)


def test_subset_summary_is_independent_of_row_order():
    """**分割して測った行を結合しても同じ集計になること。**

    `--shard` は飛び飛びに担当を割るので、結合後の並びは一括と同じ順序に
    なるが、集計が順序に依存していないことは押さえておく。
    """
    rows = [cer_row("a", i, cer) for i, cer in enumerate([0.1, 0.5, 0.2, 0.4])]

    forward = summarize_subsets(rows, ["a"])
    backward = summarize_subsets(list(reversed(rows)), ["a"])

    assert forward == backward


def test_subset_summary_separates_subsets():
    rows = [cer_row("a", 0, 0.1), cer_row("b", 0, 0.9)]

    summary = summarize_subsets(rows, ["a", "b"])

    assert summary["a"]["cer_mean"] == pytest.approx(0.1)
    assert summary["b"]["cer_mean"] == pytest.approx(0.9)


def test_subset_with_no_usable_rows_reports_zero():
    summary = summarize_subsets([cer_row("a", 0, None, status="error")], ["a"])

    assert summary["a"] == {"n": 0}


def test_reading_cer_is_averaged_separately():
    rows = [cer_row("a", 0, 0.4, reading=0.1), cer_row("a", 1, 0.6, reading=0.3)]

    summary = summarize_subsets(rows, ["a"])

    assert summary["a"]["cer_mean"] == pytest.approx(0.5)
    assert summary["a"]["cer_reading_mean"] == pytest.approx(0.2)


def test_missing_reading_cer_is_none_not_zero():
    """`pyopenjtalk` が無い環境では読みCERが付かない。**0と混同しないこと。**"""
    summary = summarize_subsets([cer_row("a", 0, 0.4)], ["a"])

    assert summary["a"]["cer_reading_mean"] is None


# ---- 抑揚 ----


def test_prosody_summary_counts_and_averages():
    rows = [prosody_row(0, 10.0, 8.0, 0.2, 0.0),
            prosody_row(1, 12.0, 14.0, 0.4, 0.1)]

    summary = summarize_run(rows)

    assert summary["n"] == 2
    assert summary["human_semitone_range"] == pytest.approx(11.0)
    assert summary["model_semitone_range"] == pytest.approx(11.0)
    assert summary["contour_similarity_mean"] == pytest.approx(0.3)
    assert summary["n_flatter_than_human"] == 1


def test_prosody_summary_drops_rows_without_f0():
    """有声フレームが足りない発話は NaN で入ってくる。分母から外す。"""
    rows = [prosody_row(0, 10.0, 8.0, 0.2, 0.0),
            prosody_row(1, float("nan"), 8.0, 0.2, 0.0)]

    summary = summarize_run(rows)

    assert summary["n"] == 1


def test_prosody_summary_compares_against_the_floor():
    """**床との差が有意かを集計に含める。** 床は文ごとに測ってある。"""
    rows = [prosody_row(i, 10.0, 10.0, 0.5, 0.0) for i in range(12)]

    summary = summarize_run(rows)

    assert summary["above_floor"]["difference"] == pytest.approx(0.5)
    assert summary["above_floor"]["significant"] is True


def test_prosody_summary_is_independent_of_row_order():
    rows = [prosody_row(i, 10.0 + i, 9.0 + i, 0.1 * i, 0.0) for i in range(6)]

    forward = summarize_run(rows)
    backward = summarize_run(list(reversed(rows)))

    assert forward["contour_similarity_mean"] == pytest.approx(
        backward["contour_similarity_mean"])
    assert forward["n_flatter_than_human"] == backward["n_flatter_than_human"]


def test_accent_rates_skip_unmeasurable_phrases():
    """核が読めなかった句（-1）は分母から外す。"""
    accent = {"expected": [1, 2, 0], "human": [1, -1, 0], "model": [1, 2, 1]}
    rows = [prosody_row(0, 10.0, 10.0, 0.1, 0.0, accent=accent)]

    summary = summarize_run(rows)

    # human が -1 の句は落ちるので分母は2
    assert summary["accent"]["model_vs_human"]["n"] == 2
    assert summary["accent"]["model_vs_human"]["rate"] == pytest.approx(0.5)


def test_accent_chance_level_comes_from_the_human_distribution():
    """当てずっぽうの水準は**人間側の分布**から出す。"""
    accent = {"expected": [0, 0], "human": [0, 0], "model": [0, 1]}
    rows = [prosody_row(0, 10.0, 10.0, 0.1, 0.0, accent=accent)]

    summary = summarize_run(rows)

    assert summary["accent"]["chance"] == pytest.approx(1.0)
    assert summary["accent"]["constant"] == pytest.approx(1.0)


def test_summary_without_accent_has_no_accent_key():
    rows = [prosody_row(0, 10.0, 10.0, 0.1, 0.0)]

    assert "accent" not in summarize_run(rows)


def test_empty_rows_report_zero_not_crash():
    summary = summarize_run([])

    assert summary["n"] == 0
    assert math.isnan(summary["human_semitone_range"] or float("nan"))
