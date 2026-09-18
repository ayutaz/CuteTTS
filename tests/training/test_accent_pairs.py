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

"""アクセント最小対の集計（A1 / `scripts/evaluate_accent_pairs.py`）。

**「区別できていない」を「正解」と数えないこと**が要点。
両方同じ核を出したら `pair_differentiated` は 0 でなければならない。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_accent_pairs.py"
_SPEC = importlib.util.spec_from_file_location("a1_eval", _PATH)
a1 = importlib.util.module_from_spec(_SPEC)
sys.modules["a1_eval"] = a1
_SPEC.loader.exec_module(a1)


def _row(pair_id: str, surface: str, expected: int, observed: int) -> dict:
    return {"pair_id": pair_id, "surface": surface, "expected": expected,
            "observed": observed, "kana": "カミ"}


def test_両方正しければ対は正解():
    summary = a1.summarize([_row("p", "神", 1, 1), _row("p", "髪", 2, 2)])
    assert summary["pair_correct"] == 1.0
    assert summary["pair_differentiated"] == 1.0
    assert summary["nucleus_match"] == 1.0


def test_同じ核を出したら区別できていない():
    """**ここが肝。** 片方が偶然当たっても「区別できた」ことにしない。"""
    summary = a1.summarize([_row("p", "神", 1, 1), _row("p", "髪", 2, 1)])
    assert summary["pair_differentiated"] == 0.0
    assert summary["pair_correct"] == 0.0
    assert summary["nucleus_match"] == 0.5


def test_違う核でも辞書と逆なら正解ではない():
    summary = a1.summarize([_row("p", "神", 1, 2), _row("p", "髪", 2, 1)])
    assert summary["pair_differentiated"] == 1.0
    assert summary["pair_correct"] == 0.0


def test_読めなかった対は分母から外す():
    rows = [_row("p", "神", 1, 1), _row("p", "髪", 2, a1.UNDETERMINED),
            _row("q", "花", 2, 2), _row("q", "鼻", 0, 0)]
    summary = a1.summarize(rows)
    assert summary["pairs"] == 1
    assert summary["pairs_incomplete"] == 1
    assert summary["pair_correct"] == 1.0
    assert summary["undetermined"] == 1
    assert summary["determined"] == 3


def test_多数決は判定できた値だけで取る():
    assert a1.majority([1, 1, 2]) == 1
    assert a1.majority([a1.UNDETERMINED, 2, 2]) == 2
    assert a1.majority([a1.UNDETERMINED]) == a1.UNDETERMINED
    assert a1.majority([1, 2]) == 1               # 同数なら小さい方


@pytest.mark.parametrize("values,expected", [([0, 0, 1], 0), ([3], 3)])
def test_多数決の素直な場合(values, expected):
    assert a1.majority(values) == expected


def test_対が片方しかなければ外す():
    summary = a1.summarize([_row("p", "神", 1, 1)])
    assert summary["pairs"] == 0
    assert summary["pairs_incomplete"] == 1
    assert summary["pair_correct"] is None
