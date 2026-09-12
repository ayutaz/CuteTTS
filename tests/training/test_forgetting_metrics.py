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

"""忘却評価の指標のテスト（R-005）。

英語は語単位のWER、中国語は文字単位のCER。
指標を取り違えると「英語が壊れた」を見逃す。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_forgetting.py"
    spec = importlib.util.spec_from_file_location("evaluate_forgetting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load()


def test_word_error_rate_is_zero_for_identical_text():
    assert module.word_error_rate("the cat sat", "the cat sat") == 0.0


def test_word_error_rate_ignores_case_and_punctuation():
    """ASRは句読点と大文字化が揺れる。それを誤りに数えない。"""
    assert module.word_error_rate("The cat sat.", "the cat sat") == 0.0


def test_word_error_rate_counts_words_not_characters():
    """1語の置換は 1/3。文字単位なら別の値になる。"""
    assert module.word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_word_error_rate_counts_deletion():
    assert module.word_error_rate("the cat sat", "the sat") == pytest.approx(1 / 3)


def test_word_error_rate_can_exceed_one():
    """挿入が参照語数を超えれば1を超える。暴走生成を隠さない。"""
    assert module.word_error_rate("hello", "hello there my old friend") == pytest.approx(4.0)


def test_word_error_rate_returns_none_for_empty_reference():
    assert module.word_error_rate("", "anything") is None


def test_character_error_rate_is_zero_for_identical_text():
    assert module.character_error_rate("今天天气很好", "今天天气很好") == 0.0


def test_character_error_rate_ignores_punctuation():
    assert module.character_error_rate("今天天气很好。", "今天天气很好") == 0.0


def test_character_error_rate_counts_characters():
    assert module.character_error_rate("今天天气很好", "今天天气不好") == pytest.approx(1 / 6)


def test_character_error_rate_returns_none_for_empty_reference():
    assert module.character_error_rate("", "何か") is None


def test_fixed_subsets_have_the_documented_size():
    """評価文は固定。**結果を見てから変更しないこと。**"""
    assert len(module.ENGLISH) == 20
    assert len(module.CHINESE) == 20


def test_english_subset_avoids_digits():
    """数字は分布外の別問題（R-010）。忘却の測定に混ぜない。"""
    assert not any(any(c.isdigit() for c in t) for t in module.ENGLISH)


def test_chinese_subset_avoids_digits():
    assert not any(any(c.isdigit() for c in t) for t in module.CHINESE)
