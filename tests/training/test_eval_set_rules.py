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

"""評価set構築の除外規則のテスト。

`has_lexical_content` は3条件で文を落とす。このうち第1条件（同一文字の
4回以上連続）は、正規表現に後方参照 `\\1` ではなく **リテラル 0x01 バイト**が
入っていたため一度も機能していなかった。実効パターンは `(.)\\x01{3,}` で、
日本語文には決して一致しない。

その結果、S0の評価setから喘ぎ声3文が落ちたのは第2条件（異なり文字比率 0.45）
だけの働きであり、この閾値は **基準線CERを見た後に選ばれている**。
規則ごとに陽性・陰性の例を固定して、同じことを繰り返さないようにする。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_eval_set.py"
    spec = importlib.util.spec_from_file_location("build_eval_set", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


has_lexical_content = _load().has_lexical_content


@pytest.mark.parametrize("text", [
    # 第1条件のみで落ちる: 漢字あり・異なり文字比率も高いが「ええええ」が4連続
    "本日は八月三十一日、気温は摂氏三十二度ですねええええ",
    "そこの角を右に曲がって、まっすぐ進んでくださいねええええ",
])
def test_repeated_character_rule_is_effective(text):
    """0x01 混入で死んでいた第1条件が実際に効くこと。

    これらは第2条件（比率0.45）・第3条件（漢字カタカナ）を通過するので、
    第1条件が働かなければ False にならない。
    """
    assert has_lexical_content(text) is False


@pytest.mark.parametrize("text", [
    "ふあぁぁぁぁっ、あぁぁ、ああぁぁぁ！んくぁぁぁぁぁぁぁぁーッ！",
    "ん、ふぁ、あ、あああああぁぁぁぁ……！",
    "うぅぅんっ、ぁ、あっ、すご、いっ、ぁ、あぁぁっ！！",
])
def test_non_verbal_utterances_are_excluded(text):
    """S0 の評価set v2 で除外された3文は、修正後も除外されること。"""
    assert has_lexical_content(text) is False


@pytest.mark.parametrize("text", [
    "でも大丈夫。気持ちは届いてるからさ、ここに",
    "今日の天気は晴れで、気温は二十五度になりますよ",
    "だったら私たちも若葉ちゃんのうちに行く！",
    "それがさくらさんの希望的観測にすぎないって可能性もありますよね？",
])
def test_ordinary_sentences_are_kept(text):
    assert has_lexical_content(text) is True


def test_too_short_is_excluded():
    assert has_lexical_content("はい。") is False


def test_kana_only_is_excluded():
    """漢字・カタカナが1文字も無い文は落とす（第3条件）。"""
    assert has_lexical_content("それはとてもよいことだとおもいますけれど") is False
