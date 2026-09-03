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

"""漢数字の読み展開のテスト（J2 / D-008）。

学習コーパスに複合漢数字は 0.32% しかなく、桁の合成規則を学べない。
tokenizerの前で仮名へ展開すればモデルは既知の仮名として読める。

**壊してはいけないもの**が2種類ある。

* `一緒` `二人` のような語の一部（単独の漢数字）
* 数詞として解釈できない並び（`百千` など）
"""

from __future__ import annotations

import pytest

from cutetts.training.reading import (
    expand_kanji_numerals,
    read_number,
)
from cutetts.training.reading import _parse_run  # noqa: PLC2701


@pytest.mark.parametrize("run,value", [
    ("三十一", 31),
    ("七十二", 72),
    ("百十五", 115),
    ("百三十九", 139),
    ("千二百八十", 1280),
    ("三千五百二十", 3520),
    ("千九百八十七", 1987),
    ("五千六百七十八", 5678),
    ("二十三万", 230_000),
    ("一万五千", 15_000),
    ("十", 10),
    ("百", 100),
    ("千", 1000),
])
def test_parse_run_reads_place_value(run, value):
    """桁の合成。これが学習データから学べないので frontend で扱う。"""
    assert _parse_run(run) == value


@pytest.mark.parametrize("run", [
    "百千",      # 位が昇順
    "十百",      # 位が昇順
    "二三",      # 数字の連続（数詞ではない）
    "万",        # 大位だけが単独
])
def test_parse_run_rejects_non_numerals(run):
    assert _parse_run(run) is None


@pytest.mark.parametrize("value,kana", [
    (0, "ゼロ"),
    (5, "ご"),
    (10, "じゅう"),
    (31, "さんじゅういち"),
    (100, "ひゃく"),
    (115, "ひゃくじゅうご"),
    (1000, "せん"),
    (1280, "せんにひゃくはちじゅう"),
    (1987, "せんきゅうひゃくはちじゅうなな"),
    (230_000, "にじゅうさんまん"),
])
def test_read_number(value, kana):
    assert read_number(value) == kana


@pytest.mark.parametrize("value,kana", [
    (300, "さんびゃく"),
    (600, "ろっぴゃく"),
    (800, "はっぴゃく"),
    (3000, "さんぜん"),
    (8000, "はっせん"),
    (3520, "さんぜんごひゃくにじゅう"),
])
def test_read_number_applies_sound_changes(value, kana):
    """連濁・促音。`三百`=さんびゃく を `さんひゃく` にしてはいけない。"""
    assert read_number(value) == kana


@pytest.mark.parametrize("text,expected", [
    ("価格は千二百八十円です", "価格はせんにひゃくはちじゅう円です"),
    ("人口はおよそ二十三万人です", "人口はおよそにじゅうさんまん人です"),
    ("北緯三十五度、東経百三十九度", "北緯さんじゅうご度、東経ひゃくさんじゅうきゅう度"),
])
def test_expand_in_sentence(text, expected):
    assert expand_kanji_numerals(text) == expected


@pytest.mark.parametrize("text", [
    "一緒に行こう",
    "二人で話した",
    "第三者の意見",
    "四苦八苦している",
    "十分に注意して",
    "一石二鳥だね",
])
def test_single_kanji_numerals_are_left_alone(text):
    """`一緒` を `いち緒` にしては壊れる。1文字の並びは展開しない。"""
    assert expand_kanji_numerals(text) == text


def test_zero_sequences_are_read_digit_by_digit():
    """電話番号の `零三` は `ゼロさん`。桁として読んではいけない。"""
    assert expand_kanji_numerals("零三の番号") == "ゼロさんの番号"


def test_text_without_numerals_is_unchanged():
    text = "でも大丈夫。気持ちは届いてるからさ、ここに"
    assert expand_kanji_numerals(text) == text


def test_counters_are_not_expanded():
    """助数詞は触らない。`三本`→`さんぼん` の音便はモデルが文脈で扱う。"""
    assert expand_kanji_numerals("三十一日") == "さんじゅういち日"
