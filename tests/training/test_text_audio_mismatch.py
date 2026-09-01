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

"""text と音声が対応していない record を弾くことを検証する。

S1 で選定した5 game のうち1 game が、text の切り出しに失敗していた。
19.4秒の音声に対して text が ``瑞羽の『瑞`` の5文字しかない、といった
record が **その game の41.6%** を占めていた（median 0.371 秒/文字）。

このデータで学習すると「短い text に長い音を返す」ことを覚え、
推論で停止しなくなる。実測では生成の15%が上限64秒に張り付き、
その区間は 0.05〜0.5 文字/秒（正常5.74）だった。
in_domain CER は 35.8%（未学習）→ 51.6% へ悪化した。

**flow loss では検知できない。** 壊れたデータも忠実に再現できるように
なるほど flow loss は下がる。S1 では dev flow が
0.8329 → 0.7909 と改善し続ける裏で CER が崩壊した。
だから入口で弾く必要がある。
"""

from __future__ import annotations

import pytest

from cutetts.training.manifest import (
    DEFAULT_MAX_SECONDS_PER_CHAR,
    VALIDATION_CODES,
    Utterance,
    validate,
)

CODE = "text_audio_mismatch"


def _utterance(duration: float, text: str, uid: str = "gol:g:u.wav") -> Utterance:
    return Utterance(utterance_id=uid, dataset_id="gol", audio_ref="a.tar::u.wav",
                     text_raw=text, speaker_id="spk", duration=duration,
                     sample_rate=48000)


def _has_mismatch(record: Utterance, **kwargs) -> bool:
    return any(issue.code == CODE for issue in validate([record], **kwargs))


# ------------------------------------------------------------ 実データの再現

def test_the_actual_broken_records_from_s1_are_rejected():
    """S1で見つかった実物。どちらも text の切り出しに失敗している。"""
    assert _has_mismatch(_utterance(19.4, "瑞羽の『瑞"))     # 3.88 秒/文字
    assert _has_mismatch(_utterance(15.7, "はぅうっ"))       # 3.92 秒/文字
    assert _has_mismatch(_utterance(9.7, "あ……"))          # 3.22 秒/文字


def test_normal_japanese_speech_is_kept():
    """自然な日本語は 0.10〜0.20 秒/文字に収まる。"""
    assert not _has_mismatch(_utterance(4.0, "これはふつうの発話です"))
    assert not _has_mismatch(_utterance(2.0, "おはようございます"))
    assert not _has_mismatch(_utterance(6.0, "車はお前が知ってるだろ？出発を確認次第、決行だ"))


# ------------------------------------------------------------------ 境界条件

def test_the_threshold_is_applied_strictly_above():
    """閾値ちょうどは通し、超えたら弾く。"""
    text = "あいうえお"                       # 5文字
    exact = DEFAULT_MAX_SECONDS_PER_CHAR * len(text)
    assert not _has_mismatch(_utterance(exact, text))
    assert _has_mismatch(_utterance(exact + 0.1, text))


def test_the_threshold_can_be_overridden():
    record = _utterance(2.0, "テスト用")       # 0.50 秒/文字
    assert _has_mismatch(record, max_seconds_per_char=0.40)
    assert not _has_mismatch(record, max_seconds_per_char=0.60)


def test_very_short_text_is_not_checked():
    """1〜2文字では比が不安定なので検査しない（間投詞を消しすぎない）。"""
    assert not _has_mismatch(_utterance(1.5, "あ"))
    assert not _has_mismatch(_utterance(2.0, "はい"))
    # 3文字からは検査する
    assert _has_mismatch(_utterance(5.0, "はいっ"))


def test_the_minimum_length_can_be_overridden():
    record = _utterance(5.0, "はいっ")
    assert _has_mismatch(record, min_chars_for_ratio=3)
    assert not _has_mismatch(record, min_chars_for_ratio=10)


def test_whitespace_is_not_counted_as_content():
    """前後の空白で比が薄まらない。"""
    assert _has_mismatch(_utterance(6.0, "  あいう  "))


# ------------------------------------------------------------- 他検査との関係

def test_a_record_can_carry_several_issues_at_once():
    """長すぎる かつ 対応していない、は両方報告する。"""
    codes = {issue.code for issue in validate([_utterance(40.0, "みじかい")],
                                              max_duration=30.0)}
    assert {"too_long", CODE} <= codes


def test_empty_text_reports_empty_not_mismatch():
    """空textは `empty_text` の担当。ゼロ除算もしない。"""
    codes = {issue.code for issue in validate([_utterance(5.0, "   ")])}
    assert "empty_text" in codes
    assert CODE not in codes


def test_zero_duration_reports_bad_duration_not_mismatch():
    codes = {issue.code for issue in validate([_utterance(0.0, "あいうえお")])}
    assert "bad_duration" in codes
    assert CODE not in codes


def test_the_code_is_registered():
    assert CODE in VALIDATION_CODES


def test_the_detail_names_the_measured_ratio():
    """artifactを後から読む人が、なぜ弾かれたかを数値で追えること。"""
    issue = next(i for i in validate([_utterance(19.4, "瑞羽の『瑞")]) if i.code == CODE)
    assert "3.8" in issue.detail
    assert "19.4" in issue.detail


@pytest.mark.parametrize("ratio,expected", [(0.20, False), (0.35, False),
                                            (0.45, True), (1.00, True)])
def test_ratios_around_the_threshold(ratio: float, expected: bool):
    text = "あいうえおかきくけこ"           # 10文字
    assert _has_mismatch(_utterance(ratio * len(text), text)) is expected
