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

"""抑揚・アクセント測定のテスト（M1）。

**測定器そのものを検証する。** このプロジェクトの誤りはすべて測定器の
欠陥から来た（R-012 / R-020 / R-021 / R-029）ので、F0推定が正しいことを
**既知のF0を持つ合成信号**で確かめる。`pyworld` が無ければ skip。

アクセント側（`accent_plan` / `pattern`）は規則だけなので依存が要らない。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cutetts.training.prosody import (
    MIN_VOICED_FRAMES,
    AccentPhrase,
    contour_similarity,
    pattern_agreement,
    resample_contour,
    semitone_contour,
    semitones,
    split_moras,
)

SAMPLE_RATE = 24000


def harmonic_tone(hz: float, seconds: float = 1.0, harmonics: int = 20,
                  sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """調波を重ねた声らしい信号。**純音では harvest が有声と判定しない**
    （実測: 調波2本の220Hzは0/101フレーム）。"""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    wave = sum(np.sin(2 * np.pi * hz * k * t) / k
               for k in range(1, harmonics + 1) if hz * k < sample_rate / 2)
    return (wave / np.abs(wave).max() * 0.8).astype(np.float64)


# ---- アクセント（依存なし） ----


def test_heiban_starts_low_then_stays_high():
    """平板型（核0）: 1モーラ目だけ低い。"""
    assert AccentPhrase(("ハ", "シ", "ヲ"), 0).pattern() == (False, True, True)


def test_atamadaka_is_high_only_on_first_mora():
    """頭高型（核1）: `箸`。"""
    assert AccentPhrase(("ハ", "シ", "ヲ"), 1).pattern() == (True, False, False)


def test_nakadaka_drops_after_the_nucleus():
    """尾高・中高（核2）: `橋`。**`箸` と区別できることがこの指標の要**。"""
    assert AccentPhrase(("ハ", "シ", "ヲ"), 2).pattern() == (False, True, False)


def test_hashi_pair_is_distinguishable():
    """同じ読みでもアクセントが違えばパターンが違う。CERでは絶対に出ない差。"""
    chopsticks = AccentPhrase(("ハ", "シ", "ヲ"), 1).pattern()
    bridge = AccentPhrase(("ハ", "シ", "ヲ"), 2).pattern()

    assert chopsticks != bridge


def test_empty_phrase_has_empty_pattern():
    assert AccentPhrase((), 0).pattern() == ()


def test_split_moras_attaches_small_kana():
    assert split_moras("チュウカ") == ["チュ", "ウ", "カ"]
    assert split_moras("キャッチ") == ["キャ", "ッ", "チ"]


def test_split_moras_of_empty_is_empty():
    assert split_moras("") == []


def test_pattern_agreement_counts_matching_moras():
    assert pattern_agreement((True, False, True), (True, False, True)) == 1.0
    assert pattern_agreement((True, False, True), (True, True, True)) == pytest.approx(2 / 3)


def test_pattern_agreement_uses_the_shorter_length():
    assert pattern_agreement((True, False), (True, False, True, True)) == 1.0


def test_pattern_agreement_of_empty_is_nan():
    assert math.isnan(pattern_agreement((), (True,)))


# ---- 輪郭（依存なし） ----


def test_semitone_contour_drops_unvoiced_frames():
    f0 = np.array([0.0, 100.0, 0.0, 200.0])

    contour = semitone_contour(f0)

    assert contour.size == 2


def test_semitone_contour_is_centred_on_the_median():
    """話者の平均音高を落とすので、**別の話者どうしでも比べられる**。"""
    low = semitone_contour(np.array([100.0, 200.0, 400.0]))
    high = semitone_contour(np.array([200.0, 400.0, 800.0]))

    assert np.allclose(low, high)


def test_semitone_contour_of_silence_is_empty():
    assert semitone_contour(np.zeros(10)).size == 0


def test_semitones_of_an_octave_is_twelve():
    assert semitones(2.0) == pytest.approx(12.0)


def test_resample_contour_preserves_shape():
    contour = np.array([0.0, 1.0, 2.0, 3.0])

    stretched = resample_contour(contour, 7)

    assert stretched.size == 7
    assert stretched[0] == pytest.approx(0.0)
    assert stretched[-1] == pytest.approx(3.0)


def test_resample_contour_of_empty_is_zeros():
    assert resample_contour(np.zeros(0), 5).tolist() == [0.0] * 5


def test_contour_similarity_is_one_for_the_same_shape_at_another_speed():
    """話速が違っても同じ抑揚なら1に近い。長さは線形に揃える。"""
    slow = np.sin(np.linspace(0, 2 * np.pi, 60))
    fast = np.sin(np.linspace(0, 2 * np.pi, 30))

    assert contour_similarity(slow, fast) == pytest.approx(1.0, abs=0.01)


def test_contour_similarity_is_negative_for_an_inverted_contour():
    rising = np.linspace(-3, 3, 40)

    assert contour_similarity(rising, -rising) == pytest.approx(-1.0, abs=0.01)


def test_contour_similarity_of_a_flat_contour_is_nan():
    """棒読み（変化なし）は相関が定義できない。**NaNを0と混同しないこと**。"""
    flat = np.zeros(40)

    assert math.isnan(contour_similarity(flat, np.linspace(0, 1, 40)))


def test_contour_similarity_needs_enough_frames():
    short = np.linspace(0, 1, MIN_VOICED_FRAMES - 1)

    assert math.isnan(contour_similarity(short, short))


# ---- F0推定（pyworld が要る） ----

pyworld = pytest.importorskip(
    "pyworld", reason="F0推定には [prosody] extra（pyworld）が要る")


@pytest.mark.parametrize("hz", [90.0, 120.0, 180.0, 250.0, 350.0, 500.0])
def test_track_f0_recovers_a_known_pitch(hz):
    """**既知のF0を誤差1%以内で当てられること。** 実測は0.01%。"""
    from cutetts.training.prosody import track_f0

    f0 = track_f0(harmonic_tone(hz), SAMPLE_RATE)
    voiced = f0[f0 > 0]

    assert voiced.size > 50
    assert abs(float(np.median(voiced)) - hz) / hz < 0.01


def test_silence_is_unvoiced():
    """**`torchaudio.detect_pitch_frequency` はここで落ちる**（全フレームを
    有声として返し、無音区間のゴミが統計を壊す）。"""
    from cutetts.training.prosody import track_f0

    f0 = track_f0(np.zeros(SAMPLE_RATE), SAMPLE_RATE)

    assert int((f0 > 0).sum()) == 0


def test_white_noise_is_mostly_unvoiced():
    from cutetts.training.prosody import track_f0

    noise = np.random.default_rng(0).normal(0, 0.1, SAMPLE_RATE)
    f0 = track_f0(noise, SAMPLE_RATE)

    assert int((f0 > 0).sum()) < 5


def test_measure_reports_a_flat_tone_as_flat():
    """一定のF0は幅がほぼ0。**「棒読み」がこの値で見える。**"""
    from cutetts.training.prosody import measure

    stats = measure(harmonic_tone(200.0), SAMPLE_RATE)

    assert stats.is_valid
    assert stats.median_hz == pytest.approx(200.0, rel=0.01)
    assert stats.semitone_sd < 0.1
    assert stats.semitone_range < 0.1


def test_edge_frames_are_dropped():
    """**端フレームを捨てないと全測定が膨らむ。**

    `harvest` は最初と最後のフレームだけ値が壊れる。一定200Hzで
    先頭 -1.13半音 / 末尾 -3.66半音 が出て、sd が 0.00005 → 0.378 になった。
    これを見逃すと「抑揚がある」と誤判定する。
    """
    from cutetts.training.prosody import EDGE_FRAMES_DROPPED, track_f0

    f0 = track_f0(harmonic_tone(200.0), SAMPLE_RATE)

    assert f0[:EDGE_FRAMES_DROPPED].tolist() == [0.0] * EDGE_FRAMES_DROPPED
    assert f0[-EDGE_FRAMES_DROPPED:].tolist() == [0.0] * EDGE_FRAMES_DROPPED


def test_measure_reports_a_sweep_as_wide():
    """上下する声は幅が大きい。"""
    from cutetts.training.prosody import measure

    sample_rate = SAMPLE_RATE
    t = np.arange(sample_rate) / sample_rate
    hz = 150.0 * np.power(2.0, np.sin(2 * np.pi * t) * 0.5)   # ±6半音
    phase = 2 * np.pi * np.cumsum(hz) / sample_rate
    wave = sum(np.sin(phase * k) / k for k in range(1, 21))
    wave = (wave / np.abs(wave).max() * 0.8).astype(np.float64)

    stats = measure(wave, sample_rate)

    assert stats.is_valid
    assert stats.semitone_sd > 2.0


def test_measure_of_silence_is_invalid():
    """無声しかない発話は NaN を返し、**0 とは区別する**。"""
    from cutetts.training.prosody import measure

    stats = measure(np.zeros(SAMPLE_RATE), SAMPLE_RATE)

    assert not stats.is_valid
    assert math.isnan(stats.semitone_sd)
