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

"""短いreferenceの延長のテスト（R-026）。

同一話者・同一文で人間と聴き比べた結果、reference長で判定が完全に分離した
（劣る側の最大3.9秒 < 近い側の最小8.2秒、「声質が違う」は劣る2/2・近い0/4）。
`runtime.prepare_reference_audio` は2秒未満だけを伸ばすので、3〜4秒は素通りする。
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from cutetts.training.reference import (
    DEFAULT_MINIMUM_SECONDS,
    duration_seconds,
    ensure_minimum_duration,
)


def _write(path, seconds: float, rate: int = 24000, channels: int = 1):
    n = int(seconds * rate)
    # 無音ではなく実際の波形。無音を足すと speaker embedding が薄まる
    t = np.linspace(0, seconds, n, endpoint=False, dtype=np.float32)
    wave = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    data = np.stack([wave] * channels, axis=-1) if channels > 1 else wave[:, None]
    sf.write(str(path), data, rate)
    return path


def test_default_minimum_matches_the_speaker_encoder_window():
    """既定の下限は speaker encoder が使う先頭8秒に合わせる。"""
    assert DEFAULT_MINIMUM_SECONDS == 8.0


def test_long_reference_is_returned_unchanged(tmp_path):
    """十分な長さなら元のpathをそのまま返す（コピーを作らない）。"""
    source = _write(tmp_path / "long.wav", 9.9)

    result = ensure_minimum_duration(source)

    assert result == source.resolve()
    assert not (tmp_path / ".ref-cache").exists()


@pytest.mark.parametrize("seconds", [3.4, 3.9, 0.5, 7.9])
def test_short_reference_is_extended_past_the_minimum(tmp_path, seconds):
    """聴取で劣った 3.4秒 / 3.9秒 を含め、下限を超えるまで伸ばす。"""
    source = _write(tmp_path / "short.wav", seconds)

    result = ensure_minimum_duration(source)

    assert result != source.resolve()
    assert duration_seconds(result) >= DEFAULT_MINIMUM_SECONDS


def test_extension_repeats_the_waveform_not_silence(tmp_path):
    """波形をそのまま繰り返す。無音で埋めると speaker embedding が薄まる。"""
    source = _write(tmp_path / "short.wav", 2.5)

    result = ensure_minimum_duration(source)
    data, _ = sf.read(str(result), dtype="float32", always_2d=True)

    # 後半に信号が残っていること
    second_half = data[len(data) // 2:]
    assert float(np.abs(second_half).max()) > 0.05


def test_extension_is_cached_and_deterministic(tmp_path):
    """同じ入力・同じ下限なら同じファイルを使い回す。"""
    source = _write(tmp_path / "short.wav", 3.0)

    first = ensure_minimum_duration(source)
    stamp = first.stat().st_mtime_ns
    second = ensure_minimum_duration(source)

    assert first == second
    assert second.stat().st_mtime_ns == stamp


def test_minimum_seconds_is_configurable(tmp_path):
    source = _write(tmp_path / "short.wav", 3.0)

    result = ensure_minimum_duration(source, minimum_seconds=12.0)

    assert duration_seconds(result) >= 12.0


def test_cache_dir_can_be_redirected(tmp_path):
    source = _write(tmp_path / "short.wav", 3.0)
    cache = tmp_path / "elsewhere"

    result = ensure_minimum_duration(source, cache_dir=cache)

    assert result.parent == cache


def test_stereo_reference_is_extended(tmp_path):
    """runtime 側が mono 化するので、ここでは channel を保ったまま伸ばす。"""
    source = _write(tmp_path / "stereo.wav", 3.0, channels=2)

    result = ensure_minimum_duration(source)
    data, _ = sf.read(str(result), dtype="float32", always_2d=True)

    assert data.shape[1] == 2
    assert duration_seconds(result) >= DEFAULT_MINIMUM_SECONDS


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ensure_minimum_duration(tmp_path / "nope.wav")


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(str(path), np.zeros((0, 1), dtype=np.float32), 24000)

    with pytest.raises(ValueError):
        ensure_minimum_duration(path)
