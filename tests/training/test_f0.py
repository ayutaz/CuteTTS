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

"""F0 のフレーム条件（M4c / `training.f0`）。

**ここが狂うと条件と音がずれる。** 特に

1. **patch の並びが latent と一致すること**（ずれると別のpatchの高さを与える）
2. **zero-init であること**（学習開始時に現行の挙動を壊さない）
3. **無声と「中央値と同じ高さ」が区別できること**（フラグが要る）
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cutetts.training.f0 import (
    F0_CLIP_OCTAVES,
    F0_FEATURE_DIM,
    F0_FRAMES_PER_LATENT,
    F0Conditioner,
    f0_features,
    patch_features,
    reference_hz,
)


def test_1_latent_frameは8フレーム():
    """latent は 80 ms、`prosody.FRAME_PERIOD_MS` は 10 ms。"""
    from cutetts.training.prosody import FRAME_PERIOD_MS

    assert F0_FRAMES_PER_LATENT * FRAME_PERIOD_MS == 80.0


def test_基準は有声の中央値():
    assert reference_hz(np.array([0.0, 100.0, 200.0, 300.0])) == 200.0
    assert reference_hz(np.zeros(4)) == 0.0


def test_無声は全部ゼロ():
    out = f0_features(np.array([0.0, 0.0]))
    assert out.shape == (2, F0_FEATURE_DIM)
    assert np.all(out == 0.0)


def test_無声と中央値の高さが区別できる():
    """**フラグが無いと両方 0 になって区別できない。**"""
    out = f0_features(np.array([0.0, 200.0]), center_hz=200.0)
    assert out[0].tolist() == [0.0, 0.0]        # 無声
    assert out[1].tolist() == [1.0, 0.0]        # 中央値と同じ高さ


def test_オクターブ比で正規化される():
    out = f0_features(np.array([100.0, 200.0, 400.0]), center_hz=200.0)
    assert out[:, 1] == pytest.approx([-1.0, 0.0, 1.0])


def test_打ち切りが効く():
    out = f0_features(np.array([25.0, 1600.0]), center_hz=200.0)
    assert out[:, 1] == pytest.approx([-F0_CLIP_OCTAVES, F0_CLIP_OCTAVES])


def test_patchの並びがlatentと一致する():
    """**patch i は frame 2i, 2i+1。** ずれると別のpatchの高さを与える。"""
    features = np.arange(8, dtype=np.float32).reshape(4, 2)
    out = patch_features(features, patch_size=2, num_patches=2)
    assert out.shape == (2, 4)
    assert out[0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert out[1].tolist() == [4.0, 5.0, 6.0, 7.0]


def test_足りないpatchはゼロで埋める():
    features = np.ones((3, 2), dtype=np.float32)
    out = patch_features(features, patch_size=2, num_patches=2)
    assert out[1].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_余ったフレームは捨てる():
    features = np.ones((9, 2), dtype=np.float32)
    assert patch_features(features, patch_size=2, num_patches=2).shape == (2, 4)


def test_conditionerはzero_init():
    """**学習開始時点で恒等**（現行と同じ挙動）から始まること。"""
    conditioner = F0Conditioner(4, 16)
    out = conditioner(torch.randn(5, 4))
    assert out.shape == (5, 16)
    assert torch.all(out == 0.0)


def test_conditionerは次元を検査する():
    conditioner = F0Conditioner(4, 16)
    with pytest.raises(ValueError):
        conditioner(torch.randn(5, 3))
    with pytest.raises(ValueError):
        conditioner(torch.randn(5))


def test_conditionerは学習すると動く():
    conditioner = F0Conditioner(4, 8)
    torch.nn.init.ones_(conditioner.proj.weight)
    out = conditioner(torch.ones(1, 4))
    assert torch.allclose(out, torch.full((1, 8), 4.0))


# ---------------------------------------------------------------- cache
#
# **latent cache と同じ形で持つ**ので、食い違いは load 時に見つけたい。


def test_cacheは書いて読める(tmp_path):
    from cutetts.training.f0 import F0CacheReader, F0CacheWriter, default_f0_meta

    meta = default_f0_meta("deadbeef")
    features = np.array([[1.0, 0.5], [0.0, 0.0], [1.0, -0.25]], dtype=np.float32)
    with F0CacheWriter(tmp_path / "f0", meta) as writer:
        writer.write("utt-1", features)
    with F0CacheReader(tmp_path / "f0") as reader:
        out = reader.read("utt-1").numpy()
    assert out.shape == (3, 2)
    assert np.allclose(out, features, atol=1e-3)     # float16 で保存する


def test_別のVAEで作ったcacheは弾く(tmp_path):
    """**decoder が変われば F0 は別のもの。** 黙って混ぜない。"""
    from cutetts.training.f0 import (
        CacheMetaMismatch,
        F0CacheReader,
        F0CacheWriter,
        default_f0_meta,
    )

    with F0CacheWriter(tmp_path / "f0", default_f0_meta("aaaa")) as writer:
        writer.write("utt-1", np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(CacheMetaMismatch):
        F0CacheReader(tmp_path / "f0", expect=default_f0_meta("bbbb"))


def test_形が違えば書けない(tmp_path):
    from cutetts.training.f0 import F0CacheWriter, default_f0_meta

    with F0CacheWriter(tmp_path / "f0", default_f0_meta("aaaa")) as writer:
        with pytest.raises(ValueError):
            writer.write("utt-1", np.zeros((2, 3), dtype=np.float32))


# ------------------------------------------------- 時間軸を保つ輪郭（測定器の修正）
#
# `semitone_contour` は**有声フレームだけを詰める**ので、有声判定が少し
# 食い違うだけで系列全体がずれる。実測で**同じ発話の VAE 往復が 0.417**まで
# 落ちた（M2 の「天井」0.382 とほぼ同じ）。


def test_同一のF0では相関1():
    from cutetts.training.prosody import contour_similarity_time

    f0 = np.array([100.0, 110.0, 0.0, 120.0, 115.0, 105.0, 100.0, 108.0,
                   112.0, 118.0, 124.0, 116.0])   # MIN_VOICED_FRAMES=10 以上
    assert contour_similarity_time(f0, f0) == pytest.approx(1.0)


def test_無声は前後から埋める():
    from cutetts.training.prosody import time_aligned_contour

    out = time_aligned_contour(np.array([100.0, 0.0, 200.0]))
    assert out.size == 3                      # **詰めずに長さを保つ**
    # 中央は 100 と 200 の線形補間（150 Hz）
    assert out[1] == pytest.approx(12.0 * np.log2(150.0 / 150.0))


def test_有声判定の食い違いに強い():
    """**ここが `semitone_contour` の弱点。** 1フレームの違いでずれない。"""
    from cutetts.training.prosody import (
        contour_similarity,
        contour_similarity_time,
        semitone_contour,
    )

    base = np.array([100.0, 105.0, 110.0, 120.0, 130.0, 125.0, 115.0, 100.0,
                     104.0, 112.0, 126.0, 118.0])
    # 3フレーム目だけ無声と判定された場合
    perturbed = base.copy()
    perturbed[2] = 0.0
    old = contour_similarity(semitone_contour(base), semitone_contour(perturbed))
    new = contour_similarity_time(base, perturbed)
    assert new > old
    assert new > 0.95


def test_全部無声なら空():
    from cutetts.training.prosody import time_aligned_contour

    assert time_aligned_contour(np.zeros(5)).size == 0
