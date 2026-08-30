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

"""latent cache の読み出しと正規化のテスト。

正規化は推論と同じ向きでなければならない（`modeling/model.py`）:

    normalized = (raw + speech_bias_factor) * speech_scaling_factor
    raw        = normalized / speech_scaling_factor - speech_bias_factor
"""

from __future__ import annotations

import pytest
import torch

from cutetts.training.dataset import (
    denormalize_latents,
    normalize_latents,
    to_patches,
)

DIM = 64


def test_normalize_matches_the_inference_formula():
    raw = torch.randn(7, DIM)
    scaling, bias = torch.tensor(2.5), torch.tensor(-0.25)
    out = normalize_latents(raw, scaling=scaling, bias=bias)
    assert torch.allclose(out, (raw + bias) * scaling)


def test_normalize_and_denormalize_round_trip():
    raw = torch.randn(11, DIM)
    scaling, bias = torch.tensor(3.0), torch.tensor(0.75)
    back = denormalize_latents(normalize_latents(raw, scaling=scaling, bias=bias),
                               scaling=scaling, bias=bias)
    assert torch.allclose(back, raw, atol=1e-5)


def test_denormalize_matches_the_inference_formula():
    """推論は `pred / scaling - bias` で戻す。"""
    normalized = torch.randn(5, DIM)
    scaling, bias = torch.tensor(2.0), torch.tensor(0.5)
    out = denormalize_latents(normalized, scaling=scaling, bias=bias)
    assert torch.allclose(out, normalized / scaling - bias)


def test_normalize_rejects_nan_factors():
    """checkpointに正規化値が無い場合、推論側は例外を投げる。学習側も同じにする。"""
    raw = torch.randn(3, DIM)
    with pytest.raises(ValueError):
        normalize_latents(raw, scaling=torch.tensor(float("nan")), bias=torch.tensor(0.0))
    with pytest.raises(ValueError):
        normalize_latents(raw, scaling=torch.tensor(1.0), bias=torch.tensor(float("nan")))


# ---------------------------------------------------------------- to_patches

def test_frames_are_grouped_into_patches():
    frames = torch.arange(8 * DIM, dtype=torch.float32).reshape(8, DIM)
    patches = to_patches(frames, patch_size=2)
    assert patches.shape == (4, 2, DIM)
    assert torch.equal(patches[0, 0], frames[0])
    assert torch.equal(patches[0, 1], frames[1])
    assert torch.equal(patches[3, 1], frames[7])


def test_odd_frame_count_is_zero_padded_at_the_end():
    """奇数frameは末尾をzero paddingする（`SegmentManager._prepare_speech_tensor` と同じ）。"""
    frames = torch.ones(5, DIM)
    patches = to_patches(frames, patch_size=2)
    assert patches.shape == (3, 2, DIM)
    assert torch.all(patches[2, 0] == 1.0)
    assert torch.all(patches[2, 1] == 0.0)


def test_patch_size_one_keeps_every_frame():
    frames = torch.randn(5, DIM)
    patches = to_patches(frames, patch_size=1)
    assert patches.shape == (5, 1, DIM)


def test_empty_input_yields_empty_patches():
    patches = to_patches(torch.zeros(0, DIM), patch_size=2)
    assert patches.shape == (0, 2, DIM)


def test_invalid_patch_size_is_rejected():
    with pytest.raises(ValueError):
        to_patches(torch.zeros(4, DIM), patch_size=0)


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        to_patches(torch.zeros(2, 4, DIM), patch_size=2)
