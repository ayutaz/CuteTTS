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

"""latent cache の読み出しと、学習で使う潜在の正規化。

P1e の cache には **生の VAE posterior mean** が入っている（正規化前）。
学習forwardへ渡す前にここで正規化する。向きは推論と同一:

    normalized = (raw + speech_bias_factor) * speech_scaling_factor      # model.py
    raw        = normalized / speech_scaling_factor - speech_bias_factor  # generation.py

正規化係数は checkpoint の buffer にあり、未設定（NaN）なら推論側は
`RuntimeError` を投げる。学習側も同じく弾く。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor

from cutetts.training.manifest import Utterance


def _check_factor(value: Tensor, name: str) -> Tensor:
    if torch.isnan(value).any():
        raise ValueError(
            f"{name} is NaN. The checkpoint does not carry acoustic normalization values."
        )
    return value


def normalize_latents(latents: Tensor, *, scaling: Tensor, bias: Tensor) -> Tensor:
    """生の VAE latent を学習・推論で使う正規化空間へ移す。"""
    _check_factor(scaling, "speech_scaling_factor")
    _check_factor(bias, "speech_bias_factor")
    return (latents + bias) * scaling


def denormalize_latents(latents: Tensor, *, scaling: Tensor, bias: Tensor) -> Tensor:
    """正規化空間の潜在を VAE の尺度へ戻す（waveform decode の直前で使う）。"""
    _check_factor(scaling, "speech_scaling_factor")
    _check_factor(bias, "speech_bias_factor")
    return latents / scaling - bias


def to_patches(frames: Tensor, *, patch_size: int) -> Tensor:
    """``[T, D]`` の latent frame 列を ``[ceil(T/P), P, D]`` の patch 列にする。

    端数は末尾を zero padding する（`SegmentManager._prepare_speech_tensor` と同じ規約）。
    """
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    if frames.dim() != 2:
        raise ValueError(f"frames must be [T, D], got {tuple(frames.shape)}")
    length, dim = frames.shape
    if length == 0:
        return frames.new_zeros((0, patch_size, dim))
    remainder = (-length) % patch_size
    if remainder:
        frames = torch.cat([frames, frames.new_zeros((remainder, dim))], dim=0)
    return frames.reshape(-1, patch_size, dim)


@dataclass(frozen=True)
class LatentSource:
    """latent cache と正規化係数をまとめて扱う。"""

    reader: object
    """`LatentCacheReader` 互換（`read(utterance_id)`、`__contains__`）。"""
    scaling: Tensor
    bias: Tensor
    patch_size: int = 2

    def __contains__(self, utterance_id: str) -> bool:
        return utterance_id in self.reader

    def patches(self, utterance_id: str) -> Tensor:
        """正規化済みの ``[n, patch, dim]`` を返す。"""
        frames = self.reader.read(utterance_id)
        normalized = normalize_latents(frames, scaling=self.scaling, bias=self.bias)
        return to_patches(normalized, patch_size=self.patch_size)


def available(source: LatentSource, records: list[Utterance]) -> Iterator[Utterance]:
    """cache に latent がある record だけを流す。"""
    for record in records:
        if record.utterance_id in source:
            yield record
