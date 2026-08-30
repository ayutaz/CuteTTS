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

"""学習sequenceの組み立て（teacher forcing）。

推論の自己回帰ループ（`inference/generation.py`）から導出した対応関係:

    hidden(prefix末尾)  -> target patch 0 を予測 / patch 0 の stop ラベル
    hidden(patch 0)     -> target patch 1 を予測
    ...
    hidden(patch N-2)   -> target patch N-1 を予測

**入力に流すのは patch 0..N-2 の N-1 個**で、N 個の target には
prefix末尾 + patch 0..N-2 の N 個の hidden が対応する。
patch N-1 は target にしかならず、入力には入らない。

sequence の並びは推論の prefix と同じ:

    [ instruction text ... , speaker slot, reference speech ... , target text ... ,
      teacher-forced target patch 0..N-2 ]

`prefix_ids` には instruction / target text を含む **text token 列全体** を渡す。
speaker slot と speech patch の位置はこのモジュールが差し込む。

padding は **右詰め**にする。推論側は batched generation のため
`padding_side="left"` だが、学習では生成しないため右詰めが自然で、
attention mask と position id の扱いも素直になる。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from cutetts.training.objectives import build_stop_targets

SPEECH_PLACEHOLDER_ID = 0
"""speech patch と speaker slot が占める位置に入れる token id。

推論側も `SegmentManager` が `audio_pad_token_id` / `batch_pad_token_id`（既定0）で
埋めており、embedding は後段で差し替えられるため値そのものは使われない。
"""


@dataclass(frozen=True)
class TrainingSample:
    """1発話ぶんの学習sample。"""

    utterance_id: str
    input_ids: Tensor
    """[L] long。text token と、speech / speaker slot のプレースホルダ。"""
    speech_mask: Tensor
    """[L] bool。speech patch の embedding を差し込む位置。"""
    speaker_slot_mask: Tensor
    """[L] bool。speaker embedding を差し込む位置。"""
    speech_latents: Tensor
    """[S, P, D] 正規化後。`speech_mask` が真の位置と順序が一致する。"""
    target_patches: Tensor
    """[N, P, D] 正規化後の学習target。"""
    target_positions: Tensor
    """[N] long。各targetを予測する hidden の sequence index。"""
    previous_cond: Tensor
    """[N, P, D] Diffusion Head の previous cond。"""
    stop_targets: Tensor
    """[N] long。"""
    prefix_length: int
    """teacher-forced target を除いた、prefix 部分の長さ。"""
    reference_patch_count: int

    @property
    def length(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def num_targets(self) -> int:
        return int(self.target_patches.shape[0])


def build_training_sample(
    *,
    utterance_id: str,
    prefix_ids: Tensor,
    reference_latents: Tensor,
    target_latents: Tensor,
    speaker_slot: bool = True,
    initial_previous_cond: Tensor | None = None,
) -> TrainingSample:
    """1発話ぶんの学習sampleを組み立てる。

    ``reference_latents`` と ``target_latents`` は **正規化後** の
    ``[n, patch, dim]`` を渡すこと（`cutetts.training.dataset.normalize_latents`）。
    """
    if target_latents.dim() != 3 or target_latents.shape[0] == 0:
        raise ValueError(
            f"target_latents must be [N, P, D] with N >= 1, got {tuple(target_latents.shape)}"
        )
    if reference_latents.dim() != 3:
        raise ValueError(
            f"reference_latents must be [R, P, D], got {tuple(reference_latents.shape)}"
        )
    if reference_latents.shape[0] and reference_latents.shape[1:] != target_latents.shape[1:]:
        raise ValueError(
            "reference and target patches must share [P, D]: "
            f"{tuple(reference_latents.shape[1:])} vs {tuple(target_latents.shape[1:])}"
        )
    if prefix_ids.dim() != 1:
        raise ValueError(f"prefix_ids must be 1-D, got {tuple(prefix_ids.shape)}")

    n_target = int(target_latents.shape[0])
    n_reference = int(reference_latents.shape[0])
    patch, dim = int(target_latents.shape[1]), int(target_latents.shape[2])
    device = target_latents.device

    # teacher forcing で入力へ流すのは target 0..N-2
    fed = target_latents[: n_target - 1]
    n_fed = int(fed.shape[0])

    prefix_len = int(prefix_ids.shape[0]) + (1 if speaker_slot else 0) + n_reference
    total_len = prefix_len + n_fed

    input_ids = torch.full((total_len,), SPEECH_PLACEHOLDER_ID, dtype=torch.long, device=device)
    speech_mask = torch.zeros(total_len, dtype=torch.bool, device=device)
    speaker_mask = torch.zeros(total_len, dtype=torch.bool, device=device)

    cursor = 0
    if speaker_slot:
        speaker_mask[cursor] = True
        cursor += 1
    if n_reference:
        speech_mask[cursor : cursor + n_reference] = True
        cursor += n_reference
    text_len = int(prefix_ids.shape[0])
    input_ids[cursor : cursor + text_len] = prefix_ids.to(device=device, dtype=torch.long)
    cursor += text_len
    assert cursor == prefix_len
    if n_fed:
        speech_mask[cursor : cursor + n_fed] = True

    speech_latents = (
        torch.cat([reference_latents, fed], dim=0)
        if n_reference
        else fed
    )
    if speech_latents.shape[0] == 0:
        speech_latents = target_latents.new_zeros((0, patch, dim))

    # target i を予測する hidden の位置
    positions = torch.empty(n_target, dtype=torch.long, device=device)
    positions[0] = prefix_len - 1
    if n_fed:
        positions[1:] = torch.arange(prefix_len, prefix_len + n_fed, device=device)

    if initial_previous_cond is None:
        first_previous = target_latents.new_zeros((patch, dim))
    else:
        if tuple(initial_previous_cond.shape) != (patch, dim):
            raise ValueError(
                "initial_previous_cond shape mismatch: "
                f"expected {(patch, dim)}, got {tuple(initial_previous_cond.shape)}"
            )
        first_previous = initial_previous_cond.to(device=device, dtype=target_latents.dtype)
    previous_cond = torch.cat(
        [first_previous.unsqueeze(0), target_latents[: n_target - 1]], dim=0
    )

    stop_targets = build_stop_targets(
        torch.ones(1, n_target, dtype=torch.bool, device=device)
    ).squeeze(0)

    return TrainingSample(
        utterance_id=utterance_id,
        input_ids=input_ids,
        speech_mask=speech_mask,
        speaker_slot_mask=speaker_mask,
        speech_latents=speech_latents,
        target_patches=target_latents,
        target_positions=positions,
        previous_cond=previous_cond,
        stop_targets=stop_targets,
        prefix_length=prefix_len,
        reference_patch_count=n_reference,
    )


@dataclass(frozen=True)
class TrainingBatch:
    """collate 済みのミニバッチ。"""

    utterance_ids: list[str]
    input_ids: Tensor
    """[B, L] 右詰めpadding。"""
    attention_mask: Tensor
    """[B, L] bool。"""
    speech_mask: Tensor
    """[B, L] bool。"""
    speaker_slot_mask: Tensor
    """[B, L] bool。"""
    speech_latents: Tensor
    """[sum(S_b), P, D]。sample順に連結。"""
    speaker_slot_count: int
    target_patches: Tensor
    """[sum(N_b), P, D]。"""
    target_batch_index: Tensor
    """[sum(N_b)] long。各targetが属する **batch 行**（hidden の取り出しに使う）。"""
    target_sample_index: Tensor
    """[sum(N_b)] long。各targetが属する **sample**（speaker の取り出しに使う）。

    packing すると1行に複数sampleが入るため、行とsampleは一致しない。"""
    speaker_slot_sample_index: Tensor
    """[num_slots] long。speaker slot の出現順に、どのsampleのものかを示す。"""
    target_positions: Tensor
    """[sum(N_b)] long。各targetを予測する hidden の sequence index。"""
    previous_cond: Tensor
    """[sum(N_b), P, D]。"""
    stop_targets: Tensor
    """[sum(N_b)] long。"""
    target_mask: Tensor
    """[sum(N_b)] bool。padding や packing 境界で無効化された target は False。"""
    attention_bias: Tensor | None = None
    """[B, 1, L, L] の加算マスク。packing で segment 境界を遮断するときだけ使う。"""
    position_ids: Tensor | None = None
    """[B, L] long。packing で segment ごとに振り直すときだけ使う。"""

    @property
    def batch_size(self) -> int:
        """行数。packing すると sample 数とは一致しない。"""
        return int(self.input_ids.shape[0])

    @property
    def num_samples(self) -> int:
        return len(self.utterance_ids)

    @property
    def num_targets(self) -> int:
        return int(self.target_patches.shape[0])


def collate(samples: list[TrainingSample], *, pad_token_id: int = SPEECH_PLACEHOLDER_ID) -> TrainingBatch:
    """`TrainingSample` を右詰めpaddingでバッチにまとめる。"""
    if not samples:
        raise ValueError("collate requires at least one sample")

    max_len = max(s.length for s in samples)
    batch = len(samples)
    device = samples[0].input_ids.device

    input_ids = torch.full((batch, max_len), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
    speech = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
    speaker = torch.zeros(batch, max_len, dtype=torch.bool, device=device)

    for row, sample in enumerate(samples):
        n = sample.length
        input_ids[row, :n] = sample.input_ids
        attention[row, :n] = True
        speech[row, :n] = sample.speech_mask
        speaker[row, :n] = sample.speaker_slot_mask

    target_index = torch.cat([
        torch.full((s.num_targets,), row, dtype=torch.long, device=device)
        for row, s in enumerate(samples)
    ])
    # padding なしの collate では行とsampleが1対1
    sample_index = target_index.clone()
    slot_sample_index = torch.tensor(
        [row for row, s in enumerate(samples) if bool(s.speaker_slot_mask.any())],
        dtype=torch.long, device=device,
    )

    return TrainingBatch(
        utterance_ids=[s.utterance_id for s in samples],
        input_ids=input_ids,
        attention_mask=attention,
        speech_mask=speech,
        speaker_slot_mask=speaker,
        speech_latents=torch.cat([s.speech_latents for s in samples], dim=0),
        speaker_slot_count=int(speaker.sum().item()),
        target_patches=torch.cat([s.target_patches for s in samples], dim=0),
        target_batch_index=target_index,
        target_sample_index=sample_index,
        speaker_slot_sample_index=slot_sample_index,
        target_positions=torch.cat([s.target_positions for s in samples], dim=0),
        previous_cond=torch.cat([s.previous_cond for s in samples], dim=0),
        stop_targets=torch.cat([s.stop_targets for s in samples], dim=0),
        target_mask=torch.ones(int(target_index.shape[0]), dtype=torch.bool, device=device),
    )
