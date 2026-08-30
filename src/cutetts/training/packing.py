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

"""sequence packing。

複数sampleを1行へ詰め、padding を減らす。**数値を変えてはならない**ので、
次の2点を守る。

* **attention を segment 境界で遮断する。** packed sample 同士が見えると
  loss が変わるだけでなく、学習が壊れる。
* **position id を segment ごとにリセットする。** RoPE は相対位置なので、
  通し番号のままだと2つ目以降のsampleが別の位置埋め込みを受ける。

packing は正しさを確認した後に足す最適化であり、
`tests/training/test_packing.py` が unpacked との一致を検証する。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from cutetts.training.collator import SPEECH_PLACEHOLDER_ID, TrainingBatch, TrainingSample


@dataclass(frozen=True)
class PackedBatch:
    """1行へ詰めたsample群。"""

    utterance_ids: list[str]
    input_ids: Tensor
    """[1, L]"""
    segment_ids: Tensor
    """[1, L] long。どのsampleに属するか。"""
    position_ids: Tensor
    """[1, L] long。segmentごとに0から振り直す。"""
    speech_mask: Tensor
    speaker_slot_mask: Tensor
    speech_latents: Tensor
    target_patches: Tensor
    target_positions: Tensor
    """[N] long。packed sequence 内の絶対位置。"""
    target_batch_index: Tensor
    target_sample_index: Tensor
    speaker_slot_sample_index: Tensor
    previous_cond: Tensor
    stop_targets: Tensor
    target_mask: Tensor
    lengths: list[int]

    @property
    def length(self) -> int:
        return int(self.input_ids.shape[1])

    def to_batch(self) -> TrainingBatch:
        """`training_forward` が受け取れる `TrainingBatch` にする。

        **segment 境界の遮断と position id のリセットを必ず載せる。**
        これが無いと packed sample 同士が attention してしまい、
        unpacked と数値が一致しない。
        """
        mask = packed_attention_mask(self.segment_ids)
        bias = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
        bias.masked_fill_(~mask, torch.finfo(torch.float32).min)
        return TrainingBatch(
            utterance_ids=self.utterance_ids,
            input_ids=self.input_ids,
            attention_mask=torch.ones_like(self.input_ids, dtype=torch.bool),
            speech_mask=self.speech_mask,
            speaker_slot_mask=self.speaker_slot_mask,
            speech_latents=self.speech_latents,
            speaker_slot_count=int(self.speaker_slot_mask.sum().item()),
            target_patches=self.target_patches,
            target_batch_index=self.target_batch_index,
            target_sample_index=self.target_sample_index,
            speaker_slot_sample_index=self.speaker_slot_sample_index,
            target_positions=self.target_positions,
            previous_cond=self.previous_cond,
            stop_targets=self.stop_targets,
            target_mask=self.target_mask,
            attention_bias=bias,
            position_ids=self.position_ids,
        )


def pack_samples(samples: list[TrainingSample]) -> PackedBatch:
    """`TrainingSample` を1行へ連結する。"""
    if not samples:
        raise ValueError("pack_samples requires at least one sample")
    device = samples[0].input_ids.device

    input_ids = torch.cat([s.input_ids for s in samples]).unsqueeze(0)
    speech = torch.cat([s.speech_mask for s in samples]).unsqueeze(0)
    speaker = torch.cat([s.speaker_slot_mask for s in samples]).unsqueeze(0)

    segment_ids = torch.cat([
        torch.full((s.length,), i, dtype=torch.long, device=device)
        for i, s in enumerate(samples)
    ]).unsqueeze(0)
    position_ids = torch.cat([
        torch.arange(s.length, dtype=torch.long, device=device) for s in samples
    ]).unsqueeze(0)

    offsets, running = [], 0
    for s in samples:
        offsets.append(running)
        running += s.length
    target_positions = torch.cat([
        s.target_positions + offset for s, offset in zip(samples, offsets)
    ])
    # hidden は1行しかないので batch index は常に0。sample index は別に持つ
    target_index = torch.zeros(int(target_positions.shape[0]), dtype=torch.long, device=device)
    sample_index = torch.cat([
        torch.full((s.num_targets,), i, dtype=torch.long, device=device)
        for i, s in enumerate(samples)
    ])
    slot_sample_index = torch.tensor(
        [i for i, s in enumerate(samples) if bool(s.speaker_slot_mask.any())],
        dtype=torch.long, device=device,
    )

    return PackedBatch(
        utterance_ids=[s.utterance_id for s in samples],
        input_ids=input_ids,
        segment_ids=segment_ids,
        position_ids=position_ids,
        speech_mask=speech,
        speaker_slot_mask=speaker,
        speech_latents=torch.cat([s.speech_latents for s in samples], dim=0),
        target_patches=torch.cat([s.target_patches for s in samples], dim=0),
        target_positions=target_positions,
        target_batch_index=target_index,
        target_sample_index=sample_index,
        speaker_slot_sample_index=slot_sample_index,
        previous_cond=torch.cat([s.previous_cond for s in samples], dim=0),
        stop_targets=torch.cat([s.stop_targets for s in samples], dim=0),
        target_mask=torch.ones(int(target_positions.shape[0]), dtype=torch.bool, device=device),
        lengths=[s.length for s in samples],
    )


def packed_attention_mask(segment_ids: Tensor) -> Tensor:
    """``[1, L]`` の segment id から ``[1, 1, L, L]`` の bool mask を作る。

    True が「見てよい」。causal かつ同一 segment のときだけ True。
    """
    if segment_ids.dim() != 2:
        raise ValueError(f"segment_ids must be [B, L], got {tuple(segment_ids.shape)}")
    batch, length = segment_ids.shape
    same = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)      # [B, L, L]
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool,
                                   device=segment_ids.device))
    return (same & causal).unsqueeze(1)


def packed_hidden_states(model, packed: PackedBatch) -> Tensor:
    """境界遮断つきで backbone を通し、hidden state を返す（検証用）。"""
    from cutetts.modeling.segments import CuteTTSSegment

    device = model.device
    segment = CuteTTSSegment(
        input_ids=packed.input_ids.to(device),
        speech_tensor=packed.speech_latents.unsqueeze(0).to(device),
        speech_mask=packed.speech_mask.to(device),
        speech_pad_mask=torch.ones(1, packed.speech_latents.shape[0],
                                   dtype=torch.bool, device=device),
        speaker_linear_mask=packed.speaker_slot_mask.to(device),
    )
    slots = int(packed.speaker_slot_mask.sum().item())
    speaker = torch.zeros(slots, model.config.lm_speaker_embedding_dim, device=device)

    flag = model.config.scale_acoustic_latent
    model.config.scale_acoustic_latent = False
    try:
        embeds, _, _ = model.prepare_input_embeds(segment, lm_speaker_embedding=speaker)
    finally:
        model.config.scale_acoustic_latent = flag

    mask = packed_attention_mask(packed.segment_ids.to(device))
    additive = torch.zeros(mask.shape, dtype=embeds.dtype, device=device)
    additive.masked_fill_(~mask, torch.finfo(embeds.dtype).min)

    out = model.forward_lm(
        inputs_embeds=embeds,
        attention_mask=additive,
        position_ids=packed.position_ids.to(device),
        use_cache=False,
    )
    return out.last_hidden_state
