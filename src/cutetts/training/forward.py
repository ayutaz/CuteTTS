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

"""学習forward。`CuteTTSModel` を teacher forcing で1 step 走らせる。

推論と共有する経路:

* `model.prepare_input_embeds` … text embedding に speech / speaker を差し込む
* `model.forward_lm`           … Qwen3 backbone
* `model.head._predict`        … Diffusion Head の velocity 予測
* `model.stop_predictor`       … 2-class stop

推論と違う点は teacher forcing であることだけで、**正規化空間・stopラベルの位置・
head のシグネチャはすべて推論と同一**にする（`collator` と `objectives` を参照）。

`model.head` は fp32 固定なので、head へ渡す前に必ず fp32 へ揃える。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from cutetts.modeling.model import CuteTTSModel
from cutetts.modeling.segments import CuteTTSSegment
from cutetts.training.collator import TrainingBatch
from cutetts.training.objectives import (
    ConditionDropoutConfig,
    FlowBatch,
    apply_condition_dropout,
    build_flow_batch,
    flow_matching_loss,
    sample_condition_dropout,
    stop_loss,
    total_loss,
)

TRAINABLE_MODULES = ("locenc", "locenc_to_lm_proj", "lm_speaker_linear",
                     "qwen_backbone", "head", "stop_predictor")
"""D-005 の主案（Patch Encoder も train）に対応する既定の学習対象。"""


@dataclass(frozen=True)
class ForwardOutput:
    loss: Tensor
    flow_loss: Tensor
    stop_loss: Tensor
    num_targets: int
    flow_rows: int


def freeze_all_but(model: CuteTTSModel, trainable: tuple[str, ...] = TRAINABLE_MODULES) -> list[str]:
    """``trainable`` に挙げた子module以外の requires_grad を落とす。

    Audio VAE と Speaker Encoder は `CuteTTSModel` の外にあるので、
    そもそもここには現れない（D-003 / D-004 の freeze は構造的に保証される）。
    """
    unknown = [name for name in trainable if not hasattr(model, name)]
    if unknown:
        raise ValueError(f"unknown module names: {unknown}")
    frozen: list[str] = []
    for name, child in model.named_children():
        flag = name in trainable
        for parameter in child.parameters():
            parameter.requires_grad_(flag)
        if not flag:
            frozen.append(name)
    return frozen


def _gather_hidden(hidden: Tensor, batch_index: Tensor, positions: Tensor) -> Tensor:
    """``[B, L, C]`` から ``(batch_index, positions)`` の位置を取り出して ``[N, C]`` にする。"""
    return hidden[batch_index, positions]


def training_forward(
    model: CuteTTSModel,
    batch: TrainingBatch,
    *,
    speaker_embeddings: Tensor | None = None,
    flow_copies: int = 4,
    stop_weight: float = 1.0,
    stop_positive_weight: float | None = None,
    dropout: ConditionDropoutConfig | None = None,
    generator: torch.Generator | None = None,
) -> ForwardOutput:
    """teacher forcing の1 step を計算して loss を返す。

    ``speaker_embeddings`` は ``[B, S]``。`batch.speaker_slot_mask` が真の行数と
    一致していなければならない。
    """
    device = model.device
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    speech_mask = batch.speech_mask.to(device)
    speaker_mask = batch.speaker_slot_mask.to(device)
    speech_latents = batch.speech_latents.to(device)
    target_patches = batch.target_patches.to(device)
    previous_cond = batch.previous_cond.to(device)
    target_index = batch.target_batch_index.to(device)
    target_positions = batch.target_positions.to(device)
    stop_targets = batch.stop_targets.to(device)
    target_mask = batch.target_mask.to(device)

    speaker = None if speaker_embeddings is None else speaker_embeddings.to(device)

    # condition dropout は sample 単位で引く
    if dropout is not None:
        drop = sample_condition_dropout(
            batch.batch_size, dropout, generator=generator, device=device
        )
        # reference の drop は speech mask の reference 部分にだけ効かせたいが、
        # 現在の collator は reference と teacher-forced target を連結して持つ。
        # そのため speaker のみ落とし、reference の扱いは packing 実装時に見直す。
        if speaker is not None:
            speaker = speaker.clone()
            speaker[drop["speaker"]] = 0.0

    # `prepare_input_embeds` は [B, L] の mask と [B, S_total, P, D] を期待するため、
    # collator の平坦な speech_latents を batch 形状へ戻す
    segment = CuteTTSSegment(
        input_ids=input_ids,
        speech_tensor=speech_latents.unsqueeze(0) if speech_latents.dim() == 3 else speech_latents,
        speech_mask=speech_mask,
        speech_pad_mask=torch.ones(
            1, speech_latents.shape[0], dtype=torch.bool, device=device
        ),
        speaker_linear_mask=speaker_mask,
    )
    # 正規化は dataset 側で済ませてあるので、ここでは二重適用しない
    scale_flag = model.config.scale_acoustic_latent
    model.config.scale_acoustic_latent = False
    try:
        input_embeds, _contains_speech, _speech_features = model.prepare_input_embeds(
            segment,
            lm_speaker_embedding=None if speaker is None else speaker[
                speaker_mask.any(dim=1)
            ],
        )
    finally:
        model.config.scale_acoustic_latent = scale_flag

    lm_out = model.forward_lm(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask.long(),
        use_cache=False,
    )
    hidden = lm_out.last_hidden_state

    z = _gather_hidden(hidden, target_index, target_positions)

    # stop は target と同じ hidden から予測する
    stop_logits_flat = model.stop_predictor(z)
    stop_logits = stop_logits_flat.unsqueeze(0)
    stop_value = stop_loss(
        stop_logits.float(),
        stop_targets.unsqueeze(0),
        target_mask.unsqueeze(0),
        positive_weight=stop_positive_weight,
    )

    head_dtype = next(model.head.parameters()).dtype
    flow_batch: FlowBatch = build_flow_batch(
        target_patches.to(head_dtype),
        z.to(head_dtype),
        previous_cond.to(head_dtype),
        None if speaker is None else speaker[target_index].to(head_dtype),
        target_mask,
        copies=flow_copies,
        generator=generator,
    )
    predicted = model.head._predict(
        flow_batch.x_t,
        flow_batch.t,
        flow_batch.z,
        flow_batch.previous_cond,
        speaker_embedding=flow_batch.speaker,
        validate_conditions=False,
    )
    flow_value = flow_matching_loss(predicted, flow_batch)

    return ForwardOutput(
        loss=total_loss(flow_value, stop_value, stop_weight=stop_weight),
        flow_loss=flow_value.detach(),
        stop_loss=stop_value.detach(),
        num_targets=batch.num_targets,
        flow_rows=flow_batch.size,
    )
