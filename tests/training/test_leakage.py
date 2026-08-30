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

"""teacher forcing に答えが漏れていないことを検証する。

S0 の学習で flow loss が 1.02 -> 0.003 まで落ちたとき、
「条件付けが target patch そのものを含んでいるのではないか」を疑った。
flow matching は原理的に velocity を完全には当てられないので、
loss が 0 に近づくのは条件付けが答えを決定している場合だけである。

ここでは **摂動** で確かめる。target patch j を大きく動かし、
target i を予測する hidden ``z[i]`` が変化するかを見る。

    z[i] が変化してよいのは j < i のときだけ

対角（j == i）が動けば、patch i を予測する条件に patch i 自身が入っている。
上三角（j > i）が動けば、因果性が壊れて未来を見ている。
"""

from __future__ import annotations

import torch

from cutetts.modeling.segments import CuteTTSSegment
from cutetts.training.collator import build_training_sample, collate
from cutetts.training.forward import _gather_hidden

from .test_forward import DIM, PATCH, SPEAKER_DIM, _prompt, _tiny_model

PERTURBATION = 5.0
"""hidden が動けば必ず検出できる大きさ。潜在は正規化済みで std 約1。"""

TOLERANCE = 1e-5


def _hidden_at_targets(model, target_latents, reference_latents):
    sample = build_training_sample(
        utterance_id="u", prompt=_prompt(6),
        reference_latents=reference_latents, target_latents=target_latents)
    batch = collate([sample])
    segment = CuteTTSSegment(
        input_ids=batch.input_ids,
        speech_tensor=batch.speech_latents.unsqueeze(0),
        speech_mask=batch.speech_mask,
        speech_pad_mask=torch.ones(1, batch.speech_latents.shape[0], dtype=torch.bool),
        speaker_linear_mask=batch.speaker_slot_mask,
    )
    scale_flag = model.config.scale_acoustic_latent
    model.config.scale_acoustic_latent = False
    try:
        embeds, _, _ = model.prepare_input_embeds(
            segment, lm_speaker_embedding=torch.zeros(1, SPEAKER_DIM))
    finally:
        model.config.scale_acoustic_latent = scale_flag
    out = model.forward_lm(inputs_embeds=embeds,
                           attention_mask=batch.attention_mask.long(),
                           position_ids=None, use_cache=False)
    return _gather_hidden(out.last_hidden_state,
                          batch.target_batch_index, batch.target_positions)


def _influence_matrix(n_target: int = 6, seed: int = 3):
    """``[j, i]`` が True なら「patch j を動かすと z[i] が動く」。"""
    model = _tiny_model()
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(n_target, PATCH, DIM, generator=g)
    reference = torch.randn(3, PATCH, DIM, generator=g)
    base = _hidden_at_targets(model, target, reference)

    rows = []
    for j in range(n_target):
        perturbed = target.clone()
        perturbed[j] += PERTURBATION
        delta = (_hidden_at_targets(model, perturbed, reference) - base).abs().amax(dim=1)
        rows.append(delta > TOLERANCE)
    return torch.stack(rows)


def test_target_patch_never_influences_its_own_prediction():
    """patch i を予測する hidden は patch i を見ていない（対角が動かない）。"""
    influence = _influence_matrix()
    diagonal = torch.diagonal(influence)
    assert not bool(diagonal.any()), (
        "target patch が自分自身の予測条件に漏れている: "
        f"影響のあった index = {torch.nonzero(diagonal).flatten().tolist()}"
    )


def test_conditioning_never_sees_future_patches():
    """未来の patch は見えない。

    ``influence[j, i]`` は「patch j を動かすと z[i] が動く」なので、
    ``j > i``（= 下三角）が真になると z[i] が未来の patch を見ている。
    """
    influence = _influence_matrix()
    future = torch.tril(influence, diagonal=-1)
    assert not bool(future.any()), (
        "未来の patch が条件に漏れている: "
        f"(j, i) = {torch.nonzero(future).tolist()}"
    )


def test_past_patches_do_influence_later_predictions():
    """逆に、過去の patch は効いていなければならない（上三角がすべて真）。

    これが無いと「何も見ていないから漏れていない」だけになり、
    上の2つのテストが空振りする。
    """
    influence = _influence_matrix()
    past = torch.triu(influence, diagonal=1)
    expected = torch.triu(torch.ones_like(influence), diagonal=1).bool()
    assert torch.equal(past, expected), (
        "過去の patch が条件に効いていない。teacher forcing が繋がっていない可能性がある:\n"
        f"{past.int()}"
    )
