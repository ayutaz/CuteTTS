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

"""sequence packing のテスト。

P2 のゴール「packing が unpacked の結果を変えないこと」を検証する。
packing は **正しさを確認した後に足す最適化** であり、
数値を変えてはならない。
"""

from __future__ import annotations

import pytest
import torch

from cutetts.training.collator import build_training_sample, collate
from cutetts.training.forward import training_forward
from cutetts.training.packing import pack_samples, packed_attention_mask

from .test_forward import DIM, PATCH, SPEAKER_DIM, _prompt, _tiny_model


def _sample(uid: str, *, n_target: int, n_reference: int, n_text: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return build_training_sample(
        utterance_id=uid,
        prompt=_prompt(n_text),
        reference_latents=torch.randn(n_reference, PATCH, DIM, generator=g),
        target_latents=torch.randn(n_target, PATCH, DIM, generator=g),
    )


# ------------------------------------------------------------------ structure

def test_packing_concatenates_samples_into_one_row():
    a = _sample("a", n_target=3, n_reference=2, n_text=4, seed=1)
    b = _sample("b", n_target=4, n_reference=3, n_text=5, seed=2)
    packed = pack_samples([a, b])
    assert packed.input_ids.shape[0] == 1
    assert packed.input_ids.shape[1] == a.length + b.length
    assert packed.segment_ids.tolist()[0] == [0] * a.length + [1] * b.length


def test_packed_target_positions_are_offset_by_the_preceding_samples():
    a = _sample("a", n_target=3, n_reference=2, n_text=4, seed=3)
    b = _sample("b", n_target=2, n_reference=1, n_text=3, seed=4)
    packed = pack_samples([a, b])
    expected = torch.cat([a.target_positions, b.target_positions + a.length])
    assert torch.equal(packed.target_positions, expected)


def test_packed_targets_keep_their_stop_labels():
    a = _sample("a", n_target=3, n_reference=2, n_text=4, seed=5)
    b = _sample("b", n_target=2, n_reference=1, n_text=3, seed=6)
    packed = pack_samples([a, b])
    assert torch.equal(packed.stop_targets, torch.cat([a.stop_targets, b.stop_targets]))


def test_attention_mask_blocks_across_segment_boundaries():
    """packed sample 同士は絶対に attention してはならない。"""
    a = _sample("a", n_target=2, n_reference=1, n_text=2, seed=7)
    b = _sample("b", n_target=2, n_reference=1, n_text=2, seed=8)
    packed = pack_samples([a, b])
    mask = packed_attention_mask(packed.segment_ids)
    # [1, 1, L, L] の bool。True が「見てよい」
    assert mask.shape == (1, 1, packed.length, packed.length)
    la = a.length
    # a の位置から b の位置は見えない
    assert not bool(mask[0, 0, :la, la:].any())
    # b の位置から a の位置も見えない
    assert not bool(mask[0, 0, la:, :la].any())


def test_attention_mask_is_causal_inside_a_segment():
    a = _sample("a", n_target=3, n_reference=1, n_text=2, seed=9)
    packed = pack_samples([a])
    mask = packed_attention_mask(packed.segment_ids)[0, 0]
    n = a.length
    for i in range(n):
        for j in range(n):
            assert bool(mask[i, j]) == (j <= i), f"({i},{j})"


def test_position_ids_restart_at_each_segment():
    a = _sample("a", n_target=2, n_reference=1, n_text=3, seed=10)
    b = _sample("b", n_target=3, n_reference=2, n_text=2, seed=11)
    packed = pack_samples([a, b])
    expected = list(range(a.length)) + list(range(b.length))
    assert packed.position_ids.tolist()[0] == expected


def test_packing_a_single_sample_matches_collate_shapes():
    a = _sample("a", n_target=3, n_reference=2, n_text=4, seed=12)
    packed = pack_samples([a])
    plain = collate([a])
    assert packed.input_ids.shape == plain.input_ids.shape
    assert torch.equal(packed.target_positions, plain.target_positions)
    assert torch.equal(packed.stop_targets, plain.stop_targets)


def test_packing_rejects_an_empty_list():
    with pytest.raises(ValueError):
        pack_samples([])


# ------------------------------------------------------- numerical equivalence

def test_packed_and_unpacked_losses_agree():
    """同じ sample 集合なら packing の有無で loss が変わらない。"""
    model = _tiny_model()
    a = _sample("a", n_target=3, n_reference=2, n_text=4, seed=20)
    b = _sample("b", n_target=4, n_reference=3, n_text=5, seed=21)
    g = torch.Generator().manual_seed(22)
    speaker = torch.randn(2, SPEAKER_DIM, generator=g)

    unpacked = training_forward(model, collate([a, b]), speaker_embeddings=speaker,
                                flow_copies=2, generator=torch.Generator().manual_seed(30))
    packed = training_forward(model, pack_samples([a, b]).to_batch(),
                              speaker_embeddings=speaker, flow_copies=2,
                              generator=torch.Generator().manual_seed(30))
    assert float(packed.flow_loss) == pytest.approx(float(unpacked.flow_loss), rel=2e-3)
    assert float(packed.stop_loss) == pytest.approx(float(unpacked.stop_loss), rel=2e-3)


def test_packed_hidden_states_match_unpacked_ones():
    """境界の遮断が効いていれば、packed の hidden は個別実行と一致する。"""
    model = _tiny_model()
    a = _sample("a", n_target=3, n_reference=2, n_text=4, seed=23)
    b = _sample("b", n_target=2, n_reference=1, n_text=3, seed=24)

    from cutetts.training.packing import packed_hidden_states

    solo_a = packed_hidden_states(model, pack_samples([a]))
    solo_b = packed_hidden_states(model, pack_samples([b]))
    together = packed_hidden_states(model, pack_samples([a, b]))

    assert torch.allclose(together[0, : a.length], solo_a[0], atol=2e-4)
    assert torch.allclose(together[0, a.length :], solo_b[0], atol=2e-4)
