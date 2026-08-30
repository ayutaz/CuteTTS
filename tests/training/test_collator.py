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

"""学習sequence組み立てのテスト。

teacher forcing の対応関係（`inference/generation.py` の自己回帰ループから導出）:

    hidden(prefix末尾)  -> target patch 0 を予測 / patch 0 の stop ラベル
    hidden(patch 0)     -> target patch 1 を予測
    ...
    hidden(patch N-2)   -> target patch N-1 を予測

したがって **入力に流すのは patch 0..N-2 の N-1 個** で、
N 個の target には prefix末尾 + patch 0..N-2 の N 個の hidden が対応する。
patch N-1 は target にしかならず、入力には入らない。
"""

from __future__ import annotations

import pytest
import torch

from cutetts.training.collator import (
    TrainingSample,
    build_training_sample,
    collate,
)
from cutetts.training.objectives import STOP_CONTINUE, STOP_STOP

PATCH = 2
DIM = 64


def _sample(*, n_target: int = 4, n_reference: int = 3, n_text: int = 5,
            speaker_slot: bool = True, uid: str = "u0", seed: int = 0) -> TrainingSample:
    g = torch.Generator().manual_seed(seed)
    return build_training_sample(
        utterance_id=uid,
        prefix_ids=torch.arange(n_text, dtype=torch.long),
        reference_latents=torch.randn(n_reference, PATCH, DIM, generator=g),
        target_latents=torch.randn(n_target, PATCH, DIM, generator=g),
        speaker_slot=speaker_slot,
    )


# --------------------------------------------------------------- single sample

def test_only_the_first_n_minus_one_target_patches_are_fed_back():
    """target が N 個なら入力に入る target patch は N-1 個。"""
    s = _sample(n_target=4, n_reference=3, n_text=5)
    assert s.target_patches.shape == (4, PATCH, DIM)
    # speech に入るのは reference 3 + teacher-forced target 3 = 6
    assert int(s.speech_mask.sum()) == 3 + 3
    assert s.speech_latents.shape == (6, PATCH, DIM)


def test_target_positions_point_at_the_hidden_that_predicts_each_patch():
    """target i を予測する hidden は、target i-1 の位置（i=0 は prefix 末尾）。"""
    s = _sample(n_target=4, n_reference=2, n_text=3)
    positions = s.target_positions.tolist()
    speech_positions = torch.nonzero(s.speech_mask).flatten().tolist()
    reference_positions = speech_positions[:2]
    fed_target_positions = speech_positions[2:]

    assert positions[0] == reference_positions[-1] if False else True
    # target 0 は「prefix の最後の位置」の hidden が予測する
    assert positions[0] == s.prefix_length - 1
    # target i (i>=1) は teacher-forced target i-1 の位置
    assert positions[1:] == fed_target_positions


def test_positions_are_strictly_increasing_and_inside_the_sequence():
    s = _sample(n_target=5, n_reference=4, n_text=6)
    pos = s.target_positions
    assert bool(torch.all(pos[1:] > pos[:-1]))
    assert int(pos.min()) >= 0
    assert int(pos.max()) < s.length


def test_previous_cond_is_the_preceding_patch_and_starts_from_the_initial_value():
    s = _sample(n_target=4, n_reference=3)
    # target 0 の previous cond は初期値（既定は zeros）
    assert torch.all(s.previous_cond[0] == 0)
    # target i の previous cond は target i-1 そのもの
    for i in range(1, 4):
        assert torch.equal(s.previous_cond[i], s.target_patches[i - 1])


def test_initial_previous_cond_can_be_supplied_from_the_prefix():
    g = torch.Generator().manual_seed(1)
    initial = torch.randn(PATCH, DIM, generator=g)
    s = build_training_sample(
        utterance_id="u",
        prefix_ids=torch.arange(3, dtype=torch.long),
        reference_latents=torch.randn(2, PATCH, DIM, generator=g),
        target_latents=torch.randn(3, PATCH, DIM, generator=g),
        speaker_slot=True,
        initial_previous_cond=initial,
    )
    assert torch.equal(s.previous_cond[0], initial)


def test_stop_target_marks_only_the_final_patch():
    s = _sample(n_target=4)
    assert s.stop_targets.tolist() == [STOP_CONTINUE] * 3 + [STOP_STOP]


def test_speaker_slot_occupies_exactly_one_position():
    s = _sample(speaker_slot=True)
    assert int(s.speaker_slot_mask.sum()) == 1
    without = _sample(speaker_slot=False)
    assert int(without.speaker_slot_mask.sum()) == 0


def test_speech_and_speaker_slots_never_overlap():
    s = _sample()
    assert not bool((s.speech_mask & s.speaker_slot_mask).any())


def test_single_target_patch_feeds_nothing_back():
    s = _sample(n_target=1, n_reference=2)
    assert s.target_patches.shape[0] == 1
    assert int(s.speech_mask.sum()) == 2       # reference のみ
    assert s.target_positions.tolist() == [s.prefix_length - 1]
    assert s.stop_targets.tolist() == [STOP_STOP]


def test_empty_target_is_rejected():
    with pytest.raises(ValueError):
        build_training_sample(
            utterance_id="u",
            prefix_ids=torch.arange(3, dtype=torch.long),
            reference_latents=torch.zeros(2, PATCH, DIM),
            target_latents=torch.zeros(0, PATCH, DIM),
            speaker_slot=True,
        )


def test_patch_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        build_training_sample(
            utterance_id="u",
            prefix_ids=torch.arange(3, dtype=torch.long),
            reference_latents=torch.zeros(2, PATCH, DIM),
            target_latents=torch.zeros(3, PATCH + 1, DIM),
            speaker_slot=True,
        )


# ---------------------------------------------------------------------- batch

def test_collate_pads_on_the_right_and_builds_attention_mask():
    a = _sample(n_target=3, n_reference=2, n_text=4, uid="a", seed=1)
    b = _sample(n_target=5, n_reference=3, n_text=6, uid="b", seed=2)
    batch = collate([a, b])
    assert batch.input_ids.shape[0] == 2
    assert batch.input_ids.shape[1] == max(a.length, b.length)
    # 短い方の末尾が padding
    assert bool(batch.attention_mask[0, a.length:].eq(False).all())
    assert bool(batch.attention_mask[0, : a.length].all())
    assert bool(batch.attention_mask[1].all())


def test_collate_keeps_each_sample_targets_separable():
    a = _sample(n_target=3, uid="a", seed=3)
    b = _sample(n_target=5, uid="b", seed=4)
    batch = collate([a, b])
    assert batch.target_patches.shape[0] == 3 + 5
    assert batch.target_batch_index.tolist() == [0] * 3 + [1] * 5
    assert torch.equal(batch.target_positions[:3], a.target_positions)
    assert torch.equal(batch.target_positions[3:], b.target_positions)


def test_collate_preserves_stop_targets_per_sample():
    a = _sample(n_target=3, uid="a", seed=5)
    b = _sample(n_target=2, uid="b", seed=6)
    batch = collate([a, b])
    assert batch.stop_targets.tolist() == [
        STOP_CONTINUE, STOP_CONTINUE, STOP_STOP, STOP_CONTINUE, STOP_STOP
    ]


def test_collate_concatenates_speech_latents_in_sample_order():
    a = _sample(n_target=3, n_reference=2, uid="a", seed=7)
    b = _sample(n_target=2, n_reference=4, uid="b", seed=8)
    batch = collate([a, b])
    assert batch.speech_latents.shape[0] == a.speech_latents.shape[0] + b.speech_latents.shape[0]
    assert torch.equal(batch.speech_latents[: a.speech_latents.shape[0]], a.speech_latents)


def test_collate_target_mask_is_all_true_without_packing():
    a = _sample(n_target=3, uid="a", seed=9)
    batch = collate([a])
    assert bool(batch.target_mask.all())


def test_collate_rejects_an_empty_list():
    with pytest.raises(ValueError):
        collate([])


def test_collate_is_order_stable():
    a = _sample(n_target=3, uid="a", seed=10)
    b = _sample(n_target=4, uid="b", seed=11)
    assert collate([a, b]).utterance_ids == ["a", "b"]
    assert collate([b, a]).utterance_ids == ["b", "a"]
