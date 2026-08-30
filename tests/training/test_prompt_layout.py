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

"""prompt 並びが推論と一致することのテスト。

**ここがずれると学習と推論で条件が食い違い、S0のゲートが無意味になる。**
"""

from __future__ import annotations

import pytest
import torch

from cutetts.training.collator import build_training_sample
from cutetts.training.prompt import PromptLayout

PATCH, DIM = 2, 64


def _layout(lead=3, mid=2, trail=5, speaker=True) -> PromptLayout:
    return PromptLayout(
        leading_ids=torch.arange(100, 100 + lead, dtype=torch.long),
        middle_ids=torch.arange(200, 200 + mid, dtype=torch.long),
        trailing_ids=torch.arange(300, 300 + trail, dtype=torch.long),
        has_speaker_slot=speaker,
    )


def test_prompt_order_matches_inference():
    """並びは leading / speaker / middle / reference / trailing。"""
    layout = _layout(lead=3, mid=2, trail=5)
    sample = build_training_sample(
        utterance_id="u", prompt=layout,
        reference_latents=torch.randn(4, PATCH, DIM),
        target_latents=torch.randn(3, PATCH, DIM),
    )
    ids = sample.input_ids.tolist()
    speech = torch.nonzero(sample.speech_mask).flatten().tolist()
    slot = torch.nonzero(sample.speaker_slot_mask).flatten().tolist()

    assert ids[0:3] == [100, 101, 102]          # leading
    assert slot == [3]                           # speaker slot
    assert ids[4:6] == [200, 201]                # middle
    assert speech[:4] == [6, 7, 8, 9]            # reference speech
    assert ids[10:15] == [300, 301, 302, 303, 304]   # trailing（target text）


def test_prefix_length_covers_everything_before_teacher_forcing():
    layout = _layout(lead=3, mid=2, trail=5)
    sample = build_training_sample(
        utterance_id="u", prompt=layout,
        reference_latents=torch.randn(4, PATCH, DIM),
        target_latents=torch.randn(3, PATCH, DIM),
    )
    assert sample.prefix_length == 3 + 1 + 2 + 4 + 5
    assert sample.target_positions[0].item() == sample.prefix_length - 1


def test_tts_layout_has_no_speaker_slot_or_reference():
    layout = _layout(lead=0, mid=0, trail=6, speaker=False)
    sample = build_training_sample(
        utterance_id="u", prompt=layout,
        reference_latents=torch.zeros(0, PATCH, DIM),
        target_latents=torch.randn(3, PATCH, DIM),
    )
    assert int(sample.speaker_slot_mask.sum()) == 0
    assert sample.reference_patch_count == 0
    assert sample.prefix_length == 6


def test_target_text_lands_after_the_reference_speech():
    """target text は reference speech より後ろでなければならない。

    推論がその順序なので、逆にすると条件付けが変わる。
    """
    layout = _layout(lead=2, mid=1, trail=4)
    sample = build_training_sample(
        utterance_id="u", prompt=layout,
        reference_latents=torch.randn(3, PATCH, DIM),
        target_latents=torch.randn(2, PATCH, DIM),
    )
    speech = torch.nonzero(sample.speech_mask).flatten().tolist()
    last_reference = speech[2]
    trailing_start = last_reference + 1
    assert sample.input_ids[trailing_start].item() == 300
