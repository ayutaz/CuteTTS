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

"""condition dropout のテスト。

論文の公開pretraining設定は condition dropout 0.1。
**何を落とすか**は論文にもコードにも規定がないため、このforkでは
speaker embedding と reference speech を対象と定め、テストで固定する。

推論側のLM-level CFGは「speakerもreferenceも無いbranch」を uncond として使う
（`inference/conditioning.build_guidance_plan`）ので、既定は joint（同時に落とす）。
"""

from __future__ import annotations

import pytest
import torch

from cutetts.training.objectives import (
    ConditionDropoutConfig,
    apply_condition_dropout,
    sample_condition_dropout,
)


def _conditions(batch: int = 6):
    g = torch.Generator().manual_seed(0)
    speaker = torch.randn(batch, 16, generator=g)
    reference = torch.randn(batch, 5, 2, 64, generator=g)
    reference_mask = torch.ones(batch, 5, dtype=torch.bool)
    return speaker, reference, reference_mask


# ------------------------------------------------------------------ sampling

def test_rate_zero_drops_nothing():
    drop = sample_condition_dropout(
        32, ConditionDropoutConfig(speaker=0.0, reference=0.0),
        generator=torch.Generator().manual_seed(1), device=torch.device("cpu"),
    )
    assert not bool(drop["speaker"].any())
    assert not bool(drop["reference"].any())


def test_rate_one_drops_everything():
    drop = sample_condition_dropout(
        32, ConditionDropoutConfig(speaker=1.0, reference=1.0),
        generator=torch.Generator().manual_seed(2), device=torch.device("cpu"),
    )
    assert bool(drop["speaker"].all())
    assert bool(drop["reference"].all())


def test_joint_dropout_uses_one_decision_for_both():
    """既定のjointでは speaker と reference が必ず同時に落ちる（推論のuncond branchと同形）。"""
    drop = sample_condition_dropout(
        512, ConditionDropoutConfig(speaker=0.5, reference=0.5, joint=True),
        generator=torch.Generator().manual_seed(3), device=torch.device("cpu"),
    )
    assert torch.equal(drop["speaker"], drop["reference"])


def test_independent_dropout_decorrelates_the_two():
    drop = sample_condition_dropout(
        512, ConditionDropoutConfig(speaker=0.5, reference=0.5, joint=False),
        generator=torch.Generator().manual_seed(4), device=torch.device("cpu"),
    )
    assert not torch.equal(drop["speaker"], drop["reference"])


def test_empirical_rate_is_close_to_the_configured_rate():
    drop = sample_condition_dropout(
        20000, ConditionDropoutConfig(speaker=0.1, reference=0.1),
        generator=torch.Generator().manual_seed(5), device=torch.device("cpu"),
    )
    rate = float(drop["speaker"].float().mean())
    assert 0.085 < rate < 0.115


def test_sampling_is_deterministic_under_the_same_generator():
    cfg = ConditionDropoutConfig(speaker=0.3, reference=0.3, joint=False)
    a = sample_condition_dropout(64, cfg, generator=torch.Generator().manual_seed(6),
                                 device=torch.device("cpu"))
    b = sample_condition_dropout(64, cfg, generator=torch.Generator().manual_seed(6),
                                 device=torch.device("cpu"))
    assert torch.equal(a["speaker"], b["speaker"])
    assert torch.equal(a["reference"], b["reference"])


def test_rate_outside_zero_one_is_rejected():
    with pytest.raises(ValueError):
        ConditionDropoutConfig(speaker=1.5)
    with pytest.raises(ValueError):
        ConditionDropoutConfig(reference=-0.1)


# ------------------------------------------------------------------ applying

def test_dropping_speaker_zeroes_only_the_selected_rows():
    speaker, reference, reference_mask = _conditions()
    drop = {"speaker": torch.tensor([True, False, False, True, False, False]),
            "reference": torch.zeros(6, dtype=torch.bool)}
    out_speaker, out_ref, out_mask = apply_condition_dropout(
        speaker, reference, reference_mask, drop
    )
    assert torch.all(out_speaker[0] == 0) and torch.all(out_speaker[3] == 0)
    for i in (1, 2, 4, 5):
        assert torch.equal(out_speaker[i], speaker[i])
    # referenceは無傷
    assert torch.equal(out_ref, reference)
    assert torch.equal(out_mask, reference_mask)


def test_dropping_reference_clears_mask_and_zeroes_latents():
    speaker, reference, reference_mask = _conditions()
    drop = {"speaker": torch.zeros(6, dtype=torch.bool),
            "reference": torch.tensor([False, True, False, False, False, True])}
    out_speaker, out_ref, out_mask = apply_condition_dropout(
        speaker, reference, reference_mask, drop
    )
    assert torch.equal(out_speaker, speaker)
    for i in (1, 5):
        assert torch.all(out_ref[i] == 0)
        assert not bool(out_mask[i].any())
    for i in (0, 2, 3, 4):
        assert torch.equal(out_ref[i], reference[i])
        assert bool(out_mask[i].all())


def test_dropping_nothing_returns_inputs_unchanged():
    speaker, reference, reference_mask = _conditions()
    drop = {"speaker": torch.zeros(6, dtype=torch.bool),
            "reference": torch.zeros(6, dtype=torch.bool)}
    out_speaker, out_ref, out_mask = apply_condition_dropout(
        speaker, reference, reference_mask, drop
    )
    assert torch.equal(out_speaker, speaker)
    assert torch.equal(out_ref, reference)
    assert torch.equal(out_mask, reference_mask)


def test_apply_does_not_mutate_its_inputs():
    speaker, reference, reference_mask = _conditions()
    speaker_before = speaker.clone()
    reference_before = reference.clone()
    mask_before = reference_mask.clone()
    drop = {"speaker": torch.ones(6, dtype=torch.bool),
            "reference": torch.ones(6, dtype=torch.bool)}
    apply_condition_dropout(speaker, reference, reference_mask, drop)
    assert torch.equal(speaker, speaker_before)
    assert torch.equal(reference, reference_before)
    assert torch.equal(reference_mask, mask_before)


def test_speaker_may_be_absent():
    _, reference, reference_mask = _conditions()
    drop = {"speaker": torch.ones(6, dtype=torch.bool),
            "reference": torch.zeros(6, dtype=torch.bool)}
    out_speaker, _, _ = apply_condition_dropout(None, reference, reference_mask, drop)
    assert out_speaker is None
