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

"""stop target / stop loss のテスト。

推論側の意味論（`inference/generation.py`）:

    stop_after_current_patch = _stop_after_current_patch(lm, last_hidden)
    ... patch を生成 ...
    if stop_after_current_patch: break

つまり **位置 i の hidden は「patch i が最終patchか」を予測する**。
ラベルが1つずれると推論の停止位置がずれるため、ずれを検出するテストを置く。
"""

from __future__ import annotations

import pytest
import torch

from cutetts.training.objectives import (
    STOP_CONTINUE,
    STOP_STOP,
    build_stop_targets,
    stop_loss,
)


def test_label_is_set_on_the_last_valid_patch_only():
    """長さNのtargetでは位置 N-1 にだけ stop が立つ。"""
    mask = torch.tensor([[True] * 5])
    targets = build_stop_targets(mask)
    assert targets.tolist() == [[STOP_CONTINUE] * 4 + [STOP_STOP]]


def test_label_position_shifted_by_one_is_detected():
    """1位置ずらした実装なら、このテストが落ちる。"""
    mask = torch.tensor([[True] * 4])
    targets = build_stop_targets(mask)
    correct = torch.tensor([[STOP_CONTINUE, STOP_CONTINUE, STOP_CONTINUE, STOP_STOP]])
    off_by_one_early = torch.tensor([[STOP_CONTINUE, STOP_CONTINUE, STOP_STOP, STOP_CONTINUE]])
    assert torch.equal(targets, correct)
    assert not torch.equal(targets, off_by_one_early)


def test_ragged_batch_marks_each_sequence_own_end():
    mask = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
        [True, False, False, False],
    ])
    targets = build_stop_targets(mask)
    assert targets[0].tolist()[:3] == [STOP_CONTINUE, STOP_CONTINUE, STOP_STOP]
    assert targets[1].tolist()[:2] == [STOP_CONTINUE, STOP_STOP]
    assert targets[2].tolist()[:1] == [STOP_STOP]


def test_single_patch_sequence_stops_immediately():
    targets = build_stop_targets(torch.tensor([[True, False, False]]))
    assert targets[0, 0].item() == STOP_STOP


def test_empty_sequence_raises():
    with pytest.raises(ValueError):
        build_stop_targets(torch.tensor([[False, False]]))


# ------------------------------------------------------------------ loss

def _logits_for(targets, mask, *, confident: float = 20.0):
    """targets を完全に当てる logits を作る。"""
    b, t = targets.shape
    logits = torch.zeros(b, t, 2)
    logits[..., STOP_CONTINUE] = torch.where(targets == STOP_CONTINUE, confident, -confident)
    logits[..., STOP_STOP] = torch.where(targets == STOP_STOP, confident, -confident)
    return logits


def test_loss_is_near_zero_for_perfect_predictions():
    mask = torch.tensor([[True] * 4])
    targets = build_stop_targets(mask)
    loss = stop_loss(_logits_for(targets, mask), targets, mask)
    assert float(loss) < 1e-6


def test_padding_positions_do_not_contribute():
    """padding位置に最悪の予測を入れてもlossが変わらない。"""
    mask = torch.tensor([[True, True, False, False]])
    targets = build_stop_targets(mask)
    logits = _logits_for(targets, mask)
    base = float(stop_loss(logits, targets, mask))

    poisoned = logits.clone()
    poisoned[0, 2] = torch.tensor([-50.0, 50.0])
    poisoned[0, 3] = torch.tensor([50.0, -50.0])
    assert float(stop_loss(poisoned, targets, mask)) == pytest.approx(base, abs=1e-9)


def test_denominator_is_the_number_of_valid_positions():
    """有効位置が2でも4でも、同じ誤差なら同じ値になる。"""
    short_mask = torch.tensor([[True, True, False, False]])
    long_mask = torch.tensor([[True] * 4])
    for mask in (short_mask, long_mask):
        targets = build_stop_targets(mask)
        logits = torch.zeros(*targets.shape, 2)  # 常に五分五分 → log 2
        assert float(stop_loss(logits, targets, mask)) == pytest.approx(0.6931, abs=1e-3)


def test_positive_weight_upweights_errors_on_the_rare_stop_class():
    """stopは系列に1つしかない。stop位置の誤りが重み分だけ重く効くこと。

    重み付き平均なので、全位置のlossが等しい入力では値は変わらない
    （重みは分子と分母の両方に入る）。効くのは**誤りが偏っているとき**で、
    そこを検証する。
    """
    mask = torch.tensor([[True] * 8])
    targets = build_stop_targets(mask)

    # stop位置(7)だけ外し、continue位置はすべて正解させる
    logits = torch.full((1, 8, 2), -10.0)
    logits[0, :7, STOP_CONTINUE] = 10.0
    logits[0, 7, STOP_CONTINUE] = 10.0      # ← 最終patchでcontinueと誤答
    logits[0, 7, STOP_STOP] = -10.0

    plain = float(stop_loss(logits, targets, mask))
    weighted = float(stop_loss(logits, targets, mask, positive_weight=7.0))
    assert weighted > plain

    # 逆にcontinue位置だけ外した場合、stopへの重み付けはlossを下げる
    # （分母の重み総和が増え、誤りはcontinue側にあるため）
    flipped = torch.full((1, 8, 2), -10.0)
    flipped[0, :, STOP_CONTINUE] = 10.0
    flipped[0, 0, STOP_CONTINUE] = -10.0    # ← 先頭でstopと誤答
    flipped[0, 0, STOP_STOP] = 10.0
    flipped[0, 7, STOP_CONTINUE] = -10.0
    flipped[0, 7, STOP_STOP] = 10.0         # 最終patchは正解
    assert float(stop_loss(flipped, targets, mask, positive_weight=7.0)) < float(
        stop_loss(flipped, targets, mask)
    )


def test_positive_weight_does_not_change_a_uniform_loss():
    """重み付き平均の性質: 全位置のlossが等しいなら重みで値は動かない。"""
    mask = torch.tensor([[True] * 8])
    targets = build_stop_targets(mask)
    logits = torch.zeros(1, 8, 2)
    assert float(stop_loss(logits, targets, mask, positive_weight=7.0)) == pytest.approx(
        float(stop_loss(logits, targets, mask)), rel=1e-6
    )


def test_positive_weight_of_one_matches_unweighted():
    mask = torch.tensor([[True] * 5])
    targets = build_stop_targets(mask)
    logits = torch.randn(1, 5, 2, generator=torch.Generator().manual_seed(0))
    assert float(stop_loss(logits, targets, mask, positive_weight=1.0)) == pytest.approx(
        float(stop_loss(logits, targets, mask)), rel=1e-6
    )


def test_loss_rejects_shape_mismatch():
    mask = torch.tensor([[True] * 3])
    targets = build_stop_targets(mask)
    with pytest.raises(ValueError):
        stop_loss(torch.zeros(1, 4, 2), targets, mask)


def test_all_padding_raises():
    mask = torch.tensor([[True, False]])
    targets = build_stop_targets(mask)
    with pytest.raises(ValueError):
        stop_loss(torch.zeros(1, 2, 2), targets, torch.zeros(1, 2, dtype=torch.bool))
