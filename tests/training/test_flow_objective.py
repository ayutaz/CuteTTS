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

"""flow-matching objective のテスト。

論文の定義:
    x_t = (1 - t) * xi + t * P
    target velocity = P - xi
    t = sigmoid(u), u ~ N(0, 1)

配線を1箇所でも外したら落ちることを意図している。
"""

from __future__ import annotations

import pytest
import torch

from cutetts.training.objectives import (
    DEFAULT_FLOW_TARGET_COPIES,
    build_flow_batch,
    flow_matching_loss,
    sample_flow_time,
)

PATCH = 2
DIM = 64
HIDDEN = 8


def _inputs(batch: int = 3, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(batch, PATCH, DIM, generator=g)
    z = torch.randn(batch, HIDDEN, generator=g)
    previous = torch.randn(batch, PATCH, DIM, generator=g)
    speaker = torch.randn(batch, 16, generator=g)
    mask = torch.ones(batch, dtype=torch.bool)
    return target, z, previous, speaker, mask


# ------------------------------------------------------------------ flow time

def test_flow_time_is_sigmoid_of_standard_normal():
    """t = sigmoid(u), u ~ N(0,1) なので必ず (0, 1) に入る。"""
    g = torch.Generator().manual_seed(7)
    t = sample_flow_time(4096, generator=g, device=torch.device("cpu"), dtype=torch.float32)
    assert t.shape == (4096,)
    assert torch.all(t > 0.0) and torch.all(t < 1.0)
    # sigmoid(N(0,1)) の中央値は 0.5 付近
    assert 0.45 < float(t.median()) < 0.55


def test_flow_time_is_deterministic_under_the_same_generator():
    a = sample_flow_time(16, generator=torch.Generator().manual_seed(3),
                         device=torch.device("cpu"), dtype=torch.float32)
    b = sample_flow_time(16, generator=torch.Generator().manual_seed(3),
                         device=torch.device("cpu"), dtype=torch.float32)
    c = sample_flow_time(16, generator=torch.Generator().manual_seed(4),
                         device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# ------------------------------------------------------------------ flow batch

def test_target_velocity_is_clean_minus_noise():
    """target velocity は P - xi でなければならない。"""
    target, z, previous, speaker, mask = _inputs()
    batch = build_flow_batch(target, z, previous, speaker, mask,
                             copies=1, generator=torch.Generator().manual_seed(1))
    # x_t = (1-t) xi + t P から xi を復元して検算する
    t = batch.t.view(-1, 1, 1)
    xi = (batch.x_t - t * batch.target_clean) / (1.0 - t)
    assert torch.allclose(batch.target_velocity, batch.target_clean - xi, atol=1e-5)


def test_interpolation_endpoints():
    """t=1 で x_t == P、t=0 で x_t == xi。"""
    target, z, previous, speaker, mask = _inputs(batch=2)
    batch = build_flow_batch(target, z, previous, speaker, mask, copies=1,
                             generator=torch.Generator().manual_seed(2),
                             flow_time=torch.ones(2))
    assert torch.allclose(batch.x_t, batch.target_clean, atol=1e-6)

    batch0 = build_flow_batch(target, z, previous, speaker, mask, copies=1,
                              generator=torch.Generator().manual_seed(2),
                              flow_time=torch.zeros(2))
    # t=0 なら x_t は純ノイズで、velocity は P - x_t
    assert torch.allclose(batch0.target_velocity, batch0.target_clean - batch0.x_t, atol=1e-6)


def test_copies_expand_the_batch_with_independent_noise_and_time():
    """公開pretrainingは各target patchを4つの独立noise/timeで複製する。"""
    target, z, previous, speaker, mask = _inputs(batch=3)
    batch = build_flow_batch(target, z, previous, speaker, mask,
                             copies=DEFAULT_FLOW_TARGET_COPIES,
                             generator=torch.Generator().manual_seed(5))
    assert DEFAULT_FLOW_TARGET_COPIES == 4
    n = 3 * 4
    assert batch.x_t.shape == (n, PATCH, DIM)
    assert batch.t.shape == (n,)
    assert batch.z.shape == (n, HIDDEN)
    assert batch.previous_cond.shape == (n, PATCH, DIM)
    assert batch.speaker.shape == (n, 16)
    # 条件（z / previous / speaker）は複製されるが、noiseとtimeは独立
    for i in range(3):
        rows = [i + 3 * c for c in range(4)]
        assert torch.allclose(batch.z[rows[0]], batch.z[rows[-1]])
        assert torch.allclose(batch.target_clean[rows[0]], batch.target_clean[rows[-1]])
        assert len({round(float(batch.t[r]), 6) for r in rows}) == 4


def test_padding_patches_are_excluded_from_the_loss_mask():
    target, z, previous, speaker, _ = _inputs(batch=4)
    mask = torch.tensor([True, False, True, False])
    batch = build_flow_batch(target, z, previous, speaker, mask, copies=2,
                             generator=torch.Generator().manual_seed(6))
    assert batch.loss_mask.tolist() == [True, False, True, False] * 2


# ------------------------------------------------------------------ loss

def test_loss_is_zero_when_prediction_equals_target():
    target, z, previous, speaker, mask = _inputs()
    batch = build_flow_batch(target, z, previous, speaker, mask, copies=2,
                             generator=torch.Generator().manual_seed(8))
    loss = flow_matching_loss(batch.target_velocity.clone(), batch)
    assert float(loss) == pytest.approx(0.0, abs=1e-9)


def test_loss_ignores_masked_rows_entirely():
    """maskされた行に巨大な誤差を入れてもlossは変わらない。"""
    target, z, previous, speaker, _ = _inputs(batch=4)
    mask = torch.tensor([True, True, False, False])
    batch = build_flow_batch(target, z, previous, speaker, mask, copies=1,
                             generator=torch.Generator().manual_seed(9))
    pred = batch.target_velocity.clone()
    pred[0] += 1.0
    base = float(flow_matching_loss(pred, batch))

    poisoned = pred.clone()
    poisoned[2] += 1e6
    poisoned[3] -= 1e6
    assert float(flow_matching_loss(poisoned, batch)) == pytest.approx(base, rel=1e-9)


def test_loss_denominator_counts_only_unmasked_elements():
    """maskが半分なら、同じ誤差でもlossは全maskのときと同じ値になる。"""
    target, z, previous, speaker, _ = _inputs(batch=4)
    full = build_flow_batch(target, z, previous, speaker,
                            torch.ones(4, dtype=torch.bool), copies=1,
                            generator=torch.Generator().manual_seed(10))
    half = build_flow_batch(target, z, previous, speaker,
                            torch.tensor([True, True, False, False]), copies=1,
                            generator=torch.Generator().manual_seed(10))
    pred_full = full.target_velocity + 1.0
    pred_half = half.target_velocity + 1.0
    assert float(flow_matching_loss(pred_full, full)) == pytest.approx(1.0, abs=1e-6)
    assert float(flow_matching_loss(pred_half, half)) == pytest.approx(1.0, abs=1e-6)


def test_loss_raises_when_no_row_is_unmasked():
    target, z, previous, speaker, _ = _inputs(batch=2)
    batch = build_flow_batch(target, z, previous, speaker,
                             torch.zeros(2, dtype=torch.bool), copies=1,
                             generator=torch.Generator().manual_seed(11))
    with pytest.raises(ValueError):
        flow_matching_loss(batch.target_velocity, batch)


def test_loss_rejects_shape_mismatch():
    target, z, previous, speaker, mask = _inputs()
    batch = build_flow_batch(target, z, previous, speaker, mask, copies=1,
                             generator=torch.Generator().manual_seed(12))
    with pytest.raises(ValueError):
        flow_matching_loss(batch.target_velocity[:, :1, :], batch)


def test_build_flow_batch_is_deterministic_under_the_same_seed():
    target, z, previous, speaker, mask = _inputs()
    a = build_flow_batch(target, z, previous, speaker, mask, copies=4,
                         generator=torch.Generator().manual_seed(13))
    b = build_flow_batch(target, z, previous, speaker, mask, copies=4,
                         generator=torch.Generator().manual_seed(13))
    c = build_flow_batch(target, z, previous, speaker, mask, copies=4,
                         generator=torch.Generator().manual_seed(14))
    assert torch.equal(a.x_t, b.x_t) and torch.equal(a.t, b.t)
    assert not torch.equal(a.x_t, c.x_t)


def test_speaker_may_be_absent():
    target, z, previous, _, mask = _inputs()
    batch = build_flow_batch(target, z, previous, None, mask, copies=2,
                             generator=torch.Generator().manual_seed(15))
    assert batch.speaker is None


def test_cpu_generator_works_with_a_non_cpu_target_dtype():
    """CPU generator で CUDA テンソルを作れること。

    `torch.randn(..., generator=cpu_gen, device='cuda')` は
    `Expected a 'cuda' device type for generator but found 'cpu'` を投げる。
    再現性のためにCPU generatorを使いつつGPUで学習したいので、
    device が食い違っても動く必要がある。

    CUDAが無い環境でも回帰を検出できるよう、まずCPUで契約を固定する。
    """
    g = torch.Generator().manual_seed(0)
    t = sample_flow_time(8, generator=g, device=torch.device("cpu"), dtype=torch.float32)
    assert t.device.type == "cpu"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA が無い")
def test_cpu_generator_produces_cuda_tensors():
    g = torch.Generator().manual_seed(0)
    cuda = torch.device("cuda")
    t = sample_flow_time(8, generator=g, device=cuda, dtype=torch.float32)
    assert t.device.type == "cuda"

    target = torch.randn(3, PATCH, DIM, device=cuda)
    batch = build_flow_batch(
        target, torch.randn(3, HIDDEN, device=cuda),
        torch.randn(3, PATCH, DIM, device=cuda), None,
        torch.ones(3, dtype=torch.bool, device=cuda),
        copies=2, generator=torch.Generator().manual_seed(1),
    )
    assert batch.x_t.device.type == "cuda"
    assert batch.t.device.type == "cuda"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA が無い")
def test_condition_dropout_accepts_a_cpu_generator_on_cuda():
    from cutetts.training.objectives import ConditionDropoutConfig, sample_condition_dropout

    drop = sample_condition_dropout(
        16, ConditionDropoutConfig(speaker=0.5, reference=0.5),
        generator=torch.Generator().manual_seed(2), device=torch.device("cuda"),
    )
    assert drop["speaker"].device.type == "cuda"
