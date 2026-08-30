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

"""checkpoint save / resume のテスト。

P2 のゴール「save/resume 前後で次stepが一致する」を、
**中断なしで走らせた場合と数値が一致するか**で検証する。
"""

from __future__ import annotations

import pytest
import torch

from cutetts.training.checkpointing import (
    TrainingState,
    export_for_inference,
    load_training_state,
    save_training_state,
)
from cutetts.training.forward import freeze_all_but, training_forward

from .test_forward import PATCH, DIM, SPEAKER_DIM, _batch, _tiny_model


def _make(seed: int = 0):
    model = _tiny_model(seed)
    freeze_all_but(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    return model, optimizer


def _step(model, optimizer, batch, speaker, generator) -> float:
    optimizer.zero_grad(set_to_none=True)
    out = training_forward(model, batch, speaker_embeddings=speaker, flow_copies=2,
                           generator=generator)
    out.loss.backward()
    optimizer.step()
    return float(out.loss)


def test_resume_reproduces_the_next_step_exactly(tmp_path):
    batch, speaker = _batch(n_target=4, n_reference=2)

    # (1) 中断なしで3 step
    model_a, opt_a = _make()
    gen_a = torch.Generator().manual_seed(100)
    torch.manual_seed(999)
    for _ in range(2):
        _step(model_a, opt_a, batch, speaker, gen_a)
    reference_loss = _step(model_a, opt_a, batch, speaker, gen_a)

    # (2) 2 step 後に保存 → 復元 → 3 step目
    model_b, opt_b = _make()
    gen_b = torch.Generator().manual_seed(100)
    torch.manual_seed(999)
    for _ in range(2):
        _step(model_b, opt_b, batch, speaker, gen_b)
    save_training_state(tmp_path / "ckpt", model=model_b, optimizer=opt_b,
                        state=TrainingState(step=2), generator=gen_b)

    model_c, opt_c = _make(seed=123)          # 別seedで初期化してから復元する
    gen_c = torch.Generator().manual_seed(1)
    restored = load_training_state(tmp_path / "ckpt", model=model_c, optimizer=opt_c,
                                   generator=gen_c)
    assert restored.step == 2
    resumed_loss = _step(model_c, opt_c, batch, speaker, gen_c)

    assert resumed_loss == pytest.approx(reference_loss, rel=1e-9), (
        f"resume mismatch: {reference_loss} vs {resumed_loss}"
    )


def test_resume_without_rng_restore_diverges(tmp_path):
    """RNG を戻さなければ一致しない。RNG保存が効いていることの裏取り。"""
    batch, speaker = _batch(n_target=4, n_reference=2)
    model_a, opt_a = _make()
    gen_a = torch.Generator().manual_seed(100)
    for _ in range(2):
        _step(model_a, opt_a, batch, speaker, gen_a)
    reference = _step(model_a, opt_a, batch, speaker, gen_a)

    model_b, opt_b = _make()
    gen_b = torch.Generator().manual_seed(100)
    for _ in range(2):
        _step(model_b, opt_b, batch, speaker, gen_b)
    save_training_state(tmp_path / "ckpt", model=model_b, optimizer=opt_b, generator=gen_b)

    model_c, opt_c = _make(seed=5)
    wrong_gen = torch.Generator().manual_seed(42)
    load_training_state(tmp_path / "ckpt", model=model_c, optimizer=opt_c,
                        generator=None)          # generator を渡さない = 復元しない
    resumed = _step(model_c, opt_c, batch, speaker, wrong_gen)
    assert resumed != pytest.approx(reference, rel=1e-9)


def test_optimizer_state_is_restored(tmp_path):
    """AdamW の moment が戻らなければ次stepは一致しない。"""
    batch, speaker = _batch(n_target=3)
    model, optimizer = _make()
    gen = torch.Generator().manual_seed(3)
    for _ in range(3):
        _step(model, optimizer, batch, speaker, gen)
    save_training_state(tmp_path / "c", model=model, optimizer=optimizer, generator=gen)

    model2, optimizer2 = _make(seed=7)
    load_training_state(tmp_path / "c", model=model2, optimizer=optimizer2, generator=None)
    for group_a, group_b in zip(optimizer.param_groups, optimizer2.param_groups):
        assert group_a["lr"] == group_b["lr"]
    assert len(optimizer2.state) == len(optimizer.state)


def test_training_state_scalars_round_trip(tmp_path):
    model, optimizer = _make()
    state = TrainingState(step=17, epoch=2, samples_seen=1024, best_metric=0.42,
                          extra={"note": "テスト"})
    save_training_state(tmp_path / "s", model=model, optimizer=optimizer, state=state)
    restored = load_training_state(tmp_path / "s", model=_tiny_model(1))
    assert restored.step == 17
    assert restored.epoch == 2
    assert restored.samples_seen == 1024
    assert restored.best_metric == pytest.approx(0.42)
    assert restored.extra["note"] == "テスト"


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_training_state(tmp_path / "nope", model=_tiny_model())


def test_metadata_json_is_written(tmp_path):
    import json
    model, optimizer = _make()
    save_training_state(tmp_path / "m", model=model, optimizer=optimizer,
                        state=TrainingState(step=5))
    meta = json.loads((tmp_path / "m" / "checkpoint.json").read_text(encoding="utf-8"))
    assert meta["step"] == 5


def test_export_requires_a_real_source_model_dir(tmp_path):
    model, _ = _make()
    with pytest.raises(FileNotFoundError):
        export_for_inference(tmp_path / "out", model=model, source_model_dir=tmp_path / "missing")
