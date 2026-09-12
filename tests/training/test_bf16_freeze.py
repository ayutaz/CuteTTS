# Copyright 2026 ayutaz
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

"""R-020: bf16パラメータを直接更新すると学習が起きないことのテスト。

公開checkpointは `qwen_backbone` と `locenc` が bf16。`AdamW` がそれを直接
更新すると、lr=2e-5 の更新量が bf16 の丸め幅（相対 2^-8）を下回るため
round-to-nearest-even が毎step更新を捨て、**同じ向きに積み上がらない**。
S0/S1の19回の学習で backbone の91%が1stepも動いていなかった。

ここでは (1) 現象そのもの、(2) `promote_to_float32` が直すこと、
(3) export で元のdtypeへ戻ること、を固定する。
"""

from __future__ import annotations

import torch

from cutetts.training.checkpointing import export_for_inference, promote_to_float32

from .test_forward import _tiny_model


LR = 2e-5
"""S0/S1 が実際に使った peak learning rate。"""


def _adamw_steps(param: torch.nn.Parameter, *, steps: int, lr: float = LR) -> float:
    """一定勾配で AdamW を回し、値が変わった要素の割合を返す。

    AdamW の更新量は lr * m/sqrt(v) で、一定勾配なら m/sqrt(v) → 1 に収束する。
    つまり毎step ほぼ lr ちょうど動かそうとする。
    """
    before = param.detach().clone()
    optimizer = torch.optim.AdamW([param], lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        param.grad = torch.ones_like(param)
        optimizer.step()
    return float((param.detach() != before).float().mean())


def test_bfloat16_parameters_do_not_move_at_the_training_lr():
    """bf16 のまま更新すると、代表的な重みの大半が動かない。"""
    # 実checkpointのbackboneと同じ規模の値域（|w| ~ 0.02）を作る
    weights = torch.linspace(0.01, 0.05, 4096)
    param = torch.nn.Parameter(weights.to(torch.bfloat16))

    moved = _adamw_steps(param, steps=50)

    # 丸め幅は |w| * 2^-8。|w|>=0.01 では更新 2e-5 が半ULPを下回る
    assert moved < 0.2, f"bf16でも{moved:.1%}動いた。前提が変わったなら R-020 を見直すこと"


def test_float32_parameters_move_at_the_same_lr():
    """fp32 なら同じ lr・同じstep数で全要素が動く。"""
    weights = torch.linspace(0.01, 0.05, 4096)
    param = torch.nn.Parameter(weights.clone())

    moved = _adamw_steps(param, steps=50)

    assert moved == 1.0, f"fp32で動いたのは{moved:.1%}"


def test_promote_to_float32_reports_original_dtypes_and_casts_the_model():
    model = _tiny_model(0)
    # tiny modelはfp32なので、実checkpointと同じ混在状態を作ってから確かめる
    model.locenc.to(dtype=torch.bfloat16)
    assert next(model.locenc.parameters()).dtype is torch.bfloat16

    original = promote_to_float32(model)

    assert next(model.locenc.parameters()).dtype is torch.float32
    locenc_keys = [k for k in original if k.startswith("locenc.")]
    assert locenc_keys, "locenc の重みが state_dict に無い"
    assert all(original[k] is torch.bfloat16 for k in locenc_keys)


def test_export_restores_the_original_dtypes(tmp_path):
    """学習は fp32、書き出しは元の dtype。config.json と整合させるため。"""
    from safetensors.torch import load_file

    source = tmp_path / "source"
    (source / "weights" / "tts").mkdir(parents=True)
    (source / "config.json").write_text("{}", encoding="utf-8")

    model = _tiny_model(0)
    model.locenc.to(dtype=torch.bfloat16)
    original = promote_to_float32(model)

    export_for_inference(tmp_path / "out", model=model, source_model_dir=source,
                         dtypes=original)
    saved = load_file(str(tmp_path / "out" / "weights" / "tts" / "model.safetensors"))

    locenc = [k for k in saved if k.startswith("locenc.") and saved[k].is_floating_point()]
    assert locenc
    assert all(saved[k].dtype is torch.bfloat16 for k in locenc)
    head = [k for k in saved if k.startswith("head.") and saved[k].is_floating_point()]
    assert head
    assert all(saved[k].dtype is torch.float32 for k in head)


def test_export_without_dtypes_keeps_current_dtypes(tmp_path):
    """`dtypes` を渡さなければ現在の dtype のまま（既存の挙動を壊さない）。"""
    from safetensors.torch import load_file

    source = tmp_path / "source"
    (source / "weights" / "tts").mkdir(parents=True)
    (source / "config.json").write_text("{}", encoding="utf-8")

    model = _tiny_model(0)
    model.locenc.to(dtype=torch.bfloat16)

    export_for_inference(tmp_path / "out", model=model, source_model_dir=source)
    saved = load_file(str(tmp_path / "out" / "weights" / "tts" / "model.safetensors"))

    locenc = [k for k in saved if k.startswith("locenc.") and saved[k].is_floating_point()]
    assert all(saved[k].dtype is torch.bfloat16 for k in locenc)
