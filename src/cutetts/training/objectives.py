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

"""学習objective: flow matching / stop / condition dropout。

論文（v2）から取る定義:

    x_t = (1 - t) * xi + t * P        xi ~ N(0, I)
    target velocity = P - xi
    t = sigmoid(u), u ~ N(0, 1)
    公開pretrainingは各target patchを4つの独立noise/timeで複製する
    condition dropout 0.1

推論コードから確認できる規約:

* Diffusion Head は **正規化後** の潜在空間で動く。
  `(latent + speech_bias_factor) * speech_scaling_factor` を適用した値を使う
  （`modeling/model.py: forward_speech_features`）。
  非正規化するのは waveform decode の直前だけ（`inference/generation.py`）。
* Head のシグネチャは
  `head._predict(x=[N,P,D], t=[N], z=[N,C], cond=[N,P,D], speaker_embedding=[N,S])`。
* stop は **位置 i の hidden が「patch i が最終patchか」** を予測する。
  推論側は patch 生成の *前* に判定し、生成後に break する
  （`inference/generation.py` の `_stop_after_current_patch` と `break`）。

論文にもコードにも規定が無く、このforkで決めた事項（04章「自分で決める必要がある箇所」）:

* padding patch は loss の分子からも **分母からも** 除く（`loss_mask`）。
* stop の class imbalance は `positive_weight` で対処する。
  stop は系列あたり1つしかないため、既定では重み付けせず、
  学習側が系列長に応じて明示的に指定する。
* condition dropout の対象は speaker embedding と reference speech。
  既定は joint（同時に落とす）で、推論の LM-level CFG の uncond branch と同形にする。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

DEFAULT_FLOW_TARGET_COPIES = 4
"""公開pretrainingの target patch 複製数。"""

STOP_CONTINUE = 0
STOP_STOP = 1
"""2-class Stop Predictor のクラス割り当て。

推論側は `torch.argmax(stop_logits, dim=1).item() == 1` を停止と解釈するので、
stop は必ず index 1 でなければならない（`inference/generation.py`）。
"""


# --------------------------------------------------------------- flow matching

@dataclass(frozen=True)
class FlowBatch:
    """flow-matching の1 stepぶんの入力。すべて先頭次元 N で揃う。"""

    x_t: Tensor
    """[N, P, D] 補間された noisy latent（正規化空間）。"""
    t: Tensor
    """[N] flow time。"""
    target_clean: Tensor
    """[N, P, D] clean target patch P（正規化空間）。"""
    target_velocity: Tensor
    """[N, P, D] 学習target。P - xi。"""
    z: Tensor
    """[N, C] LM hidden state。"""
    previous_cond: Tensor
    """[N, P, D] 直前patch。系列先頭は zeros または prefix 末尾の speech patch。"""
    speaker: Tensor | None
    """[N, S] speaker embedding。条件が無ければ None。"""
    loss_mask: Tensor
    """[N] bool。False の行は loss の分子・分母どちらにも入れない。"""

    @property
    def size(self) -> int:
        return int(self.x_t.shape[0])


def _generator_device(generator: torch.Generator | None) -> torch.device | None:
    """generator が縛られている device。None なら device 指定なし。"""
    return None if generator is None else generator.device


def _randn(shape, *, generator, device, dtype) -> Tensor:
    """generator の device と出力 device が食い違っても動く randn。

    `torch.randn` は generator と device が一致していないと
    ``Expected a 'cuda' device type for generator but found 'cpu'`` を投げる。
    CPU generator で再現性を担保しつつ CUDA 上で学習したいので、
    generator の device で作ってから移す。
    """
    gen_device = _generator_device(generator)
    if generator is None or gen_device is None or gen_device.type == torch.device(device or "cpu").type:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
    out = torch.randn(shape, generator=generator, device=gen_device, dtype=torch.float32)
    return out.to(device=device, dtype=dtype)


def sample_flow_time(
    n: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``t = sigmoid(u), u ~ N(0, 1)`` を n 個サンプルする。"""
    u = _randn((n,), generator=generator, device=device, dtype=dtype)
    return torch.sigmoid(u)


def build_flow_batch(
    target_patches: Tensor,
    z: Tensor,
    previous_cond: Tensor,
    speaker: Tensor | None,
    patch_mask: Tensor,
    *,
    copies: int = DEFAULT_FLOW_TARGET_COPIES,
    generator: torch.Generator | None = None,
    flow_time: Tensor | None = None,
) -> FlowBatch:
    """target patch を ``copies`` 個の独立 noise / time で複製して FlowBatch を作る。

    ``target_patches`` は **正規化後** の潜在（`[B, P, D]`）を渡すこと。
    ``flow_time`` を渡すとサンプリングを行わずその値を使う（テスト用）。
    """
    if copies < 1:
        raise ValueError(f"copies must be >= 1, got {copies}")
    if target_patches.dim() != 3:
        raise ValueError(f"target_patches must be [B, P, D], got {tuple(target_patches.shape)}")
    if previous_cond.shape != target_patches.shape:
        raise ValueError(
            "previous_cond must match target_patches: "
            f"{tuple(previous_cond.shape)} vs {tuple(target_patches.shape)}"
        )
    batch = target_patches.shape[0]
    if z.shape[0] != batch or patch_mask.shape[0] != batch:
        raise ValueError("z and patch_mask must share the batch dimension with target_patches")
    if speaker is not None and speaker.shape[0] != batch:
        raise ValueError("speaker must share the batch dimension with target_patches")

    clean = target_patches.repeat(copies, 1, 1)
    z_rep = z.repeat(copies, 1)
    prev_rep = previous_cond.repeat(copies, 1, 1)
    speaker_rep = None if speaker is None else speaker.repeat(copies, 1)
    mask_rep = patch_mask.repeat(copies)

    n = clean.shape[0]
    if flow_time is None:
        t = sample_flow_time(n, generator=generator, device=clean.device, dtype=clean.dtype)
    else:
        t = flow_time.repeat(copies) if flow_time.shape[0] == batch else flow_time
        if t.shape[0] != n:
            raise ValueError(f"flow_time must have {batch} or {n} entries, got {t.shape[0]}")
        t = t.to(device=clean.device, dtype=clean.dtype)

    noise = _randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
    t_view = t.view(-1, 1, 1)
    x_t = (1.0 - t_view) * noise + t_view * clean
    velocity = clean - noise

    return FlowBatch(
        x_t=x_t,
        t=t,
        target_clean=clean,
        target_velocity=velocity,
        z=z_rep,
        previous_cond=prev_rep,
        speaker=speaker_rep,
        loss_mask=mask_rep,
    )


def flow_matching_loss(predicted_velocity: Tensor, batch: FlowBatch) -> Tensor:
    """masked MSE。

    分母は **マスクが真の行に属する要素数** であり、padding patch は
    分子からも分母からも外れる。
    """
    if predicted_velocity.shape != batch.target_velocity.shape:
        raise ValueError(
            "predicted_velocity shape mismatch: "
            f"{tuple(predicted_velocity.shape)} vs {tuple(batch.target_velocity.shape)}"
        )
    valid = int(batch.loss_mask.sum().item())
    if valid == 0:
        raise ValueError("flow_matching_loss requires at least one unmasked row")

    error = (predicted_velocity - batch.target_velocity) ** 2
    weights = batch.loss_mask.to(error.dtype).view(-1, *([1] * (error.dim() - 1)))
    per_row = error * weights
    denominator = valid * int(error[0].numel())
    return per_row.sum() / denominator


# ------------------------------------------------------------------------ stop

def build_stop_targets(patch_mask: Tensor) -> Tensor:
    """``[B, T]`` の有効patch maskから stop ラベル ``[B, T]`` を作る。

    **位置 i のラベルは「patch i が最終patchか」** を表す。
    各系列の最後の有効patchにだけ `STOP_STOP` を置く。
    """
    if patch_mask.dim() != 2:
        raise ValueError(f"patch_mask must be [B, T], got {tuple(patch_mask.shape)}")
    lengths = patch_mask.sum(dim=1)
    if bool((lengths == 0).any()):
        raise ValueError("every sequence must contain at least one valid patch")

    targets = torch.full(patch_mask.shape, STOP_CONTINUE, dtype=torch.long,
                         device=patch_mask.device)
    last_index = lengths - 1
    targets.scatter_(1, last_index.unsqueeze(1), STOP_STOP)
    return targets


def stop_loss(
    stop_logits: Tensor,
    stop_targets: Tensor,
    patch_mask: Tensor,
    *,
    positive_weight: float | None = None,
) -> Tensor:
    """2-class の cross entropy。padding位置は分子・分母から外す。

    ``positive_weight`` は stop クラスの重み。stop は系列に1つしかないため、
    系列が長いほど不均衡が強くなる。既定は重み付けなし。
    """
    if stop_logits.dim() != 3 or stop_logits.shape[-1] != 2:
        raise ValueError(f"stop_logits must be [B, T, 2], got {tuple(stop_logits.shape)}")
    if stop_logits.shape[:2] != stop_targets.shape:
        raise ValueError(
            "stop_logits and stop_targets disagree: "
            f"{tuple(stop_logits.shape[:2])} vs {tuple(stop_targets.shape)}"
        )
    if patch_mask.shape != stop_targets.shape:
        raise ValueError(
            "patch_mask and stop_targets disagree: "
            f"{tuple(patch_mask.shape)} vs {tuple(stop_targets.shape)}"
        )
    if int(patch_mask.sum().item()) == 0:
        raise ValueError("stop_loss requires at least one valid position")

    weight = None
    if positive_weight is not None:
        if positive_weight <= 0.0:
            raise ValueError("positive_weight must be positive")
        weight = torch.tensor([1.0, float(positive_weight)],
                              dtype=stop_logits.dtype, device=stop_logits.device)

    per_position = F.cross_entropy(
        stop_logits.reshape(-1, 2),
        stop_targets.reshape(-1),
        weight=weight,
        reduction="none",
    ).view(stop_targets.shape)

    valid = patch_mask.to(per_position.dtype)
    numerator = (per_position * valid).sum()
    if weight is None:
        denominator = valid.sum()
    else:
        # 重み付き平均の分母は、有効位置の重みの合計にする
        denominator = (weight[stop_targets] * valid).sum()
    return numerator / denominator


def total_loss(flow: Tensor, stop: Tensor, *, stop_weight: float = 1.0) -> Tensor:
    """flow loss と stop loss を合成する。

    重みは論文に規定が無い。既定を1.0とし、Stage 0 で調整する。
    """
    if stop_weight < 0.0:
        raise ValueError("stop_weight must be non-negative")
    return flow + stop_weight * stop


# ------------------------------------------------------------- condition dropout

@dataclass(frozen=True)
class ConditionDropoutConfig:
    """condition dropout の設定。公開pretrainingの既定は 0.1。"""

    speaker: float = 0.1
    reference: float = 0.1
    joint: bool = True
    """True なら speaker と reference を同じ判定で落とす（推論のuncond branchと同形）。"""

    def __post_init__(self) -> None:
        for name in ("speaker", "reference"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} dropout rate must be in [0, 1], got {value}")


def sample_condition_dropout(
    batch_size: int,
    config: ConditionDropoutConfig,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> dict[str, Tensor]:
    """条件ごとの drop マスクを引く。True が「落とす」。"""
    def draw(rate: float) -> Tensor:
        if rate <= 0.0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        if rate >= 1.0:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        gen_device = _generator_device(generator)
        if generator is not None and gen_device is not None and                 gen_device.type != torch.device(device or "cpu").type:
            u = torch.rand(batch_size, generator=generator, device=gen_device).to(device)
        else:
            u = torch.rand(batch_size, generator=generator, device=device)
        return u < rate

    speaker = draw(config.speaker)
    reference = speaker.clone() if config.joint else draw(config.reference)
    return {"speaker": speaker, "reference": reference}


def apply_condition_dropout(
    speaker: Tensor | None,
    reference_latents: Tensor,
    reference_mask: Tensor,
    drop: dict[str, Tensor],
) -> tuple[Tensor | None, Tensor, Tensor]:
    """drop マスクに従って条件を落とす。入力は変更しない。

    speaker は zero vector に、reference は latent を zero にして mask を落とす。
    zero にするのは、推論の uncond branch が speaker slot に zeros を入れる
    （`modeling/model.py: prepare_input_embeds`）のと揃えるため。
    """
    speaker_out = speaker
    if speaker is not None:
        speaker_out = speaker.clone()
        speaker_out[drop["speaker"]] = 0.0

    reference_out = reference_latents.clone()
    mask_out = reference_mask.clone()
    dropped = drop["reference"]
    reference_out[dropped] = 0.0
    mask_out[dropped] = False
    return speaker_out, reference_out, mask_out
