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

"""学習forwardのテスト。実 `CuteTTSModel` を縮小configで組み立ててCPUで回す。

P2 のゴールのうち次を検証する:

* deterministic tiny batch で loss が再現する
* 学習対象moduleにだけ gradient が流れる
* 1 utterance を overfit できる
"""

from __future__ import annotations

import copy

import pytest
import torch

from cutetts.modeling.configuration import CuteTTSConfig
from cutetts.modeling.model import CuteTTSModel
from cutetts.training.collator import build_training_sample, collate
from cutetts.training.forward import TRAINABLE_MODULES, freeze_all_but, training_forward

PATCH = 2
DIM = 64
SPEAKER_DIM = 256


def _prompt(n_text: int = 5, *, speaker: bool = True):
    """テスト用の PromptLayout。text token を leading/trailing に分けて置く。"""
    from cutetts.training.prompt import PromptLayout
    lead = n_text // 2
    trail = n_text - lead
    return PromptLayout(
        leading_ids=torch.arange(lead, dtype=torch.long),
        middle_ids=torch.zeros(0, dtype=torch.long),
        trailing_ids=torch.arange(20, 20 + trail, dtype=torch.long),
        has_speaker_slot=speaker,
    )


def _tiny_model(seed: int = 0) -> CuteTTSModel:
    """公開configと同じ構造キーのまま、層数とhiddenだけ縮めたモデル。"""
    torch.manual_seed(seed)
    lm = dict(
        model_type="qwen3", hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_hidden_layers=2, max_position_embeddings=2048, rms_norm_eps=1e-6,
        rope_theta=10000.0, vocab_size=64, tie_word_embeddings=True,
    )
    config = CuteTTSConfig(
        lm_config=lm, attn_implementation="eager", torch_dtype="float32",
        lm_keep_num_hidden_layers=2, acoustic_latent_dim=DIM, extended_vocab_size=64,
        locenc_enabled=True, locenc_patch_size=PATCH, locenc_layers=1,
        locenc_hidden_dim=64, locenc_ffn_dim=128, locenc_num_heads=4, locenc_num_kv_heads=2,
        diff_head_kind="audio_dit", diff_dit_patch_size=PATCH, diff_dit_layers=1,
        diff_dit_hidden_dim=64, diff_dit_ffn_dim=128, diff_dit_num_heads=4,
        diff_dit_num_kv_heads=2, diff_dit_speaker_adaln_zero_enabled=True,
        diff_dit_speaker_embedding_dim=SPEAKER_DIM,
        lm_speaker_linear_enabled=True, lm_speaker_embedding_dim=SPEAKER_DIM,
        scale_acoustic_latent=True, two_class_stop_predictor=True,
    )
    model = CuteTTSModel(config)
    model.speech_scaling_factor.fill_(1.0)
    model.speech_bias_factor.fill_(0.0)
    return model.eval()


def _batch(*, n_target: int = 5, n_reference: int = 3, seed: int = 0, uid: str = "u0"):
    g = torch.Generator().manual_seed(seed)
    sample = build_training_sample(
        utterance_id=uid,
        prompt=_prompt(6),
        reference_latents=torch.randn(n_reference, PATCH, DIM, generator=g),
        target_latents=torch.randn(n_target, PATCH, DIM, generator=g),
    )
    speaker = torch.randn(1, SPEAKER_DIM, generator=g)
    return collate([sample]), speaker


# ------------------------------------------------------------------- basic run

def test_forward_produces_finite_losses():
    model = _tiny_model()
    batch, speaker = _batch()
    out = training_forward(model, batch, speaker_embeddings=speaker, flow_copies=2,
                           generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.flow_loss)
    assert torch.isfinite(out.stop_loss)
    assert out.num_targets == 5
    assert out.flow_rows == 10          # 5 target x 2 copies


def test_loss_is_reproducible_for_a_deterministic_tiny_batch():
    model = _tiny_model()
    batch, speaker = _batch()
    first = training_forward(model, batch, speaker_embeddings=speaker, flow_copies=2,
                             generator=torch.Generator().manual_seed(7))
    second = training_forward(model, batch, speaker_embeddings=speaker, flow_copies=2,
                              generator=torch.Generator().manual_seed(7))
    assert float(first.loss) == pytest.approx(float(second.loss), rel=1e-9)


def test_different_noise_seeds_change_the_flow_loss():
    model = _tiny_model()
    batch, speaker = _batch()
    a = training_forward(model, batch, speaker_embeddings=speaker, flow_copies=2,
                         generator=torch.Generator().manual_seed(1))
    b = training_forward(model, batch, speaker_embeddings=speaker, flow_copies=2,
                         generator=torch.Generator().manual_seed(2))
    assert float(a.flow_loss) != pytest.approx(float(b.flow_loss), rel=1e-9)


# ---------------------------------------------------------------- gradients

def test_gradient_reaches_every_trainable_module():
    model = _tiny_model()
    freeze_all_but(model)
    batch, speaker = _batch()
    out = training_forward(model, batch, speaker_embeddings=speaker, flow_copies=2,
                           generator=torch.Generator().manual_seed(3))
    out.loss.backward()
    touched = {
        name.split(".")[0]
        for name, p in model.named_parameters()
        if p.grad is not None and float(p.grad.abs().sum()) > 0.0
    }
    assert set(TRAINABLE_MODULES) <= touched


def test_freezing_a_module_stops_its_gradient():
    model = _tiny_model()
    freeze_all_but(model, trainable=("qwen_backbone", "head", "stop_predictor",
                                     "locenc_to_lm_proj", "lm_speaker_linear"))
    batch, speaker = _batch()
    out = training_forward(model, batch, speaker_embeddings=speaker, flow_copies=2,
                           generator=torch.Generator().manual_seed(4))
    out.loss.backward()
    locenc_grads = [p.grad for _, p in model.locenc.named_parameters()]
    assert all(g is None for g in locenc_grads)
    assert any(p.grad is not None for p in model.head.parameters())


def test_freeze_all_but_rejects_unknown_module_names():
    model = _tiny_model()
    with pytest.raises(ValueError):
        freeze_all_but(model, trainable=("does_not_exist",))


def test_stop_predictor_receives_gradient_from_the_stop_loss_only():
    """stop_weight=0 なら stop_predictor に勾配が来ない。"""
    model = _tiny_model()
    freeze_all_but(model)
    batch, speaker = _batch()
    out = training_forward(model, batch, speaker_embeddings=speaker, flow_copies=2,
                           stop_weight=0.0, generator=torch.Generator().manual_seed(5))
    out.loss.backward()
    total = sum(float(p.grad.abs().sum()) for p in model.stop_predictor.parameters()
                if p.grad is not None)
    assert total == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------- overfit

def test_a_single_utterance_can_be_overfit():
    """同じsampleを繰り返し学習すると loss が明確に下がる。"""
    model = _tiny_model()
    freeze_all_but(model)
    batch, speaker = _batch(n_target=3, n_reference=2)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-3
    )

    def step(seed: int) -> float:
        optimizer.zero_grad(set_to_none=True)
        out = training_forward(model, batch, speaker_embeddings=speaker, flow_copies=4,
                               generator=torch.Generator().manual_seed(seed))
        out.loss.backward()
        optimizer.step()
        return float(out.flow_loss)

    first = step(0)
    for i in range(1, 60):
        step(i)
    # 同じnoise seedで比較する
    optimizer.zero_grad(set_to_none=True)
    final = float(training_forward(model, batch, speaker_embeddings=speaker, flow_copies=4,
                                   generator=torch.Generator().manual_seed(0)).flow_loss)
    assert final < first * 0.7, f"flow loss did not drop: {first:.4f} -> {final:.4f}"


def test_batch_of_two_runs_and_separates_targets():
    model = _tiny_model()
    g = torch.Generator().manual_seed(11)
    a = build_training_sample(
        utterance_id="a", prompt=_prompt(5),
        reference_latents=torch.randn(2, PATCH, DIM, generator=g),
        target_latents=torch.randn(3, PATCH, DIM, generator=g))
    b = build_training_sample(
        utterance_id="b", prompt=_prompt(7),
        reference_latents=torch.randn(3, PATCH, DIM, generator=g),
        target_latents=torch.randn(4, PATCH, DIM, generator=g))
    batch = collate([a, b])
    speaker = torch.randn(2, SPEAKER_DIM, generator=g)
    out = training_forward(model, batch, speaker_embeddings=speaker, flow_copies=1,
                           generator=torch.Generator().manual_seed(12))
    assert out.num_targets == 7
    assert torch.isfinite(out.loss)


# ------------------------------------------------------ F0 の条件（M4c）
#
# **zero-init が no-op であること**が最重要。ここが崩れると、公開
# checkpointからの継続学習が最初の step で壊れる。


def _batch_with_f0(*, n_target: int = 5, n_reference: int = 3, seed: int = 0):
    from cutetts.training.f0 import F0_FEATURE_DIM

    g = torch.Generator().manual_seed(seed)
    sample = build_training_sample(
        utterance_id="u0",
        prompt=_prompt(6),
        reference_latents=torch.randn(n_reference, PATCH, DIM, generator=g),
        target_latents=torch.randn(n_target, PATCH, DIM, generator=g),
        target_f0=torch.randn(n_target, PATCH * F0_FEATURE_DIM, generator=g),
    )
    speaker = torch.randn(1, SPEAKER_DIM, generator=g)
    return collate([sample]), speaker


def test_zero_initのF0条件は何も変えない():
    """**恒等から始まること。** ここが崩れると継続学習の前提が壊れる。"""
    from cutetts.training.f0 import F0_FEATURE_DIM, F0Conditioner

    model = _tiny_model()
    batch, speaker = _batch_with_f0()
    hidden = int(model.lm_speaker_linear.out_features)
    conditioner = F0Conditioner(PATCH * F0_FEATURE_DIM, hidden)

    without = training_forward(model, batch, speaker_embeddings=speaker,
                               generator=torch.Generator().manual_seed(7))
    with_zero = training_forward(model, batch, speaker_embeddings=speaker,
                                 generator=torch.Generator().manual_seed(7),
                                 f0_conditioner=conditioner)
    assert float(with_zero.loss) == pytest.approx(float(without.loss), rel=1e-6)


def test_学習後のF0条件は結果を変える():
    from cutetts.training.f0 import F0_FEATURE_DIM, F0Conditioner

    model = _tiny_model()
    batch, speaker = _batch_with_f0()
    hidden = int(model.lm_speaker_linear.out_features)
    conditioner = F0Conditioner(PATCH * F0_FEATURE_DIM, hidden)
    torch.nn.init.normal_(conditioner.proj.weight, std=0.5)

    without = training_forward(model, batch, speaker_embeddings=speaker,
                               generator=torch.Generator().manual_seed(7))
    with_f0 = training_forward(model, batch, speaker_embeddings=speaker,
                               generator=torch.Generator().manual_seed(7),
                               f0_conditioner=conditioner)
    assert float(with_f0.loss) != pytest.approx(float(without.loss), rel=1e-6)


def test_F0条件に勾配が流れる():
    """**grad が来なければ学習されない**（zero-init なので永遠に 0 のまま）。"""
    from cutetts.training.f0 import F0_FEATURE_DIM, F0Conditioner

    model = _tiny_model()
    batch, speaker = _batch_with_f0()
    hidden = int(model.lm_speaker_linear.out_features)
    conditioner = F0Conditioner(PATCH * F0_FEATURE_DIM, hidden)
    out = training_forward(model, batch, speaker_embeddings=speaker,
                           generator=torch.Generator().manual_seed(7),
                           f0_conditioner=conditioner)
    out.loss.backward()
    grad = conditioner.proj.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_F0条件が無いbatchでは無視される():
    from cutetts.training.f0 import F0_FEATURE_DIM, F0Conditioner

    model = _tiny_model()
    batch, speaker = _batch()            # target_f0 は None
    assert batch.target_f0 is None
    conditioner = F0Conditioner(PATCH * F0_FEATURE_DIM,
                                int(model.lm_speaker_linear.out_features))
    torch.nn.init.normal_(conditioner.proj.weight, std=0.5)
    without = training_forward(model, batch, speaker_embeddings=speaker,
                               generator=torch.Generator().manual_seed(7))
    with_f0 = training_forward(model, batch, speaker_embeddings=speaker,
                               generator=torch.Generator().manual_seed(7),
                               f0_conditioner=conditioner)
    assert float(with_f0.loss) == pytest.approx(float(without.loss), rel=1e-6)


def test_F0条件の形が違えば落ちる():
    from cutetts.training.f0 import F0_FEATURE_DIM

    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        build_training_sample(
            utterance_id="u0", prompt=_prompt(6),
            reference_latents=torch.randn(3, PATCH, DIM, generator=g),
            target_latents=torch.randn(5, PATCH, DIM, generator=g),
            target_f0=torch.randn(4, PATCH * F0_FEATURE_DIM, generator=g),
        )


def _train_speaker_adaln(model, std: float = 0.05) -> None:
    """`speaker_adaln` を 0 でない値にする（tiny model は未学習なので）。

    **zero-init のままだと speaker へ勾配が流れない。** gate が 0 なので
    枝ごと消え、`d(gate)/d(speaker) = W_gate = 0` になる。
    公開checkpointでは学習済み（|w| 平均 2.06e-02、最大 1.05）なので
    実機ではこの問題は起きない。
    """
    for name, param in model.head.named_parameters():
        if "speaker_adaln" in name:
            torch.nn.init.normal_(param, std=std)


def test_head側のF0条件もzero_initならno_op():
    """**差し込む場所を変えても恒等から始まること。**"""
    from cutetts.training.f0 import F0_FEATURE_DIM, F0Conditioner

    model = _tiny_model()
    batch, speaker = _batch_with_f0()
    conditioner = F0Conditioner(PATCH * F0_FEATURE_DIM, SPEAKER_DIM)

    without = training_forward(model, batch, speaker_embeddings=speaker,
                               generator=torch.Generator().manual_seed(7))
    with_zero = training_forward(model, batch, speaker_embeddings=speaker,
                                 generator=torch.Generator().manual_seed(7),
                                 f0_head_conditioner=conditioner)
    assert float(with_zero.loss) == pytest.approx(float(without.loss), rel=1e-6)


def test_head側のF0条件に勾配が流れる():
    from cutetts.training.f0 import F0_FEATURE_DIM, F0Conditioner

    model = _tiny_model()
    batch, speaker = _batch_with_f0()
    _train_speaker_adaln(model)
    conditioner = F0Conditioner(PATCH * F0_FEATURE_DIM, SPEAKER_DIM)
    out = training_forward(model, batch, speaker_embeddings=speaker,
                           generator=torch.Generator().manual_seed(7),
                           f0_head_conditioner=conditioner)
    out.loss.backward()
    grad = conditioner.proj.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_head側の条件は学習すると結果を変える():
    from cutetts.training.f0 import F0_FEATURE_DIM, F0Conditioner

    model = _tiny_model()
    batch, speaker = _batch_with_f0()
    _train_speaker_adaln(model)
    conditioner = F0Conditioner(PATCH * F0_FEATURE_DIM, SPEAKER_DIM)
    torch.nn.init.normal_(conditioner.proj.weight, std=0.5)
    without = training_forward(model, batch, speaker_embeddings=speaker,
                               generator=torch.Generator().manual_seed(7))
    with_f0 = training_forward(model, batch, speaker_embeddings=speaker,
                               generator=torch.Generator().manual_seed(7),
                               f0_head_conditioner=conditioner)
    assert float(with_f0.loss) != pytest.approx(float(without.loss), rel=1e-6)


# ------------------------------------------------- 条件の dropout（M4e）
#
# 条件づけを入れるとモデルが条件に依存し、**条件が外れると素のモデルより
# 悪くなる**（実測で輪郭 +0.019、長さ 6.81秒 対 人間 5.56秒）。
# 実運用では良い条件を供給できないので、**落としても劣化しない**ように学習する。


def test_dropout1なら条件が消える():
    """全部落とすので、条件を渡さないのと同じ結果になる。

    **比較の相手も dropout を通す。** 抽選は flow のノイズと同じ generator を
    消費するので（既存の condition dropout と同じ作り）、通さない呼び出しと
    比べると乱数列がずれて別の loss になる。
    """
    from cutetts.training.f0 import F0_FEATURE_DIM, F0Conditioner

    model = _tiny_model()
    batch, speaker = _batch_with_f0()
    conditioner = F0Conditioner(PATCH * F0_FEATURE_DIM,
                                int(model.lm_speaker_linear.out_features))
    torch.nn.init.normal_(conditioner.proj.weight, std=0.5)
    torch.nn.init.zeros_(conditioner.proj.bias)      # bias があると 0 入力でも動く

    without = training_forward(model, batch, speaker_embeddings=speaker,
                               generator=torch.Generator().manual_seed(7),
                               f0_conditioner=None, f0_dropout=1.0)
    dropped = training_forward(model, batch, speaker_embeddings=speaker,
                               generator=torch.Generator().manual_seed(7),
                               f0_conditioner=conditioner, f0_dropout=1.0)
    assert float(dropped.loss) == pytest.approx(float(without.loss), rel=1e-6)


def test_dropout0なら全部残る():
    from cutetts.training.f0 import F0_FEATURE_DIM, F0Conditioner

    model = _tiny_model()
    batch, speaker = _batch_with_f0()
    conditioner = F0Conditioner(PATCH * F0_FEATURE_DIM,
                                int(model.lm_speaker_linear.out_features))
    torch.nn.init.normal_(conditioner.proj.weight, std=0.5)
    kept = training_forward(model, batch, speaker_embeddings=speaker,
                            generator=torch.Generator().manual_seed(7),
                            f0_conditioner=conditioner, f0_dropout=0.0)
    same = training_forward(model, batch, speaker_embeddings=speaker,
                            generator=torch.Generator().manual_seed(7),
                            f0_conditioner=conditioner)
    assert float(kept.loss) == pytest.approx(float(same.loss), rel=1e-6)


def test_dropoutはbatchを壊さない():
    """**元の batch を書き換えない**（次の step で条件が消えていては困る）。"""
    from cutetts.training.f0 import F0_FEATURE_DIM, F0Conditioner

    model = _tiny_model()
    batch, speaker = _batch_with_f0()
    before = batch.target_f0.clone()
    conditioner = F0Conditioner(PATCH * F0_FEATURE_DIM,
                                int(model.lm_speaker_linear.out_features))
    training_forward(model, batch, speaker_embeddings=speaker,
                     generator=torch.Generator().manual_seed(7),
                     f0_conditioner=conditioner, f0_dropout=1.0)
    assert torch.equal(batch.target_f0, before)
