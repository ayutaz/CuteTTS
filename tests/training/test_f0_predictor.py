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

"""テキストからの韻律予測（M4b / `training.f0_predictor`）。

**長さを正規化した輪郭を予測する。** 輪郭の指標は長さを揃えてから相関を
取るので、継続時間のモデル化が要らない。ここが崩れると比較が成り立たない。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cutetts.training.f0_predictor import (
    CONTOUR_POINTS,
    F0Predictor,
    contour_from_f0,
    encode_text,
    text_vocabulary,
)


def test_語彙は0を空けておく():
    """**0 は padding。** ここを埋めると padding が文字として学習される。"""
    vocab = text_vocabulary(["アイ", "ウ"])
    assert 0 not in vocab.values()
    assert 1 not in vocab.values()


def test_符号化は長さを揃える():
    vocab = text_vocabulary(["アイウ"])
    assert encode_text("アイウ", vocab, 5).tolist()[3:] == [0, 0]
    assert len(encode_text("アイウエオカキ", vocab, 4)) == 4


def test_未知の文字は1になる():
    vocab = text_vocabulary(["ア"])
    assert encode_text("ン", vocab, 1).tolist() == [1]


def test_輪郭は長さに依らず同じ次元():
    short = contour_from_f0(np.full(12, 200.0) + np.arange(12), CONTOUR_POINTS)
    long = contour_from_f0(np.full(80, 200.0) + np.arange(80), CONTOUR_POINTS)
    assert short is not None and long is not None
    assert short.shape == long.shape == (CONTOUR_POINTS,)


def test_有声が足りなければNone():
    assert contour_from_f0(np.zeros(20)) is None
    assert contour_from_f0(np.array([200.0] * 5 + [0.0] * 15)) is None


def test_予測器は輪郭の次元を出す():
    model = F0Predictor(vocab_size=10)
    out = model(torch.randint(1, 10, (3, 16)))
    assert out.shape == (3, CONTOUR_POINTS)


def test_paddingは平均から外れる():
    """**外さないと短い文ほど 0 に引っ張られる。**"""
    model = F0Predictor(vocab_size=10).eval()
    ids = torch.randint(1, 10, (1, 8))
    padded = torch.cat([ids, torch.zeros(1, 24, dtype=torch.long)], dim=1)
    with torch.no_grad():
        assert torch.allclose(model(ids), model(padded), atol=1e-5)


def test_辞書の輪郭は平板と頭高で違う():
    pytest.importorskip("pyopenjtalk", reason="[ja] extra が要る")
    from cutetts.training.f0_predictor import dictionary_contour

    a = dictionary_contour("桜が咲いた。")
    b = dictionary_contour("箸を持つ。")
    assert a is not None and b is not None
    assert not np.allclose(a, b)
