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

"""テキストから韻律の輪郭を予測する（M4b）。

M4c は「韻律を**与えられたら**従えるか」を測る。実運用では与えるものが無いので、
**テキストから作る**必要がある。ここはその予測器。

**長さを正規化した輪郭を予測する。** 輪郭の指標（`contour_similarity`）は
長さを揃えてから相関を取るので、**継続時間のモデル化が要らない**。
64点に伸縮した半音の輪郭だけを出す。

    片仮名（frontend と同じ表現）→ 文字埋め込み → 双方向GRU → 64点の輪郭

**比べる相手は辞書**（`prosody.accent_plan` から作る理想化した輪郭）。
辞書は人間と 44.8% しか一致しない（アクセント核で）。
予測器がそれを超えなければ、**テキストから韻律を取る道は細い**。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "CONTOUR_POINTS",
    "F0Predictor",
    "contour_from_f0",
    "dictionary_contour",
    "encode_text",
    "text_vocabulary",
]

#: 輪郭を表す点数。**長さを正規化してある**ので、文の長短に依らず同じ次元。
#:
#: 64点は 12.5 Hz なら 5.1 秒ぶん。評価setの人間音声は平均 5.56 秒なので、
#: だいたい1点が1 latent frame に対応する粗さ。
CONTOUR_POINTS = 64


def text_vocabulary(texts: list[str]) -> dict[str, int]:
    """文字 → id。**0 は padding**、1 は未知。"""
    seen = sorted({ch for text in texts for ch in text})
    return {ch: index + 2 for index, ch in enumerate(seen)}


def encode_text(text: str, vocab: dict[str, int], length: int) -> np.ndarray:
    """文字列を id 列へ。長ければ切り、短ければ 0 で埋める。"""
    ids = [vocab.get(ch, 1) for ch in text[:length]]
    return np.array(ids + [0] * (length - len(ids)), dtype=np.int64)


def contour_from_f0(f0_hz: np.ndarray, points: int = CONTOUR_POINTS
                    ) -> np.ndarray | None:
    """F0（Hz、無声 0）から**長さを正規化した半音の輪郭**へ。

    無声は前後から補間する（`prosody.time_aligned_contour` と同じ規約）。
    有声フレームが足りなければ ``None``。
    """
    from cutetts.training.prosody import (
        MIN_VOICED_FRAMES,
        resample_contour,
        time_aligned_contour,
    )

    array = np.asarray(f0_hz, dtype=np.float64)
    if int((array > 0).sum()) < MIN_VOICED_FRAMES:
        return None
    contour = time_aligned_contour(array)
    if contour.size < 2:
        return None
    return resample_contour(contour, points).astype(np.float32)


def dictionary_contour(text: str, points: int = CONTOUR_POINTS
                       ) -> np.ndarray | None:
    """**辞書から作る理想化した輪郭**（比較の相手）。

    アクセント句ごとに「核まで上がって核の後で下がる」段を作る。
    日本語の標準的な型（頭高・中高・尾高・平板）をそのまま段差にしたもので、
    細かい抑揚は入らない。**これを超えられるかが M4b の問い。**
    """
    from cutetts.training.alignment import phrase_plan

    try:
        phrases = phrase_plan(text)
    except Exception:
        return None
    values: list[float] = []
    for phrase in phrases:
        count = len(phrase.moras)
        if count == 0:
            continue
        nucleus = phrase.internal_nucleus
        for position in range(1, count + 1):
            if nucleus == 0:                     # 平板: 1モーラ目だけ低い
                values.append(-1.0 if position == 1 else 0.5)
            elif position <= nucleus:            # 核まで高い
                values.append(1.0 if position > 1 or nucleus == 1 else 0.0)
            else:                                # 核の後で下がる
                values.append(-1.5)
    if len(values) < 2:
        return None
    from cutetts.training.prosody import resample_contour

    array = np.asarray(values, dtype=np.float64)
    array = array - array.mean()
    return resample_contour(array, points).astype(np.float32)


class F0Predictor(nn.Module):
    """片仮名 → 長さを正規化した輪郭（``[points]``）。

    **小さくしてある。** 23万発話しかないので、大きくすると覚えるだけになる。
    """

    def __init__(self, vocab_size: int, *, embed_dim: int = 64,
                 hidden_dim: int = 128, points: int = CONTOUR_POINTS,
                 layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.points = int(points)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.points),
        )

    def forward(self, ids: Tensor) -> Tensor:
        mask = (ids != 0)
        lengths = mask.sum(dim=1).clamp(min=1)
        embedded = self.embedding(ids)
        # **padding を詰めてから GRU へ渡す。** 双方向なので逆方向が padding を
        # 先に読み、詰めないと**実位置の表現まで変わる**（同じ文でも padding の
        # 長さで出力が変わった）。平均から外すだけでは足りない。
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        hidden, _ = self.encoder(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)
        # 詰め戻すと長さが入力より短くなりうるので、mask も合わせる
        limit = hidden.size(1)
        weights = mask[:, :limit].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return self.head(pooled)
