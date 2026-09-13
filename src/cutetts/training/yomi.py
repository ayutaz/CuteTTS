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

"""語の読み付与 frontend（J3 / D-034）。

**誤読は聴取4回すべてで最多の指摘だった**（盲検A/Bで 9/20 = 45%）。
主因は byte-fallback による文字の分解（[R-027]）。`華` は tokenizer に
単独pieceを持たず `<0xE8><0x8F><0xAF>` の3断片になり、モデルは文字として
見ていないので読みを引けない。gol 30万文で単独pieceを持たない文字は1,115種ある。

同一モデル・同一reference・同一seedで**表記だけ**を替えると直る:

    ……中華ですね     → ASR「シュカですね」  ✗
    ……ちゅうかですね  → ASR「中華ですね」    ○

**置換は最小限にする。** 全文を仮名にするとモデルの学習分布
（漢字かな混在）から外れる。判定条件を実測で比べた結果:

| 条件 | 誤読4語を捕まえる | 正しい6語を誤爆 |
|---|---|---|
| **文字が byte-fallback**（採用） | **3/4** | **0/6** |
| 語が vocab に無い | 4/4 | **5/6**（`大丈夫` `気持ち` まで置換してしまう） |

**`砂肝` 型は取りこぼす**（`砂` `肝` はどちらも語彙にあるが、語としては誤読する）。
再現率を上げるには誤爆が増えるので、適合率を優先した。

    >>> from cutetts.training.yomi import ReadingAssigner
    >>> assigner = ReadingAssigner.from_model_dir("model/CuteTTS")
    >>> assigner.apply("それじゃ湊さんを案内したら")
    'それじゃミナトさんを案内したら'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: 置換しない品詞。記号は `read` が `、` になるため触ると文が壊れる
#: （`――` や `…` が読点に化ける）。
SKIP_POS = frozenset({"記号", "フィラー", "その他"})

#: 置換する語の最小長。**1**（1文字語も置換する）。
#:
#: 当初は「`一` `生` のような多音字を壊しうる」として2にしたが、実測すると
#: **多音字は語彙にあるので判定に掛からない**。1文字で fallback になるのは
#: `顔` `頃` `塩` `歳` のような語彙に無い漢字だけだった。
#: 一方 `湊` `聖`（どちらも実際に誤読した人名）は1文字なので、2では逃す。
#: 30万文のうち3,000文で測ると、置換が起きる文は 24.2% → 25.9% と
#: 1.7pt しか増えない。**再現率の利得が誤爆のコストを上回る。**
MIN_SURFACE_LENGTH = 1


def _load_vocab(model_dir: str | Path) -> frozenset[str]:
    """checkpointのtokenizerが単独pieceとして持つ文字の集合。"""
    import sentencepiece as spm

    path = Path(model_dir) / "tokenizer" / "tokenizer.model"
    if not path.is_file():
        raise FileNotFoundError("tokenizer が無い: {}".format(path))
    processor = spm.SentencePieceProcessor()
    processor.load(str(path))
    return frozenset(processor.id_to_piece(i) for i in range(processor.get_piece_size()))


@dataclass
class ReadingAssigner:
    """byte-fallback を含む語だけを読み（片仮名）へ置き換える。

    ``vocab`` は checkpoint の tokenizer が単独pieceとして持つ文字の集合。
    ここに無い文字を含む語が置換対象になる。
    """

    vocab: frozenset[str]
    skip_pos: frozenset[str] = SKIP_POS
    min_length: int = MIN_SURFACE_LENGTH
    _replaced: list[tuple[str, str]] = field(default_factory=list, repr=False)

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, **kwargs) -> "ReadingAssigner":
        return cls(vocab=_load_vocab(model_dir), **kwargs)

    def needs_reading(self, surface: str, pos: str) -> bool:
        """この語を読みへ置き換えるべきか。"""
        if pos in self.skip_pos:
            return False
        if len(surface) < self.min_length:
            return False
        return any(char not in self.vocab for char in surface)

    def apply(self, text: str) -> str:
        """文中の該当語を読みへ置き換える。該当が無ければ元の文をそのまま返す。

        `run_frontend` は語を分割して `string` と `read` を返す。
        置換しない語は `string` をそのまま連結するので、**記号や空白は保たれる**。
        """
        import pyopenjtalk

        self._replaced = []
        words = pyopenjtalk.run_frontend(text)
        if not words:
            return text

        parts: list[str] = []
        for word in words:
            surface = word.get("string") or ""
            reading = word.get("read") or ""
            if surface and reading and self.needs_reading(surface, word.get("pos") or ""):
                parts.append(reading)
                self._replaced.append((surface, reading))
            else:
                parts.append(surface)
        return "".join(parts)

    @property
    def replaced(self) -> list[tuple[str, str]]:
        """直前の `apply` で置き換えた (表記, 読み) の一覧。"""
        return list(self._replaced)


def assign_readings(text: str, model_dir: str | Path = "model/CuteTTS") -> str:
    """1回だけ使うとき用の薄い入口。**繰り返し呼ぶなら `ReadingAssigner` を使い回す**
    （tokenizerの読み込みが毎回走る）。"""
    return ReadingAssigner.from_model_dir(model_dir).apply(text)
