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

#: 読みを平仮名で書くか。**False**（既定は `pyopenjtalk` が返す片仮名のまま）。
#:
#: 学習コーパスは平仮名が主（manifest全文で片仮名は仮名の3.4%）なので、
#: 片仮名は分布から外れて不利ではないかと考え、300文で対照実験した。
#: **支持されなかった。**
#:
#: | 条件 | 素CER | 対 J3なし |
#: |---|---:|---|
#: | J3なし | 25.86% | — |
#: | **片仮名** | **23.22%** | **-2.64pt**（有意） |
#: | 平仮名 | 23.73% | -2.13pt（有意） |
#:
#: 片仮名 → 平仮名は +0.51pt、95%CI [-0.82, +1.83] で**有意差なし**。
#: 悪化した文数も80で同じだった。**表記は効かない。**
#: token数が同じ（`キレイ` も `きれい` も4 piece、fallbackなし）ことと整合する。
#: 測定値が良い片仮名を既定にし、切り替えは残す。
USE_HIRAGANA = False

#: 置換する語の最小長。**1**（1文字語も置換する）。
#:
#: 当初は「`一` `生` のような多音字を壊しうる」として2にしたが、実測すると
#: **多音字は語彙にあるので判定に掛からない**。1文字で fallback になるのは
#: `顔` `頃` `塩` `歳` のような語彙に無い漢字だけだった。
#: 一方 `湊` `聖`（どちらも実際に誤読した人名）は1文字なので、2では逃す。
#: 30万文のうち3,000文で測ると、置換が起きる文は 24.2% → 25.9% と
#: 1.7pt しか増えない。**再現率の利得が誤爆のコストを上回る。**
MIN_SURFACE_LENGTH = 1


def to_hiragana(text: str) -> str:
    """片仮名を平仮名にする。長音符 `ー` などはそのまま残す。"""
    return "".join(
        chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in text)


#: 読み比較で落とす文字（句読点・記号・空白）。
#: `evaluate_japanese_cer.normalize` と揃えてある。
_PUNCT_FOR_READING = frozenset(
    "、。「」『』・…‥！？!?,.-―ー~〜"
    + chr(0x22) + chr(0x27) + "()（）"
    + chr(0x20) + chr(0x3000) + chr(0x09) + chr(0x0A)
)


def reading_form(text: str) -> str:
    """文を**読み**（平仮名）へ正規化する。表記の違いを消すための比較用。

    仮名で入力すると **ASR も仮名で書き戻す**ので、漢字の参照文に対する
    素のCERは「発音は正しいのに表記が違う」を誤りと数える（R-029）。
    両辺をこの形にしてから比べると、その分が落ちる。

        >>> reading_form("楓寺の御先様")
        'かえでてらのごさきさま'
        >>> reading_form("カエデテラの御先様")
        'かえでてらのごさきさま'

    **これは万能ではない。** 同音異義（`聞く` / `効く`）の誤りは見えなくなる。
    素のCERと**併記する**こと。長音は `read` のまま残す（`pron` は
    仮名入力と漢字入力で長音化が食い違い、別の欠陥を持ち込む:
    `王様`→`オーサマ` に対し `おうさま`→`オウサマ`）。
    """
    import pyopenjtalk

    words = pyopenjtalk.run_frontend(text)
    if not words:
        return ""
    parts = []
    for word in words:
        if (word.get("pos") or "") in SKIP_POS:
            continue                       # 記号の `read` は `、` なので落とす
        parts.append(word.get("read") or word.get("string") or "")
    return "".join(
        c for c in to_hiragana("".join(parts)) if c not in _PUNCT_FOR_READING)


#: frontend の種類。**学習と推論で同じ値を使う**（食い違うと条件が変わる）。
#:
#: * ``"none"``: 何もしない。**学習の既定**（学習21回すべてこれ）
#: * ``"yomi"``: J3（語の読み付与）→ J2（漢数字の展開）。**推論の既定**
#: * ``"kana_full"``: **全文を片仮名にするが記号は置かない。** `accent` から
#:   記号だけを抜いた対照。`yomi` は漢字を 18.16% → 14.65% しか減らさない
#:   （`accent` は 0%）ので、`yomi` との差には仮名化の効果が混ざる
#: * ``"kana_full_clean"``: `kana_full` から**語境界をまたぐ長音化を除いた**もの。
#:   **効いているのは片仮名の忠実さなので**（R-047）、その欠陥を直す意味がある
#: * ``"accent"``: 全文を片仮名にし、アクセント核に記号を置く（M4a）
#: * ``"accent_clean"``: `accent` と同じだが、**語境界をまたぐ長音化を止める**。
#:   `コトモーシエテ` ではなく `コトモオシエテ`（実測で約4,400箇所/20,000文）
#: * ``"accent_shuffled"``: **核の位置を偽の位置へ動かした対照**。記号の数と
#:   句の構造は `accent` と同じ。読みCERが落ちたままなら効いていたのは
#:   「区切りがあること」で、戻るなら「アクセントの内容」
FRONTEND_MODES = ("none", "yomi", "kana_full", "kana_full_clean", "accent",
                  "accent_shuffled", "accent_clean")


def frontend_text(text: str, mode: str = "none", *,
                  assigner: "ReadingAssigner | None" = None) -> str:
    """`mode` に従って frontend を掛ける。**学習と推論の共通入口。**

    **学習は長らく `text_raw` をそのまま使っていた**（`mode="none"`）。
    一方で推論は J2/J3 を掛けるので、**学習データの 29.9% の文で表記が
    食い違っていた**（2026-09-18 実測）。J2/J3 が再学習なしで効いたのは
    仮名が学習分布の中にあるからで、**新しい記号は分布に無いので
    学習側にも同じ frontend を通さないと効かない**（M4a）。

    Args:
        text: 元のテキスト。
        mode: :data:`FRONTEND_MODES` のいずれか。
        assigner: ``"yomi"`` で使う。``None`` なら読み付与を飛ばす。
    """
    if mode not in FRONTEND_MODES:
        raise ValueError(f"未知の frontend: {mode}（{FRONTEND_MODES}）")
    if mode == "none":
        return text
    if mode in ("kana_full", "kana_full_clean", "accent", "accent_shuffled",
                "accent_clean"):
        from cutetts.training.accent import NUCLEUS_MARK, accent_marked_text

        return accent_marked_text(
            text,
            mark="" if mode.startswith("kana_full") else NUCLEUS_MARK,
            shuffle=mode == "accent_shuffled",
            merge_across_words=not mode.endswith("_clean"),
        ).text
    return apply_frontend(text, assigner=assigner, expand_numerals=True)


def apply_frontend(text: str, *, assigner: "ReadingAssigner | None" = None,
                   expand_numerals: bool = True) -> str:
    """読み付与（J3）→ 漢数字の展開（J2）の順で frontend を掛ける。

    **順序を逆にすると数詞が壊れる。** J3 は `run_frontend` で形態素解析を
    やり直すので、J2 が作った仮名列（`せんにひゃくはちじゅう`）を再解釈して
    **漢数字を復活させる**（`せんにひゃく八ジュウ`）。小書き仮名（`ゅ`）が
    byte-fallback なのが引き金で、`じゅう` が置換対象になる。

    実測（`checkpoints/s1v2-fp32-30000` の vocab）:

    | 原文 | J2 → J3（誤り） | **J3 → J2（本関数）** |
    |---|---|---|
    | 千二百八十円 | せんにひゃく八ジュウエン | **せんにひゃくはちじゅうエン** |
    | 八月三十一日 | 八月さんジュウ一日 | **八月さんじゅういち日** |

    数詞を含まない文では両者は一致する（`中華料理` → `チュウカ料理`）。
    """
    from cutetts.training.reading import expand_kanji_numerals

    spoken = assigner.apply(text) if assigner is not None else text
    return expand_kanji_numerals(spoken) if expand_numerals else spoken


def _load_vocab(model_dir: str | Path) -> frozenset[str]:
    """checkpointのtokenizerが単独pieceとして持つ文字の集合。

    **これは byte-fallback の近似であって実測ではない。** SentencePiece は
    NFKC正規化してから照合するので、piece に無くても fallback にならない文字が
    ある（`…`→`..`×3、`！`→`!`、`？`→`?`、全角数字→半角）。
    実測すると評価300文のうち**判定が変わるのは2文**（`％`→`パーセント` など）で、
    記号は `skip_pos` が先に弾くため実害は出ていない（R-030）。
    厳密に測るなら `sp.encode(surface)` に `<0x..>` が出るかを見ること。
    """
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
    hiragana: bool = USE_HIRAGANA
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
                if self.hiragana:
                    reading = to_hiragana(reading)
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
