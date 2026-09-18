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

"""frontend の適用順（`yomi.apply_frontend`）。

**J2 → J3 の順だと数詞が壊れる。** J3 は `run_frontend` で形態素解析を
やり直すので、J2 が作った仮名列を再解釈して漢数字を復活させる。
実測（`checkpoints/s1v2-fp32-30000`）:

    千二百八十円  J2→J3: せんにひゃく八ジュウエン / J3→J2: せんにひゃくはちじゅうエン
    八月三十一日  J2→J3: 八月さんジュウ一日      / J3→J2: 八月さんじゅういち日

**ここが逆に戻ると、実運用の合成で数詞が読めなくなる。**
"""

from __future__ import annotations

import pytest

from cutetts.training.reading import expand_kanji_numerals
from cutetts.training.yomi import apply_frontend


class _StubAssigner:
    """J3 の壊し方だけを真似る。

    実物は `pyopenjtalk` と checkpoint の語彙を要するので、ここでは
    「仮名列を再解釈して漢数字を復活させる」挙動だけを固定して持つ。
    """

    def __init__(self):
        self.seen: list[str] = []

    def apply(self, text: str) -> str:
        self.seen.append(text)
        return text.replace("はちじゅう", "八ジュウ").replace("千", "セン")


def test_J3を先に掛けるのでJ2の出力は再解釈されない():
    stub = _StubAssigner()
    out = apply_frontend("価格は千二百八十円です。", assigner=stub)
    assert "八ジュウ" not in out
    # J3 が見るのは**原文**でなければならない（J2の出力ではない）
    assert stub.seen == ["価格は千二百八十円です。"]


def test_逆順なら壊れることを記録しておく():
    """この順（J2 → J3）に戻してはいけない。"""
    stub = _StubAssigner()
    broken = stub.apply(expand_kanji_numerals("価格は千二百八十円です。"))
    assert "八ジュウ" in broken


def test_assignerが無ければJ2だけが掛かる():
    out = apply_frontend("価格は千二百八十円です。", assigner=None)
    assert out == expand_kanji_numerals("価格は千二百八十円です。")


def test_expand_numeralsを切るとJ3だけになる():
    stub = _StubAssigner()
    out = apply_frontend("千円", assigner=stub, expand_numerals=False)
    assert out == "セン円"


def test_数詞が無い文では順序で結果が変わらない():
    stub_a, stub_b = _StubAssigner(), _StubAssigner()
    forward = apply_frontend("中華料理のお店へ。", assigner=stub_a)
    backward = stub_b.apply(expand_kanji_numerals("中華料理のお店へ。"))
    assert forward == backward


def test_J3を2回掛けると壊れる():
    """**呼び出し側で `assigner.apply` を足してはいけない。**

    `apply_frontend` / `frontend_text` の中で既に J3 が掛かっている。
    2回目は J2 の出力（仮名列）を再解釈するので、逆順と同じ壊れ方をする。
    `evaluate_prosody.py` に実際に入っていた（prosody set 240文中2文で発火）。
    """
    stub = _StubAssigner()
    once = apply_frontend("価格は千二百八十円です。", assigner=stub)
    assert "八ジュウ" not in once
    assert "八ジュウ" in stub.apply(once)


# ---------------------------------------------------------------- 仮名化の分離
#
# **`yomi` は「仮名化」ではない。** J3 は byte-fallback を含む語だけを
# 置換するので、評価set 200文の実測で漢字は 18.16% → 14.65% しか減らない
# （`accent` は 0%）。`accent` と `yomi` の差には**仮名化の効果**が混ざる。
# `kana_full` はその分離のための対照（全文仮名・記号なし）。


def test_kana_fullは記号を置かない():
    pytest.importorskip("pyopenjtalk", reason="[ja] extra が要る")
    from cutetts.training.accent import NUCLEUS_MARK
    from cutetts.training.yomi import frontend_text

    text = "箸を持つ手と、橋を渡る足。"
    assert NUCLEUS_MARK not in frontend_text(text, "kana_full")
    assert NUCLEUS_MARK in frontend_text(text, "accent")


def test_kana_fullはaccentから記号を抜いたものと同じ():
    pytest.importorskip("pyopenjtalk", reason="[ja] extra が要る")
    from cutetts.training.accent import NUCLEUS_MARK
    from cutetts.training.yomi import frontend_text

    text = "明日の待ち合わせは、駅の南口で大丈夫ですか。"
    assert frontend_text(text, "kana_full") == \
        frontend_text(text, "accent").replace(NUCLEUS_MARK, "")


def test_yomiは漢字を全部消さない():
    """**「仮名化のみ」と書いてはいけない。** J3 の対象は fallback を含む語だけ。"""
    pytest.importorskip("pyopenjtalk", reason="[ja] extra が要る")
    from cutetts.training.yomi import frontend_text

    text = "良い子と友達になりましたね。"
    assert any("一" <= ch <= "鿿" for ch in frontend_text(text, "yomi"))
    assert not any("一" <= ch <= "鿿"
                   for ch in frontend_text(text, "kana_full"))


def test_kana_full_cleanは記号なしで長音化を直す():
    """**効いているのは片仮名の忠実さ**（R-047）なので、その欠陥を直す版。

    記号は入れず、語境界をまたぐ長音化だけを止める。
    """
    pytest.importorskip("pyopenjtalk", reason="[ja] extra が要る")
    from cutetts.training.accent import NUCLEUS_MARK
    from cutetts.training.yomi import frontend_text

    text = "その目で見たことを、そのまま言え"
    clean = frontend_text(text, "kana_full_clean")
    assert NUCLEUS_MARK not in clean
    assert "コトオ" in clean
    assert "コトー" not in clean
    # 記号を抜いた `accent_clean` と一致する
    assert clean == frontend_text(text, "accent_clean").replace(NUCLEUS_MARK, "")


def test_kana_full_cleanは語の中の長音を残す():
    pytest.importorskip("pyopenjtalk", reason="[ja] extra が要る")
    from cutetts.training.yomi import frontend_text

    assert "ショーヒゼー" in frontend_text("消費税込みです。", "kana_full_clean")
