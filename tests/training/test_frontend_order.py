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
