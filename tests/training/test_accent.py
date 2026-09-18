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

"""アクセント核つき片仮名テキスト（M4a / `training.accent`）。

**この表現が壊れると、M4a の学習がまるごと無駄になる。**
実測で確認した3つの欠陥を回帰として残す。

1. 促音が `フ` になった（`letters` は MMS_FA 用で、`_fill_geminates` が
   次の子音を入れる。**音素から作らないといけない**）
2. 長音が `オ` / `エ` になった（`ショオヒゼエ`。学習分布から外れる）
3. 読点・句点が消えた（`か。` と `か？` で抑揚が違う）
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyopenjtalk", reason="[ja] extra が要る")

from cutetts.training.accent import (  # noqa: E402
    LONG_VOWEL,
    NUCLEUS_MARK,
    accent_marked_text,
    kana_of_phonemes,
)


def test_同音異義のアクセントが区別される():
    """**J3 では両方 `ハシ` になって区別が消える**（R-039）。核の位置で分かれる。"""
    hashi_bashi = accent_marked_text("箸を持つ手と、橋を渡る足。").text
    assert "ハ'シ" in hashi_bashi      # 箸 = 頭高（核1）
    assert "ハシ'" in hashi_bashi      # 橋 = 尾高（核2）


def test_促音は小さいツになる():
    """`letters` を使うと `エフ` になった（`_fill_geminates` の副作用）。"""
    marked = accent_marked_text("えっ、本当ですか。").text
    assert "ッ" in marked
    assert "フ" not in marked


def test_長音は長音記号になる():
    """`ショオヒゼエ` ではなく `ショーヒゼー`。"""
    marked = accent_marked_text("消費税込みです。").text
    assert LONG_VOWEL in marked
    assert "ゼエ" not in marked


def test_読点と句点が残る():
    marked = accent_marked_text("箸を持つ手と、橋を渡る足。").text
    assert "、" in marked
    assert marked.endswith("。")


def test_疑問符が残る():
    """**`か。` と `か？` で抑揚が違う**ので落とせない。"""
    assert accent_marked_text("これ、本当に大丈夫なの？").text.endswith("？")


def test_平板には核記号が付かない():
    """平板（0型）は句の中で下がらない。"""
    marked = accent_marked_text("桜が咲いた。")
    assert marked.marks <= 1        # `咲いた` 側のみ


def test_核記号は1文字():
    assert len(NUCLEUS_MARK) == 1


def test_漢字が残らない():
    """全文を片仮名にする。**漢字が残ると読みの手がかりが二重になる。**"""
    marked = accent_marked_text("明日の待ち合わせは、駅の南口で大丈夫ですか。").text
    assert not any("一" <= ch <= "鿿" for ch in marked)


def test_未知の音素は残して数える():
    """黙って消さない。表の漏れに気づけるようにする。"""
    marked = accent_marked_text("こんにちは。")
    assert marked.unknown == ()
    assert marked.moras > 0


def test_音素からの変換():
    assert kana_of_phonemes(("h", "a")) == "ハ"
    assert kana_of_phonemes(("sh", "i")) == "シ"
    assert kana_of_phonemes(("N",)) == "ン"
    assert kana_of_phonemes(("cl",)) == "ッ"
    assert kana_of_phonemes(("s", "U")) == "ス"      # 無声化
    assert kana_of_phonemes(("zzz",)) is None


@pytest.mark.parametrize("text", [
    "そう言われましてもぉー……困りますね。",
    "第3回の会議は9時からです。",
    "中華料理のお店へ、湊さんと行きませんか。",
    "ふぁいと！",
    "ヴァイオリンを弾く。",
    "ティッシュを取って。",
])
def test_実データ風の文で未知の音素が出ない(text):
    """**表の漏れは黙って壊れる。** 促音・拗音・外来音を通す。"""
    marked = accent_marked_text(text)
    assert marked.unknown == (), f"{text}: {marked.unknown}"
    assert marked.moras > 0
