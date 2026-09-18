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


# ---------------------------------------------------------------- shuffle 対照
#
# **記号が読みに効いた理由を切り分けるための対照。**
# 記号の数と句の構造は同じまま、核の位置だけを偽の位置へ動かす。
# これが壊れていると「区切りの効果」と「内容の効果」が分離できない。


def test_shuffleは記号の数を変えない():
    text = "明日の待ち合わせは、駅の南口で大丈夫ですか。"
    assert accent_marked_text(text, shuffle=True).marks == \
        accent_marked_text(text).marks


def test_shuffleはモーラ数を変えない():
    text = "箸を持つ手と、橋を渡る足。"
    assert accent_marked_text(text, shuffle=True).moras == \
        accent_marked_text(text).moras


def test_shuffleは核の位置を動かす():
    """同音異義の区別が**消える**のが狙い。"""
    marked = accent_marked_text("箸を持つ手と、橋を渡る足。", shuffle=True).text
    assert marked != accent_marked_text("箸を持つ手と、橋を渡る足。").text


def test_shuffleは同じ句に同じ偽位置を割り当てる():
    """再現できないと、同じ語が文ごとに違う読みになって条件が濁る。"""
    a = accent_marked_text("駅の南口で待っています。", shuffle=True).text
    b = accent_marked_text("駅の南口で待っています。", shuffle=True).text
    assert a == b


def test_shuffleは平板に記号を付けない():
    """核が無い句に記号を足すと、記号の数が変わってしまう。"""
    text = "桜が咲いた。"
    assert accent_marked_text(text, shuffle=True).marks == \
        accent_marked_text(text).marks


@pytest.mark.parametrize("text", [
    "そう言われましてもぉー……困りますね。",
    "中華料理のお店へ、湊さんと行きませんか。",
    "ティッシュを取って。",
])
def test_shuffleでも未知の音素が出ない(text):
    marked = accent_marked_text(text, shuffle=True)
    assert marked.unknown == (), f"{text}: {marked.unknown}"


# ------------------------------------------------- 語境界をまたぐ長音化（accent_clean）
#
# 長音規則は「直前と同じ母音の裸母音」を `ー` にする。この規則は
# **語境界をまたいでも発火し**、実測（20,000文）で約4,400箇所を潰していた。
#
#   コトモ**オ**シエテ → コトモ**ー**シエテ   `教えて` の頭が消える
#
# **既定の挙動は変えられない** — 現行最良の checkpoint はこの規則で
# 学習してあるので、既定を直すと学習と推論が食い違う（F1 の教訓）。


def test_既定は語境界をまたいで長音化する():
    """**回帰の防波堤。** ここが変わると現行 checkpoint の前提が崩れる。"""
    marked = accent_marked_text("貴官のことも教えていただけませんか？").text
    assert "コトモーシエテ" in marked


def test_cleanは語の先頭で長音化を止める():
    marked = accent_marked_text("貴官のことも教えていただけませんか？",
                                merge_across_words=False).text
    assert "コトモオシエテ" in marked
    assert "コトモーシエテ" not in marked


def test_cleanでも語の中の長音は残る():
    """`ショーヒゼー` は語の中の長音。**止めてはいけない。**"""
    marked = accent_marked_text("消費税込みです。", merge_across_words=False).text
    assert "ショーヒゼー" in marked.replace(NUCLEUS_MARK, "")


def test_cleanは核の位置を変えない():
    """直すのは仮名だけ。アクセントの判定には触らない。"""
    text = "貴官のことも教えていただけませんか？"
    assert accent_marked_text(text, merge_across_words=False).marks == \
        accent_marked_text(text).marks


def test_cleanはモーラ数を変えない():
    """`ー` も1モーラなので、潰れが解けても数は同じ。"""
    text = "とにかく動かず、静かに、じっと、していろ"
    assert accent_marked_text(text, merge_across_words=False).moras == \
        accent_marked_text(text).moras


def test_cleanは助動詞の境界では止めない():
    """**自分で入れた欠陥の回帰（1回目）。**

    `でしょう` は NJD で `でしょ` + `う`（助動詞）に分かれる。語境界で
    一律に長音化を止めると `デショオ` になる（正しくは `デショー`）。
    実測で 10,000文中 113箇所がこの形だった。
    """
    marked = accent_marked_text("どなた様でしょうか？", merge_across_words=False).text
    assert "デショー" in marked.replace(NUCLEUS_MARK, "")
    assert "デショオ" not in marked


def test_cleanは助詞のヲを飲み込まない():
    """**自分で入れた欠陥の回帰（2回目）。**

    1回目の修正で「自立語だけ止める」にしたら、助詞の `を` が飲み込まれて
    `ことを` が `コトー` になった（正しくは `コトオ`）。`を` は頻出なので
    実害が大きい。**助動詞だけを許す**のが両方を満たす。
    """
    marked = accent_marked_text("その目で見たことを、そのまま言え",
                                merge_across_words=False).text
    bare = marked.replace(NUCLEUS_MARK, "")
    assert "コトオ" in bare
    assert "コトー" not in bare


def test_cleanは自立語の境界では止める():
    """内容語の頭が守られていること（本来直したかった欠陥）。"""
    marked = accent_marked_text("貴官のことも教えていただけませんか？",
                                merge_across_words=False).text
    assert "コトモオシエテ" in marked.replace(NUCLEUS_MARK, "")


def test_既定は助詞のヲも飲み込む():
    """**現行最良 checkpoint の前提を固定する。**

    `accent`（既定）では `ことを` が `コトー` になる。これは欠陥だが、
    `m4a-accent` はこの入力で学習してあるので**推論も同じでなければ
    ならない**。直した版は `accent_clean` として別に測る。
    """
    bare = accent_marked_text("その目で見たことを、そのまま言え").text.replace(
        NUCLEUS_MARK, "")
    assert "コトー" in bare
