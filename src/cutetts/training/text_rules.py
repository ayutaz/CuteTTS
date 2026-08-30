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

"""日本語テキストの除外・markupルール（D-016）。

``docs/japanese-training/data-inventory.md`` 「前処理で除外・変換が必要な発話」節を、
``data/raw/gol/metadata.tsv``（7,405,094行）の全走査で裏を取ってから実装したもの。
:mod:`cutetts.training.manifest` の :func:`~cutetts.training.manifest.validate` が
遅延importで参照する。

このmoduleは **文字列だけを扱う**。音声にもmodelにも依存しないので、実データ無しで
決定的にテストできる。

実測（gol-dataset全件、metadata.tsv）
------------------------------------

markupの出現は7,405,094発話中のごく一部（0.05%前後）だが、engineごとに記法が違い
1種類ではない。走査で確認できた記法は次のとおり:

===============================  =======  ==========================================
記法                             実測件数  意味
===============================  =======  ==========================================
``<rよみ>表層</r>``               2,547    ルビ。**読みがタグ側、表層がタグの中身**
``<ハ>``                          2,211    ハートマーク（.）のマクロ。非言語装飾
``%0``                            1,523    同じくハートマーク相当の装飾コード
``[n]``                             332    改行制御
``<d>…</d>`` / ``<d0018>…</d>``     328    用語集リンク。中身は本文なので残す
``%bd``                             252    **主人公名の差し替え変数**。音声と不一致
``@ruby``                           123    ルビの残骸。区切り文字が無く読みを復元できない
``</s>`` / ``<s36>``                 65    表示スタイルのspan。中身は本文なので残す
``$s:21;`` ``$b;`` ``$bd;`` 等    1,000+   文字サイズ・色・太字の制御コード
``[rb,表層,よみ]``                   11+   ルビ（別記法）
``[表層/よみ]``                      35+   ルビ（別記法）
``[size 大]`` ``[quake]``           20+   演出コマンド
``[sel_init …][sel_text …]``        20+   選択肢挿入コマンド
``[name キャラ名 voice_id]``         11+   話者名・voice fileの指定
``<center>`` ``<e>`` ``<k>``        40+   レイアウト系タグ
===============================  =======  ==========================================

``%`` は「90%以上」のような **本物のパーセント記号** としても出るため、
:data:`MARKUP_PATTERN` の ``%`` 分岐は直前が数字のときにマッチしない。

未対応と決めたもの（理由つき）
------------------------------

* ``@ruby`` は ``空の@ruby門で〜`` のように **区切りが無い**。表層と読みの境界を
  決められないので :func:`extract_ruby` は拾わない（markupとしては除去する）。
* ``[現在|今、ここ]`` ``[今も:・]`` のような ``|`` / ``:`` 区切りのルビは、実例を見ると
  左右どちらが表層かがgameごとに逆転している（``[・|？]`` は傍点なので右が表層、
  ``[現在|今、ここ]`` は左が読み）。誤った読みを学習に混ぜる方が害が大きいので
  :func:`extract_ruby` は拾わず、除去だけする。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

__all__ = [
    "GENERIC_SPEAKER_LABELS",
    "MARKUP_PATTERN",
    "NAME_PLACEHOLDER_PATTERN",
    "NON_LEXICAL_CHARS",
    "PUNCTUATION_CATEGORIES",
    "contains_markup",
    "contains_name_placeholder",
    "extract_ruby",
    "find_markup",
    "generic_speaker_ids",
    "gol_speaker_hash",
    "has_name_placeholder",
    "is_punctuation_only",
    "strip_markup",
]


# --------------------------------------------------------------------------------------
# 1. 句読点・記号のみの発話
# --------------------------------------------------------------------------------------

#: 「言語内容が無い」とみなすUnicode general categoryの先頭文字。
#:
#: * ``P`` 句読点（``…`` ``！`` ``―`` …）
#: * ``S`` 記号（``♪`` ``♡`` ``～`` ``■`` …）
#: * ``Z`` 空白
#: * ``C`` 制御文字・未割当
PUNCTUATION_CATEGORIES: frozenset[str] = frozenset({"P", "S", "Z", "C"})

#: 単独では発音できない仮名・記号。categoryとしては文字（Lo/Lm）だが読みを持たない。
#:
#: ``……っ！？`` を「句読点のみ」と判定するために要る。促音・長音・踊り字は直前の母音が
#: 無ければ音にならないので、これだけの列は言語内容が無いとみなす。
#: 小書き母音（``ぁぃぅぇぉ``）は単独でも発音できるので **含めない**。
NON_LEXICAL_CHARS: frozenset[str] = frozenset("っッーｰ々〻ゝゞヽヾ")


def is_punctuation_only(text: str, *, allow_non_lexical_kana: bool = True) -> bool:
    """句読点・記号・空白のみ（例 ``"…………"`` ``"……っ！？"`` ``"・・・"``）なら ``True``。

    空文字・空白のみも ``True`` を返す（呼び出し側は ``empty_text`` と併せて弾く）。

    Args:
        text: 判定対象。
        allow_non_lexical_kana: ``True`` なら :data:`NON_LEXICAL_CHARS`
            （促音・長音・踊り字）を句読点と同じ扱いにする。``False`` にすると
            ``docs/japanese-training/data-inventory.md`` が報告した
            「句読点・記号のみ 152,605件」に対応する狭い判定になる
            （こちらの実測は 152,623件。差は18件）。

    Returns:
        言語内容を持つ文字が1つも無ければ ``True``。
    """
    stripped = text.strip()
    if not stripped:
        return True
    for char in stripped:
        if unicodedata.category(char)[0] in PUNCTUATION_CATEGORIES:
            continue
        if allow_non_lexical_kana and char in NON_LEXICAL_CHARS:
            continue
        return False
    return True


# --------------------------------------------------------------------------------------
# 2. markup
# --------------------------------------------------------------------------------------

# ルビ: <rよみ>表層</r>。表層側に <d0018>…</d> が入れ子になる実例がある。
_RUBY_ANGLE_RE = re.compile(r"<r([^<>\s]{1,32})>(.{0,64}?)</r>")

# ルビ: [rb,表層,よみ]
_RUBY_RB_RE = re.compile(r"\[rb,([^\[\],\s]{1,24}),([^\[\],\s]{1,24})\]")

# ルビ: [表層/よみ]
_RUBY_SLASH_RE = re.compile(r"\[([^\[\]/|:,\s]{1,24})/([^\[\]/|:,\s]{1,24})\]")

# ルビ（左右どちらが表層か不定）: [・|？] [現在|今、ここ] [今も:・] [まる|○]
_RUBY_AMBIGUOUS_RE = re.compile(r"\[([^\[\]|:\s]{1,24})[|:]([^\[\]|:\s]{1,24})\]")

#: 傍点・圏点として使われる装飾文字。``[まる|○]`` のどちら側が本文かの判定に使う。
_BOUTEN_CHARS = frozenset("・･。゛゜○●◯◎﹅﹆＾^、,")

# 山括弧タグ全般。</r> <d0018> </d> <s36> </s> <center> <ハ> <e> <k> <a> …
_ANGLE_TAG = r"</?[^<>\s]{1,32}>"

# 角括弧コマンド全般。[n] [f5] [size 大] [quake] [sel_text label="x" text="y"] [name A b]
_BRACKET_CMD = r"\[[^\[\]\n]{0,64}\]"

# %系コード。%bd %0 %C %D1149 …。直前が数字なら本物の「％」（「90%以上」）なので
# マッチさせない。ただし ``%0%0%0`` のような連鎖では直前の数字が前のcodeの一部なので、
# 「``%`` に続かない数字」だけを除外する入れ子lookbehindにしてある。
_PERCENT_CODE = r"(?<!(?<!%)[0-9０-９])%(?:bd|[A-Za-z][0-9]*|[0-9]+)"

# $系コード。$s:21; $sd; $b; $bd; $c:255/64/80,48/48/48;
# 誤検出を避けるため末尾の ``;`` を必須にしている。
_DOLLAR_CODE = r"\$[A-Za-z_][A-Za-z0-9_]*(?::[^;\n]{0,64})?;"

# 区切りの無いルビ残骸。
_AT_RUBY = r"@ruby"

#: 構造化markupの1トークンにマッチする。:func:`find_markup` はこれを ``findall`` する。
MARKUP_PATTERN: re.Pattern[str] = re.compile(
    "|".join((_BRACKET_CMD, _ANGLE_TAG, _AT_RUBY, _PERCENT_CODE, _DOLLAR_CODE))
)


def find_markup(text: str) -> list[str]:
    """検出したmarkupトークンを出現順に返す。

    ``<rあやせ>綾瀬</r>`` は ``["<rあやせ>", "</r>"]`` のように **開きタグと閉じタグを
    別トークン** として返す（data-inventory.mdの内訳表と同じ数え方）。
    """
    return MARKUP_PATTERN.findall(text)


def contains_markup(text: str) -> bool:
    """markupを1つでも含むなら ``True``。

    :func:`cutetts.training.manifest.validate` が ``markup`` code の判定に使う名前。
    """
    return MARKUP_PATTERN.search(text) is not None


def _strip_inner_tags(text: str) -> str:
    """ルビの表層側に残る ``<d0018>…</d>`` のような入れ子タグだけを落とす。"""
    return re.sub(_ANGLE_TAG, "", text)


def _resolve_ambiguous_ruby(match: re.Match[str]) -> str:
    """``[A|B]`` / ``[A:B]`` から本文になる側を選ぶ（heuristic・低信頼）。

    片側が傍点だけなら他方を残す（``[・|？]`` → ``？``、``[まる|○]`` → ``まる``）。
    両側が本文なら左を残す（``[現在|今、ここ]`` → ``現在``）。
    どちらが表層かはgameごとに逆転するため :func:`extract_ruby` では使わない。
    """
    left, right = match.group(1), match.group(2)
    left_bouten = all(char in _BOUTEN_CHARS for char in left)
    right_bouten = all(char in _BOUTEN_CHARS for char in right)
    if left_bouten and not right_bouten:
        return right
    if right_bouten and not left_bouten:
        return left
    return left


def strip_markup(text: str, *, keep_ruby_surface: bool = True) -> str:
    """markupを除去した本文を返す。

    Args:
        text: 元テキスト。
        keep_ruby_surface: ``True`` なら ``<rかな>漢字</r>`` から表層（``漢字``）を残す。
            ``False`` ならルビ構造を **表層ごと** 落とす。

    Returns:
        markupを除いた本文。除去で生じた連続空白は1つに畳み、前後を ``strip`` する。
    """
    if keep_ruby_surface:
        result = _RUBY_ANGLE_RE.sub(lambda m: _strip_inner_tags(m.group(2)), text)
        result = _RUBY_RB_RE.sub(r"\1", result)
        result = _RUBY_SLASH_RE.sub(r"\1", result)
        result = _RUBY_AMBIGUOUS_RE.sub(_resolve_ambiguous_ruby, result)
    else:
        result = _RUBY_ANGLE_RE.sub("", text)
        result = _RUBY_RB_RE.sub("", result)
        result = _RUBY_SLASH_RE.sub("", result)
        result = _RUBY_AMBIGUOUS_RE.sub("", result)

    # 残りは中身を持たないか、中身がそのまま本文であるmarkup。タグだけ落とせばよい。
    result = MARKUP_PATTERN.sub("", result)
    result = re.sub(r"[ \t　]{2,}", " ", result)
    return result.strip()


def extract_ruby(text: str) -> list[tuple[str, str]]:
    """``(表層, 読み)`` の組を出現順に返す。読み情報（J2）の材料。

    拾うのは境界が曖昧でない3記法だけ（module docstringの「未対応」参照）:

    * ``<rよみ>表層</r>``
    * ``[rb,表層,よみ]``
    * ``[表層/よみ]``

    Returns:
        重複を除かない ``(表層, 読み)`` のlist。1つも無ければ空list。
    """
    found: list[tuple[int, str, str]] = []
    for match in _RUBY_ANGLE_RE.finditer(text):
        surface = _strip_inner_tags(match.group(2)).strip()
        reading = match.group(1).strip()
        if surface and reading:
            found.append((match.start(), surface, reading))
    for pattern in (_RUBY_RB_RE, _RUBY_SLASH_RE):
        for match in pattern.finditer(text):
            surface = match.group(1).strip()
            reading = match.group(2).strip()
            if surface and reading:
                found.append((match.start(), surface, reading))
    found.sort(key=lambda item: item[0])
    return [(surface, reading) for _, surface, reading in found]


# --------------------------------------------------------------------------------------
# 3. 主人公名の差し替え変数
# --------------------------------------------------------------------------------------

#: 主人公名などの差し替え変数にマッチする。
#:
#: gol-datasetで実在するのは ``%bd``（252件）だけ。他のalternativeは他engineでよく使う
#: 記法への保険で、gol-dataset全件走査では1件もヒットしない（実測）。
#: ``$bd;`` は太字終了タグであって名前ではないので、意図的に含めていない。
NAME_PLACEHOLDER_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![0-9A-Za-z０-９])%bd(?![0-9A-Za-z])"
    r"|%name%"
    r"|\$\{\s*name\s*\}"
    r"|[＜<【]\s*主人公\s*[＞>】]"
)


def has_name_placeholder(text: str) -> bool:
    """``%bd`` のような主人公名差し替え変数を含むか。

    テキストと音声が一致しない（音声側は具体的な名前を喋っている、または読み飛ばして
    いる）ため、学習からは除外する。
    """
    return NAME_PLACEHOLDER_PATTERN.search(text) is not None


def contains_name_placeholder(text: str) -> bool:
    """:func:`has_name_placeholder` の別名。

    :func:`cutetts.training.manifest.validate` がこの名前で呼ぶ。
    """
    return has_name_placeholder(text)


# --------------------------------------------------------------------------------------
# 4. 総称ラベル話者
# --------------------------------------------------------------------------------------

#: 総称ラベル語。1つのIDに **複数の異なる声** が混ざるため、speaker条件付けを壊す。
#:
#: data-inventory.md の「総称ラベル91件 / 47.4 h / 37,923発話」に対応する。
#: 声の識別子として使えないだけでテキスト自体は正常なので、speaker条件を外した
#: 学習に使う余地は残る（判断はP1d以降）。
GENERIC_SPEAKER_LABELS: tuple[str, ...] = (
    "？？？",
    "???",
    "？",
    "？？",
    "全員",
    "一同",
    "二人",
    "三人",
    "男の子",
    "女の子",
    "男",
    "女",
    "男性",
    "女性",
    "少年",
    "少女",
    "青年",
    "老人",
    "母",
    "父",
    "母親",
    "父親",
    "兄",
    "姉",
    "弟",
    "妹",
    "モブ",
    "群衆",
    "村人",
    "町人",
    "店員",
    "店主",
    "兵士",
    "衛兵",
    "騎士",
    "医者",
    "看護師",
    "教師",
    "先生",
    "生徒",
    "学生",
    "客",
    "通行人",
    "アナウンス",
    "ナレーション",
    "ナレーター",
    "システム",
    "声",
    "謎の声",
    "男の声",
    "女の声",
    "男子生徒",
    "女子生徒",
    "男子",
    "女子",
    "子供",
    "その他",
    "不明",
    "なし",
    "一般人",
    "住民",
    "記者",
    "司会",
    "神父",
    "シスター",
    "メイド",
    "執事",
    "運転手",
    "警官",
    "刑事",
    "部下",
    "社員",
    "上司",
    "同僚",
    "女将",
    "主人",
    "奥さん",
    "おばさん",
    "おじさん",
    "おじいさん",
    "おばあさん",
    "男たち",
    "女たち",
    "人々",
    "観客",
    "生徒たち",
    "子供たち",
)


def gol_speaker_hash(label: str) -> str:
    """gol-datasetの ``speaker`` 列を表示名から復元する。

    実測で確認済みの生成方式は ``SHA-256(表示名のUTF-8)`` の16進digest先頭32桁を
    大文字にしたもの。例: ``gol_speaker_hash("女の子")`` は
    ``"E9291C8A0748FB406D1AE76437CCC344"``。
    """
    return hashlib.sha256(label.encode("utf-8")).hexdigest()[:32].upper()


def generic_speaker_ids() -> frozenset[str]:
    """:data:`GENERIC_SPEAKER_LABELS` を :func:`gol_speaker_hash` にかけた集合。

    :func:`cutetts.training.manifest.validate` の ``generic_speaker_ids`` 引数に渡す。
    """
    return frozenset(gol_speaker_hash(label) for label in GENERIC_SPEAKER_LABELS)
