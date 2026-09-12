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

"""漢数字を読み（仮名）へ展開する frontend（J2 / D-008）。

**なぜ学習では直らないか。** 評価の `out_of_domain` 12文は全てが
複合漢数字（`二十三` `千二百八十`）を含むのに、学習コーパスでは 0.32%
（235,016発話中752例）しかなく、しかも `二千二十一貫` のような特殊用例。
桁の合成規則を学べる分布ではない（gol全体10,654hに外挿しても約34時間分）。

一方 `十五分`→`15分`、`四トン`→`4トン` のような単純な数は既に読める。
**欠けているのは桁の合成規則だけ**なので、tokenizerの前で仮名へ展開すれば
モデルは既知の仮名として読める。**再学習を要しない。**

対象は数詞のみ。助数詞（`本` `匹` `日`）の音便は展開しない
（`三本`→`さんぼん` のような変化はモデルが文脈から扱う）。

    >>> expand_kanji_numerals("価格は千二百八十円です")
    '価格はせんにひゃくはちじゅう円です'
"""

from __future__ import annotations

import re

DIGITS = {
    "〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
LARGE_UNITS = {"万": 10**4, "億": 10**8, "兆": 10**12}

NUMERAL_CHARS = set(DIGITS) | set(SMALL_UNITS) | set(LARGE_UNITS)
_RUN = re.compile(f"[{''.join(NUMERAL_CHARS)}]+")

DIGIT_KANA = {
    0: "ゼロ", 1: "いち", 2: "に", 3: "さん", 4: "よん",
    5: "ご", 6: "ろく", 7: "なな", 8: "はち", 9: "きゅう",
}

#: 位の読み。連濁・促音（音便）は `_UNIT_EXCEPTIONS` で上書きする。
UNIT_KANA = {10: "じゅう", 100: "ひゃく", 1000: "せん",
             10**4: "まん", 10**8: "おく", 10**12: "ちょう"}

#: (前の数, 位) -> 読み。日本語の数詞に固有の音便。
_UNIT_EXCEPTIONS = {
    (3, 100): "さんびゃく",
    (6, 100): "ろっぴゃく",
    (8, 100): "はっぴゃく",
    (3, 1000): "さんぜん",
    (8, 1000): "はっせん",
    (1, 10**12): "いっちょう",
    (8, 10**12): "はっちょう",
}


def _parse_run(run: str) -> int | None:
    """漢数字の並びを整数にする。解釈できなければ ``None``。"""
    total = 0
    section = 0
    current = 0
    seen = False
    last_small = None              # 直前の小位。降順でなければ数詞ではない
    last_large = None              # 直前の大位。同じく降順を要求する
    for char in run:
        if char in DIGITS:
            if current:            # 二三 のような並びは数詞ではない
                return None
            current = DIGITS[char]
            seen = True
        elif char in SMALL_UNITS:
            unit = SMALL_UNITS[char]
            if last_small is not None and unit >= last_small:
                return None        # 十百 / 百百 のような並びは数詞ではない
            if current == 0:
                current = 1        # 十 / 百 / 千 の単独は 1 を補う
            section += current * unit
            current = 0
            last_small = unit
            seen = True
        else:                      # 万 / 億 / 兆
            unit = LARGE_UNITS[char]
            if last_large is not None and unit >= last_large:
                return None        # 万億 のような並びは数詞ではない
            section += current
            if section == 0:
                return None        # 位だけが先に来るのは数詞ではない
            total += section * unit
            section = 0
            current = 0
            last_small = None      # 大位を跨ぐと小位は仕切り直し
            last_large = unit
            seen = True
    if not seen:
        return None
    return total + section + current


def _read_below_10000(value: int) -> str:
    """1〜9999 の読み。"""
    parts: list[str] = []
    for unit in (1000, 100, 10):
        digit, value = divmod(value, unit)
        if not digit:
            continue
        exception = _UNIT_EXCEPTIONS.get((digit, unit))
        if exception:
            parts.append(exception)
        elif digit == 1:
            parts.append(UNIT_KANA[unit])       # 一百 ではなく ひゃく
        else:
            parts.append(DIGIT_KANA[digit] + UNIT_KANA[unit])
    if value:
        parts.append(DIGIT_KANA[value])
    return "".join(parts)


def read_number(value: int) -> str:
    """整数を日本語の読み（仮名）にする。"""
    if value == 0:
        return DIGIT_KANA[0]
    parts: list[str] = []
    for unit in (10**12, 10**8, 10**4):
        section, value = divmod(value, unit)
        if not section:
            continue
        exception = _UNIT_EXCEPTIONS.get((section, unit))
        if exception:
            parts.append(exception)
        else:
            parts.append(_read_below_10000(section) + UNIT_KANA[unit])
    if value:
        parts.append(_read_below_10000(value))
    return "".join(parts)


def _read_digitwise(run: str) -> str:
    """`零三` のような桁を持たない並びを1文字ずつ読む（電話番号など）。"""
    return "".join(DIGIT_KANA[DIGITS[c]] for c in run)


def expand_kanji_numerals(text: str) -> str:
    """文中の漢数字を読みへ展開する。数詞と解釈できない並びは触らない。

    `一緒` `二人` のような語の一部は単独の漢数字なので、**1文字だけの並びは
    展開しない**。誤って `一` を `いち` にすると `一緒` が壊れる。
    """
    def replace(match: re.Match) -> str:
        run = match.group(0)
        if len(run) == 1:
            return run                          # 一緒 / 二人 などを壊さない
        if any(c in "〇零" for c in run):
            if all(c in DIGITS for c in run):
                return _read_digitwise(run)     # 零三 → ゼロさん
            return run
        value = _parse_run(run)
        if value is None:
            return run
        return read_number(value)

    return _RUN.sub(replace, text)


def to_arabic_numerals(text: str) -> str:
    """漢数字をアラビア数字へ正規化する。**CERの比較を数字表記に依存させない。**

    ASRは音声を聞いて `1280円` と書くが、評価の参照テキストは `千二百八十円`。
    正しく読めているほど文字列が一致しなくなるため、素のCERでは
    「数字が読めるようになったこと」を検出できない（J2の効果が測れない）。
    参照側と仮説側の両方にこれを掛けてから比較する。両側に同じ変換を掛けるので、
    `一緒`→`1緒` のように語の一部が変わっても一致判定は壊れない。
    むしろ `四トン`（参照）と `4トン`（ASR出力）が一致するようになる。

        >>> to_arabic_numerals("価格は千二百八十円")
        '価格は1280円'
    """
    def replace(match: re.Match) -> str:
        run = match.group(0)
        if len(run) == 1 and run in DIGITS:
            return str(DIGITS[run])
        if any(c in "〇零" for c in run) and all(c in DIGITS for c in run):
            return "".join(str(DIGITS[c]) for c in run)
        value = _parse_run(run)
        return run if value is None else str(value)

    return _RUN.sub(replace, text)
