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

"""アクセント核をテキストに明示する（M4a）。

**モデルには韻律を受け取る経路が無い。** 抑揚は7軸すべてで動かず
（lr / batch / 学習対象 / データ量19倍 / 計算量4倍 / `flow_copies` /
`condition_dropout`）、**同じ台詞の別テイクを参照に渡しても写さなかった**
（R-041 / D-047）。そこで入力に明示する。

    箸を持つ手と、橋を渡る足。
      → ハ'シオモ'ツテ'ト、ハシ'オワタルアシ。

* 全文を片仮名にする
* **アクセント核の直後に `'` を置く**（句の内部で下がる位置。平板は無印）
* **長音は `ー`**（`ショオヒゼエ` ではなく `ショーヒゼー`。学習分布に近い形）
* 読点は `pau` の位置から、文末の記号は元テキストから復元する
  （`か。` と `か？` で抑揚が変わるので落とさない）

## 情報源は full-context label だけにする

核の位置と仮名の**両方**を full-context から作る（D-042 と同じ理由）。
`run_frontend` の `pron` は `ねぇー` → `ネー` と正規化してモーラ数を変えるので
使わない（実測で2,000文中8.8%がずれる）。

## `letters` ではなく音素から作る

`alignment.Mora.letters` は **MMS_FA 用**で、`_fill_geminates` が促音のモーラに
**次の子音**を入れている（`えっ` の `っ` が `h` になる）。
片仮名へ写すときにこれを使うと `エッ` が `エフ` になる。
**`Mora.phonemes` を直接読む。**

## 変換表の網羅

`tests/training/test_accent.py` が gol コーパスに現れるモーラを
**全部写せること**を検査する。未知の音素は `KeyError` にせず
**ローマ字のまま残して数を報告する**（黙って消すと気づけない）。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

__all__ = [
    "LONG_VOWEL",
    "NUCLEUS_MARK",
    "SHUFFLE_SEED",
    "MarkedText",
    "accent_marked_text",
    "kana_of_phonemes",
]

#: アクセント核の直後に置く記号。**1文字**にする（token数を増やしにくい）。
NUCLEUS_MARK = "'"

#: 長音記号。直前と同じ母音が続いたときに使う。
LONG_VOWEL = "ー"

#: 核の位置をずらす対照（`shuffle=True`）の種。
#:
#: **記号が読みに効いた理由を切り分けるための対照。** 記号の数と句の構造は
#: そのままに、**核の位置だけを偽の位置へ動かす**。読みCERが落ちたままなら
#: 効いていたのは「区切りがあること」で、戻るなら「アクセントの内容」。
SHUFFLE_SEED = 20260918

_VOWELS = ("a", "i", "u", "e", "o")

# 子音ごとの5段。
_ROWS: dict[str, tuple[str, str, str, str, str]] = {
    "": ("ア", "イ", "ウ", "エ", "オ"),
    "k": ("カ", "キ", "ク", "ケ", "コ"),
    "g": ("ガ", "ギ", "グ", "ゲ", "ゴ"),
    "s": ("サ", "スィ", "ス", "セ", "ソ"),
    "z": ("ザ", "ズィ", "ズ", "ゼ", "ゾ"),
    "t": ("タ", "ティ", "トゥ", "テ", "ト"),
    "d": ("ダ", "ディ", "ドゥ", "デ", "ド"),
    "n": ("ナ", "ニ", "ヌ", "ネ", "ノ"),
    "h": ("ハ", "ヒ", "フ", "ヘ", "ホ"),
    "b": ("バ", "ビ", "ブ", "ベ", "ボ"),
    "p": ("パ", "ピ", "プ", "ペ", "ポ"),
    "m": ("マ", "ミ", "ム", "メ", "モ"),
    "y": ("ヤ", "イ", "ユ", "イェ", "ヨ"),
    "r": ("ラ", "リ", "ル", "レ", "ロ"),
    "w": ("ワ", "ウィ", "ウ", "ウェ", "ウォ"),
    "f": ("ファ", "フィ", "フ", "フェ", "フォ"),
    "v": ("ヴァ", "ヴィ", "ヴ", "ヴェ", "ヴォ"),
    "sh": ("シャ", "シ", "シュ", "シェ", "ショ"),
    "j": ("ジャ", "ジ", "ジュ", "ジェ", "ジョ"),
    "ch": ("チャ", "チ", "チュ", "チェ", "チョ"),
    "ts": ("ツァ", "ツィ", "ツ", "ツェ", "ツォ"),
    "ky": ("キャ", "キ", "キュ", "キェ", "キョ"),
    "gy": ("ギャ", "ギ", "ギュ", "ギェ", "ギョ"),
    "ny": ("ニャ", "ニ", "ニュ", "ニェ", "ニョ"),
    "hy": ("ヒャ", "ヒ", "ヒュ", "ヒェ", "ヒョ"),
    "by": ("ビャ", "ビ", "ビュ", "ビェ", "ビョ"),
    "py": ("ピャ", "ピ", "ピュ", "ピェ", "ピョ"),
    "my": ("ミャ", "ミ", "ミュ", "ミェ", "ミョ"),
    "ry": ("リャ", "リ", "リュ", "リェ", "リョ"),
    "dy": ("ヂャ", "ヂ", "ヂュ", "ヂェ", "ヂョ"),
    "ty": ("テャ", "ティ", "テュ", "テェ", "テョ"),
    # 拗音の外来音。**コーパスに 0.15% 出た**（`kwu` 570回 / `gwu` 357回）
    "kw": ("クヮ", "クィ", "ク", "クェ", "クォ"),
    "gw": ("グヮ", "グィ", "グ", "グェ", "グォ"),
}

#: 母音を持たないモーラ。
_STANDALONE: dict[str, str] = {
    "N": "ン",    # 撥音
    "cl": "ッ",   # 促音
    "q": "ッ",    # 促音（別の符号化）
}


def _build_table() -> dict[tuple[str, ...], str]:
    table: dict[tuple[str, ...], str] = {}
    for consonant, row in _ROWS.items():
        for vowel, kana in zip(_VOWELS, row):
            if consonant:
                table[(consonant, vowel)] = kana
            else:
                table[(vowel,)] = kana
    for phoneme, kana in _STANDALONE.items():
        table[(phoneme,)] = kana
    # 子音だけのモーラ（言い落ち）は**ウ段**で代表させる
    for consonant, row in _ROWS.items():
        if consonant:
            table.setdefault((consonant,), row[2])
    return table


_TABLE = _build_table()

#: 無声化した母音。full-context では大文字で来る。
_DEVOICED = {"I": "i", "U": "u"}

_SIL = frozenset({"sil", "xx"})
_PAUSE = "pau"
_PHONEME = re.compile(r"\-(.*?)\+")
_FIELD_A = re.compile(r"/A:([-\d]+)\+(\d+)\+(\d+)")
_FIELD_F = re.compile(r"/F:(\d+)_(\d+)#\d+_\d+@(\d+)_(\d+)")
_FIELD_I = re.compile(r"/I:\d+-\d+@(\d+)\+")

#: 文末に残す記号。**`か。` と `か？` で抑揚が違う**ので落とさない。
_FINAL_MARKS = "。．.！!？?…"


def _normalize(phonemes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_DEVOICED.get(p, p) for p in phonemes)


def kana_of_phonemes(phonemes: tuple[str, ...]) -> str | None:
    """音素のモーラを片仮名へ。表に無ければ ``None``。"""
    return _TABLE.get(_normalize(phonemes))


def _vowel_of(phonemes: tuple[str, ...]) -> str | None:
    """そのモーラの母音。持たなければ ``None``。"""
    normalized = _normalize(phonemes)
    if not normalized:
        return None
    last = normalized[-1]
    return last if last in _VOWELS else None


@dataclass(frozen=True)
class MarkedText:
    """`accent_marked_text` の結果。"""

    text: str
    """アクセント核つきの片仮名テキスト。"""

    moras: int
    """写したモーラ数。"""

    unknown: tuple[str, ...]
    """表に無かった音素（ローマ字のまま埋めてある）。"""

    marks: int
    """置いた核記号の数。"""


def _parse(text: str):
    """full-context label を (アクセント句のモーラ列, 直後にpauがあるか) へ。"""
    import pyopenjtalk

    phrases: list[tuple[list[tuple[str, ...]], int]] = []
    pauses: set[int] = set()
    key = None
    nucleus = 0
    moras: list[tuple[str, ...]] = []
    pending: list[str] = []
    position = 0

    def flush_mora():
        nonlocal pending
        if pending:
            moras.append(tuple(pending))
            pending = []

    def flush_phrase():
        nonlocal moras
        flush_mora()
        if moras:
            phrases.append((moras, nucleus))
        moras = []

    for label in pyopenjtalk.extract_fullcontext(text):
        found = _PHONEME.search(label)
        if found is None:
            continue
        name = found.group(1)
        if name in _SIL:
            continue
        if name == _PAUSE:
            flush_phrase()
            key = None
            pauses.add(len(phrases))      # この句の後に間がある
            continue
        field_a, field_f = _FIELD_A.search(label), _FIELD_F.search(label)
        if not (field_a and field_f):
            continue
        field_i = _FIELD_I.search(label)
        current = (int(field_i.group(1)) if field_i else 0, int(field_f.group(3)))
        if current != key:
            flush_phrase()
            key = current
            nucleus = int(field_f.group(2))
        mora_index = int(field_a.group(2))
        if mora_index != position:
            flush_mora()
            position = mora_index
        pending.append(name)
    flush_phrase()
    return phrases, pauses


def _word_start_moras(text: str) -> set[int]:
    """**語の先頭にあたるモーラの位置**（0始まり・文全体の通し番号）。

    長音規則は「直前と同じ母音の裸母音」を `ー` にするが、この規則は
    **語境界をまたいでも発火する**。実測（20,000文）で

        コトモ**オ**シエテ  →  コトモ**ー**シエテ   `教えて` の頭が消える
        トニカク**ウ**ゴカズ →  トニカク**ー**ゴカズ  `動かず` の頭が消える

    のように約4,400箇所が潰れていた。語の先頭では止められるようにする。

    NJD の `mora_size` の累和で境界を出す。ラベル側のモーラ総数と合わない
    ときは空集合を返す（**黙って別の位置を止めるより、止めない方が安全**）。
    """
    import pyopenjtalk

    starts: set[int] = set()
    total = 0
    try:
        features = pyopenjtalk.run_frontend(text)
    except Exception:
        return set()
    for item in features:
        size = int(item.get("mora_size") or 0)
        if size <= 0:
            continue
        starts.add(total)
        total += size
    return starts


def _shuffled_nucleus(mora_count: int, true_nucleus: int, kana: str) -> int:
    """核の位置を**偽の位置**へ動かす（対照用）。

    * 記号の数は変えない（`true_nucleus == 0` なら 0 のまま）
    * 句の長さも変えない（1〜`mora_count - 1` の範囲に収める）
    * 同じ句には常に同じ偽位置を割り当てる（再現できるようにする）
    """
    if true_nucleus == 0 or mora_count < 3:
        return true_nucleus
    digest = hashlib.sha256(f"{SHUFFLE_SEED}:{kana}".encode("utf-8")).hexdigest()
    candidates = [i for i in range(1, mora_count) if i != true_nucleus]
    if not candidates:
        return true_nucleus
    return candidates[int(digest[:8], 16) % len(candidates)]


def accent_marked_text(text: str, *, mark: str = NUCLEUS_MARK,
                       pause: str = "、", shuffle: bool = False,
                       merge_across_words: bool = True) -> MarkedText:
    """アクセント核つきの片仮名テキストを作る。

    Args:
        text: 元のテキスト。
        mark: 核の直後に置く記号。
        pause: 間（`pau`）の位置に置く記号。
        shuffle: ``True`` なら**核の位置を偽の位置へ動かす**（対照。
            記号の数と句の構造は同じまま）。
        merge_across_words: ``True``（既定）なら語境界をまたいでも同じ母音を
            長音へ潰す。**既定を変えてはいけない** — 現行最良の checkpoint は
            この挙動で学習してある。``False`` は `_word_start_moras` を使って
            語の先頭で止める（`accent_clean`）。

    Returns:
        :class:`MarkedText`。**表に無い音素はローマ字のまま残す**。
    """
    phrases, pauses = _parse(text)
    parts: list[str] = []
    unknown: list[str] = []
    moras = marks = 0
    previous_vowel: str | None = None
    word_starts: set[int] = set() if merge_across_words else _word_start_moras(text)

    for index, (mora_list, nucleus) in enumerate(phrases):
        if index in pauses and parts:
            parts.append(pause)
            previous_vowel = None
        # 平板と尾高は句の中では区別できない（`AccentPhrase.internal_nucleus` と同じ規約）
        internal = nucleus if 0 < nucleus < len(mora_list) else 0
        if shuffle:
            kana_key = "".join("".join(p) for p in mora_list)
            internal = _shuffled_nucleus(len(mora_list), internal, kana_key)
        for position, phonemes in enumerate(mora_list, start=1):
            vowel = _vowel_of(phonemes)
            if (len(_normalize(phonemes)) == 1 and vowel
                    and vowel == previous_vowel and moras not in word_starts):
                kana = LONG_VOWEL
            else:
                kana = kana_of_phonemes(phonemes)
                if kana is None:
                    unknown.append("".join(phonemes))
                    kana = "".join(phonemes)
            parts.append(kana)
            previous_vowel = vowel
            moras += 1
            if position == internal:
                parts.append(mark)
                marks += 1
    # 文末の記号は残す（`か。` と `か？` で抑揚が違う）
    stripped = text.rstrip()
    if stripped and stripped[-1] in _FINAL_MARKS:
        parts.append(stripped[-1])
    return MarkedText("".join(parts), moras, tuple(unknown), marks)
