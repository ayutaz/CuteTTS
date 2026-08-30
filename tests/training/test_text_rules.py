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

"""``cutetts.training.text_rules`` のテスト。

実データ（``data/raw/``）には依存しない。ただし入力文字列は
``data/raw/gol/metadata.tsv`` の全件走査で実在を確認した実例をそのまま使っている。
"""

from __future__ import annotations

import pytest

from cutetts.training.text_rules import (
    GENERIC_SPEAKER_LABELS,
    MARKUP_PATTERN,
    NAME_PLACEHOLDER_PATTERN,
    contains_markup,
    contains_name_placeholder,
    extract_ruby,
    find_markup,
    generic_speaker_ids,
    gol_speaker_hash,
    has_name_placeholder,
    is_punctuation_only,
    strip_markup,
)

# --------------------------------------------------------------------------------------
# gol_speaker_hash / generic_speaker_ids
# --------------------------------------------------------------------------------------

#: 実データ側で確認済みの (表示名, speaker ID) 対応。
KNOWN_SPEAKER_HASHES = {
    "女の子": "E9291C8A0748FB406D1AE76437CCC344",
    "？？？": "3FE3A9D5FF5031B4A1A9F594CB565C8F",
    "店員": "2ED99C463A5B2128EB3C93D053712877",
    "全員": "F7AF6675EC30C90BEA20E2A36C3E0202",
}


@pytest.mark.parametrize(("label", "expected"), sorted(KNOWN_SPEAKER_HASHES.items()))
def test_gol_speaker_hash_matches_measured_ids(label: str, expected: str) -> None:
    assert gol_speaker_hash(label) == expected


def test_gol_speaker_hash_shape() -> None:
    digest = gol_speaker_hash("適当な名前")
    assert len(digest) == 32
    assert digest == digest.upper()
    assert all(char in "0123456789ABCDEF" for char in digest)


def test_gol_speaker_hash_is_utf8_based() -> None:
    """UTF-8以外のencodingでhashしていないことを、非ASCIIな別名で二重に確認する。"""
    import hashlib

    assert gol_speaker_hash("ナレーション") == (
        hashlib.sha256("ナレーション".encode("utf-8")).hexdigest()[:32].upper()
    )


def test_generic_speaker_ids_contains_known_hashes() -> None:
    ids = generic_speaker_ids()
    for expected in KNOWN_SPEAKER_HASHES.values():
        assert expected in ids


def test_generic_speaker_labels_are_unique_and_hash_without_collision() -> None:
    assert len(set(GENERIC_SPEAKER_LABELS)) == len(GENERIC_SPEAKER_LABELS)
    assert len(generic_speaker_ids()) == len(GENERIC_SPEAKER_LABELS)


@pytest.mark.parametrize(
    "label",
    [
        "？？？",
        "全員",
        "女の子",
        "男の子",
        "ナレーション",
        "モブ",
        "謎の声",
        "おばあさん",
        "子供たち",
        "システム",
    ],
)
def test_required_generic_labels_present(label: str) -> None:
    assert label in GENERIC_SPEAKER_LABELS
    assert gol_speaker_hash(label) in generic_speaker_ids()


def test_generic_speaker_ids_is_frozenset() -> None:
    assert isinstance(generic_speaker_ids(), frozenset)


# --------------------------------------------------------------------------------------
# is_punctuation_only
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "…………",
        "……っ！？",
        "",
        "   ",
        "　",
        "・・・",
        "。",
        "！？！？",
        "―――",
        "～～♪",
        "っ……",
        "──ッ！！",
    ],
)
def test_is_punctuation_only_true(text: str) -> None:
    assert is_punctuation_only(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "こんにちは",
        "……あ",
        "ぁぁぁ……",  # 小書き母音は単独で発音できるので言語内容ありとみなす
        "A",
        "１２３",
        "……間もなく金剛町、金剛町です。",
    ],
)
def test_is_punctuation_only_false(text: str) -> None:
    assert is_punctuation_only(text) is False


def test_is_punctuation_only_strict_mode_excludes_sokuon() -> None:
    """狭い判定（data-inventory.mdの152,605件に対応する側）では促音を言語内容扱いする。"""
    assert is_punctuation_only("……っ！？", allow_non_lexical_kana=False) is False
    assert is_punctuation_only("…………", allow_non_lexical_kana=False) is True
    assert is_punctuation_only("", allow_non_lexical_kana=False) is True


# --------------------------------------------------------------------------------------
# find_markup / contains_markup
# --------------------------------------------------------------------------------------

# すべて metadata.tsv に実在する行（一部は先頭のみ）。
REAL_MARKUP_SAMPLES = {
    "はぁい、ぴちぴちらっぴ〜！キミの守り神様、<rあやせ>綾瀬</r>だにょ〜ん！": ["<rあやせ>", "</r>"],
    "うふっ、雫にお任せっ<ハ>": ["<ハ>"],
    "……お出口は左側です。[n]We will soon make a breif stop at Kongo-cho...": ["[n]"],
    "だが我は違う。<d>クーリングオフ</d>にも対応しておるぞ": ["<d>", "</d>"],
    "いいか、落ち着いて%bdの指示に従え！": ["%bd"],
    "<s36>なぜなら皆さん、と〜っても弱いからデス！</s>": ["<s36>", "</s>"],
    "空の@ruby門で“そらかど”、蒼は蒼穹の“そう”よ": ["@ruby"],
    "おしあわせに%0明日の朝迎えに来るからね%0%0": ["%0", "%0", "%0"],
    "私の[rb,膣内,なか]に出してください": ["[rb,膣内,なか]"],
    "はいはい、１００[Ｇ/ゴールド]だよっ！": ["[Ｇ/ゴールド]"],
    "[name 魅夕 miy_0369]ももちゃんさえよければいいですよー？": ["[name 魅夕 miy_0369]"],
    "萌黄ちゃん……っ！！ひしっ！！[quake]": ["[quake]"],
}


@pytest.mark.parametrize(("text", "expected"), sorted(REAL_MARKUP_SAMPLES.items()))
def test_find_markup_on_real_samples(text: str, expected: list[str]) -> None:
    assert find_markup(text) == expected
    assert contains_markup(text) is True


def test_find_markup_nested_ruby_returns_tokens_in_order() -> None:
    text = "みこっちゃん、その編入生って<rぶげいか><d0018>武芸科</d></r>？"
    assert find_markup(text) == ["<rぶげいか>", "<d0018>", "</d>", "</r>"]


def test_find_markup_dollar_control_codes() -> None:
    text = "……現時点では$c:255/64/80,48/48/48;$b;存在しない$bd;$c;ことがわかりました。"
    assert find_markup(text) == ["$c:255/64/80,48/48/48;", "$b;", "$bd;", "$c;"]


@pytest.mark.parametrize(
    "text",
    [
        "天然由来成分90%以上配合で、お肌にとっても優しいのっ♪",
        "成功する確率は70%くらいなんだ",
        "いいえ、100%日本製ですよ",
        "こんにちは。今日はいい天気ですね",
        "……………………",
    ],
)
def test_find_markup_no_false_positive(text: str) -> None:
    """本物のパーセント記号や普通の日本語をmarkupと誤認しない。"""
    assert find_markup(text) == []
    assert contains_markup(text) is False


def test_markup_pattern_is_compiled_regex() -> None:
    import re

    assert isinstance(MARKUP_PATTERN, re.Pattern)


# --------------------------------------------------------------------------------------
# strip_markup
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "はぁい、ぴちぴちらっぴ〜！キミの守り神様、<rあやせ>綾瀬</r>だにょ〜ん！",
            "はぁい、ぴちぴちらっぴ〜！キミの守り神様、綾瀬だにょ〜ん！",
        ),
        (
            "みこっちゃん、その編入生って<rぶげいか><d0018>武芸科</d></r>？",
            "みこっちゃん、その編入生って武芸科？",
        ),
        (
            "だが我は違う。<d>クーリングオフ</d>にも対応しておるぞ",
            "だが我は違う。クーリングオフにも対応しておるぞ",
        ),
        ("<s36>なぜなら皆さん、と〜っても弱いからデス！</s>", "なぜなら皆さん、と〜っても弱いからデス！"),
        ("うふっ、雫にお任せっ<ハ>", "うふっ、雫にお任せっ"),
        ("おしあわせに%0明日の朝迎えに来るからね%0%0", "おしあわせに明日の朝迎えに来るからね"),
        ("いいか、落ち着いて%bdの指示に従え！", "いいか、落ち着いての指示に従え！"),
        ("私の[rb,膣内,なか]に出してください", "私の膣内に出してください"),
        ("はいはい、１００[Ｇ/ゴールド]だよっ！", "はいはい、１００Ｇだよっ！"),
        ("[name 魅夕 miy_0369]ももちゃんさえよければいいですよー？", "ももちゃんさえよければいいですよー？"),
        (
            "……現時点では$c:255/64/80,48/48/48;$b;存在しない$bd;$c;ことがわかりました。",
            "……現時点では存在しないことがわかりました。",
        ),
        ("空の@ruby門で“そらかど”、蒼は蒼穹の“そう”よ", "空の門で“そらかど”、蒼は蒼穹の“そう”よ"),
        # 傍点markup。片側が傍点だけなら本文側を残す。
        ("今日の私は、ちゃんとできてた[・|？]", "今日の私は、ちゃんとできてた？"),
        ("特殊能力は低め[まる|○]です", "特殊能力は低めまるです"),
    ],
)
def test_strip_markup_keeps_body(text: str, expected: str) -> None:
    assert strip_markup(text) == expected


def test_strip_markup_is_idempotent() -> None:
    text = "みこっちゃん、その編入生って<rぶげいか><d0018>武芸科</d></r>？"
    once = strip_markup(text)
    assert strip_markup(once) == once


def test_strip_markup_without_ruby_surface_drops_the_whole_ruby() -> None:
    text = "キミの守り神様、<rあやせ>綾瀬</r>だにょ〜ん！"
    assert strip_markup(text, keep_ruby_surface=False) == "キミの守り神様、だにょ〜ん！"
    assert strip_markup(text, keep_ruby_surface=True) == "キミの守り神様、綾瀬だにょ〜ん！"


def test_strip_markup_leaves_plain_text_untouched() -> None:
    text = "こんにちは。今日はいい天気ですね"
    assert strip_markup(text) == text


def test_strip_markup_collapses_whitespace_left_behind() -> None:
    assert strip_markup("  [n]  こんにちは  [n]  ") == "こんにちは"


def test_strip_markup_result_has_no_markup_left() -> None:
    for text in REAL_MARKUP_SAMPLES:
        assert contains_markup(strip_markup(text)) is False


# --------------------------------------------------------------------------------------
# extract_ruby
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("キミの守り神様、<rあやせ>綾瀬</r>だにょ〜ん！", [("綾瀬", "あやせ")]),
        (
            "その編入生って<rぶげいか><d0018>武芸科</d></r>？それとも<rぎこうか><d0019>技巧科</d></r>？",
            [("武芸科", "ぶげいか"), ("技巧科", "ぎこうか")],
        ),
        ("昔はの、我を祀るための<rやしろ>社</r>があったのじゃ", [("社", "やしろ")]),
        ("私の[rb,膣内,なか]に出してください", [("膣内", "なか")]),
        ("はいはい、１００[Ｇ/ゴールド]だよっ！", [("Ｇ", "ゴールド")]),
        ("……忍法[天神雷光/アラハバキ]！！", [("天神雷光", "アラハバキ")]),
        ("こんにちは", []),
    ],
)
def test_extract_ruby(text: str, expected: list[tuple[str, str]]) -> None:
    assert extract_ruby(text) == expected


def test_extract_ruby_returns_in_document_order() -> None:
    text = "[rb,美湖,みこ]と<rやしろ>社</r>と[Ｇ/ゴールド]"
    assert extract_ruby(text) == [("美湖", "みこ"), ("社", "やしろ"), ("Ｇ", "ゴールド")]


def test_extract_ruby_skips_ambiguous_notations() -> None:
    """``@ruby`` と ``|`` / ``:`` 区切りは表層と読みの境界が決まらないので拾わない。"""
    assert extract_ruby("空の@ruby門で“そらかど”、蒼は蒼穹の“そう”よ") == []
    assert extract_ruby("今日の私は、ちゃんとできてた[・|？]") == []
    assert extract_ruby("その“拳銃”は、撃ち抜く対象の“[現在|今、ここ]”を破壊する") == []


def test_extract_ruby_surface_matches_strip_markup() -> None:
    """抽出した表層は strip_markup の結果に必ず残っている（読みと本文の整合）。"""
    text = "その編入生って<rぶげいか><d0018>武芸科</d></r>？それとも<rぎこうか><d0019>技巧科</d></r>？"
    body = strip_markup(text)
    for surface, _reading in extract_ruby(text):
        assert surface in body


# --------------------------------------------------------------------------------------
# has_name_placeholder
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "%bd！？",
        "%bd",
        "……%bd、ファラ",
        "いいか、落ち着いて%bdの指示に従え！",
        "ちゅぷ、んくっ……%bd",
    ],
)
def test_has_name_placeholder_true(text: str) -> None:
    assert has_name_placeholder(text) is True
    assert contains_name_placeholder(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "こんにちは",
        # $bd; は太字終了タグであって名前変数ではない
        "……現時点では$c:255/64/80,48/48/48;$b;存在しない$bd;$c;ことがわかりました。",
        "おしあわせに%0明日の朝迎えに来るからね%0%0",
        "%Cあたしはもう、眠り姫なんかじゃない……",
        "天然由来成分90%以上配合で、お肌にとっても優しいのっ♪",
        "ふぇっ！？%D1149な、なんでもないよ",
    ],
)
def test_has_name_placeholder_false(text: str) -> None:
    assert has_name_placeholder(text) is False
    assert contains_name_placeholder(text) is False


def test_name_placeholder_pattern_is_compiled_regex() -> None:
    import re

    assert isinstance(NAME_PLACEHOLDER_PATTERN, re.Pattern)


# --------------------------------------------------------------------------------------
# manifest.validate との結合
# --------------------------------------------------------------------------------------


def test_manifest_validate_picks_up_text_rules() -> None:
    """``manifest.validate`` が遅延importで呼ぶ関数名が揃っていることの回帰テスト。"""
    from cutetts.training.manifest import Utterance, validate

    records = [
        Utterance(
            utterance_id="gol:g:ok",
            dataset_id="gol",
            audio_ref="a.wav",
            text_raw="こんにちは。今日はいい天気ですね",
            speaker_id="A" * 32,
            duration=3.0,
            sample_rate=48000,
        ),
        Utterance(
            utterance_id="gol:g:punct",
            dataset_id="gol",
            audio_ref="b.wav",
            text_raw="……っ！？",
            speaker_id="A" * 32,
            duration=3.0,
            sample_rate=48000,
        ),
        Utterance(
            utterance_id="gol:g:markup",
            dataset_id="gol",
            audio_ref="c.wav",
            text_raw="キミの守り神様、<rあやせ>綾瀬</r>だにょ〜ん！",
            speaker_id="A" * 32,
            duration=3.0,
            sample_rate=48000,
        ),
        Utterance(
            utterance_id="gol:g:name",
            dataset_id="gol",
            audio_ref="d.wav",
            text_raw="いいか、落ち着いて%bdの指示に従え！",
            speaker_id="A" * 32,
            duration=3.0,
            sample_rate=48000,
        ),
        Utterance(
            utterance_id="gol:g:generic",
            dataset_id="gol",
            audio_ref="e.wav",
            text_raw="こんにちは",
            speaker_id=gol_speaker_hash("女の子"),
            duration=3.0,
            sample_rate=48000,
        ),
    ]
    issues = validate(records, generic_speaker_ids=generic_speaker_ids())
    by_id = {(issue.utterance_id, issue.code) for issue in issues}

    assert not [code for uid, code in by_id if uid == "gol:g:ok"]
    assert ("gol:g:punct", "punctuation_only") in by_id
    assert ("gol:g:markup", "markup") in by_id
    assert ("gol:g:name", "name_placeholder") in by_id
    assert ("gol:g:name", "markup") in by_id  # %bd はmarkupでもある
    assert ("gol:g:generic", "generic_speaker") in by_id
