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

"""語の読み付与のテスト（J3 / D-034）。

判定（`needs_reading`）は vocab を渡すだけで試せるので、`pyopenjtalk` を
必要としない。文への適用（`apply`）は `pyopenjtalk` を使うので、
入っていなければ skip する（`[ja]` extra）。

**壊してはいけないもの**が2種類ある。

* 記号（`――` `…` は `read` が `、` になるため、触ると文が壊れる）
* 語彙にある語（`大丈夫` `気持ち` を仮名にすると学習分布から外れる）
"""

from __future__ import annotations

import pytest

from cutetts.training.yomi import (
    DEFAULT_FRONTEND,
    FRONTEND_MODES,
    MIN_SURFACE_LENGTH,
    SKIP_POS,
    USE_HIRAGANA,
    ReadingAssigner,
    frontend_from_model_dir,
    reading_form,
    to_hiragana,
)

# checkpoint の tokenizer を模した語彙。
# **仮名は全部持たせる**（実物も持っている。ここを欠くと `を` `です` まで
# 置換対象になり、判定の意図と違うものを測ってしまう）。
# `華` `湊` `聖` `会` `釈` `滅` を**持たない**ことがこのテストの前提。
_KANA = "".join(chr(c) for c in range(0x3041, 0x3097))          # ひらがな
_KATAKANA = "".join(chr(c) for c in range(0x30A1, 0x30F7))      # カタカナ
VOCAB = frozenset(
    _KANA + _KATAKANA + "ー、。！？…―"
    + "中国大丈夫気持届砂肝学生室案内風味浅深人目上一生開今日語"
)


@pytest.fixture
def assigner() -> ReadingAssigner:
    return ReadingAssigner(vocab=VOCAB)


def test_single_character_words_are_included():
    """既定は1文字も対象。`湊` `聖` は1文字で、実際に誤読した人名。

    当初2にしたが、実測で多音字（`一` `生`）は語彙にあるので
    判定に掛からないと分かった。
    """
    assert MIN_SURFACE_LENGTH == 1


def test_symbols_are_never_replaced():
    """記号の `read` は `、`。置換すると `――` や `…` が読点に化ける。"""
    assert "記号" in SKIP_POS


def test_word_with_missing_character_needs_reading(assigner):
    """`華` が語彙に無いので `中華` は置換対象。"""
    assert assigner.needs_reading("中華", "名詞") is True


def test_word_fully_in_vocab_is_left_alone(assigner):
    """`大丈夫` は全文字が語彙にあるので触らない。"""
    assert assigner.needs_reading("大丈夫", "名詞") is False


@pytest.mark.parametrize("surface", ["湊", "聖"])
def test_single_character_name_needs_reading(surface):
    """1文字の人名も捕まえる。どちらも聴取で実際に誤読した。"""
    assigner = ReadingAssigner(vocab=VOCAB)
    assert assigner.needs_reading(surface, "名詞") is True


@pytest.mark.parametrize("pos", sorted(SKIP_POS))
def test_skipped_pos_is_never_replaced(assigner, pos):
    """語彙に無い文字を含んでいても、記号・フィラーは置換しない。"""
    assert assigner.needs_reading("華", pos) is False


def test_min_length_can_be_raised(assigner):
    """1文字を除外したいときは設定で変えられる。"""
    strict = ReadingAssigner(vocab=VOCAB, min_length=2)
    assert strict.needs_reading("湊", "名詞") is False
    assert strict.needs_reading("中華", "名詞") is True


def test_default_is_katakana():
    """既定は `pyopenjtalk` の素の出力（片仮名）。

    「学習コーパスは平仮名が主なので平仮名が有利」という仮説を300文で
    対照実験したが、**+0.51pt / 95%CI [-0.82, +1.83] で有意差なし**だった（D-037）。
    測定値が良い片仮名を既定にしてある。
    """
    assert USE_HIRAGANA is False


def test_to_hiragana_converts_only_katakana():
    """長音符・漢字・平仮名・記号は素通しする。"""
    assert to_hiragana("チュウカ") == "ちゅうか"
    assert to_hiragana("コーヒー") == "こーひー"
    assert to_hiragana("中華、はい") == "中華、はい"


def test_to_hiragana_keeps_small_kana():
    """拗音・促音（`ッ` `ャ`）も対応する平仮名になる。"""
    assert to_hiragana("キャッチ") == "きゃっち"


def test_missing_tokenizer_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReadingAssigner.from_model_dir(tmp_path)


# ---- ここから下は pyopenjtalk（[ja] extra）が要る ----

pyopenjtalk = pytest.importorskip(
    "pyopenjtalk", reason="読み付与には [ja] extra（pyopenjtalk-plus）が要る")


def test_apply_replaces_only_the_target_word(assigner):
    result = assigner.apply("砂肝をオイスター風味で中華ですね")

    assert "チュウカ" in result
    # 語彙にある語はそのまま
    assert "風味" in result
    assert [surface for surface, _ in assigner.replaced] == ["中華"]


def test_apply_keeps_symbols(assigner):
    """`――` `…` `？` `！` が読点や半角に化けないこと。"""
    text = "――中華ですね？……はい！"

    result = assigner.apply(text)

    for symbol in ("――", "……", "？", "！"):
        assert symbol in result


def test_apply_leaves_ordinary_sentences_unchanged(assigner):
    """語彙に収まる会話文は1文字も変えない。"""
    text = "でも大丈夫。気持ちは届いてるからさ"

    assert assigner.apply(text) == text
    assert assigner.replaced == []


def test_apply_can_write_hiragana():
    """切り替えは残してある（既定ではない）。"""
    assigner = ReadingAssigner(vocab=VOCAB, hiragana=True)

    assert "ちゅうか" in assigner.apply("中華ですね")


def test_apply_records_what_it_replaced(assigner):
    assigner.apply("中華ですね")

    assert assigner.replaced == [("中華", "チュウカ")]


def test_replaced_is_reset_between_calls(assigner):
    assigner.apply("中華ですね")
    assigner.apply("でも大丈夫")

    assert assigner.replaced == []


def test_apply_is_deterministic(assigner):
    text = "それじゃ湊さんを案内したら"

    assert assigner.apply(text) == assigner.apply(text)


def test_empty_text_is_returned_as_is(assigner):
    assert assigner.apply("") == ""


# ---- 読みレベルの比較形（R-029） ----


def test_reading_form_erases_notation_difference():
    """仮名で書いても漢字で書いても同じ形になる。

    **これが無いと J3 の効果を測れない。** 仮名を入力すると ASR も仮名で
    書き戻すので、漢字の参照文に対する素のCERは正しい発音を誤りと数える。
    """
    assert reading_form("綺麗さっぱり") == reading_form("キレイさっぱり")
    assert reading_form("痺れるような") == reading_form("しびれるような")


def test_reading_form_keeps_genuine_reading_errors():
    """**読み自体が違えば違う形のまま**。表記の違いだけを消す。

    `楓寺` を `カエデジ` と読むのは実際の誤読で、これを一致にしてしまうと
    測定器として使えない。
    """
    assert reading_form("楓寺の御先様") != reading_form("カエデジの御先様")


def test_reading_form_drops_punctuation():
    assert reading_form("こんにちは、東京") == reading_form("こんにちは東京")


def test_reading_form_is_hiragana():
    assert reading_form("中華") == "ちゅうか"


def test_reading_form_of_empty_text_is_empty():
    assert reading_form("") == ""


# --- checkpoint が学習時の frontend を名乗る（2026-09-26） --------------------
#
# 公開した `m4h-prosody` は `accent` で学習したのに config.json が何も持たず、
# `synthesize_japanese.py` が素の漢字を渡していた。model card に載せた
# 読みCER 7.12% が手順どおりでは出ない状態だったので、記録する側と読む側の
# 両方をテストで留める。


def _write_config(tmp_path, payload):
    import json

    (tmp_path / "config.json").write_text(json.dumps(payload, ensure_ascii=False),
                                          encoding="utf-8")
    return tmp_path


def test_frontend_from_model_dirは記録された表記を返す(tmp_path):
    _write_config(tmp_path, {"model_type": "cutetts", "japanese_frontend": "accent"})
    assert frontend_from_model_dir(tmp_path) == "accent"


def test_frontend_from_model_dirは記録が無ければNoneを返す(tmp_path):
    """**古い checkpoint は名乗れない。** 呼ぶ側が既定を当てる。"""
    _write_config(tmp_path, {"model_type": "cutetts"})
    assert frontend_from_model_dir(tmp_path) is None


def test_frontend_from_model_dirはconfigが無ければNoneを返す(tmp_path):
    assert frontend_from_model_dir(tmp_path) is None


def test_frontend_from_model_dirは未知の表記を弾く(tmp_path):
    """**黙って既定に落とさない。** 綴り違いを無視すると食い違いが再発する。"""
    _write_config(tmp_path, {"japanese_frontend": "accent_v2"})
    with pytest.raises(ValueError, match="japanese_frontend"):
        frontend_from_model_dir(tmp_path)


def test_frontend_from_model_dirは壊れたconfigでNoneを返す(tmp_path):
    (tmp_path / "config.json").write_text("{ではない", encoding="utf-8")
    assert frontend_from_model_dir(tmp_path) is None


def test_既定の表記は公開checkpointと同じ():
    """**公開しているのは `accent` で学習した checkpoint だけ。**

    記録の無い checkpoint に当てる既定がこれと違うと、公開したモデルを
    そのまま使った人が学習と別の表記を渡すことになる。
    """
    assert DEFAULT_FRONTEND == "accent"
    assert DEFAULT_FRONTEND in FRONTEND_MODES
