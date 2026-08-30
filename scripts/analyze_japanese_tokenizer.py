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

"""P1b: 公式SentencePiece Tokenizerの日本語coverage測定。

``docs/japanese-training/08-execution-plan.md`` の P1b と
``docs/japanese-training/02-continual-training-strategy.md`` 第4節 (T0) を実装する。

測るもの:

* ``<unk>`` の文単位 / 文字単位の発生率
* **byte fallback** の文単位 / 文字単位の発生率
  （このtokenizerは256個の ``<0xXX>`` pieceを持つ。語彙に無い文字は ``<unk>`` にならず
  UTF-8 byteへ分解される。したがって「``<unk>`` が0である」ことはcoverageの証明にならない）
* 文字あたりtoken数（平均・分布）と token長の P50 / P95 / P99 / max
* 文字種別（漢字・ひらがな・カタカナ・半角カタカナ・ASCII英字・数字・記号・絵文字…）coverage
* Unicode正規化（NFKC）前後の差
* 既存special token（``<|im_start|>`` / ``<|im_end|>`` / ``<|endofprompt|>``）とcorpusの衝突
* promptテンプレート込みの実効text予算
  （``processor.py`` の実テンプレートと ``SegmentManager`` をそのまま使って実測する）

CPUのみで動く。TTS本体のweightはloadしない。

実行:

    .venv/Scripts/python.exe scripts/analyze_japanese_tokenizer.py \
        --config configs/japanese/tokenizer-coverage.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

import numpy as np
import yaml

#: repository root（scripts/ の1階層上）。configの相対pathはここ基準で解決する。
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT / "src") not in sys.path:  # editable installでなくても動くように
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cutetts.training.artifacts import (  # noqa: E402  (sys.path調整の後でimportする)
    file_checksum,
    new_run_dir,
    write_metrics,
    write_run_metadata,
)

PHASE_DEFAULT = "p1b"

# =============================================================================
# 文字種分類
# =============================================================================

#: 文字種class名。``other`` が既定。
CHAR_CLASSES: tuple[str, ...] = (
    "other",
    "whitespace",
    "symbol",
    "digit",
    "ascii_letter",
    "fullwidth_latin",
    "hiragana",
    "katakana",
    "halfwidth_katakana",
    "kanji",
    "emoji",
)

_CLASS_INDEX: dict[str, int] = {name: index for index, name in enumerate(CHAR_CLASSES)}

_UNICODE_LIMIT = 0x110000

#: (class名, 開始codepoint, 終了codepoint) を **適用順** に並べたもの。後勝ち。
_CHAR_CLASS_RANGES: tuple[tuple[str, int, int], ...] = (
    # --- 記号・約物 -----------------------------------------------------------
    ("symbol", 0x0021, 0x002F),  # ! " # $ % & ' ( ) * + , - . /
    ("symbol", 0x003A, 0x0040),  # : ; < = > ? @
    ("symbol", 0x005B, 0x0060),  # [ \ ] ^ _ `
    ("symbol", 0x007B, 0x007E),  # { | } ~
    ("symbol", 0x00A1, 0x00BF),  # ラテン1の記号
    ("symbol", 0x00D7, 0x00D7),
    ("symbol", 0x00F7, 0x00F7),
    ("symbol", 0x2000, 0x206F),  # general punctuation
    ("symbol", 0x20A0, 0x20CF),  # 通貨記号
    ("symbol", 0x2100, 0x214F),  # letterlike
    ("symbol", 0x2150, 0x218F),  # 数字形（ローマ数字など）
    ("symbol", 0x2190, 0x21FF),  # 矢印
    ("symbol", 0x2200, 0x22FF),  # 数学記号
    ("symbol", 0x2300, 0x23FF),  # misc technical
    ("symbol", 0x2460, 0x24FF),  # 囲み英数字
    ("symbol", 0x2500, 0x257F),  # 罫線
    ("symbol", 0x25A0, 0x25FF),  # 幾何学模様
    ("symbol", 0x3000, 0x303F),  # CJK約物（、。「」…）
    ("symbol", 0x3200, 0x33FF),  # 囲みCJK・CJK互換（℃ ㍿ など）
    ("symbol", 0xFE10, 0xFE6F),  # 縦書き用約物・小字形
    ("symbol", 0xFF01, 0xFF20),  # 全角記号
    ("symbol", 0xFF3B, 0xFF40),
    ("symbol", 0xFF5B, 0xFF65),  # 全角記号 + 半角約物（｡ ｢ ｣ ､ ･）
    ("symbol", 0xFFE0, 0xFFEF),
    # --- 数字 ----------------------------------------------------------------
    ("digit", 0x0030, 0x0039),
    ("digit", 0xFF10, 0xFF19),
    # --- ラテン英字 -----------------------------------------------------------
    ("ascii_letter", 0x0041, 0x005A),
    ("ascii_letter", 0x0061, 0x007A),
    ("fullwidth_latin", 0xFF21, 0xFF3A),
    ("fullwidth_latin", 0xFF41, 0xFF5A),
    # --- 仮名 ----------------------------------------------------------------
    ("hiragana", 0x3041, 0x309F),
    ("katakana", 0x30A0, 0x30FF),
    ("katakana", 0x31F0, 0x31FF),  # 小書きカタカナ拡張
    ("halfwidth_katakana", 0xFF66, 0xFF9F),
    # --- 漢字 ----------------------------------------------------------------
    ("kanji", 0x3400, 0x4DBF),  # 拡張A
    ("kanji", 0x4E00, 0x9FFF),  # 基本
    ("kanji", 0xF900, 0xFAFF),  # 互換漢字
    ("kanji", 0x20000, 0x2A6DF),  # 拡張B
    ("kanji", 0x2A700, 0x2EBEF),  # 拡張C-F
    ("kanji", 0x2F800, 0x2FA1F),  # 互換漢字補助
    # --- 絵文字 --------------------------------------------------------------
    ("emoji", 0x2600, 0x26FF),
    ("emoji", 0x2700, 0x27BF),
    ("emoji", 0x2B00, 0x2BFF),
    ("emoji", 0xFE0F, 0xFE0F),  # variation selector-16
    ("emoji", 0x1F000, 0x1FAFF),
    # --- 空白（記号rangeより後に置いて上書きする） -------------------------------
    ("whitespace", 0x0009, 0x000D),
    ("whitespace", 0x0020, 0x0020),
    ("whitespace", 0x00A0, 0x00A0),
    ("whitespace", 0x2000, 0x200A),
    ("whitespace", 0x3000, 0x3000),
    # --- 個別の上書き ---------------------------------------------------------
    ("kanji", 0x3005, 0x3007),  # 々 〆 〇
    ("symbol", 0x30FB, 0x30FB),  # ・（カタカナrange内だが約物）
)


@lru_cache(maxsize=1)
def build_char_class_table() -> np.ndarray:
    """codepoint → :data:`CHAR_CLASSES` のindex を引くlookup table（read-only）。"""
    table = np.full(_UNICODE_LIMIT, _CLASS_INDEX["other"], dtype=np.uint8)
    for name, start, end in _CHAR_CLASS_RANGES:
        table[start : end + 1] = _CLASS_INDEX[name]
    table.flags.writeable = False
    return table


def classify_char(char: str) -> str:
    """1文字の文字種class名を返す。:func:`build_char_class_table` と必ず一致する。"""
    if len(char) != 1:
        raise ValueError(f"classify_char expects exactly one character, got {len(char)}.")
    return CHAR_CLASSES[int(build_char_class_table()[ord(char)])]


# =============================================================================
# tokenizer coverageの状態
# =============================================================================

#: 1文字をtokenizeしたときの状態。``covered`` 以外は日本語入力の情報欠落リスク。
COVERAGE_STATES: tuple[str, ...] = ("covered", "byte_fallback", "unk", "dropped")


def coverage_state(
    piece_ids: Sequence[int],
    *,
    unk_id: int,
    byte_ids: frozenset[int] | set[int],
    dummy_prefix_id: int | None = None,
) -> str:
    """1文字分のtoken IDから coverage状態を判定する。

    ``dropped`` は正規化で文字自体が消えた場合（例: U+3000 全角空白、BOM）。
    SentencePieceは語彙に無い文字を ``<unk>`` ではなくUTF-8 byteへ落とすため、
    ``byte_fallback`` が実質的な「未知文字」を表す。
    """
    ids = [int(index) for index in piece_ids if dummy_prefix_id is None or int(index) != dummy_prefix_id]
    if not ids:
        return "dropped"
    if any(index == unk_id for index in ids):
        return "unk"
    if any(index in byte_ids for index in ids):
        return "byte_fallback"
    return "covered"


def decode_byte_fallback(byte_values: Sequence[int] | np.ndarray) -> tuple[np.ndarray, int]:
    """byte fallback pieceのbyte列を、実際に潰れた **文字** のcodepointへ戻す。

    corpus中で byte fallback になった文字を文脈込みで正確に数えるために使う。
    ``<0xXX>`` pieceは常に1文字分が連続して並ぶので、byte pieceだけを出現順に
    連結した列はUTF-8として自己同期する（間に挟まる通常pieceは無視してよい）。

    Returns:
        (codepoint配列, 不正なUTF-8断片として捨てたlead byteの数)
    """
    values = np.asarray(byte_values, dtype=np.int64)
    if values.size == 0:
        return np.zeros(0, dtype=np.int64), 0
    total = int(values.size)
    is_continuation = (values >= 0x80) & (values < 0xC0)
    lead_positions = np.nonzero(~is_continuation)[0]
    if lead_positions.size == 0:
        return np.zeros(0, dtype=np.int64), total
    lead = values[lead_positions]

    lengths = np.ones_like(lead)
    lengths[lead >= 0xC0] = 2
    lengths[lead >= 0xE0] = 3
    lengths[lead >= 0xF0] = 4

    def _at(offset: int) -> tuple[np.ndarray, np.ndarray]:
        index = np.minimum(lead_positions + offset, total - 1)
        return values[index] & 0x3F, is_continuation[index]

    trail1, ok1 = _at(1)
    trail2, ok2 = _at(2)
    trail3, ok3 = _at(3)

    codepoints = np.where(
        lengths == 1,
        lead,
        np.where(
            lengths == 2,
            ((lead & 0x1F) << 6) | trail1,
            np.where(
                lengths == 3,
                ((lead & 0x0F) << 12) | (trail1 << 6) | trail2,
                ((lead & 0x07) << 18) | (trail1 << 12) | (trail2 << 6) | trail3,
            ),
        ),
    )
    valid = (lead_positions + lengths) <= total
    valid &= (lengths < 2) | ok1
    valid &= (lengths < 3) | ok2
    valid &= (lengths < 4) | ok3
    valid &= (codepoints >= 0) & (codepoints < _UNICODE_LIMIT)
    valid &= (codepoints < 0xD800) | (codepoints > 0xDFFF)
    return codepoints[valid], int((~valid).sum())


class TokenEncoder(Protocol):
    """:class:`CorpusAccumulator` が要求する最小のtokenizer interface。"""

    vocab_size: int
    unk_id: int
    byte_ids: frozenset[int]
    byte_values: dict[int, int]
    dummy_prefix_id: int | None

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        ...

    def encode_pieces(self, text: str) -> list[str]:
        ...


_BYTE_PIECE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")

#: SentencePieceのdummy prefix（U+2581 LOWER ONE EIGHTH BLOCK）。
DUMMY_PREFIX = "▁"


class SentencePieceEncoder:
    """``tokenizer.model`` を直接読む高速encoder（HF wrapperを介さない）。

    corpus全走査ではHF ``PreTrainedTokenizer.encode`` は1件あたり約50 usかかるが、
    ``SentencePieceProcessor.encode`` のbatch呼び出しは約3 usで済む。
    日本語本文にはspecial tokenが（衝突検査で確認する範囲では）出現しないため、
    集計にはこちらを使い、promptテンプレートの実測だけHF側を使う。
    """

    def __init__(self, model_path: str | Path) -> None:
        import sentencepiece as spm

        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.model_path)
        self.vocab_size = int(self.sp.get_piece_size())
        self.unk_id = int(self.sp.unk_id())
        self.byte_values = {
            index: int(self.sp.id_to_piece(index)[3:5], 16)
            for index in range(self.vocab_size)
            if _BYTE_PIECE_RE.match(self.sp.id_to_piece(index))
        }
        self.byte_ids = frozenset(self.byte_values)
        dummy = int(self.sp.piece_to_id(DUMMY_PREFIX))
        self.dummy_prefix_id = dummy if dummy != self.unk_id else None

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return self.sp.encode(list(texts), out_type=int)

    def encode_pieces(self, text: str) -> list[str]:
        return list(self.sp.encode(text, out_type=str))

    def id_to_piece(self, index: int) -> str:
        return str(self.sp.id_to_piece(int(index)))


# =============================================================================
# 集計ユーティリティ
# =============================================================================

#: token長histogramの上限（超過分は overflow として別に数える）。
TOKEN_HIST_CAP = 4096

#: token/文字 比のhistogram倍率と上限（2000 = 20.00 token/char）。
RATIO_HIST_SCALE = 100
RATIO_HIST_CAP = 2000


def segment_sums(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """``values[starts[i]:ends[i]]`` の合計をまとめて求める。

    ``np.add.reduceat`` は長さ0のsegmentを正しく扱えないので累積和の差で計算する。
    """
    cumulative = np.concatenate(([0], np.cumsum(np.asarray(values, dtype=np.int64))))
    return cumulative[ends] - cumulative[starts]


def percentiles_from_histogram(
    counts: Sequence[int] | np.ndarray,
    quantiles: Sequence[float],
) -> dict[str, int]:
    """整数値histogramからnearest-rank percentileを求める。

    ``counts[v]`` は値 ``v`` の出現数。戻り値のkeyは ``p50`` / ``p95`` / ``p99`` 形式。
    要素が無い場合はすべて0を返す。
    """
    counts_array = np.asarray(counts, dtype=np.int64)
    total = int(counts_array.sum())
    cumulative = np.cumsum(counts_array)
    result: dict[str, int] = {}
    for quantile in quantiles:
        if not 0.0 < quantile <= 1.0:
            raise ValueError(f"quantile must be in (0, 1], got {quantile}.")
        key = f"p{quantile * 100:g}"
        if total == 0:
            result[key] = 0
            continue
        rank = min(total, max(1, math.ceil(quantile * total)))
        result[key] = int(np.searchsorted(cumulative, rank, side="left"))
    return result


def histogram_mean(counts: Sequence[int] | np.ndarray) -> float:
    """整数値histogramの平均。空なら0.0。"""
    counts_array = np.asarray(counts, dtype=np.int64)
    total = int(counts_array.sum())
    if total == 0:
        return 0.0
    values = np.arange(counts_array.shape[0], dtype=np.int64)
    return float((values * counts_array).sum() / total)


def safe_ratio(numerator: float, denominator: float) -> float:
    """0除算を0.0にする比。"""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def count_special_token_hits(texts: Sequence[str], specials: Sequence[str]) -> Counter:
    """corpus本文にspecial token文字列がそのまま出現する件数（文単位）。"""
    hits: Counter = Counter()
    for text in texts:
        if "<" not in text:
            continue
        for special in specials:
            if special in text:
                hits[special] += 1
    return hits


# =============================================================================
# corpus走査
# =============================================================================


@dataclass
class CorpusRow:
    """corpus 1行分（走査に必要な列だけ）。"""

    line_number: int
    text: str
    duration: float | None = None


def iter_tsv_rows(
    path: str | Path,
    *,
    text_column: str = "text",
    duration_column: str | None = "duration",
    encoding: str = "utf-8",
    limit: int | None = None,
) -> Iterator[CorpusRow]:
    """ヘッダ付きTSVから text（と duration）を取り出す。

    列数が合わない行は読み飛ばし、``line_number`` はヘッダを1行目とした実ファイル行番号。
    """
    path = Path(path)
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            return
        header = header_line.rstrip("\r\n").split("\t")
        if text_column not in header:
            raise ValueError(f"Column {text_column!r} not found in header {header}.")
        text_index = header.index(text_column)
        duration_index = header.index(duration_column) if duration_column in header else None
        expected = len(header)
        emitted = 0
        for offset, line in enumerate(handle, start=2):
            if limit is not None and emitted >= limit:
                return
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != expected:
                continue
            duration: float | None = None
            if duration_index is not None:
                try:
                    duration = float(parts[duration_index])
                except ValueError:
                    duration = None
            emitted += 1
            yield CorpusRow(line_number=offset, text=parts[text_index], duration=duration)


def batched(rows: Iterator[CorpusRow], size: int) -> Iterator[list[CorpusRow]]:
    """iteratorを固定長のlistへまとめる。"""
    if size <= 0:
        raise ValueError("batch size must be positive.")
    batch: list[CorpusRow] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# =============================================================================
# 本体
# =============================================================================


@dataclass
class UnkSample:
    """``unk_samples.txt`` に残す実例。"""

    line_number: int
    text: str
    unk_tokens: int
    byte_tokens: int


class CorpusAccumulator:
    """corpusをbatch単位で走査し、P1bのT0項目を積み上げる。

    実データに依存しないよう、tokenizerは :class:`TokenEncoder` protocolで受け取る。
    """

    def __init__(
        self,
        encoder: TokenEncoder,
        *,
        sample_limit: int = 200,
        special_tokens: Sequence[str] = (),
        nfkc: bool = True,
        nfkc_limit: int | None = None,
    ) -> None:
        self.encoder = encoder
        self.sample_limit = int(sample_limit)
        self.special_tokens = tuple(special_tokens)
        self.nfkc = bool(nfkc)
        self.nfkc_limit = nfkc_limit

        self.class_table = build_char_class_table()
        self.byte_flag = np.zeros(encoder.vocab_size, dtype=bool)
        if encoder.byte_ids:
            self.byte_flag[np.fromiter(encoder.byte_ids, dtype=np.int64)] = True
        self.unk_flag = np.zeros(encoder.vocab_size, dtype=bool)
        self.unk_flag[int(encoder.unk_id)] = True
        self.byte_value_lookup = np.full(encoder.vocab_size, -1, dtype=np.int64)
        for token_id, byte_value in getattr(encoder, "byte_values", {}).items():
            self.byte_value_lookup[int(token_id)] = int(byte_value)

        # --- 全体量 ---
        self.n_sentences = 0
        self.n_empty_sentences = 0
        self.n_chars = 0
        self.n_tokens = 0
        self.n_unk_tokens = 0
        self.n_byte_tokens = 0
        self.n_sentences_with_unk = 0
        self.n_sentences_with_byte = 0
        self.total_duration = 0.0
        self.n_duration_rows = 0

        # --- 分布 ---
        self.token_hist = np.zeros(TOKEN_HIST_CAP + 1, dtype=np.int64)
        self.token_overflow = 0
        self.max_tokens = 0
        self.ratio_hist = np.zeros(RATIO_HIST_CAP + 1, dtype=np.int64)
        self.char_hist = np.zeros(_UNICODE_LIMIT, dtype=np.int64)

        # --- 文脈込みで byte fallback になった文字 ---
        self.fallback_char_hist = np.zeros(_UNICODE_LIMIT, dtype=np.int64)
        self.n_fallback_chars = 0
        self.n_invalid_fallback_leads = 0

        # --- 文字種別 ---
        n_classes = len(CHAR_CLASSES)
        self.class_sentences = np.zeros(n_classes, dtype=np.int64)
        self.class_sentences_with_byte = np.zeros(n_classes, dtype=np.int64)

        # --- special token衝突 ---
        self.special_hits: Counter = Counter()

        # --- NFKC比較 ---
        self.nfkc_compared = 0
        self.nfkc_changed_texts = 0
        self.nfkc_tokens_before = 0
        self.nfkc_tokens_after = 0
        self.nfkc_chars_before = 0
        self.nfkc_chars_after = 0
        self.nfkc_unk_after = 0
        self.nfkc_byte_after = 0
        self.nfkc_sentences_with_byte_after = 0
        self.nfkc_token_increase = 0
        self.nfkc_token_decrease = 0

        self.samples: list[UnkSample] = []

    # -- 走査 ---------------------------------------------------------------

    def update(self, rows: Sequence[CorpusRow]) -> None:
        """1 batch分を集計する。"""
        if not rows:
            return
        texts = [row.text for row in rows]
        token_ids = self.encoder.encode_batch(texts)
        token_lengths = np.fromiter(
            (len(ids) for ids in token_ids), dtype=np.int64, count=len(token_ids)
        )
        total_tokens = int(token_lengths.sum())
        flat = np.fromiter(chain.from_iterable(token_ids), dtype=np.int64, count=total_tokens)
        token_ends = np.cumsum(token_lengths)
        token_starts = token_ends - token_lengths

        is_unk = self.unk_flag[flat] if total_tokens else np.zeros(0, dtype=bool)
        is_byte = self.byte_flag[flat] if total_tokens else np.zeros(0, dtype=bool)
        per_unk = segment_sums(is_unk, token_starts, token_ends)
        per_byte = segment_sums(is_byte, token_starts, token_ends)

        char_lengths = np.fromiter((len(text) for text in texts), dtype=np.int64, count=len(texts))
        joined = "".join(texts)
        codepoints = np.frombuffer(joined.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
        char_ends = np.cumsum(char_lengths)
        char_starts = char_ends - char_lengths

        # --- 全体量 ---
        self.n_sentences += len(rows)
        self.n_empty_sentences += int((char_lengths == 0).sum())
        self.n_chars += int(char_lengths.sum())
        self.n_tokens += total_tokens
        self.n_unk_tokens += int(is_unk.sum())
        self.n_byte_tokens += int(is_byte.sum())
        self.n_sentences_with_unk += int((per_unk > 0).sum())
        self.n_sentences_with_byte += int((per_byte > 0).sum())

        # --- byte fallbackで潰れた文字を文脈込みで復元する ---
        if total_tokens and bool(is_byte.any()):
            fallback_codepoints, invalid = decode_byte_fallback(self.byte_value_lookup[flat[is_byte]])
            if fallback_codepoints.size:
                self.fallback_char_hist += np.bincount(fallback_codepoints, minlength=_UNICODE_LIMIT)
            self.n_fallback_chars += int(fallback_codepoints.size)
            self.n_invalid_fallback_leads += invalid

        durations = [row.duration for row in rows if row.duration is not None]
        if durations:
            self.total_duration += float(sum(durations))
            self.n_duration_rows += len(durations)

        # --- 分布 ---
        clipped = np.minimum(token_lengths, TOKEN_HIST_CAP)
        self.token_hist += np.bincount(clipped, minlength=TOKEN_HIST_CAP + 1)
        self.token_overflow += int((token_lengths > TOKEN_HIST_CAP).sum())
        if token_lengths.size:
            self.max_tokens = max(self.max_tokens, int(token_lengths.max()))
        non_empty = char_lengths > 0
        if bool(non_empty.any()):
            ratios = token_lengths[non_empty] / char_lengths[non_empty] * RATIO_HIST_SCALE
            buckets = np.minimum(np.rint(ratios).astype(np.int64), RATIO_HIST_CAP)
            self.ratio_hist += np.bincount(buckets, minlength=RATIO_HIST_CAP + 1)
        if codepoints.size:
            self.char_hist += np.bincount(codepoints, minlength=_UNICODE_LIMIT)

        # --- 文字種別（文単位） ---
        if codepoints.size:
            classes = self.class_table[codepoints]
            has_byte = per_byte > 0
            for class_index in np.unique(classes):
                mask = classes == class_index
                contains = segment_sums(mask, char_starts, char_ends) > 0
                self.class_sentences[int(class_index)] += int(contains.sum())
                self.class_sentences_with_byte[int(class_index)] += int((contains & has_byte).sum())

        # --- special token衝突 ---
        if self.special_tokens:
            self.special_hits.update(count_special_token_hits(texts, self.special_tokens))

        # --- 実例 ---
        if len(self.samples) < self.sample_limit:
            flagged = np.nonzero((per_byte > 0) | (per_unk > 0))[0]
            for index in flagged[: self.sample_limit - len(self.samples)]:
                row = rows[int(index)]
                self.samples.append(
                    UnkSample(
                        line_number=row.line_number,
                        text=row.text,
                        unk_tokens=int(per_unk[int(index)]),
                        byte_tokens=int(per_byte[int(index)]),
                    )
                )

        # --- NFKC ---
        if self.nfkc:
            remaining = None if self.nfkc_limit is None else self.nfkc_limit - self.nfkc_compared
            if remaining is None or remaining > 0:
                subset = texts if remaining is None else texts[:remaining]
                self._update_nfkc(
                    subset, token_lengths[: len(subset)], char_lengths[: len(subset)]
                )

    def _update_nfkc(
        self,
        texts: Sequence[str],
        token_lengths_before: np.ndarray,
        char_lengths_before: np.ndarray,
    ) -> None:
        normalized = [unicodedata.normalize("NFKC", text) for text in texts]
        self.nfkc_compared += len(texts)
        self.nfkc_changed_texts += sum(
            1 for before, after in zip(texts, normalized) if before != after
        )
        self.nfkc_tokens_before += int(token_lengths_before.sum())
        self.nfkc_chars_before += int(char_lengths_before.sum())

        token_ids = self.encoder.encode_batch(normalized)
        token_lengths = np.fromiter(
            (len(ids) for ids in token_ids), dtype=np.int64, count=len(token_ids)
        )
        total_tokens = int(token_lengths.sum())
        flat = np.fromiter(chain.from_iterable(token_ids), dtype=np.int64, count=total_tokens)
        ends = np.cumsum(token_lengths)
        starts = ends - token_lengths
        is_unk = self.unk_flag[flat] if total_tokens else np.zeros(0, dtype=bool)
        is_byte = self.byte_flag[flat] if total_tokens else np.zeros(0, dtype=bool)

        self.nfkc_tokens_after += total_tokens
        self.nfkc_chars_after += sum(len(text) for text in normalized)
        self.nfkc_unk_after += int(is_unk.sum())
        self.nfkc_byte_after += int(is_byte.sum())
        self.nfkc_sentences_with_byte_after += int((segment_sums(is_byte, starts, ends) > 0).sum())
        delta = token_lengths - token_lengths_before
        self.nfkc_token_increase += int((delta > 0).sum())
        self.nfkc_token_decrease += int((delta < 0).sum())

    # -- 結果 ---------------------------------------------------------------

    def corpus_metrics(self) -> dict[str, Any]:
        """文単位・token単位の集計結果。"""
        quantiles = (0.5, 0.95, 0.99, 0.999)
        token_percentiles = percentiles_from_histogram(self.token_hist, quantiles)
        ratio_percentiles = {
            key: value / RATIO_HIST_SCALE
            for key, value in percentiles_from_histogram(self.ratio_hist, quantiles).items()
        }
        return {
            "sentences": self.n_sentences,
            "empty_sentences": self.n_empty_sentences,
            "characters": self.n_chars,
            "tokens": self.n_tokens,
            "unk_tokens": self.n_unk_tokens,
            "byte_fallback_tokens": self.n_byte_tokens,
            "sentences_with_unk": self.n_sentences_with_unk,
            "sentences_with_byte_fallback": self.n_sentences_with_byte,
            "unk_sentence_rate": safe_ratio(self.n_sentences_with_unk, self.n_sentences),
            "byte_fallback_sentence_rate": safe_ratio(self.n_sentences_with_byte, self.n_sentences),
            "unk_token_rate": safe_ratio(self.n_unk_tokens, self.n_tokens),
            "byte_fallback_token_rate": safe_ratio(self.n_byte_tokens, self.n_tokens),
            "byte_fallback_characters": self.n_fallback_chars,
            "invalid_fallback_leads": self.n_invalid_fallback_leads,
            "char_byte_fallback_rate": safe_ratio(self.n_fallback_chars, self.n_chars),
            "char_unk_rate": safe_ratio(self.n_unk_tokens, self.n_chars),
            "tokens_per_char_mean": safe_ratio(self.n_tokens, self.n_chars),
            "chars_per_token_mean": safe_ratio(self.n_chars, self.n_tokens),
            "tokens_per_char_sentence_mean": histogram_mean(self.ratio_hist) / RATIO_HIST_SCALE,
            "tokens_per_char_percentiles": ratio_percentiles,
            "tokens_per_sentence_mean": histogram_mean(self.token_hist),
            "tokens_per_sentence_percentiles": token_percentiles,
            "tokens_per_sentence_max": self.max_tokens,
            "tokens_per_sentence_over_histogram_cap": self.token_overflow,
            "chars_per_sentence_mean": safe_ratio(self.n_chars, self.n_sentences),
            "total_duration_seconds": self.total_duration,
            "duration_rows": self.n_duration_rows,
            "chars_per_second": safe_ratio(self.n_chars, self.total_duration),
            "tokens_per_second": safe_ratio(self.n_tokens, self.total_duration),
            "special_token_hits": {
                name: int(self.special_hits.get(name, 0)) for name in self.special_tokens
            },
        }

    def nfkc_metrics(self) -> dict[str, Any]:
        """NFKC適用前後の差分。"""
        if not self.nfkc or self.nfkc_compared == 0:
            return {"enabled": False}
        return {
            "enabled": True,
            "sentences_compared": self.nfkc_compared,
            "sentences_changed_by_nfkc": self.nfkc_changed_texts,
            "sentence_change_rate": safe_ratio(self.nfkc_changed_texts, self.nfkc_compared),
            "characters_before": self.nfkc_chars_before,
            "characters_after": self.nfkc_chars_after,
            "tokens_before": self.nfkc_tokens_before,
            "tokens_after": self.nfkc_tokens_after,
            "token_delta": self.nfkc_tokens_after - self.nfkc_tokens_before,
            "token_delta_rate": safe_ratio(
                self.nfkc_tokens_after - self.nfkc_tokens_before, self.nfkc_tokens_before
            ),
            "tokens_per_char_after": safe_ratio(self.nfkc_tokens_after, self.nfkc_chars_after),
            "unk_tokens_after": self.nfkc_unk_after,
            "byte_fallback_tokens_after": self.nfkc_byte_after,
            "sentences_with_byte_fallback_after": self.nfkc_sentences_with_byte_after,
            "sentences_with_more_tokens": self.nfkc_token_increase,
            "sentences_with_fewer_tokens": self.nfkc_token_decrease,
        }


def in_context_coverage(
    char_hist: np.ndarray,
    fallback_char_hist: np.ndarray,
    *,
    unk_tokens: int = 0,
    top_n: int = 50,
) -> dict[str, Any]:
    """corpus本文で **実際に** byte fallbackへ落ちた文字の集計（主指標）。

    :func:`isolated_character_coverage` は1文字だけを渡した場合の判定なので、
    「その文字を含む複数文字pieceが存在する」場合を過大に未知扱いする。
    こちらは実際のtoken列から復元しているので過大評価がない。
    """
    table = build_char_class_table()
    total_chars = int(char_hist.sum())
    total_fallback = int(fallback_char_hist.sum())
    codepoints = np.nonzero(fallback_char_hist)[0]

    per_class: dict[str, dict[str, Any]] = {
        name: {"occurrences": 0, "fallback_occurrences": 0} for name in CHAR_CLASSES
    }
    for index, name in enumerate(CHAR_CLASSES):
        mask = table == index
        per_class[name]["occurrences"] = int(char_hist[mask].sum())
        per_class[name]["fallback_occurrences"] = int(fallback_char_hist[mask].sum())
    for entry in per_class.values():
        entry["fallback_rate"] = safe_ratio(entry["fallback_occurrences"], entry["occurrences"])
        entry["share_of_all_characters"] = safe_ratio(entry["occurrences"], total_chars)

    order = np.argsort(fallback_char_hist[codepoints])[::-1] if codepoints.size else []
    top = [
        {
            "char": chr(int(codepoints[index])),
            "codepoint": f"U+{int(codepoints[index]):04X}",
            "name": _unicode_name(chr(int(codepoints[index]))),
            "class": CHAR_CLASSES[int(table[int(codepoints[index])])],
            "count": int(fallback_char_hist[codepoints[index]]),
            "occurrences": int(char_hist[codepoints[index]]),
            "fallback_share_of_char": safe_ratio(
                int(fallback_char_hist[codepoints[index]]), int(char_hist[codepoints[index]])
            ),
        }
        for index in list(order)[:top_n]
    ]
    return {
        "total_characters": total_chars,
        "byte_fallback_characters": total_fallback,
        "char_byte_fallback_rate": safe_ratio(total_fallback, total_chars),
        "char_unk_rate": safe_ratio(unk_tokens, total_chars),
        "distinct_fallback_characters": int(codepoints.size),
        "by_class": per_class,
        "top_fallback": top,
    }


def isolated_character_coverage(
    encoder: TokenEncoder,
    char_hist: np.ndarray,
    *,
    top_uncovered: int = 50,
) -> dict[str, Any]:
    """出現した全distinct文字を **1文字だけ** tokenizeしたときのcoverage。

    「その文字がvocabularyに単独pieceとして存在するか」に近い指標で、
    文脈込みの :func:`in_context_coverage` の上界になる。
    vocabulary拡張の候補文字を選ぶときはこちらを見る。
    """
    table = build_char_class_table()
    codepoints = np.nonzero(char_hist)[0]
    characters = [chr(int(codepoint)) for codepoint in codepoints]
    encoded = encoder.encode_batch(characters) if characters else []

    total_chars = int(char_hist.sum())
    by_state_distinct: Counter = Counter()
    by_state_occurrences: Counter = Counter()
    per_class: dict[str, dict[str, Any]] = {
        name: {
            "distinct": 0,
            "occurrences": 0,
            "distinct_by_state": {state: 0 for state in COVERAGE_STATES},
            "occurrences_by_state": {state: 0 for state in COVERAGE_STATES},
        }
        for name in CHAR_CLASSES
    }
    uncovered: list[tuple[int, str, str, str]] = []  # (count, char, class, state)

    for codepoint, character, ids in zip(codepoints, characters, encoded):
        count = int(char_hist[codepoint])
        state = coverage_state(
            ids,
            unk_id=encoder.unk_id,
            byte_ids=encoder.byte_ids,
            dummy_prefix_id=encoder.dummy_prefix_id,
        )
        class_name = CHAR_CLASSES[int(table[int(codepoint)])]
        by_state_distinct[state] += 1
        by_state_occurrences[state] += count
        entry = per_class[class_name]
        entry["distinct"] += 1
        entry["occurrences"] += count
        entry["distinct_by_state"][state] += 1
        entry["occurrences_by_state"][state] += count
        if state != "covered":
            uncovered.append((count, character, class_name, state))

    for entry in per_class.values():
        occurrences = entry["occurrences"]
        not_covered = occurrences - entry["occurrences_by_state"]["covered"]
        entry["share_of_all_characters"] = safe_ratio(occurrences, total_chars)
        entry["uncovered_occurrences"] = not_covered
        entry["uncovered_occurrence_rate"] = safe_ratio(not_covered, occurrences)
        entry["byte_fallback_occurrence_rate"] = safe_ratio(
            entry["occurrences_by_state"]["byte_fallback"], occurrences
        )
        entry["distinct_covered_rate"] = safe_ratio(
            entry["distinct_by_state"]["covered"], entry["distinct"]
        )

    uncovered.sort(key=lambda item: item[0], reverse=True)
    top = [
        {
            "char": character,
            "codepoint": f"U+{ord(character):04X}",
            "name": _unicode_name(character),
            "class": class_name,
            "state": state,
            "count": count,
            "pieces": encoder.encode_pieces(character),
        }
        for count, character, class_name, state in uncovered[:top_uncovered]
    ]

    uncovered_occurrences = total_chars - by_state_occurrences.get("covered", 0)
    return {
        "distinct_characters": int(codepoints.size),
        "total_characters": total_chars,
        "distinct_by_state": {
            state: int(by_state_distinct.get(state, 0)) for state in COVERAGE_STATES
        },
        "occurrences_by_state": {
            state: int(by_state_occurrences.get(state, 0)) for state in COVERAGE_STATES
        },
        "char_unk_rate": safe_ratio(by_state_occurrences.get("unk", 0), total_chars),
        "char_byte_fallback_rate": safe_ratio(
            by_state_occurrences.get("byte_fallback", 0), total_chars
        ),
        "char_dropped_rate": safe_ratio(by_state_occurrences.get("dropped", 0), total_chars),
        "char_uncovered_rate": safe_ratio(uncovered_occurrences, total_chars),
        "by_class": per_class,
        "top_uncovered": top,
    }


def _unicode_name(character: str) -> str:
    try:
        return unicodedata.name(character)
    except ValueError:
        return ""


def sentence_class_metrics(accumulator: CorpusAccumulator) -> dict[str, Any]:
    """文字種を含む文が byte fallback を出す率（文単位のcoverage）。"""
    result: dict[str, Any] = {}
    for index, name in enumerate(CHAR_CLASSES):
        sentences = int(accumulator.class_sentences[index])
        with_byte = int(accumulator.class_sentences_with_byte[index])
        result[name] = {
            "sentences_containing": sentences,
            "sentences_containing_rate": safe_ratio(sentences, accumulator.n_sentences),
            "sentences_with_byte_fallback": with_byte,
            "byte_fallback_rate_given_class": safe_ratio(with_byte, sentences),
        }
    return result


# =============================================================================
# promptテンプレートと実効text予算
# =============================================================================


@dataclass
class _PromptProbe:
    """``CuteTTSProcessor`` のprompt組み立てだけを再利用するための最小stub。

    ``_text_only_prompt`` / ``_reference_prompt_segments`` は
    ``segment_manager`` / ``tokenizer`` / ``text_suffix_token`` しか参照しないため、
    Audio VAEをloadせずに **推論と同一のテンプレート** を測れる。
    テンプレート文字列をこのscriptへ複製しないことが目的。
    """

    segment_manager: Any
    tokenizer: Any
    text_suffix_token: str


def reference_patch_count(
    seconds: float,
    *,
    sample_rate: int,
    speech_compress_rate: int,
    audio_patch_size: int,
) -> int:
    """reference音声の秒数 → LM sequence上のpatch数。

    ``prepare_reference_audio`` は先頭30秒へcropし、``SegmentManager`` はpatch境界へ
    zero paddingするので ceil を2回かける。
    """
    if seconds < 0:
        raise ValueError("seconds must be non-negative.")
    latent_frames = math.ceil(seconds * sample_rate / speech_compress_rate)
    return math.ceil(latent_frames / audio_patch_size)


def effective_text_budget(
    max_length: int,
    *,
    template_tokens: int,
    reference_tokens: int = 0,
    decode_reserve: int = 0,
) -> int:
    """prefix長制限から、日本語text本文に使えるtoken数を求める（下限0）。"""
    return max(
        0, int(max_length) - int(template_tokens) - int(reference_tokens) - int(decode_reserve)
    )


def measure_prompt_budget(
    tokenizer_dir: str | Path,
    model_config: dict[str, Any],
    *,
    probe_texts: Sequence[str],
    reference_seconds: float,
    max_decode_length: int,
    chars_per_token: float,
) -> dict[str, Any]:
    """推論と同じテンプレート・同じ ``SegmentManager`` で実効text予算を測る。"""
    import torch  # 重いimportをmoduleロード時に持ち込まない

    from cutetts.inference.conditioning import build_guidance_plan
    from cutetts.modeling.processor import CuteTTSProcessor
    from cutetts.modeling.segments import SegmentManager, SegmentManagerConfig
    from cutetts.modeling.tokenizer import CuteTTSSentencePieceTokenizer

    segment_config = SegmentManagerConfig(**model_config["processor"]["segment"])
    manager = SegmentManager(segment_config)
    tokenizer = CuteTTSSentencePieceTokenizer.from_pretrained(str(tokenizer_dir))
    probe = _PromptProbe(
        segment_manager=manager,
        tokenizer=tokenizer,
        text_suffix_token="<|endofprompt|>",
    )

    sample_rate = int(model_config["sample_rate"])
    compress_rate = int(model_config["processor"]["speech_compress_rate"])
    patch_size = int(segment_config.audio_patch_size)
    reference_patches = reference_patch_count(
        reference_seconds,
        sample_rate=sample_rate,
        speech_compress_rate=compress_rate,
        audio_patch_size=patch_size,
    )
    latent_frames = math.ceil(reference_seconds * sample_rate / compress_rate)
    reference_segment = manager.create_speech_segment(
        torch.zeros((1, latent_frames, segment_config.audio_feat_dim))
    )
    if int(reference_segment.total_length) != reference_patches:
        raise AssertionError(
            f"reference patch mismatch: {reference_segment.total_length} != {reference_patches}"
        )

    tts_overheads: list[int] = []
    clone_overheads: list[int] = []
    per_probe: list[dict[str, Any]] = []
    for text in probe_texts:
        text_tokens = len(tokenizer.encode(text))
        tts_prompt = CuteTTSProcessor._text_only_prompt(probe, text)
        tts_total = len(tokenizer.encode(tts_prompt))
        clone_segments = CuteTTSProcessor._reference_prompt_segments(probe, text, reference_segment)
        clone_total = sum(int(segment.total_length) for segment in clone_segments)
        tts_overheads.append(tts_total - text_tokens)
        clone_overheads.append(clone_total - text_tokens - reference_patches)
        per_probe.append(
            {
                "text": text,
                "characters": len(text),
                "text_tokens": text_tokens,
                "tts_prefix_tokens": tts_total,
                "voice_clone_prefix_tokens": clone_total,
            }
        )

    # 空textでの下限（テンプレートそのものの長さ）も残す。
    empty_tts = len(tokenizer.encode(CuteTTSProcessor._text_only_prompt(probe, "")))
    empty_clone = sum(
        int(segment.total_length)
        for segment in CuteTTSProcessor._reference_prompt_segments(probe, "", reference_segment)
    )

    tts_overhead = max(tts_overheads) if tts_overheads else empty_tts
    clone_overhead = max(clone_overheads) if clone_overheads else empty_clone - reference_patches
    max_length = int(segment_config.max_length)

    budgets: dict[str, Any] = {}
    for mode, template_tokens, reference_tokens in (
        ("tts", tts_overhead, 0),
        ("voice_clone", clone_overhead, reference_patches),
    ):
        prefix_only = effective_text_budget(
            max_length, template_tokens=template_tokens, reference_tokens=reference_tokens
        )
        with_decode = effective_text_budget(
            max_length,
            template_tokens=template_tokens,
            reference_tokens=reference_tokens,
            decode_reserve=max_decode_length,
        )
        budgets[mode] = {
            "template_tokens": int(template_tokens),
            "reference_tokens": int(reference_tokens),
            "text_token_budget_prefix_only": prefix_only,
            "text_token_budget_reserving_decode": with_decode,
            "text_char_budget_prefix_only": int(prefix_only * chars_per_token),
            "text_char_budget_reserving_decode": int(with_decode * chars_per_token),
        }

    plan = build_guidance_plan("voice_clone", "lm", 2.0)
    return {
        "segment_max_length": max_length,
        "sample_rate": sample_rate,
        "speech_compress_rate": compress_rate,
        "audio_patch_size": patch_size,
        "latent_frame_rate_hz": sample_rate / compress_rate,
        "patch_rate_hz": sample_rate / compress_rate / patch_size,
        "reference_seconds": reference_seconds,
        "reference_latent_frames": latent_frames,
        "reference_patches": reference_patches,
        "speaker_slot_tokens": 1,
        "max_decode_length": int(max_decode_length),
        "max_decode_seconds": max_decode_length / (sample_rate / compress_rate / patch_size),
        "empty_text_tts_prefix_tokens": empty_tts,
        "empty_text_voice_clone_prefix_tokens": empty_clone,
        "probe_texts": per_probe,
        "budgets": budgets,
        "lm_cfg_branches": 2 if plan.uses_lm_cfg else 1,
        "chars_per_token_used": chars_per_token,
    }


# =============================================================================
# 3分岐の判定
# =============================================================================

BRANCH_LABELS = {
    1: "既存Tokenizerを維持して継続学習する",
    2: "既存token ID互換のvocabulary拡張を設計する",
    3: "reading / G2P を入力へ追加する",
}


def recommend_branch(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    """02章第4節の3分岐のうち、coverage実測から選べるものを決める。

    分岐3（reading/G2P）は「読み精度」の問題であり、coverage測定では判定できない。
    ここでは分岐1と2だけを決め、3は別ゲート（S0の聴取・CER）へ回すことを明示する。
    """
    corpus = metrics["corpus"]
    coverage = metrics["character_coverage"]["in_context"]
    budget = metrics["prompt_budget"]["budgets"]["voice_clone"]["text_token_budget_prefix_only"]
    p99_ratio = safe_ratio(corpus["tokens_per_sentence_percentiles"]["p99"], budget)

    checks = [
        {
            "name": "char_unk_rate",
            "description": "文字単位の <unk> 率",
            "value": coverage["char_unk_rate"],
            "threshold": float(thresholds["unk_char_rate_max"]),
            "passed": coverage["char_unk_rate"] <= float(thresholds["unk_char_rate_max"]),
        },
        {
            "name": "char_byte_fallback_rate",
            "description": "文字単位の byte fallback 率（文脈込み実測・実質的な未知文字率）",
            "value": coverage["char_byte_fallback_rate"],
            "threshold": float(thresholds["fallback_char_rate_max"]),
            "passed": coverage["char_byte_fallback_rate"]
            <= float(thresholds["fallback_char_rate_max"]),
        },
        {
            "name": "tokens_per_char",
            "description": "平均 token / 文字",
            "value": corpus["tokens_per_char_mean"],
            "threshold": float(thresholds["tokens_per_char_max"]),
            "passed": corpus["tokens_per_char_mean"] <= float(thresholds["tokens_per_char_max"]),
        },
        {
            "name": "p99_within_budget",
            "description": "token長P99 が実効text予算に占める割合",
            "value": p99_ratio,
            "threshold": float(thresholds["p99_budget_ratio_max"]),
            "passed": p99_ratio <= float(thresholds["p99_budget_ratio_max"]),
        },
    ]
    failed = [check for check in checks if not check["passed"]]
    branch = 1 if not failed else 2
    reasons = [
        (
            f"{check['description']} = {check['value']:.6g} "
            f"({'<=' if check['passed'] else '>'} 閾値 {check['threshold']:.6g})"
        )
        for check in checks
    ]
    return {
        "branch": branch,
        "branch_label": BRANCH_LABELS[branch],
        "checks": checks,
        "failed_checks": [check["name"] for check in failed],
        "reasons": reasons,
        "reading_gate_note": (
            "分岐3 (reading/G2P追加) は文字coverageでは判定できない。"
            "漢字の読み分け誤りが支配的かどうかは S0 の聴取と ASR CER で判断する。"
        ),
    }


# =============================================================================
# 出力
# =============================================================================


def _format_rate(value: float) -> str:
    if value == 0:
        return "0"
    if value < 1e-4:
        return f"{value:.3e}"
    return f"{value * 100:.4f}%"


def render_report(metrics: dict[str, Any]) -> str:
    """人が読む ``report.md`` を組み立てる。"""
    corpus = metrics["corpus"]
    coverage = metrics["character_coverage"]["in_context"]
    isolated = metrics["character_coverage"]["isolated"]
    per_class = coverage["by_class"]
    isolated_class = isolated["by_class"]
    sentence_class = metrics["sentence_character_class"]
    nfkc = metrics["nfkc"]
    budget = metrics["prompt_budget"]
    decision = metrics["decision"]
    run = metrics["run"]

    lines: list[str] = []
    add = lines.append
    add("# P1b: 日本語Tokenizer coverage report")
    add("")
    add(f"生成: {run['finished_at']}")
    add("")
    add("## 0. 結論")
    add("")
    add(f"**推奨する分岐: {decision['branch']}. {decision['branch_label']}**")
    add("")
    for reason in decision["reasons"]:
        add(f"- {reason}")
    add("")
    add(f"- {decision['reading_gate_note']}")
    add("")
    add("根拠となる主要な実測値:")
    add("")
    add("| 指標 | 値 |")
    add("|---|---|")
    add(f"| 走査した文数 | {corpus['sentences']:,} |")
    add(f"| 走査した文字数 | {corpus['characters']:,} |")
    add(f"| 文字単位 `<unk>` 率 | {_format_rate(coverage['char_unk_rate'])} |")
    add(f"| 文単位 `<unk>` 率 | {_format_rate(corpus['unk_sentence_rate'])} |")
    add(f"| 文字単位 byte fallback 率（文脈込み実測） | {_format_rate(coverage['char_byte_fallback_rate'])} |")
    add(f"| 文単位 byte fallback 率 | {_format_rate(corpus['byte_fallback_sentence_rate'])} |")
    add(
        f"| 1文字単独では表せない文字の出現率（上界） | "
        f"{_format_rate(isolated['char_uncovered_rate'])} |"
    )
    add(f"| 平均 token / 文字 | {corpus['tokens_per_char_mean']:.4f} |")
    add(f"| 平均 文字 / token | {corpus['chars_per_token_mean']:.4f} |")
    add(
        "| text token長 P50 / P95 / P99 / max | "
        f"{corpus['tokens_per_sentence_percentiles']['p50']} / "
        f"{corpus['tokens_per_sentence_percentiles']['p95']} / "
        f"{corpus['tokens_per_sentence_percentiles']['p99']} / "
        f"{corpus['tokens_per_sentence_max']} |"
    )
    add(
        "| 実効text予算 (voice_clone, prefix制限のみ) | "
        f"{budget['budgets']['voice_clone']['text_token_budget_prefix_only']:,} token |"
    )
    add("")

    add("## 1. 測定条件")
    add("")
    add(f"- corpus: `{run['corpus_path']}`（{run['corpus_name']}）")
    limit_text = f"{run['limit']:,}" if run["limit"] is not None else "全件"
    add(f"- 走査行数: **{corpus['sentences']:,}**（limit = {limit_text}）")
    add(f"- tokenizer: `{run['tokenizer_dir']}`")
    add(f"- vocab size: {run['vocab_size']}（extended vocab {run['extended_vocab_size']}）")
    add(f"- byte fallback piece 数: {run['byte_piece_count']}")
    add(f"- 走査時間: {run['scan_seconds']:.1f} 秒 / 全体 {run['elapsed_seconds']:.1f} 秒")
    add("")
    add(
        "このtokenizerは `<0xXX>` 形式のbyte fallback pieceを "
        f"{run['byte_piece_count']} 個持つ。語彙に無い文字は `<unk>` にならずUTF-8 byteへ分解されるため、"
        "**`<unk>` 率が0でもcoverageの証明にはならない**。"
        "以降ではbyte fallback率を実質的な未知文字率として扱う。"
    )
    add("")

    add("## 2. 文字種別 coverage")
    add("")
    add("2つの見方を分けて出す。")
    add("")
    add(
        "- **文脈込み（実測）**: corpusのtoken列から、実際に `<0xXX>` へ潰れた文字を復元して数えたもの。"
        "これが日本語入力で情報が落ちる本当の割合。"
    )
    add(
        "- **単独文字（上界）**: その文字を1文字だけtokenizeしたときにbyteへ落ちるか。"
        "複数文字pieceに含まれていれば文脈次第で救われるので、実測の上界にあたる。"
        "vocabulary拡張の候補文字を選ぶときはこちらを見る。"
    )
    add("")
    add(
        "| 文字種 | 延べ出現数 | 全文字比 | fallback率(文脈込み) | 異なり文字数 | "
        "単独で表せる異なり文字の割合 | 単独ならfallbackする文字の出現率(上界) | "
        "この文字種を含む文がfallbackを出す率 |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in CHAR_CLASSES:
        entry = per_class[name]
        if entry["occurrences"] == 0:
            continue
        iso = isolated_class[name]
        add(
            f"| {name} | {entry['occurrences']:,} | {entry['share_of_all_characters'] * 100:.4f}% | "
            f"{_format_rate(entry['fallback_rate'])} | {iso['distinct']:,} | "
            f"{iso['distinct_covered_rate'] * 100:.2f}% | "
            f"{_format_rate(iso['uncovered_occurrence_rate'])} | "
            f"{_format_rate(sentence_class[name]['byte_fallback_rate_given_class'])} |"
        )
    add("")
    add("### 実際に byte fallback になった文字の上位（文脈込み）")
    add("")
    if coverage["top_fallback"]:
        add("| 文字 | codepoint | 文字種 | fallback回数 | corpus内の総出現数 | fallbackした割合 |")
        add("|---|---|---|---:|---:|---:|")
        for item in coverage["top_fallback"][:30]:
            add(
                f"| `{item['char']}` | {item['codepoint']} | {item['class']} | {item['count']:,} | "
                f"{item['occurrences']:,} | {item['fallback_share_of_char'] * 100:.2f}% |"
            )
    else:
        add("byte fallbackになった文字は検出されなかった。")
    add("")
    add("### 単独ではvocabularyに無い文字の上位（拡張候補）")
    add("")
    if isolated["top_uncovered"]:
        add("| 文字 | codepoint | 文字種 | 状態 | corpus内出現数 | 単独tokenize結果 |")
        add("|---|---|---|---|---:|---|")
        for item in isolated["top_uncovered"][:30]:
            pieces = " ".join(item["pieces"])
            add(
                f"| `{item['char']}` | {item['codepoint']} | {item['class']} | {item['state']} | "
                f"{item['count']:,} | `{pieces}` |"
            )
    else:
        add("すべての出現文字が単独pieceで表現できる。")
    add("")

    add("## 3. 系列長")
    add("")
    ratio = corpus["tokens_per_char_percentiles"]
    add(
        f"- 平均 token / 文字: **{corpus['tokens_per_char_mean']:.4f}**"
        f"（文ごとの比の平均 {corpus['tokens_per_char_sentence_mean']:.4f}）"
    )
    add(
        f"- token / 文字 の P50 / P95 / P99: "
        f"{ratio['p50']:.2f} / {ratio['p95']:.2f} / {ratio['p99']:.2f}"
    )
    add(
        f"- 1文あたり token 平均 {corpus['tokens_per_sentence_mean']:.2f}, "
        f"文字 平均 {corpus['chars_per_sentence_mean']:.2f}"
    )
    percentiles = corpus["tokens_per_sentence_percentiles"]
    add(
        f"- token長 P50 {percentiles['p50']} / P95 {percentiles['p95']} / P99 {percentiles['p99']} / "
        f"P99.9 {percentiles['p99.9']} / max {corpus['tokens_per_sentence_max']}"
    )
    if corpus["duration_rows"]:
        add(
            f"- corpus実測の発話速度: {corpus['chars_per_second']:.2f} 文字/秒, "
            f"{corpus['tokens_per_second']:.2f} token/秒"
            f"（duration列 {corpus['duration_rows']:,} 行, "
            f"合計 {corpus['total_duration_seconds'] / 3600:.1f} 時間）"
        )
    add("")

    add("## 4. Unicode正規化（NFKC）前後の差")
    add("")
    if not nfkc.get("enabled"):
        add("NFKC比較は無効化されている。")
    else:
        add(f"- 比較した文数: {nfkc['sentences_compared']:,}")
        add(
            f"- NFKCで文字列が変化した文: {nfkc['sentences_changed_by_nfkc']:,} "
            f"({_format_rate(nfkc['sentence_change_rate'])})"
        )
        add(
            f"- token数: {nfkc['tokens_before']:,} → {nfkc['tokens_after']:,} "
            f"（差 {nfkc['token_delta']:+,}, {nfkc['token_delta_rate'] * 100:+.4f}%）"
        )
        add(
            f"- token数が増えた文 {nfkc['sentences_with_more_tokens']:,} / "
            f"減った文 {nfkc['sentences_with_fewer_tokens']:,}"
        )
        add(
            f"- byte fallback token: {corpus['byte_fallback_tokens']:,} → "
            f"{nfkc['byte_fallback_tokens_after']:,}、"
            f"`<unk>`: {corpus['unk_tokens']:,} → {nfkc['unk_tokens_after']:,}"
        )
        add("")
        add(
            "SentencePiece model自身がNFKC相当の正規化を内蔵しているため、"
            "外側でNFKCをかけても結果はほとんど変わらない。下の実例で確認できる。"
        )
    add("")
    add("### 正規化の実例（tokenizer内蔵normalizerの挙動）")
    add("")
    add("| 入力 | pieces |")
    add("|---|---|")
    for item in metrics["probe_strings"]:
        pieces = " ".join(item["pieces"])
        add(f"| `{item['text']}` | `{pieces}` ({item['tokens']} token) |")
    add("")

    add("## 5. special tokenの衝突")
    add("")
    add("| special token | corpus本文に出現した文数 |")
    add("|---|---:|")
    for name, count in corpus["special_token_hits"].items():
        add(f"| `{name}` | {count:,} |")
    add("")

    add("## 6. promptテンプレート込みの実効text予算")
    add("")
    add(
        f"`SegmentManagerConfig.max_length` = **{budget['segment_max_length']:,}**、"
        f"latent {budget['latent_frame_rate_hz']:.1f} Hz / patch {budget['audio_patch_size']} → "
        f"**{budget['patch_rate_hz']:.2f} patch/秒**。"
    )
    add(
        f"reference {budget['reference_seconds']:.0f} 秒 = "
        f"{budget['reference_latent_frames']} latent frame = "
        f"**{budget['reference_patches']} patch**、"
        f"speaker slot {budget['speaker_slot_tokens']} token。"
    )
    add("")
    add(
        "| mode | テンプレート | reference | text予算 (prefix制限のみ) | "
        f"text予算 (decode {budget['max_decode_length']} patchも引く) |"
    )
    add("|---|---:|---:|---:|---:|")
    for mode in ("tts", "voice_clone"):
        entry = budget["budgets"][mode]
        add(
            f"| {mode} | {entry['template_tokens']} token | {entry['reference_tokens']} patch | "
            f"{entry['text_token_budget_prefix_only']:,} token "
            f"(≒ {entry['text_char_budget_prefix_only']:,} 文字) | "
            f"{entry['text_token_budget_reserving_decode']:,} token "
            f"(≒ {entry['text_char_budget_reserving_decode']:,} 文字) |"
        )
    add("")
    add(
        f"`max_decode_length` = {budget['max_decode_length']} patch は約 "
        f"{budget['max_decode_seconds']:.0f} 秒の音声に相当する。"
    )
    if corpus["duration_rows"]:
        speech_chars = int(budget["max_decode_seconds"] * corpus["chars_per_second"])
        add(
            f"corpus実測の {corpus['chars_per_second']:.2f} 文字/秒 で換算すると、"
            f"この長さで喋れるのは約 **{speech_chars:,} 文字**。"
            "つまり実運用の上限を決めるのは text予算ではなく `max_decode_length` 側である。"
        )
    add("")
    add("テンプレート実測（`src/cutetts/modeling/processor.py` のprompt組み立てをそのまま使用）:")
    add("")
    for item in budget["probe_texts"]:
        add(
            f"- text {item['characters']} 文字 / {item['text_tokens']} token → "
            f"tts prefix {item['tts_prefix_tokens']} token, "
            f"voice_clone prefix {item['voice_clone_prefix_tokens']} token"
        )
    add("")

    add("## 7. 判定")
    add("")
    add("| check | 値 | 閾値 | 判定 |")
    add("|---|---:|---:|---|")
    for check in decision["checks"]:
        add(
            f"| {check['description']} | {check['value']:.6g} | {check['threshold']:.6g} | "
            f"{'OK' if check['passed'] else 'NG'} |"
        )
    add("")
    add(f"**分岐 {decision['branch']}: {decision['branch_label']}**")
    add("")
    add("次のS0では、この結論に基づき入力テキスト形式を決める（02章 J0 / J1 / J2）。")
    add("")
    return "\n".join(lines) + "\n"


def write_unk_samples(path: Path, samples: Sequence[UnkSample], encoder: TokenEncoder) -> None:
    """``<unk>`` / byte fallback を出した文の実例をUTF-8で書く。

    各文について、その文で実際に byte へ潰れた文字を復元して併記する。
    """
    lines = [
        "# P1b: <unk> / byte fallback を出した文の実例",
        "#",
        "# このtokenizerはbyte fallback pieceを持つため <unk> はほぼ出ない。",
        "# 実質的な未知文字は byte(<0xXX>) へ分解された文字である。",
        "#   text   : 元テキスト",
        "#   pieces : tokenizeした結果",
        "#   lost   : この文でbyteへ潰れた文字（重複除去）",
        f"# 件数: {len(samples)}",
        "",
    ]
    for index, sample in enumerate(samples, start=1):
        token_ids = encoder.encode_batch([sample.text])[0]
        byte_values = [encoder.byte_values[i] for i in token_ids if i in encoder.byte_values]
        codepoints, _ = decode_byte_fallback(byte_values)
        lost = sorted({chr(int(codepoint)) for codepoint in codepoints})
        lines.append(
            f"[{index:04d}] line={sample.line_number} "
            f"unk={sample.unk_tokens} byte={sample.byte_tokens}"
        )
        lines.append(f"  text   : {sample.text}")
        lines.append(f"  pieces : {' '.join(encoder.encode_pieces(sample.text))}")
        if lost:
            lines.append(f"  lost   : {' '.join(lost)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 設定とmain
# =============================================================================

#: 固有名詞・数字・日付・単位・URL・全角半角の分割を見るための固定probe。
PROBE_STRINGS: tuple[str, ...] = (
    "こんにちは、世界！",
    "東京特許許可局",
    "令和6年12月31日、午後3時45分",
    "2024年11月3日 12:34:56",
    "気温は25.5℃、湿度は60%です。",
    "1,234,567円を支払った。",
    "https://example.com/path?query=1 を開いてください。",
    "CuteTTSはOPPO Mente Labが公開したTTSです。",
    "ＡＢＣ１２３ｱｲｳｴｵ",
    "彼は「そうだね」と言った……。",
    "絵文字😀と記号★を含む文。",
    "ヴァイオリンとﾊﾞｲｵﾘﾝ",
    "促音っ・撥音ん・長音ー・拗音ゃゅょ",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML設定を読む。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must be a mapping.")
    return config


def resolve_path(path: str | Path) -> Path:
    """相対pathをrepository root基準で解決する。"""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1b: 日本語Tokenizer coverage 測定")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "japanese" / "tokenizer-coverage.yaml"),
        help="YAML設定ファイル",
    )
    parser.add_argument("--limit", type=int, default=None, help="走査する行数の上限（configを上書き）")
    parser.add_argument("--corpus", default=None, help="corpus path（configを上書き）")
    parser.add_argument("--tokenizer-dir", default=None, help="tokenizerディレクトリ（configを上書き）")
    parser.add_argument("--output-root", default=None, help="artifact root（configを上書き）")
    parser.add_argument("--no-nfkc", action="store_true", help="NFKC比較を行わない")
    parser.add_argument("--checksum", action="store_true", help="corpusのsha256も計算する（1.68GBで数分）")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windowsのcp932コンソールで落ちないように
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    started = time.time()
    started_at = datetime.now().astimezone().isoformat()

    config = load_config(args.config)
    corpus_config = dict(config.get("corpus", {}))
    output_config = dict(config.get("output", {}))
    normalization_config = dict(config.get("normalization", {}))
    budget_config = dict(config.get("budget", {}))

    tokenizer_dir = resolve_path(args.tokenizer_dir or config["tokenizer_dir"])
    model_config_path = resolve_path(config["model_config"])
    corpus_path = resolve_path(args.corpus or corpus_config["path"])
    limit = args.limit if args.limit is not None else corpus_config.get("limit")
    batch_size = int(corpus_config.get("batch_size", 50000))
    sample_limit = int(output_config.get("sample_limit", 200))
    progress_interval = int(output_config.get("progress_interval", 500000))
    special_tokens = list(config.get("special_tokens", []))
    use_nfkc = bool(normalization_config.get("nfkc", True)) and not args.no_nfkc
    nfkc_limit = normalization_config.get("limit")
    want_checksum = bool(corpus_config.get("checksum", False)) or args.checksum
    phase = str(output_config.get("phase", PHASE_DEFAULT))

    for path in (tokenizer_dir / "tokenizer.model", model_config_path, corpus_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))

    print(f"[p1b] tokenizer : {tokenizer_dir}")
    print(f"[p1b] corpus    : {corpus_path}")
    print(f"[p1b] limit     : {limit if limit is not None else 'all rows'}")

    encoder = SentencePieceEncoder(tokenizer_dir / "tokenizer.model")
    accumulator = CorpusAccumulator(
        encoder,
        sample_limit=sample_limit,
        special_tokens=special_tokens,
        nfkc=use_nfkc,
        nfkc_limit=nfkc_limit,
    )

    rows = iter_tsv_rows(
        corpus_path,
        text_column=str(corpus_config.get("text_column", "text")),
        duration_column=corpus_config.get("duration_column", "duration"),
        encoding=str(corpus_config.get("encoding", "utf-8")),
        limit=limit,
    )
    next_report = progress_interval
    for batch in batched(rows, batch_size):
        accumulator.update(batch)
        if progress_interval > 0 and accumulator.n_sentences >= next_report:
            print(f"[p1b] {accumulator.n_sentences:,} rows / {time.time() - started:.0f}s", flush=True)
            next_report += progress_interval
    scan_seconds = time.time() - started
    print(f"[p1b] scan done: {accumulator.n_sentences:,} rows in {scan_seconds:.1f}s", flush=True)

    corpus_metrics = accumulator.corpus_metrics()
    coverage = {
        "in_context": in_context_coverage(
            accumulator.char_hist,
            accumulator.fallback_char_hist,
            unk_tokens=accumulator.n_unk_tokens,
        ),
        "isolated": isolated_character_coverage(encoder, accumulator.char_hist),
    }
    prompt_budget = measure_prompt_budget(
        tokenizer_dir,
        model_config,
        probe_texts=list(budget_config.get("probe_texts", [])),
        reference_seconds=float(budget_config.get("reference_seconds", 30.0)),
        max_decode_length=int(budget_config.get("max_decode_length", 750)),
        chars_per_token=corpus_metrics["chars_per_token_mean"],
    )
    probe_strings = [
        {
            "text": text,
            "pieces": encoder.encode_pieces(text),
            "tokens": len(encoder.encode_batch([text])[0]),
        }
        for text in PROBE_STRINGS
    ]

    finished_at = datetime.now().astimezone().isoformat()
    elapsed = time.time() - started
    corpus_stat = corpus_path.stat()
    metrics: dict[str, Any] = {
        "run": {
            "phase": phase,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed,
            "scan_seconds": scan_seconds,
            "corpus_name": corpus_config.get("name", corpus_path.stem),
            "corpus_path": str(corpus_path),
            "limit": limit,
            "tokenizer_dir": str(tokenizer_dir),
            "vocab_size": encoder.vocab_size,
            "extended_vocab_size": int(model_config["architecture"]["extended_vocab_size"]),
            "byte_piece_count": len(encoder.byte_ids),
            "unk_id": encoder.unk_id,
            "nfkc_enabled": use_nfkc,
        },
        "corpus": corpus_metrics,
        "character_coverage": coverage,
        "sentence_character_class": sentence_class_metrics(accumulator),
        "nfkc": accumulator.nfkc_metrics(),
        "prompt_budget": prompt_budget,
        "probe_strings": probe_strings,
    }
    metrics["decision"] = recommend_branch(metrics, dict(config.get("decision", {})))

    run_dir = new_run_dir(
        phase,
        root=resolve_path(args.output_root or output_config.get("root", "artifacts")),
    )
    write_metrics(run_dir, metrics)
    (run_dir / "report.md").write_text(render_report(metrics), encoding="utf-8")
    write_unk_samples(run_dir / "unk_samples.txt", accumulator.samples, encoder)

    corpus_inputs: dict[str, Any] = {
        "path": str(corpus_path),
        "size_bytes": corpus_stat.st_size,
        "mtime": datetime.fromtimestamp(corpus_stat.st_mtime).astimezone().isoformat(),
        "rows_scanned": accumulator.n_sentences,
        "limit": limit,
        "checksum_mode": "sha256" if want_checksum else "size_mtime",
        "sha256": file_checksum(corpus_path) if want_checksum else None,
    }
    tokenizer_inputs = {
        name: file_checksum(tokenizer_dir / name)
        for name in (
            "tokenizer.model",
            "added_tokens.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
        )
        if (tokenizer_dir / name).is_file()
    }
    write_run_metadata(
        run_dir,
        phase=phase,
        command=list(sys.argv),
        seed=None,
        inputs={
            "tokenizer_dir": str(tokenizer_dir),
            "tokenizer_files_sha256": tokenizer_inputs,
            "model_config": {
                "path": str(model_config_path),
                "sha256": file_checksum(model_config_path),
            },
            "corpus": corpus_inputs,
            "config": {
                "path": str(Path(args.config).resolve()),
                "sha256": file_checksum(args.config),
            },
        },
        extra={
            "note": (
                "run.json の started_at は metadata書き込み時刻。実際の開始/終了は "
                "metrics.json の run.started_at / run.finished_at を見ること。"
            ),
            "corpus_checksum_note": (
                "corpusは1.68GB / 7.4M行のためsha256を既定では計算しない。"
                "inputs.json では size_bytes + mtime で同一性を担保する"
                "（--checksum を付ければsha256も記録する）。"
            ),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed,
            "rows_scanned": accumulator.n_sentences,
            "config_path": str(Path(args.config).resolve()),
        },
    )

    decision = metrics["decision"]
    print(f"[p1b] artifacts : {run_dir}")
    print(f"[p1b] rows      : {accumulator.n_sentences:,}")
    print(f"[p1b] char unk  : {coverage['in_context']['char_unk_rate']:.3e}")
    print(f"[p1b] char byte : {coverage['in_context']['char_byte_fallback_rate']:.3e}")
    print(f"[p1b] tok/char  : {corpus_metrics['tokens_per_char_mean']:.4f}")
    print(f"[p1b] decision  : branch {decision['branch']} - {decision['branch_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
