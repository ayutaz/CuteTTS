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

"""日本語継続学習の共通データ入口（JSONL manifest）。

``docs/japanese-training/03-data-and-frontend.md`` 第2節のschemaを、P1dで実際に扱う
2 dataset（gol-dataset / moe-speech-plus）へ絞って確定させたもの。
学習・評価・前処理（latent cache）はすべてこのmanifestを入口にする。

重要な前提（``docs/japanese-training/data-inventory.md`` の確認済み事項）:

* ``speaker_id`` は **声の識別子ではない**。gol は ``SHA-256(キャラクター表示名)[:32]``、
  moe は ``uuid4().hex[:8]`` で、同一声優が別IDになる。zero-shot splitを切るときは
  ``speaker_id`` ではなく :attr:`Utterance.voice_cluster_id`（Speaker Encoder由来）を使う。
* 音声は書庫（tar/zip）の中にあることが多いため、:attr:`Utterance.audio_ref` は
  ``"<archive path>::<member>"`` 形式も取れる文字列にしてある。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Iterator

from cutetts.training.artifacts import file_checksum

__all__ = [
    "ARCHIVE_SEPARATOR",
    "DATASET_IDS",
    "SplitCounts",
    "UNASSIGNED_SPLIT",
    "Utterance",
    "VALIDATION_CODES",
    "ValidationIssue",
    "load_manifest",
    "manifest_checksum",
    "split_audio_ref",
    "summarize",
    "validate",
    "write_manifest",
]

#: manifestが受け付けるdataset ID。これ以外は ``unknown_dataset`` として検出する。
DATASET_IDS: tuple[str, ...] = ("gol", "moe")

#: 書庫内memberを指すときの区切り。``<archive path>::<member>``。
ARCHIVE_SEPARATOR = "::"

#: :func:`summarize` で ``split`` 未設定のrecordをまとめるkey。
UNASSIGNED_SPLIT = "unassigned"

# 省略できない（``to_json`` で必ず出力される）field。
_REQUIRED_FIELDS: tuple[str, ...] = (
    "utterance_id",
    "dataset_id",
    "audio_ref",
    "text_raw",
    "speaker_id",
    "duration",
    "sample_rate",
)


@dataclass(frozen=True)
class Utterance:
    """manifest 1行分。``text_raw`` は監査用の不変fieldとして必ず残す。"""

    utterance_id: str
    """グローバル一意なID。例 ``"gol:<game_id>:<basename>"`` / ``"moe:<uuid>:<n>"``。"""

    dataset_id: str
    """``"gol"`` または ``"moe"``。"""

    audio_ref: str
    """解決可能な音声参照。書庫内なら ``"<archive path>::<member>"``。"""

    text_raw: str
    """元データのテキスト。正規化しても **この値は書き換えない**。"""

    speaker_id: str
    """データセット由来の話者ID。声の識別子ではない（module docstring参照）。"""

    duration: float
    """秒。"""

    sample_rate: int
    """元音声のsample rate（gol 48000 / moe 44100）。学習時は24 kHzへ変換する。"""

    language: str = "ja"

    text_normalized: str | None = None

    reading: str | None = None

    voice_cluster_id: str | None = None
    """Speaker Encoder embeddingのクラスタID。**完全連結**で作る「同じ声」の単位。

    :class:`~cutetts.training.pairing.PairSampler` が reference と target を
    選ぶ単位。クラスタ内の全ペアが閾値を満たすので、別の声が混ざらない。
    """

    split_group_id: str | None = None
    """split を切る単位。**単連結**で作る、より粗いグループ。

    ``voice_cluster_id`` より粗く、1つの split_group が複数の voice_cluster を含む。
    「同じ声かもしれない話者」を1つに寄せることで、同じ声が train と zero-shot へ
    分かれるのを防ぐ（D-015）。

    2種類必要な理由は非対称な失敗にある:

    * PairSampler の単位が粗いと、別の声を reference にして学習する
      （「このreferenceの声で別の声を出せ」）
    * split の単位が細かいと、同じ声が train と zero-shot に現れて
      zero-shot が zero-shot でなくなる

    片方の粒度では両方を同時に満たせないため、単位を分ける。
    """

    quality_score: float | None = None
    """moeの speechMOS 等。gol側は現状 ``None``。"""

    split: str | None = None

    game_id: str | None = None

    source_checksum: str | None = None

    def to_json(self) -> dict:
        """JSON化する。値が ``None`` のfieldは **省略** する（行を短く保つため）。"""
        payload: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            payload[field.name] = value
        return payload

    @classmethod
    def from_json(cls, obj: dict) -> "Utterance":
        """:meth:`to_json` の逆変換。未知のkeyは前方互換のため無視する。"""
        known = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in obj.items() if key in known}
        missing = [name for name in _REQUIRED_FIELDS if name not in kwargs]
        if missing:
            raise ValueError(f"Manifest record is missing required fields: {', '.join(missing)}")
        kwargs["duration"] = float(kwargs["duration"])
        kwargs["sample_rate"] = int(kwargs["sample_rate"])
        if kwargs.get("quality_score") is not None:
            kwargs["quality_score"] = float(kwargs["quality_score"])
        return cls(**kwargs)


def split_audio_ref(audio_ref: str) -> tuple[str, str | None]:
    """``audio_ref`` を ``(container, member)`` に分ける。書庫でなければ member は ``None``。"""
    if ARCHIVE_SEPARATOR in audio_ref:
        container, _, member = audio_ref.partition(ARCHIVE_SEPARATOR)
        return container, member
    return audio_ref, None


def write_manifest(path: str | Path, records: Iterable[Utterance]) -> int:
    """JSONLとして書き出し、書いた件数を返す。``records`` はiteratorでよい。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), ensure_ascii=False, sort_keys=False))
            handle.write("\n")
            count += 1
    return count


def load_manifest(path: str | Path) -> Iterator[Utterance]:
    """JSONLを逐次読む。数百万行を想定するので **全件をメモリに載せない**。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON line ({exc})") from exc
            try:
                yield Utterance.from_json(obj)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc


@dataclass(frozen=True)
class ValidationIssue:
    """1 recordの1問題。1 recordが複数のissueを持ち得る。"""

    code: str
    utterance_id: str
    detail: str


#: :func:`validate` が返し得る ``code`` の全集合。
VALIDATION_CODES: tuple[str, ...] = (
    "empty_text",
    "punctuation_only",
    "markup",
    "name_placeholder",
    "too_short",
    "too_long",
    "duplicate_id",
    "bad_duration",
    "bad_sample_rate",
    "unknown_dataset",
    "generic_speaker",
)

# text_rules.py（別担当が後から追加）に期待する関数名。code -> 関数名。
# いずれも ``str -> bool``（truthyならissue）を想定する。
_TEXT_RULE_FUNCTIONS: tuple[tuple[str, str], ...] = (
    ("punctuation_only", "is_punctuation_only"),
    ("markup", "contains_markup"),
    ("name_placeholder", "contains_name_placeholder"),
)


def _load_text_rules() -> Any | None:
    """``cutetts.training.text_rules`` を遅延importする。未実装なら ``None``。

    text_rules.py はP1dの別担当が後から追加するため、無い状態でも
    :func:`validate` が動くようにしてある（該当codeの判定だけskipされる）。
    """
    try:
        from cutetts.training import text_rules
    except ImportError:
        return None
    return text_rules


def validate(
    records: Iterable[Utterance],
    *,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    generic_speaker_ids: frozenset[str] = frozenset(),
) -> list[ValidationIssue]:
    """全recordを検査してissueを返す。

    Args:
        records: 検査対象。iteratorでよい（ID重複検出のためutterance_idのみ保持する）。
        min_duration: これ未満は ``too_short``（golでは0〜1秒が4.39%）。
        max_duration: これ超は ``too_long``。
        generic_speaker_ids: 総称ラベル話者のID集合。1 IDに複数の声が混在するため
            ``generic_speaker`` として除外候補にする。

    Returns:
        検出順のissue list。問題が無ければ空list。1 recordが複数issueを持ち得る。
    """
    text_rules = _load_text_rules()
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()

    for record in records:
        uid = record.utterance_id

        if uid in seen_ids:
            issues.append(ValidationIssue("duplicate_id", uid, "utterance_id appears more than once"))
        else:
            seen_ids.add(uid)

        if record.dataset_id not in DATASET_IDS:
            issues.append(
                ValidationIssue(
                    "unknown_dataset",
                    uid,
                    f"dataset_id={record.dataset_id!r} is not one of {DATASET_IDS}",
                )
            )

        if record.speaker_id in generic_speaker_ids:
            issues.append(
                ValidationIssue(
                    "generic_speaker",
                    uid,
                    f"speaker_id={record.speaker_id!r} is a generic label shared by multiple voices",
                )
            )

        text = record.text_raw or ""
        if not text.strip():
            issues.append(ValidationIssue("empty_text", uid, "text_raw is empty or whitespace only"))
        elif text_rules is not None:
            for code, func_name in _TEXT_RULE_FUNCTIONS:
                func = getattr(text_rules, func_name, None)
                if func is None:
                    continue
                result = func(text)
                if not result:
                    continue
                detail = f"text_rules.{func_name} matched"
                if not isinstance(result, bool):
                    detail = f"{detail}: {result!r}"
                issues.append(ValidationIssue(code, uid, detail))

        duration = record.duration
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0.0:
            issues.append(
                ValidationIssue("bad_duration", uid, f"duration={duration!r} is not a positive finite number")
            )
        else:
            if duration < min_duration:
                issues.append(ValidationIssue("too_short", uid, f"duration={duration:.3f}s < {min_duration}s"))
            if duration > max_duration:
                issues.append(ValidationIssue("too_long", uid, f"duration={duration:.3f}s > {max_duration}s"))

        sample_rate = record.sample_rate
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            issues.append(
                ValidationIssue("bad_sample_rate", uid, f"sample_rate={sample_rate!r} is not a positive int")
            )

    return issues


def manifest_checksum(path: str | Path) -> str:
    """manifest fileのsha256。run artifactの ``inputs.json`` に記録する。"""
    return file_checksum(path)


@dataclass(frozen=True)
class SplitCounts:
    """split別の件数。"""

    total: int
    by_split: dict[str, int]


def summarize(records: Iterable[Utterance]) -> dict:
    """レポート用の集計。件数・dataset別・split別・話者数・合計時間(hours)を返す。"""
    total = 0
    total_seconds = 0.0
    by_dataset: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    seconds_by_dataset: Counter[str] = Counter()
    speakers: set[tuple[str, str]] = set()
    voice_clusters: set[str] = set()

    for record in records:
        total += 1
        by_dataset[record.dataset_id] += 1
        by_split[record.split if record.split is not None else UNASSIGNED_SPLIT] += 1
        speakers.add((record.dataset_id, record.speaker_id))
        if record.voice_cluster_id is not None:
            voice_clusters.add(record.voice_cluster_id)
        duration = record.duration
        if isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0.0:
            total_seconds += float(duration)
            seconds_by_dataset[record.dataset_id] += float(duration)

    counts = SplitCounts(total=total, by_split=dict(sorted(by_split.items())))
    return {
        "total": counts.total,
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_split": counts.by_split,
        "speakers": len(speakers),
        "voice_clusters": len(voice_clusters),
        "hours": round(total_seconds / 3600.0, 4),
        "hours_by_dataset": {
            dataset: round(seconds / 3600.0, 4) for dataset, seconds in sorted(seconds_by_dataset.items())
        },
    }
