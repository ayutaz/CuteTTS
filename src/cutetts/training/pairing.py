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

"""reference/target ペアのサンプラ（P1d、R-004 leakage対策）。

voice cloning学習の1 sampleは「あるtarget発話」と「同じ声のreference」の組で決まる。
このmoduleはmanifestのrecord列からその組を **決定的に** 作り、
``docs/japanese-training/07-risks-and-decisions.md`` R-004（reference/target leakage）が
挙げる失敗を構造的に潰す。

設計上の前提（``docs/japanese-training/data-inventory.md``「影響 0」「影響 3」）:

* **speaker IDは声の識別子ではない。** gol は ``SHA-256(キャラクター表示名)[:32]``、
  moe は ``uuid4().hex[:8]``。同一声優が別IDに割れるし、総称ラベルでは複数の声が
  1 IDへ混ざる。よってグループ化キーは固定せず、``group_key`` で
  ``"speaker_id"`` と ``"voice_cluster_id"``（P1eのSpeaker Encoder由来）を
  **切り替えられる** ようにしてある。クラスタIDが揃った時点で
  ``group_key="voice_cluster_id"`` に変えるだけで、samplerは声単位になる。
* **1発話をそのままreferenceにすると推論と乖離する。** golの発話は平均5.18秒・
  中央値4.55秒だが、推論側 :func:`cutetts.runtime.prepare_reference_audio` は
  VAE用に先頭30秒を想定している。そこで data-inventory.md の対応案のうち
  **案A（同一グループの複数発話を連結してreferenceにする）** を実装した。
  :class:`PairSampler` は ``target_reference_seconds`` に最も近くなるよう
  ``max_reference_utterances`` 件までを連結する。連結そのもの（音声のconcat）は
  latent cache側（P1e/P2）の仕事で、ここは **どの発話を何件使うか** だけを決める。

leakage対策は3層になっている:

1. reference候補から target と **同じ utterance_id** を必ず除く
2. グループ内の record を utterance_id で重複排除してから使う
   （manifestに重複行があっても reference_group 内に同じ発話が2度入らない）
3. :func:`assert_no_leakage` を学習ループ・artifact書き出しの直前に呼べるようにする

「同一録音の近重複」（別IDだが同じ音源）はこの層では検出できない。それは
voiceクラスタリング（P1e）とmanifestの ``source_checksum`` の担当で、ここでは
``exclude_group_ids`` と ``group_key`` の切り替えで受け止める。
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, fields
from itertools import islice
from typing import Iterable, Iterator

from cutetts.training.manifest import Utterance

__all__ = [
    "DEFAULT_GROUP_KEY",
    "GROUP_KEYS",
    "PairSampler",
    "ReferenceTargetPair",
    "assert_no_leakage",
    "group_size_histogram",
]

#: 既定のグループ化キー。voiceクラスタが揃うまでの暫定値。
DEFAULT_GROUP_KEY = "speaker_id"

#: 実際に使うことを想定しているキー。他のfield名も指定できるが、この2つ以外は用途外。
GROUP_KEYS: tuple[str, ...] = ("speaker_id", "voice_cluster_id")

# group_key として受け付ける値（Utteranceのfield名）。typoを早期に弾くため。
_UTTERANCE_FIELDS: frozenset[str] = frozenset(field.name for field in fields(Utterance))


@dataclass(frozen=True)
class ReferenceTargetPair:
    """1学習sample分の「target発話」と「referenceに使う発話群」。

    ``reference_group`` は **1件以上**。複数件のときは、その順に連結して1本の
    referenceにする想定（案A）。順序はsamplerが決めた順のまま保持する。
    """

    target: Utterance
    """再構成対象の発話。この音声とテキストがlossの教師になる。"""

    reference_group: tuple[Utterance, ...]
    """referenceに使う発話（1件以上）。この順に連結する。targetは絶対に含まれない。"""

    group_id: str
    """サンプリングに使ったグループキーの値（speaker_id か voice_cluster_id）。"""

    def __post_init__(self) -> None:
        if not self.reference_group:
            raise ValueError("reference_group must contain at least one utterance.")

    @property
    def reference_seconds(self) -> float:
        """``reference_group`` のduration合計（秒）。連結後のreference長の見積り。"""
        return float(sum(float(record.duration) for record in self.reference_group))

    def to_json(self) -> dict:
        """pair provenanceをartifactへ残すための最小表現（P1dの判断ゲート R-004）。"""
        return {
            "group_id": self.group_id,
            "target_id": self.target.utterance_id,
            "reference_ids": [record.utterance_id for record in self.reference_group],
            "reference_seconds": round(self.reference_seconds, 4),
        }


def _group_value(record: Utterance, group_key: str) -> str | None:
    """``record`` のグループキー値。未設定（None / 空白のみ）なら ``None``。"""
    value = getattr(record, group_key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{record.utterance_id}: {group_key}={value!r} must be a string or None."
        )
    if not value.strip():
        return None
    return value


def _check_group_key(group_key: str) -> None:
    """``group_key`` が :class:`Utterance` のfield名かを検査する（typoの早期検出）。"""
    if group_key not in _UTTERANCE_FIELDS:
        raise ValueError(
            f"group_key={group_key!r} is not a field of Utterance. "
            f"Expected one of {sorted(_UTTERANCE_FIELDS)}."
        )


def _build_groups(records: Iterable[Utterance], group_key: str) -> dict[str, list[Utterance]]:
    """``group_key`` でグループ化する。

    * キー値が ``None`` / 空白のみのrecordは落とす（``voice_cluster_id`` 未付与の行など）
    * ``utterance_id`` が重複するrecordは **最初の1件だけ** 残す
      （重複行の報告は :func:`cutetts.training.manifest.validate` の担当。
      ここでは reference_group 内に同じ発話が2度入るのを防ぐのが目的）
    * グループ内は ``utterance_id`` 昇順に整列する。これで入力の並び順が変わっても
      サンプリング結果が変わらない
    """
    _check_group_key(group_key)

    groups: dict[str, list[Utterance]] = {}
    seen_ids: set[str] = set()
    for record in records:
        value = _group_value(record, group_key)
        if value is None:
            continue
        if record.utterance_id in seen_ids:
            continue
        seen_ids.add(record.utterance_id)
        groups.setdefault(value, []).append(record)

    for members in groups.values():
        members.sort(key=lambda record: record.utterance_id)
    return groups


def group_size_histogram(
    records: Iterable[Utterance],
    *,
    group_key: str = DEFAULT_GROUP_KEY,
) -> dict[int, int]:
    """「グループの発話数」→「そのサイズのグループ数」のヒストグラム。

    ``min_utterances_per_group`` や ``exclude_group_ids`` を決めるための素の分布なので、
    :class:`PairSampler` の除外条件は **適用しない**（キー未設定と重複IDだけ落とす）。

    Returns:
        サイズ昇順のdict。例 ``{1: 2, 3: 1}`` は「1発話のグループが2つ、3発話が1つ」。
    """
    sizes = Counter(len(members) for members in _build_groups(records, group_key).values())
    return dict(sorted(sizes.items()))


def assert_no_leakage(pairs: Iterable[ReferenceTargetPair]) -> None:
    """pairがleakageしていないことを検査する。学習ループ・artifact出力の直前に呼ぶ。

    Raises:
        AssertionError: 次のいずれかのとき。

            * ``target.utterance_id`` が自分の ``reference_group`` に含まれる
            * ``reference_group`` 内に重複した ``utterance_id`` がある
    """
    for pair in pairs:
        reference_ids = [record.utterance_id for record in pair.reference_group]
        target_id = pair.target.utterance_id
        if target_id in reference_ids:
            raise AssertionError(
                f"reference/target leakage: target {target_id!r} appears in its own "
                f"reference_group (group_id={pair.group_id!r})."
            )
        if len(set(reference_ids)) != len(reference_ids):
            duplicated = sorted({uid for uid in reference_ids if reference_ids.count(uid) > 1})
            raise AssertionError(
                f"reference_group of target {target_id!r} repeats utterance(s) "
                f"{duplicated} (group_id={pair.group_id!r})."
            )


class PairSampler:
    """manifest recordから reference/target ペアを決定的にサンプリングする。

    サンプリングは **復元抽出**（i.i.d.）。学習streamをそのまま回す用途を想定していて、
    :meth:`sample` は :meth:`iter_pairs` の先頭 ``n`` 件と厳密に一致する。
    ``seed`` が同じなら呼ぶたびに同じ列が出る（RNGは呼び出しごとに作り直す）。
    """

    def __init__(
        self,
        records: Iterable[Utterance],
        *,
        seed: int,
        group_key: str = DEFAULT_GROUP_KEY,
        min_utterances_per_group: int = 2,
        target_reference_seconds: float = 10.0,
        max_reference_utterances: int = 8,
        exclude_group_ids: frozenset[str] = frozenset(),
    ) -> None:
        """
        Args:
            records: manifestのrecord列。iteratorでよい（内部で1度だけ実体化する）。
            seed: 乱数seed。artifactの ``run.json`` に記録すること。
            group_key: グループ化に使う :class:`Utterance` のfield名。
                ``"speaker_id"``（既定）か ``"voice_cluster_id"``。
                voiceクラスタが揃ったら後者へ切り替える。
            min_utterances_per_group: これ未満のグループは使わない。**2以上必須**
                （1にするとtarget以外のreference候補が無くなり、leakage無しでは
                pairを作れないため）。
            target_reference_seconds: 連結後のreferenceの目標長（秒）。
                推論側の30秒想定と実発話長5秒前後の乖離を埋めるための値（案A）。
            max_reference_utterances: 1 referenceに連結する発話の上限。
            exclude_group_ids: 使わないグループキー値（総称ラベル話者など）。

        Raises:
            ValueError: 引数が不正なとき（``group_key`` がfield名でない、
                ``min_utterances_per_group < 2`` など）。
        """
        _check_group_key(group_key)
        if min_utterances_per_group < 2:
            raise ValueError(
                "min_utterances_per_group must be >= 2: a group with a single utterance "
                "cannot provide a reference that differs from the target."
            )
        if not target_reference_seconds > 0.0:
            raise ValueError("target_reference_seconds must be positive.")
        if max_reference_utterances < 1:
            raise ValueError("max_reference_utterances must be >= 1.")

        self.seed = int(seed)
        self.group_key = group_key
        self.min_utterances_per_group = int(min_utterances_per_group)
        self.target_reference_seconds = float(target_reference_seconds)
        self.max_reference_utterances = int(max_reference_utterances)
        self.exclude_group_ids: frozenset[str] = frozenset(exclude_group_ids)

        groups = _build_groups(records, group_key)
        self._groups: dict[str, list[Utterance]] = {
            group_id: members
            for group_id, members in sorted(groups.items())
            if group_id not in self.exclude_group_ids
            and len(members) >= self.min_utterances_per_group
        }
        self._group_ids: tuple[str, ...] = tuple(self._groups)
        # 発話一様サンプリング用のflat index。(group_id, グループ内index)。
        self._flat: tuple[tuple[str, int], ...] = tuple(
            (group_id, index)
            for group_id in self._group_ids
            for index in range(len(self._groups[group_id]))
        )

    def eligible_groups(self) -> dict[str, list[Utterance]]:
        """サンプリング対象のグループ。``group_id`` 昇順、グループ内は utterance_id 昇順。

        ``min_utterances_per_group`` 未満のグループと ``exclude_group_ids`` を除外したもの。
        ``group_key`` の値が ``None``（や空文字）の record は除外する。

        Returns:
            呼び出しごとの新しいdict/list（呼び出し側が壊してもsamplerは壊れない）。
        """
        return {group_id: list(members) for group_id, members in self._groups.items()}

    def sample(self, n: int) -> list[ReferenceTargetPair]:
        """``n`` 件のペアを返す。同じ引数・同じseedなら常に同じ結果。

        :meth:`iter_pairs` の先頭 ``n`` 件（``speaker_uniform=True``）と一致する。
        復元抽出なので、``n`` が発話数を超えても同じtargetが再登場するだけで問題ない。
        """
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return []
        return list(islice(self.iter_pairs(), n))

    def iter_pairs(self, *, speaker_uniform: bool = True) -> Iterator[ReferenceTargetPair]:
        """ペアを無限に生成する。**終端は無い** ので ``islice`` などで打ち切ること。

        Args:
            speaker_uniform: ``True`` ならまずグループを一様に選び、その中からtargetを選ぶ
                （発話数の多い話者へ偏らせない）。``False`` なら全発話から一様に選ぶ
                （データ分布そのまま。発話数の多い話者が優勢になる）。

        Raises:
            ValueError: 対象グループが1つも無いとき。設定ミスを黙って空streamにしない。
        """
        if not self._group_ids:
            raise ValueError(
                "No eligible groups: check group_key, min_utterances_per_group and "
                "exclude_group_ids against the manifest."
            )
        return self._iter_pairs(speaker_uniform)

    def _iter_pairs(self, speaker_uniform: bool) -> Iterator[ReferenceTargetPair]:
        rng = random.Random(self.seed)
        group_ids = self._group_ids
        flat = self._flat
        while True:
            if speaker_uniform:
                group_id = group_ids[rng.randrange(len(group_ids))]
                members = self._groups[group_id]
                target_index = rng.randrange(len(members))
            else:
                group_id, target_index = flat[rng.randrange(len(flat))]
                members = self._groups[group_id]
            yield self._build_pair(group_id, members, target_index, rng)

    def _build_pair(
        self,
        group_id: str,
        members: list[Utterance],
        target_index: int,
        rng: random.Random,
    ) -> ReferenceTargetPair:
        """targetを除く候補をshuffleし、目標長へ最も近くなるところまで連結する（案A）。"""
        target = members[target_index]
        candidates = [
            record
            for index, record in enumerate(members)
            if index != target_index and record.utterance_id != target.utterance_id
        ]
        rng.shuffle(candidates)

        chosen: list[Utterance] = []
        total = 0.0
        for candidate in candidates:
            if len(chosen) >= self.max_reference_utterances:
                break
            # 1件目は無条件に採る（reference_groupは必ず1件以上）。
            if chosen and total >= self.target_reference_seconds:
                break
            chosen.append(candidate)
            total += float(candidate.duration)

        # 最後の1件で目標を大きく行き過ぎたなら外す（外した方が目標長へ近い場合のみ）。
        if len(chosen) >= 2:
            without = total - float(chosen[-1].duration)
            if abs(without - self.target_reference_seconds) < abs(
                total - self.target_reference_seconds
            ):
                chosen.pop()

        return ReferenceTargetPair(
            target=target,
            reference_group=tuple(chosen),
            group_id=group_id,
        )
