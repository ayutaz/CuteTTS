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

"""speaker IDではなく「声」でsplitを切るためのvoiceクラスタリング（D-015 / R-004）。

解いている問題（``docs/japanese-training/data-inventory.md`` 「影響 0」で確認済み）:

* gol-dataset の ``speaker`` は ``SHA-256(キャラクター表示名)[:32]``。
  同名異キャラが1 IDへ統合され、総称ラベル（『女の子』等）は
  **複数の声が1 IDに混在** する。
* moe-speech-plus の話者IDは ``uuid4().hex[:8]`` のランダム値で、READMEに
  「同一声優・同一キャラでも別IDを割り当てる」と明記されている。

どちらも **speaker-disjoint splitがvoice-actor-disjointを保証しない** ため、
zero-shot評価が楽観側にバイアスする。対策として、frozenの公式Speaker Encoder
（ECAPA student, 16 kHz -> 256-dim）のembeddingで

1. speaker IDごとの重心（:class:`SpeakerProfile`）を作り、
2. 重心間のcosine類似度で単連結クラスタリングし（:func:`cluster_speakers`）、
3. splitの単位をIDではなく **voiceクラスタ** へ移す（:func:`assign_clusters` が
   :attr:`cutetts.training.manifest.Utterance.voice_cluster_id` を埋める）。

同時に、話者内のばらつき（:attr:`SpeakerProfile.dispersion`）で総称ラベル話者を
自動検出する（:func:`suspicious_speakers`、D-016）。

このmoduleはembeddingを **生成しない**。P1eのspeaker embedding cache
（``read(utterance_id) -> [256]`` と ``__contains__`` を持つreader）を入力に取るだけなので、
torchにもcheckpointにも依存せず、numpyだけで決定的に動く。

閾値について（**提案値であり、実データで再校正すること**）:

``threshold=0.70`` と ``dispersion_threshold=0.35`` は初期値にすぎない。
特にdispersionは、1 IDに等量の2声（相互cosine ``s``）が混ざったとき

    dispersion = 1 - sqrt((1 + s) / 2)

にしかならない（``s = 0`` でも 0.293）。つまり既定 0.35 は「3声以上の混在」を狙った
強めの閾値で、2声混在まで拾うには 0.25 前後へ下げる必要がある。実データの分布を
見てからP1dで確定する（D-015 / D-016）。
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol, Sequence

import numpy as np

from cutetts.training.manifest import Utterance

__all__ = [
    "DEFAULT_CLUSTER_THRESHOLD",
    "DEFAULT_LINKAGE",
    "LINKAGE_MODES",
    "cluster_cohesion",
    "DEFAULT_MAX_ZERO_SHOT_SHARE",
    "DEFAULT_SEEN_FRACTION",
    "DEFAULT_ZERO_SHOT_FRACTION",
    "SPLIT_GROUP_PREFIX",
    "assign_split_groups",
    "assign_splits_by_cluster",
    "cross_split_similarity",
    "split_leakage",
    "DEFAULT_DISPERSION_THRESHOLD",
    "DEFAULT_MAX_SAMPLES_PER_SPEAKER",
    "SPEAKER_EMBEDDING_DIM",
    "SpeakerEmbeddingReader",
    "SpeakerProfile",
    "VOICE_CLUSTER_PREFIX",
    "assign_clusters",
    "build_profiles",
    "cluster_members",
    "cluster_speakers",
    "cluster_summary",
    "cosine_similarity_matrix",
    "suspicious_speakers",
]

#: 公式Speaker Encoder（ECAPA student）の出力次元。参考値で、強制はしない。
SPEAKER_EMBEDDING_DIM = 256

#: :func:`cluster_speakers` の既定閾値（提案値。module docstring参照）。
DEFAULT_CLUSTER_THRESHOLD = 0.70
_MAX_AGGLOMERATIVE_SPEAKERS = 30_000
"""平均/完全連結は [N,N] を実体化する。N=30,000 で約1.8 GB。"""

#: :func:`suspicious_speakers` の既定閾値（提案値。module docstring参照）。
DEFAULT_DISPERSION_THRESHOLD = 0.35

#: 1 speakerあたりに読むembeddingの上限（既定）。
DEFAULT_MAX_SAMPLES_PER_SPEAKER = 32

#: cluster_idのprefix。``"vc:<クラスタ内で辞書順最小のspeaker_id>"`` になる。
VOICE_CLUSTER_PREFIX = "vc:"

# 類似度をblockで計算するときの行数。N=19,349（gol実測）でも
# 512 x 19,349 x 4 byte = 約40 MBに収まり、[N,N]を実体化せずに済む。
_SIMILARITY_BLOCK_ROWS = 512

LINKAGE_MODES = ("average", "complete", "single")
DEFAULT_LINKAGE = "complete"
"""既定は完全連結。**クラスタ内の全ペアが閾値以上**であることを保証する。

単連結（``"single"``）は **推移的に併合する**ため、A~B と B~C が閾値を満たせば
cos(A,C) が低くても A,B,C が1つになる。話者数が増えるとこの連鎖が効き、
S1の実データ（1,110話者）では **45話者・86.1時間・trainの30.5%** が
1クラスタになった。そのクラスタのcentroid間で cos>=0.92 を満たすペアは
3.4%（最小0.614）しかなく、大半が別の声だった。

voiceクラスタは :class:`~cutetts.training.pairing.PairSampler` が
reference と target を選ぶ単位なので、別の声が混ざると
**「このreferenceの声で別の声を出せ」と教えることになる**。
S1の実データではペアの34.8%がこの状態だった。

完全連結なら「閾値未満のペアが同居しない」ことが構成上保証される。
S1実測（閾値0.92、1,110話者）:

===================== ========== ============ ==============
linkage               クラスタ数  最大話者数   ペア不一致
===================== ========== ============ ==============
single                     974           42        26.9%
average                  1,006            4        13.1%
complete                 1,021            2         8.3%
===================== ========== ============ ==============

完全連結の残る8.3%は全ペアが cos>=0.92 なので「別IDの同じ声」であり、
D-015が意図した動作。分割しすぎる副作用は
:attr:`~cutetts.training.manifest.Utterance.split_group_id`
（単連結の粗いグループ）がsplit側で吸収する。
"""

# ゼロ長ベクトルを弾く閾値。
_NORM_EPS = 1e-12


class SpeakerEmbeddingReader(Protocol):
    """P1eのspeaker embedding cache readerに期待する最小の契約。

    ``SpeakerEmbeddingCacheReader`` 互換ならよく、実装への依存はない
    （テストではdictを包んだfakeを渡す）。
    """

    def __contains__(self, utterance_id: str) -> bool:  # pragma: no cover - Protocol定義
        ...

    def read(self, utterance_id: str) -> "np.ndarray":  # pragma: no cover - Protocol定義
        ...


@dataclass(frozen=True)
class SpeakerProfile:
    """speaker ID 1つぶんの「声」の要約。"""

    speaker_id: str
    """データセット由来の話者ID（声の識別子ではない）。"""

    centroid: "np.ndarray"
    """L2正規化済みの重心embedding ``[D]``（公式encoderなら ``D = 256``）。"""

    n_samples: int
    """重心の計算に **実際に使った** embedding数（``max_samples_per_speaker`` 以下）。"""

    dispersion: float
    """話者内のばらつき。``1 - mean(cosine(各サンプル, centroid))``。

    0に近いほど1つの声。大きいほど1 IDに複数の声が混ざっている疑いが強い
    （総称ラベルの検出に使う。:func:`suspicious_speakers`）。
    ``n_samples == 1`` のときは定義上 ``0.0``。
    """


def _l2_normalize(vector: "np.ndarray") -> "np.ndarray | None":
    """L2正規化する。ノルムが0（相当）または非有限なら ``None``。"""
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= _NORM_EPS:
        return None
    return vector / norm


def _speaker_rng(seed: int, speaker_id: str) -> "np.random.Generator":
    """speaker IDごとの独立なRNG。``(seed, speaker_id)`` だけで決まる。

    speaker単位に分けるのは、選択結果を他話者の件数や登場順に依存させないため。
    """
    digest = hashlib.sha256(f"{int(seed)}:{speaker_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


class _Reservoir:
    """1 speakerぶんのreservoir sampling（Algorithm R）。"""

    __slots__ = ("capacity", "items", "rng", "seen")

    def __init__(self, rng: "np.random.Generator", capacity: int) -> None:
        self.rng = rng
        self.capacity = capacity
        self.items: list[str] = []
        self.seen = 0

    def offer(self, utterance_id: str) -> None:
        if len(self.items) < self.capacity:
            self.items.append(utterance_id)
        else:
            index = int(self.rng.integers(0, self.seen + 1))
            if index < self.capacity:
                self.items[index] = utterance_id
        self.seen += 1


def build_profiles(
    reader: SpeakerEmbeddingReader,
    records: Iterable[Utterance],
    *,
    max_samples_per_speaker: int = DEFAULT_MAX_SAMPLES_PER_SPEAKER,
    seed: int = 0,
) -> dict[str, SpeakerProfile]:
    """speaker IDごとの :class:`SpeakerProfile` を作る。

    Args:
        reader: speaker embedding cache（``read(utterance_id) -> [D]`` と ``in``）。
        records: manifestのrecord。iteratorでよい（全件をメモリに載せない）。
        max_samples_per_speaker: 1 speakerから読むembeddingの上限。超過分は
            reservoir samplingで **seed固定の決定的な選択** を行う。
            ``reader.read`` の呼び出しは1 speakerあたりこの数を超えない。
        seed: サンプリングseed。同じ ``seed`` と同じrecord列なら結果は完全に一致する。

    Returns:
        ``speaker_id -> SpeakerProfile``。cacheに1件も無かったspeakerと、
        使えるembeddingが1件も無かったspeakerは **含まれない**。

    Raises:
        ValueError: ``max_samples_per_speaker`` が1未満のとき、embeddingが1次元でないとき、
            次元がrecord間で揃っていないとき。

    Notes:
        cacheに無いutteranceは ``__contains__`` で除外するため ``read`` されない。
        ``in`` がTrueなのに ``read`` が失敗する場合はcacheの不整合なので、例外は
        そのまま伝播させる（黙って捨てない）。
    """
    if max_samples_per_speaker < 1:
        raise ValueError("max_samples_per_speaker must be >= 1.")

    reservoirs: dict[str, _Reservoir] = {}
    for record in records:
        utterance_id = record.utterance_id
        if utterance_id not in reader:
            continue
        speaker_id = record.speaker_id
        reservoir = reservoirs.get(speaker_id)
        if reservoir is None:
            reservoir = _Reservoir(_speaker_rng(seed, speaker_id), max_samples_per_speaker)
            reservoirs[speaker_id] = reservoir
        reservoir.offer(utterance_id)

    profiles: dict[str, SpeakerProfile] = {}
    dim: int | None = None
    for speaker_id in sorted(reservoirs):
        vectors: list[np.ndarray] = []
        # 読み出し順もsortして固定する（reservoirのslot位置に浮動小数の和を依存させない）。
        for utterance_id in sorted(reservoirs[speaker_id].items):
            embedding = np.asarray(reader.read(utterance_id), dtype=np.float64)
            if embedding.ndim != 1:
                raise ValueError(
                    f"Speaker embedding for {utterance_id!r} must be 1-D, "
                    f"got shape {tuple(embedding.shape)}."
                )
            if dim is None:
                dim = int(embedding.shape[0])
            elif int(embedding.shape[0]) != dim:
                raise ValueError(
                    f"Speaker embedding dim mismatch for {utterance_id!r}: "
                    f"{embedding.shape[0]} != {dim}."
                )
            unit = _l2_normalize(embedding)
            if unit is None:
                continue
            vectors.append(unit)

        if not vectors:
            continue

        stacked = np.stack(vectors)
        centroid = _l2_normalize(stacked.mean(axis=0))
        if centroid is None:
            # 完全に打ち消し合う場合（例: v と -v だけ）。重心が定義できないのでprofileを作らない。
            continue
        dispersion = float(1.0 - float(np.mean(stacked @ centroid)))
        profiles[speaker_id] = SpeakerProfile(
            speaker_id=speaker_id,
            centroid=centroid.astype(np.float32),
            n_samples=len(vectors),
            dispersion=dispersion,
        )
    return profiles


def _stack_centroids(profiles: dict[str, SpeakerProfile]) -> tuple[list[str], "np.ndarray"]:
    """``(speaker_id昇順のID list, [N,D] の正規化済みfloat32行列)``。"""
    speaker_ids = sorted(profiles)
    if not speaker_ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    matrix = np.stack(
        [np.asarray(profiles[speaker_id].centroid, dtype=np.float32).reshape(-1) for speaker_id in speaker_ids]
    )
    return speaker_ids, matrix


def cosine_similarity_matrix(profiles: dict[str, SpeakerProfile]) -> tuple[list[str], "np.ndarray"]:
    """重心間のcosine類似度行列を返す。

    Returns:
        ``(speaker_id昇順のlist, [N,N] のfloat32行列)``。行列は対称で対角は厳密に1.0、
        値は ``[-1, 1]`` にclipされる。``profiles`` が空なら ``([], shape (0,0) の配列)``。

    Notes:
        ``[N,N]`` を実体化するので大規模（gol実測 N = 19,349 なら約1.5 GB）では重い。
        クラスタリングだけが目的なら、行列を作らない :func:`cluster_speakers` を使う。
    """
    speaker_ids, matrix = _stack_centroids(profiles)
    if not speaker_ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)
    similarity = (similarity + similarity.T) * 0.5  # 浮動小数由来の非対称を潰す
    np.fill_diagonal(similarity, 1.0)
    return speaker_ids, similarity.astype(np.float32)


class _UnionFind:
    """path compression + union by size。併合順に依存しない連結成分を返す。"""

    __slots__ = ("parent", "size")

    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, node: int) -> int:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def _agglomerative_groups(matrix: "np.ndarray", threshold: float, linkage: str) -> list[list[int]]:
    """平均連結／完全連結で凝集型クラスタリングし、メンバのindex集合を返す。

    ``matrix`` は行が単位ベクトルの重心 ``[N, D]``。

    平均連結のクラスタ間類似度は、単位ベクトルの **和** から厳密に計算できる::

        mean_{a in A, b in B} cos(a, b) = (sum_A . sum_B) / (|A| * |B|)

    完全連結は Lance-Williams の更新で ``sim(C, X) = min(sim(A, X), sim(B, X))``。
    どちらも「閾値を満たすペアが無くなったら停止」する。
    """
    count = matrix.shape[0]
    if count > _MAX_AGGLOMERATIVE_SPEAKERS:
        raise ValueError(
            f"{linkage} linkage は [N,N] を実体化するため N<={_MAX_AGGLOMERATIVE_SPEAKERS} "
            f"を想定している（N={count}）。大規模では linkage='single' を使うこと。"
        )
    similarity = np.clip(matrix @ matrix.T, -1.0, 1.0).astype(np.float32)
    similarity = (similarity + similarity.T) * 0.5
    np.fill_diagonal(similarity, -np.inf)

    sums = matrix.astype(np.float32).copy()
    sizes = np.ones(count, dtype=np.float32)
    active = np.ones(count, dtype=bool)
    groups: list[list[int]] = [[index] for index in range(count)]

    while True:
        flat = int(np.argmax(similarity))
        left, right = divmod(flat, count)
        if not np.isfinite(similarity[left, right]) or similarity[left, right] < threshold:
            break
        # right を left へ併合する
        groups[left].extend(groups[right])
        groups[right] = []
        sums[left] += sums[right]
        sizes[left] += sizes[right]
        active[right] = False
        similarity[right, :] = -np.inf
        similarity[:, right] = -np.inf

        if linkage == "average":
            updated = (sums @ sums[left]) / (sizes * sizes[left])
        else:  # complete
            updated = np.minimum(similarity[left], similarity[right])
        updated[~active] = -np.inf
        updated[left] = -np.inf
        similarity[left, :] = updated
        similarity[:, left] = updated

    return [members for members in groups if members]


def cluster_speakers(
    profiles: dict[str, SpeakerProfile],
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    linkage: str = DEFAULT_LINKAGE,
) -> dict[str, str]:
    """重心のcosine類似度でクラスタリングする。

    Args:
        profiles: :func:`build_profiles` の結果。
        threshold: この値 **以上** の類似度を持つクラスタ同士を併合する。
        linkage: 併合条件。``"average"``（既定）はクラスタ間の平均cos、
            ``"complete"`` は最小cos、``"single"`` は1ペアでも閾値を満たせば併合。
            単連結は推移的なので連鎖で巨大クラスタを作る（:data:`DEFAULT_LINKAGE` 参照）。

    Returns:
        ``speaker_id -> cluster_id``。``profiles`` の全speakerが必ず含まれ、
        どこにも併合されなかったspeakerは自分1人のクラスタになる。

    Raises:
        ValueError: ``threshold`` が非有限のとき、``linkage`` が未知のとき。

    Notes:
        cluster_idは ``"vc:<クラスタ内で辞書順最小のspeaker_id>"``。同じ入力なら
        併合順に依存せず常に同じIDになる。``threshold`` を下げると併合が進み、
        上げると分割が細かくなる（partitionは refinement 関係）。
        単連結は ``[N,N]`` を実体化せずblockごとに計算するが、
        平均／完全連結は ``[N,N]`` を持つ（:data:`_MAX_AGGLOMERATIVE_SPEAKERS` 参照）。
    """
    if not np.isfinite(threshold):
        raise ValueError("threshold must be a finite number.")
    if linkage not in LINKAGE_MODES:
        raise ValueError(f"linkage must be one of {LINKAGE_MODES}, got {linkage!r}")

    speaker_ids, matrix = _stack_centroids(profiles)
    count = len(speaker_ids)
    if count == 0:
        return {}

    if linkage != "single":
        labels: dict[str, str] = {}
        for members in _agglomerative_groups(matrix, threshold, linkage):
            cluster_id = min(speaker_ids[index] for index in members)
            for index in members:
                labels[speaker_ids[index]] = f"{VOICE_CLUSTER_PREFIX}{cluster_id}"
        return labels

    union_find = _UnionFind(count)
    for start in range(0, count, _SIMILARITY_BLOCK_ROWS):
        stop = min(start + _SIMILARITY_BLOCK_ROWS, count)
        block = matrix[start:stop] @ matrix.T
        rows, cols = np.nonzero(block >= threshold)
        for row, col in zip(rows.tolist(), cols.tolist()):
            left = start + row
            if col > left:  # 上三角だけ見れば連結成分は変わらない
                union_find.union(left, col)

    # 連結成分ごとに、辞書順最小のspeaker_idをcluster_idにする。
    root_label: dict[int, str] = {}
    for index, speaker_id in enumerate(speaker_ids):
        root = union_find.find(index)
        current = root_label.get(root)
        if current is None or speaker_id < current:
            root_label[root] = speaker_id
    return {
        speaker_id: f"{VOICE_CLUSTER_PREFIX}{root_label[union_find.find(index)]}"
        for index, speaker_id in enumerate(speaker_ids)
    }


def suspicious_speakers(
    profiles: dict[str, SpeakerProfile],
    *,
    dispersion_threshold: float = DEFAULT_DISPERSION_THRESHOLD,
) -> list[str]:
    """1 IDに複数の声が混ざっている疑いのあるspeaker IDを返す（総称ラベル検出、D-016）。

    Args:
        profiles: :func:`build_profiles` の結果。
        dispersion_threshold: :attr:`SpeakerProfile.dispersion` がこの値を
            **超える** speakerを疑わしいとみなす。

    Returns:
        speaker_idの昇順list（決定的）。

    Notes:
        ``n_samples == 1`` のspeakerはdispersionが常に0なので決して載らない。
        サンプル数が少ないほど過小評価になる（``max_samples_per_speaker`` を小さくすると
        検出漏れが増える）点にも注意する。
    """
    return sorted(
        speaker_id for speaker_id, profile in profiles.items() if profile.dispersion > dispersion_threshold
    )


def assign_clusters(records: Iterable[Utterance], mapping: dict[str, str]) -> Iterator[Utterance]:
    """``voice_cluster_id`` を埋めたrecordをyieldする。

    Args:
        records: manifestのrecord（iteratorでよい）。
        mapping: :func:`cluster_speakers` の ``speaker_id -> cluster_id``。

    Yields:
        ``mapping`` にspeaker_idがあれば ``voice_cluster_id`` を差し替えたrecord、
        無ければ **元のrecordをそのまま**（``voice_cluster_id`` は ``None`` のまま）。
        cacheに音声が無くprofileを作れなかったspeakerがここに落ちるので、zero-shot splitを
        切る側は ``voice_cluster_id is None`` のrecordを必ず扱うこと。
    """
    for record in records:
        cluster_id = mapping.get(record.speaker_id)
        if cluster_id is None:
            yield record
        else:
            yield dataclasses.replace(record, voice_cluster_id=cluster_id)


def cluster_members(mapping: dict[str, str]) -> dict[str, list[str]]:
    """``cluster_id -> speaker_id昇順list``。cluster_idの昇順で並ぶ。"""
    members: dict[str, list[str]] = {}
    for speaker_id in sorted(mapping):
        members.setdefault(mapping[speaker_id], []).append(speaker_id)
    return {cluster_id: members[cluster_id] for cluster_id in sorted(members)}


def cluster_summary(mapping: dict[str, str]) -> dict:
    """クラスタリング結果の集計（artifactの ``metrics.json`` にそのまま入る形）。

    Returns:
        ``speakers`` / ``clusters`` / ``largest_cluster_size`` / ``largest_cluster_id`` /
        ``multi_speaker_clusters``（2人以上を含むクラスタ数）/ ``singleton_clusters`` /
        ``speakers_in_multi_speaker_clusters`` / ``size_histogram``
        （クラスタサイズ -> クラスタ数。keyはJSON化のためstr）。
        ``mapping`` が空なら件数はすべて0で ``largest_cluster_id`` は ``None``。

    Notes:
        ``multi_speaker_clusters`` が0なら「別IDの同一声」を1件も検出できていない、
        つまり閾値が高すぎるかサンプル数が足りない可能性がある（R-004）。
    """
    sizes: Counter[str] = Counter(mapping.values())
    if not sizes:
        return {
            "speakers": 0,
            "clusters": 0,
            "largest_cluster_size": 0,
            "largest_cluster_id": None,
            "multi_speaker_clusters": 0,
            "singleton_clusters": 0,
            "speakers_in_multi_speaker_clusters": 0,
            "size_histogram": {},
        }

    largest_size = max(sizes.values())
    # 同サイズが複数あるときは辞書順最小のcluster_idを採る（決定的にするため）。
    largest_id = min(cluster_id for cluster_id, size in sizes.items() if size == largest_size)
    histogram: Counter[int] = Counter(sizes.values())
    return {
        "speakers": len(mapping),
        "clusters": len(sizes),
        "largest_cluster_size": largest_size,
        "largest_cluster_id": largest_id,
        "multi_speaker_clusters": sum(1 for size in sizes.values() if size >= 2),
        "singleton_clusters": sum(1 for size in sizes.values() if size == 1),
        "speakers_in_multi_speaker_clusters": sum(size for size in sizes.values() if size >= 2),
        "size_histogram": {str(size): count for size, count in sorted(histogram.items())},
    }


# ------------------------------------------------------- クラスタ単位のsplit

DEFAULT_ZERO_SHOT_FRACTION = 0.12
DEFAULT_SEEN_FRACTION = 0.05
DEFAULT_MAX_ZERO_SHOT_SHARE = 0.05
"""1クラスタがzero-shot splitへ持ち込める発話数の上限（全体に対する割合）。

単連結クラスタリングは連鎖で巨大クラスタを作ることがある（S1では45話者・
90.7時間・全体の28%が1クラスタになった）。それがzero-shotへ落ちると
split構成が壊れるため、大きすぎるクラスタはtrain側へ固定する。
"""


def _bucket(seed: int, key: str) -> float:
    """seedとキーから [0,1) の決定的な値を作る。"""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_splits_by_cluster(
    cluster_ids: Sequence[str],
    *,
    seed: int,
    zero_shot_fraction: float = DEFAULT_ZERO_SHOT_FRACTION,
    seen_fraction: float = DEFAULT_SEEN_FRACTION,
    max_zero_shot_share: float = DEFAULT_MAX_ZERO_SHOT_SHARE,
) -> list[str]:
    """発話ごとの ``voice_cluster_id`` 列から split を決める。

    **1つのクラスタは必ず1つのsplitにだけ現れる。** speaker_id単位で切ると
    「別IDの同じ声」がtrainとzero-shotへ分かれ、zero-shotがzero-shotで
    なくなる（D-015）。S1の実データでは 21クラスタ・73,001発話（37.1%）が
    この漏れに該当していた。

    Args:
        cluster_ids: 発話と同じ並びの ``voice_cluster_id``。
        seed: 決定的な割り当てのためのseed。
        zero_shot_fraction: zero-shotへ回すクラスタの割合（dev/testで折半）。
        seen_fraction: train クラスタ内で seen 評価へ回す発話の割合。
        max_zero_shot_share: 1クラスタがzero-shotへ持ち込める発話数の上限
            （全体に対する割合）。これを超えるクラスタはtrainへ固定する。

    Returns:
        ``cluster_ids`` と同じ長さの split 列。

    Raises:
        ValueError: 割合が範囲外のとき。
    """
    for name, value in (("zero_shot_fraction", zero_shot_fraction),
                        ("seen_fraction", seen_fraction),
                        ("max_zero_shot_share", max_zero_shot_share)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1], got {value}")

    total = len(cluster_ids)
    if total == 0:
        return []
    sizes: Counter[str] = Counter(cluster_ids)
    cap = max_zero_shot_share * total

    cluster_split: dict[str, str] = {}
    for cluster_id in sizes:
        if sizes[cluster_id] > cap:
            cluster_split[cluster_id] = "train"       # 巨大クラスタはtrainへ固定
            continue
        value = _bucket(seed, cluster_id)
        if value < zero_shot_fraction / 2:
            cluster_split[cluster_id] = "test-zero-shot"
        elif value < zero_shot_fraction:
            cluster_split[cluster_id] = "dev-zero-shot"
        else:
            cluster_split[cluster_id] = "train"

    position: Counter[str] = Counter()
    splits: list[str] = []
    for cluster_id in cluster_ids:
        assigned = cluster_split[cluster_id]
        if assigned != "train":
            splits.append(assigned)
            continue
        index = position[cluster_id]
        position[cluster_id] += 1
        inner = _bucket(seed, f"{cluster_id}#{index}")
        if inner < seen_fraction / 2:
            splits.append("test-seen")
        elif inner < seen_fraction:
            splits.append("dev-seen")
        else:
            splits.append("train")
    return splits


def split_leakage(cluster_ids: Sequence[str], splits: Sequence[str]) -> dict[str, list[str]]:
    """zero-shot と他splitをまたぐクラスタを列挙する。空なら漏れ無し。"""
    if len(cluster_ids) != len(splits):
        raise ValueError("cluster_ids と splits の長さが一致しない")
    seen: dict[str, set[str]] = defaultdict(set)
    for cluster_id, split in zip(cluster_ids, splits):
        seen[cluster_id].add(split)
    zero = {"dev-zero-shot", "test-zero-shot"}
    return {cluster_id: sorted(values) for cluster_id, values in seen.items()
            if (values & zero) and (values - zero)}


def cluster_cohesion(
    profiles: dict[str, SpeakerProfile],
    mapping: dict[str, str],
) -> dict[str, dict]:
    """クラスタごとの内部凝集度を測る。連鎖が残っていないかの検証に使う。

    Args:
        profiles: :func:`build_profiles` の結果。
        mapping: :func:`cluster_speakers` の結果。

    Returns:
        ``cluster_id -> {"speakers", "min_cos", "mean_cos"}``。
        話者が1人のクラスタは ``min_cos`` / ``mean_cos`` が ``None``。

    Notes:
        ``min_cos`` が閾値を大きく下回るクラスタは、中継役を経由して繋がった
        「別の声の寄せ集め」である可能性が高い。単連結ではこれが起こりうる。
    """
    members: dict[str, list[str]] = defaultdict(list)
    for speaker_id, cluster_id in mapping.items():
        if speaker_id in profiles:
            members[cluster_id].append(speaker_id)

    report: dict[str, dict] = {}
    for cluster_id, speakers in members.items():
        if len(speakers) < 2:
            report[cluster_id] = {"speakers": len(speakers), "min_cos": None, "mean_cos": None}
            continue
        block = np.stack([profiles[s].centroid for s in speakers]).astype(np.float32)
        similarity = np.clip(block @ block.T, -1.0, 1.0)
        upper = similarity[np.triu_indices(len(speakers), 1)]
        report[cluster_id] = {
            "speakers": len(speakers),
            "min_cos": float(upper.min()),
            "mean_cos": float(upper.mean()),
        }
    return report


SPLIT_GROUP_PREFIX = "sg:"


def assign_split_groups(
    records: Iterable[Utterance],
    mapping: dict[str, str],
) -> Iterator[Utterance]:
    """``split_group_id`` を埋めたrecordをyieldする。

    Args:
        records: manifestのrecord。
        mapping: **単連結**の :func:`cluster_speakers` 結果
            （``linkage="single"``）。``voice_cluster_id`` 用の
            完全連結マッピングとは別に作ること。

    Yields:
        ``split_group_id`` を差し替えたrecord。speaker_idがmappingに無ければ
        元のrecordをそのまま返す。
    """
    for record in records:
        group_id = mapping.get(record.speaker_id)
        if group_id is None:
            yield record
        else:
            yield dataclasses.replace(
                record, split_group_id=group_id.replace(VOICE_CLUSTER_PREFIX,
                                                        SPLIT_GROUP_PREFIX, 1))


def cross_split_similarity(
    profiles: dict[str, SpeakerProfile],
    speaker_splits: dict[str, str],
    *,
    zero_shot_splits: Iterable[str] = ("dev-zero-shot", "test-zero-shot"),
) -> dict:
    """zero-shot と train 側で、話者centroidがどれだけ似ているかを測る。

    split を細かい単位で切ると「同じ声」が両側に現れる。この関数はその
    残留漏れを検出する。``max_cos`` が閾値以上なら、同じ声が境界をまたいでいる。

    Args:
        profiles: :func:`build_profiles` の結果。
        speaker_splits: ``speaker_id -> split``。
        zero_shot_splits: zero-shot 扱いする split 名。

    Returns:
        ``{"max_cos", "train_speakers", "zero_shot_speakers", "pairs_at_92"}``。
        どちらかの側が空なら ``max_cos`` は ``None``。
    """
    zero = set(zero_shot_splits)
    train_ids = [s for s, split in speaker_splits.items()
                 if split not in zero and s in profiles]
    zero_ids = [s for s, split in speaker_splits.items()
                if split in zero and s in profiles]
    if not train_ids or not zero_ids:
        return {"max_cos": None, "train_speakers": len(train_ids),
                "zero_shot_speakers": len(zero_ids), "pairs_at_92": 0}
    left = np.stack([profiles[s].centroid for s in train_ids]).astype(np.float32)
    right = np.stack([profiles[s].centroid for s in zero_ids]).astype(np.float32)
    similarity = np.clip(left @ right.T, -1.0, 1.0)
    return {
        "max_cos": float(similarity.max()),
        "train_speakers": len(train_ids),
        "zero_shot_speakers": len(zero_ids),
        "pairs_at_92": int((similarity >= 0.92).sum()),
    }
