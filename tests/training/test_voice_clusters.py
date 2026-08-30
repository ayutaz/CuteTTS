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

"""voiceクラスタリングの決定的テスト（合成embeddingのみ。実データ・torch不要）。"""

from __future__ import annotations

import numpy as np
import pytest

from cutetts.training.manifest import Utterance
from cutetts.training.voice_clusters import (
    SPEAKER_EMBEDDING_DIM,
    VOICE_CLUSTER_PREFIX,
    assign_clusters,
    build_profiles,
    cluster_members,
    cluster_speakers,
    cluster_summary,
    cosine_similarity_matrix,
    suspicious_speakers,
)

# --- 合成データのヘルパ -------------------------------------------------------


class FakeReader:
    """``SpeakerEmbeddingCacheReader`` 互換のfake。読んだIDを記録する。"""

    def __init__(self, table: dict[str, np.ndarray]) -> None:
        self._table = dict(table)
        self.reads: list[str] = []

    def __contains__(self, utterance_id: str) -> bool:
        return utterance_id in self._table

    def read(self, utterance_id: str) -> np.ndarray:
        self.reads.append(utterance_id)
        return self._table[utterance_id]


def orthonormal_voices(count: int, *, dim: int = SPEAKER_EMBEDDING_DIM, seed: int = 0) -> np.ndarray:
    """互いに直交する単位ベクトル ``[count, dim]``。「まったく別の声」の代用。"""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(dim, count)))
    return np.ascontiguousarray(basis.T)


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def jittered(center: np.ndarray, count: int, *, jitter: float, rng: np.random.Generator) -> list[np.ndarray]:
    """``center`` の周りに散らした発話embedding。ノルムはわざと1にしない。"""
    return [
        unit(center + jitter * rng.normal(size=center.shape)) * float(rng.uniform(0.5, 4.0))
        for _ in range(count)
    ]


def make_record(utterance_id: str, speaker_id: str, **overrides) -> Utterance:
    """検証を通る最小構成のrecord。"""
    base = {
        "utterance_id": utterance_id,
        "dataset_id": "gol",
        "audio_ref": f"data/raw/gol/audio/game0001.tar::{utterance_id}.wav",
        "text_raw": "今日はいい天気ですね。",
        "speaker_id": speaker_id,
        "duration": 5.18,
        "sample_rate": 48000,
    }
    base.update(overrides)
    return Utterance(**base)


def build_dataset(
    layout: dict[str, list[np.ndarray]],
    *,
    utterances_per_voice: int = 4,
    jitter: float = 0.01,
    seed: int = 0,
) -> tuple[FakeReader, list[Utterance]]:
    """``speaker_id -> [その話者に混ざっている声の中心, ...]`` から reader と records を作る。

    1話者に複数の中心を渡すと「1 IDに複数の声が混在」した総称ラベルを再現できる。
    """
    rng = np.random.default_rng(seed)
    table: dict[str, np.ndarray] = {}
    records: list[Utterance] = []
    for speaker_id in sorted(layout):
        index = 0
        for center in layout[speaker_id]:
            for embedding in jittered(center, utterances_per_voice, jitter=jitter, rng=rng):
                utterance_id = f"{speaker_id}:{index:04d}"
                table[utterance_id] = embedding
                records.append(make_record(utterance_id, speaker_id))
                index += 1
    return FakeReader(table), records


# --- build_profiles -----------------------------------------------------------


def test_centroid_is_l2_normalized_even_though_inputs_are_not() -> None:
    voices = orthonormal_voices(3)
    reader, records = build_dataset({"spk-a": [voices[0]], "spk-b": [voices[1]], "spk-c": [voices[2]]})
    # 入力embeddingのノルムは1ではない（jittered が意図的にスケールしている）。
    assert not np.allclose([np.linalg.norm(v) for v in reader._table.values()], 1.0)

    profiles = build_profiles(reader, records)

    assert set(profiles) == {"spk-a", "spk-b", "spk-c"}
    for profile in profiles.values():
        assert profile.centroid.ndim == 1
        assert profile.centroid.shape[0] == SPEAKER_EMBEDDING_DIM
        assert float(np.linalg.norm(profile.centroid)) == pytest.approx(1.0, abs=1e-6)


def test_build_profiles_never_reads_more_than_max_samples_per_speaker() -> None:
    voices = orthonormal_voices(2)
    reader, records = build_dataset(
        {"spk-a": [voices[0]], "spk-b": [voices[1]]}, utterances_per_voice=50
    )

    profiles = build_profiles(reader, records, max_samples_per_speaker=8, seed=0)

    assert len(reader.reads) == 16  # 2話者 x 8
    assert len(set(reader.reads)) == 16  # 同じutteranceを二度読まない
    for speaker_id in ("spk-a", "spk-b"):
        assert profiles[speaker_id].n_samples == 8
        assert sum(1 for uid in reader.reads if uid.startswith(f"{speaker_id}:")) == 8


def test_build_profiles_is_deterministic_for_a_fixed_seed() -> None:
    voices = orthonormal_voices(2)
    reader_a, records = build_dataset({"spk-a": [voices[0]], "spk-b": [voices[1]]}, utterances_per_voice=40)
    reader_b = FakeReader(reader_a._table)

    first = build_profiles(reader_a, records, max_samples_per_speaker=5, seed=7)
    second = build_profiles(reader_b, list(records), max_samples_per_speaker=5, seed=7)

    assert reader_a.reads == reader_b.reads
    assert set(first) == set(second)
    for speaker_id in first:
        assert np.array_equal(first[speaker_id].centroid, second[speaker_id].centroid)
        assert first[speaker_id].dispersion == second[speaker_id].dispersion


def test_build_profiles_seed_changes_the_selected_subset() -> None:
    voices = orthonormal_voices(1)
    reader_a, records = build_dataset({"spk-a": [voices[0]]}, utterances_per_voice=60)
    reader_b = FakeReader(reader_a._table)

    build_profiles(reader_a, records, max_samples_per_speaker=6, seed=0)
    build_profiles(reader_b, records, max_samples_per_speaker=6, seed=1)

    assert set(reader_a.reads) != set(reader_b.reads)


def test_uncached_utterances_are_skipped_without_reading() -> None:
    voices = orthonormal_voices(2)
    reader, records = build_dataset({"spk-a": [voices[0]], "spk-b": [voices[1]]}, utterances_per_voice=4)
    # spk-b のembeddingをcacheから丸ごと落とし、spk-a も1件だけ落とす。
    reader._table = {
        key: value
        for key, value in reader._table.items()
        if not key.startswith("spk-b:") and key != "spk-a:0000"
    }
    records.append(make_record("spk-c:0000", "spk-c"))  # cacheに一切無い話者

    profiles = build_profiles(reader, records)

    assert set(profiles) == {"spk-a"}
    assert profiles["spk-a"].n_samples == 3
    assert "spk-a:0000" not in reader.reads
    assert all(uid.startswith("spk-a:") for uid in reader.reads)


class ArrayLike:
    """``__array__`` だけを持つオブジェクト（torch.Tensorの代用）。"""

    def __init__(self, data) -> None:
        self._data = np.asarray(data, dtype=np.float32)

    def __array__(self, dtype=None, **kwargs) -> np.ndarray:
        return self._data if dtype is None else self._data.astype(dtype)


def test_reader_may_return_any_array_like() -> None:
    """実装側の ``SpeakerEmbeddingCacheReader.read`` は torch.Tensor を返す。"""
    center = orthonormal_voices(1)[0]
    reader = FakeReader({"u0": ArrayLike(center), "u1": list(center * 3.0)})

    profiles = build_profiles(reader, [make_record("u0", "spk-a"), make_record("u1", "spk-a")])

    assert profiles["spk-a"].n_samples == 2
    assert float(np.linalg.norm(profiles["spk-a"].centroid)) == pytest.approx(1.0, abs=1e-6)
    assert profiles["spk-a"].dispersion == pytest.approx(0.0, abs=1e-6)


def test_single_sample_speaker_has_zero_dispersion() -> None:
    voices = orthonormal_voices(1)
    reader, records = build_dataset({"spk-a": [voices[0]]}, utterances_per_voice=1)

    profile = build_profiles(reader, records)["spk-a"]

    assert profile.n_samples == 1
    assert profile.dispersion == pytest.approx(0.0, abs=1e-9)


def test_zero_vector_embeddings_are_dropped() -> None:
    voices = orthonormal_voices(1)
    reader, records = build_dataset({"spk-a": [voices[0]]}, utterances_per_voice=3)
    reader._table["spk-a:0000"] = np.zeros(SPEAKER_EMBEDDING_DIM)
    reader._table["spk-b:0000"] = np.zeros(SPEAKER_EMBEDDING_DIM)
    records.append(make_record("spk-b:0000", "spk-b"))

    profiles = build_profiles(reader, records)

    assert set(profiles) == {"spk-a"}  # 全部ゼロの spk-b はprofileを作らない
    assert profiles["spk-a"].n_samples == 2


def test_build_profiles_rejects_bad_embeddings() -> None:
    reader, records = build_dataset({"spk-a": [orthonormal_voices(1)[0]]}, utterances_per_voice=2)

    with pytest.raises(ValueError, match="max_samples_per_speaker"):
        build_profiles(reader, records, max_samples_per_speaker=0)

    two_dim = FakeReader({"u0": np.ones((2, SPEAKER_EMBEDDING_DIM))})
    with pytest.raises(ValueError, match="1-D"):
        build_profiles(two_dim, [make_record("u0", "spk-a")])

    mismatched = FakeReader({"u0": np.ones(256), "u1": np.ones(192)})
    with pytest.raises(ValueError, match="dim mismatch"):
        build_profiles(mismatched, [make_record("u0", "spk-a"), make_record("u1", "spk-b")])


# --- cosine_similarity_matrix -------------------------------------------------


def test_cosine_similarity_matrix_is_sorted_symmetric_and_unit_diagonal() -> None:
    voices = orthonormal_voices(3)
    reader, records = build_dataset({"spk-c": [voices[0]], "spk-a": [voices[1]], "spk-b": [voices[2]]})
    profiles = build_profiles(reader, records)

    speaker_ids, similarity = cosine_similarity_matrix(profiles)

    assert speaker_ids == ["spk-a", "spk-b", "spk-c"]
    assert similarity.shape == (3, 3)
    assert np.allclose(np.diag(similarity), 1.0)
    assert np.array_equal(similarity, similarity.T)
    assert similarity.min() >= -1.0 and similarity.max() <= 1.0
    off_diagonal = similarity[~np.eye(3, dtype=bool)]
    assert np.all(np.abs(off_diagonal) < 0.2)  # 直交する声どうしは似ていない


def test_cosine_similarity_matrix_of_empty_profiles() -> None:
    speaker_ids, similarity = cosine_similarity_matrix({})

    assert speaker_ids == []
    assert similarity.shape == (0, 0)


# --- cluster_speakers ---------------------------------------------------------


def test_three_separated_voice_groups_become_three_clusters() -> None:
    voices = orthonormal_voices(3)
    # 1つの声につき2つのspeaker ID（moeの「同一声優でも別ID」を再現）。
    layout = {
        "spk-a1": [voices[0]],
        "spk-a2": [voices[0]],
        "spk-b1": [voices[1]],
        "spk-b2": [voices[1]],
        "spk-c1": [voices[2]],
        "spk-c2": [voices[2]],
    }
    reader, records = build_dataset(layout, utterances_per_voice=6)
    profiles = build_profiles(reader, records)

    mapping = cluster_speakers(profiles, threshold=0.70)

    assert set(mapping) == set(layout)
    assert len(set(mapping.values())) == 3
    assert mapping["spk-a1"] == mapping["spk-a2"] == f"{VOICE_CLUSTER_PREFIX}spk-a1"
    assert mapping["spk-b1"] == mapping["spk-b2"] == f"{VOICE_CLUSTER_PREFIX}spk-b1"
    assert mapping["spk-c1"] == mapping["spk-c2"] == f"{VOICE_CLUSTER_PREFIX}spk-c1"
    assert cluster_members(mapping)[f"{VOICE_CLUSTER_PREFIX}spk-a1"] == ["spk-a1", "spk-a2"]

    summary = cluster_summary(mapping)
    assert summary["speakers"] == 6
    assert summary["clusters"] == 3
    assert summary["largest_cluster_size"] == 2
    assert summary["multi_speaker_clusters"] == 3
    assert summary["singleton_clusters"] == 0
    assert summary["speakers_in_multi_speaker_clusters"] == 6
    assert summary["size_histogram"] == {"2": 3}


def test_threshold_is_monotonic_and_partitions_are_nested() -> None:
    # 既知のcosineを持つ3つの中心: sim(A,C)=0.8, sim(B,C)=0.6, sim(A,B)=0。
    basis = orthonormal_voices(2, seed=3)
    centers = {
        "spk-a": basis[0],
        "spk-b": basis[1],
        "spk-c": unit(0.8 * basis[0] + 0.6 * basis[1]),
    }
    reader, records = build_dataset({key: [value] for key, value in centers.items()}, jitter=0.0)
    profiles = build_profiles(reader, records)

    loose = cluster_speakers(profiles, threshold=0.50)
    middle = cluster_speakers(profiles, threshold=0.70)
    tight = cluster_speakers(profiles, threshold=0.90)

    assert len(set(loose.values())) == 1  # 全部つながる
    assert len(set(middle.values())) == 2  # A-C だけ残る
    assert len(set(tight.values())) == 3  # 全部ばらける
    assert middle["spk-a"] == middle["spk-c"] != middle["spk-b"]

    # 単調性: 閾値を上げるとクラスタ数は減らない。
    counts = [len(set(cluster_speakers(profiles, threshold=t).values())) for t in (0.3, 0.5, 0.7, 0.9, 0.99)]
    assert counts == sorted(counts)

    # 入れ子性: 厳しい閾値で同じクラスタなら、緩い閾値でも必ず同じクラスタ。
    for finer, coarser in ((tight, middle), (middle, loose)):
        for left in profiles:
            for right in profiles:
                if finer[left] == finer[right]:
                    assert coarser[left] == coarser[right]


def test_cluster_ids_are_deterministic_and_order_independent() -> None:
    voices = orthonormal_voices(3)
    layout = {"spk-z": [voices[0]], "spk-m": [voices[0]], "spk-a": [voices[1]], "spk-q": [voices[2]]}
    reader, records = build_dataset(layout, utterances_per_voice=5)
    profiles = build_profiles(reader, records)

    first = cluster_speakers(profiles)
    second = cluster_speakers(profiles)
    shuffled = cluster_speakers({key: profiles[key] for key in reversed(list(profiles))})

    assert first == second == shuffled
    # 同じ声の2 IDは、辞書順最小のIDでラベルされる。
    assert first["spk-z"] == first["spk-m"] == f"{VOICE_CLUSTER_PREFIX}spk-m"
    # 併合されなかった話者は自分1人のクラスタ。
    assert first["spk-a"] == f"{VOICE_CLUSTER_PREFIX}spk-a"


def test_cluster_speakers_edge_cases() -> None:
    assert cluster_speakers({}) == {}

    voices = orthonormal_voices(1)
    reader, records = build_dataset({"spk-a": [voices[0]]})
    profiles = build_profiles(reader, records)
    assert cluster_speakers(profiles) == {"spk-a": f"{VOICE_CLUSTER_PREFIX}spk-a"}

    with pytest.raises(ValueError, match="finite"):
        cluster_speakers(profiles, threshold=float("nan"))


# --- suspicious_speakers ------------------------------------------------------


def test_speaker_mixing_several_voices_is_flagged_as_suspicious() -> None:
    voices = orthonormal_voices(4)
    layout = {
        "spk-clean1": [voices[0]],
        "spk-clean2": [voices[1]],
        # 総称ラベル（『女の子』等）の再現: 1 IDに3つの声が混ざっている。
        "spk-generic": [voices[1], voices[2], voices[3]],
    }
    reader, records = build_dataset(layout, utterances_per_voice=6)
    profiles = build_profiles(reader, records, max_samples_per_speaker=32)

    assert profiles["spk-clean1"].dispersion < 0.05
    assert profiles["spk-clean2"].dispersion < 0.05
    # 等量3声・相互直交なら理論値は 1 - 1/sqrt(3) = 0.4226。
    assert profiles["spk-generic"].dispersion == pytest.approx(1.0 - 1.0 / np.sqrt(3.0), abs=0.03)

    assert suspicious_speakers(profiles, dispersion_threshold=0.35) == ["spk-generic"]
    assert suspicious_speakers(profiles, dispersion_threshold=0.99) == []
    assert suspicious_speakers(profiles, dispersion_threshold=0.0) == [
        "spk-clean1",
        "spk-clean2",
        "spk-generic",
    ]


def test_two_voice_mixture_dispersion_matches_the_documented_formula() -> None:
    # docstringに書いた「等量2声・相互cosine s なら dispersion = 1 - sqrt((1+s)/2)」の確認。
    # s = 0 では 0.293 にしかならず、既定の0.35では検出できない（閾値校正の根拠）。
    voices = orthonormal_voices(2, seed=11)
    reader, records = build_dataset({"spk-two": [voices[0], voices[1]]}, utterances_per_voice=8, jitter=0.0)
    profiles = build_profiles(reader, records)

    dispersion = profiles["spk-two"].dispersion
    assert dispersion == pytest.approx(1.0 - np.sqrt(0.5), abs=1e-6)
    assert suspicious_speakers(profiles, dispersion_threshold=0.35) == []
    assert suspicious_speakers(profiles, dispersion_threshold=0.25) == ["spk-two"]


# --- assign_clusters / cluster_summary ---------------------------------------


def test_assign_clusters_fills_known_speakers_and_leaves_others_none() -> None:
    records = [
        make_record("gol:0001", "spk-a"),
        make_record("gol:0002", "spk-b"),
        make_record("gol:0003", "spk-unknown"),
    ]
    mapping = {"spk-a": "vc:spk-a", "spk-b": "vc:spk-a"}

    assigned = list(assign_clusters(iter(records), mapping))

    assert [record.voice_cluster_id for record in assigned] == ["vc:spk-a", "vc:spk-a", None]
    assert [record.utterance_id for record in assigned] == ["gol:0001", "gol:0002", "gol:0003"]
    # voice_cluster_id 以外は素通し。元のrecordも書き換わらない。
    assert assigned[0].text_raw == records[0].text_raw
    assert assigned[0].speaker_id == records[0].speaker_id
    assert assigned[0].duration == records[0].duration
    assert [record.voice_cluster_id for record in records] == [None, None, None]


def test_assign_clusters_overwrites_a_stale_voice_cluster_id() -> None:
    records = [make_record("gol:0001", "spk-a", voice_cluster_id="vc:old")]

    assigned = list(assign_clusters(records, {"spk-a": "vc:new"}))

    assert assigned[0].voice_cluster_id == "vc:new"


def test_cluster_summary_reports_sizes() -> None:
    mapping = {
        "spk-a": "vc:spk-a",
        "spk-b": "vc:spk-a",
        "spk-c": "vc:spk-a",
        "spk-d": "vc:spk-d",
        "spk-e": "vc:spk-e",
        "spk-f": "vc:spk-e",
    }

    summary = cluster_summary(mapping)

    assert summary == {
        "speakers": 6,
        "clusters": 3,
        "largest_cluster_size": 3,
        "largest_cluster_id": "vc:spk-a",
        "multi_speaker_clusters": 2,
        "singleton_clusters": 1,
        "speakers_in_multi_speaker_clusters": 5,
        "size_histogram": {"1": 1, "2": 1, "3": 1},
    }


def test_cluster_summary_of_empty_mapping() -> None:
    summary = cluster_summary({})

    assert summary["speakers"] == 0
    assert summary["clusters"] == 0
    assert summary["largest_cluster_id"] is None
    assert summary["size_histogram"] == {}


def test_end_to_end_split_unit_is_the_voice_cluster_not_the_speaker_id() -> None:
    """R-004の本題: 別IDの同一声が、splitで必ず同じ側に入ることを確認する。"""
    voices = orthonormal_voices(2)
    layout = {"spk-x1": [voices[0]], "spk-x2": [voices[0]], "spk-y": [voices[1]]}
    reader, records = build_dataset(layout, utterances_per_voice=5)

    profiles = build_profiles(reader, records)
    mapping = cluster_speakers(profiles, threshold=0.70)
    assigned = list(assign_clusters(records, mapping))

    clusters_by_speaker: dict[str, set[str]] = {}
    for record in assigned:
        assert record.voice_cluster_id is not None
        clusters_by_speaker.setdefault(record.speaker_id, set()).add(record.voice_cluster_id)

    assert clusters_by_speaker["spk-x1"] == clusters_by_speaker["spk-x2"]
    assert clusters_by_speaker["spk-x1"] != clusters_by_speaker["spk-y"]
    # speaker ID単位でsplitすると x1/x2 が train と zero-shot に割れるが、
    # cluster単位なら割れない。
    assert len({next(iter(value)) for value in clusters_by_speaker.values()}) == 2
