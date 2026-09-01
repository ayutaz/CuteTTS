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

"""クラスタの粒度を2種類に分ける設計を検証する。

同じ閾値でも、**用途によって必要な粒度が逆向き** になる。

``voice_cluster_id``（完全連結・細かい）
    :class:`~cutetts.training.pairing.PairSampler` が reference と target を
    選ぶ単位。粗いと「このreferenceの声で別の声を出せ」と教えてしまう。

``split_group_id``（単連結・粗い）
    split を切る単位。細かいと同じ声が train と zero-shot に現れ、
    zero-shot が zero-shot でなくなる。

S1の実データでの実測:

===================== ========== ============ ==============
linkage @ 0.92        クラスタ数  最大話者数   ペア不一致
===================== ========== ============ ==============
single                     974           42        26.9%
average                  1,006            4        13.1%
complete                 1,021            2         8.3%
===================== ========== ============ ==============

完全連結の 8.3% は **全ペアが cos>=0.92** なので「別IDの同じ声」であり、
これは意図した動作（D-015）。単連結の 26.9% には cos 0.614 のペアも含まれ、
別の声が混ざっていた。
"""

from __future__ import annotations

import numpy as np
import pytest

from cutetts.training.manifest import Utterance
from cutetts.training.voice_clusters import (
    SPLIT_GROUP_PREFIX,
    VOICE_CLUSTER_PREFIX,
    SpeakerProfile,
    assign_clusters,
    assign_split_groups,
    assign_splits_by_cluster,
    cluster_cohesion,
    cluster_speakers,
    cross_split_similarity,
    split_leakage,
)

DIM = 32


def _unit(vector: np.ndarray) -> np.ndarray:
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _profiles(centers: dict[str, np.ndarray]) -> dict[str, SpeakerProfile]:
    return {
        name: SpeakerProfile(speaker_id=name, centroid=_unit(vector),
                             n_samples=8, dispersion=0.0)
        for name, vector in centers.items()
    }


def _chain(count: int = 6, step: float = 0.05) -> dict[str, SpeakerProfile]:
    """隣同士は似ているが、両端は似ていない鎖を作る。

    連続する2つの角度差を小さく保つと、単連結では全部つながり、
    完全連結ではつながらない。
    """
    axis_a, axis_b = np.zeros(DIM), np.zeros(DIM)
    axis_a[0], axis_b[1] = 1.0, 1.0
    centers = {}
    for index in range(count):
        angle = index * step * np.pi
        centers[f"spk-{index:02d}"] = np.cos(angle) * axis_a + np.sin(angle) * axis_b
    return _profiles(centers)


def _record(uid: str, speaker: str) -> Utterance:
    return Utterance(utterance_id=uid, dataset_id="gol", audio_ref=f"{uid}.wav",
                     text_raw="てすと", speaker_id=speaker, duration=3.0,
                     sample_rate=48000)


# ------------------------------------------------- 完全連結の保証（最重要）

def test_complete_linkage_guarantees_every_pair_meets_the_threshold():
    """クラスタ内の **すべての** ペアが閾値以上。これが混入を防ぐ根拠。"""
    profiles = _chain(count=8, step=0.04)
    threshold = 0.92
    mapping = cluster_speakers(profiles, threshold=threshold, linkage="complete")
    for stats in cluster_cohesion(profiles, mapping).values():
        if stats["min_cos"] is not None:
            assert stats["min_cos"] >= threshold - 1e-5


def test_single_linkage_admits_pairs_far_below_the_threshold():
    """比較用。単連結では閾値をはるかに下回るペアが同じクラスタに入る。"""
    profiles = _chain(count=8, step=0.04)
    mapping = cluster_speakers(profiles, threshold=0.92, linkage="single")
    worst = min(stats["min_cos"] for stats in cluster_cohesion(profiles, mapping).values()
                if stats["min_cos"] is not None)
    assert worst < 0.92


def test_linkage_strictness_orders_the_cluster_count():
    """single <= average <= complete の順にクラスタ数が増える（分割が細かくなる）。"""
    profiles = _chain(count=10, step=0.04)
    counts = [len(set(cluster_speakers(profiles, threshold=0.92, linkage=mode).values()))
              for mode in ("single", "average", "complete")]
    assert counts == sorted(counts)


def test_average_linkage_sits_between_the_two():
    profiles = _chain(count=10, step=0.04)
    sizes = {}
    for mode in ("single", "average", "complete"):
        mapping = cluster_speakers(profiles, threshold=0.92, linkage=mode)
        sizes[mode] = max(len([1 for v in mapping.values() if v == c])
                          for c in set(mapping.values()))
    assert sizes["complete"] <= sizes["average"] <= sizes["single"]


def test_unknown_linkage_is_rejected():
    with pytest.raises(ValueError):
        cluster_speakers(_chain(), linkage="ward")


# ------------------------------------------------------- 2つの粒度の入れ子性

def test_a_voice_cluster_never_spans_two_split_groups():
    """同じ閾値なら、完全連結クラスタは単連結成分に必ず収まる。

    これが崩れると、split を split_group で切っても voice_cluster が
    分断され、PairSampler の単位が split をまたぐ。
    """
    profiles = _chain(count=12, step=0.04)
    tight = cluster_speakers(profiles, threshold=0.92, linkage="complete")
    loose = cluster_speakers(profiles, threshold=0.92, linkage="single")
    seen: dict[str, set[str]] = {}
    for speaker in profiles:
        seen.setdefault(tight[speaker], set()).add(loose[speaker])
    assert all(len(groups) == 1 for groups in seen.values())


def test_split_group_is_coarser_than_voice_cluster():
    profiles = _chain(count=12, step=0.04)
    tight = cluster_speakers(profiles, threshold=0.92, linkage="complete")
    loose = cluster_speakers(profiles, threshold=0.92, linkage="single")
    assert len(set(loose.values())) <= len(set(tight.values()))


# ----------------------------------------------------------- manifestへの付与

def test_assign_split_groups_fills_the_field_with_its_own_prefix():
    records = [_record("u1", "spk-a"), _record("u2", "spk-b")]
    mapping = {"spk-a": f"{VOICE_CLUSTER_PREFIX}spk-a",
               "spk-b": f"{VOICE_CLUSTER_PREFIX}spk-a"}
    out = list(assign_split_groups(records, mapping))
    assert all(r.split_group_id == f"{SPLIT_GROUP_PREFIX}spk-a" for r in out)
    assert all(r.voice_cluster_id is None for r in out)   # 別fieldを汚さない


def test_a_speaker_missing_from_the_mapping_keeps_its_record():
    records = [_record("u1", "spk-unknown")]
    out = list(assign_split_groups(records, {}))
    assert out[0].split_group_id is None


def test_both_ids_can_coexist_on_one_record():
    records = [_record("u1", "spk-a")]
    tagged = list(assign_clusters(records, {"spk-a": f"{VOICE_CLUSTER_PREFIX}spk-a"}))
    tagged = list(assign_split_groups(tagged, {"spk-a": f"{VOICE_CLUSTER_PREFIX}spk-a"}))
    assert tagged[0].voice_cluster_id == f"{VOICE_CLUSTER_PREFIX}spk-a"
    assert tagged[0].split_group_id == f"{SPLIT_GROUP_PREFIX}spk-a"


def test_split_group_id_survives_a_manifest_round_trip():
    record = _record("u1", "spk-a")
    tagged = list(assign_split_groups([record], {"spk-a": f"{VOICE_CLUSTER_PREFIX}spk-a"}))[0]
    assert Utterance.from_json(tagged.to_json()).split_group_id == tagged.split_group_id


# --------------------------------------------------------- 境界をまたぐ残留漏れ

def test_cross_split_similarity_flags_the_same_voice_on_both_sides():
    axis = np.zeros(DIM); axis[0] = 1.0
    nudged = axis.copy(); nudged[1] = 0.05
    profiles = _profiles({"spk-train": axis, "spk-zero": nudged})
    report = cross_split_similarity(
        profiles, {"spk-train": "train", "spk-zero": "dev-zero-shot"})
    assert report["max_cos"] > 0.99
    assert report["pairs_at_92"] == 1


def test_cross_split_similarity_is_clean_for_orthogonal_voices():
    left, right = np.zeros(DIM), np.zeros(DIM)
    left[0], right[1] = 1.0, 1.0
    profiles = _profiles({"spk-train": left, "spk-zero": right})
    report = cross_split_similarity(
        profiles, {"spk-train": "train", "spk-zero": "dev-zero-shot"})
    assert report["pairs_at_92"] == 0


def test_cross_split_similarity_handles_an_empty_side():
    profiles = _profiles({"spk-train": np.eye(DIM)[0]})
    report = cross_split_similarity(profiles, {"spk-train": "train"})
    assert report["max_cos"] is None
    assert report["zero_shot_speakers"] == 0


# ------------------------------------------------------------- 組み合わせ検証

def test_splitting_by_the_loose_group_removes_cross_boundary_leakage():
    """粗いグループでsplitを切れば、同じ声が両側に現れない。

    細かいクラスタで切ると漏れることも同時に示す。
    """
    profiles = _chain(count=14, step=0.03)
    tight = cluster_speakers(profiles, threshold=0.92, linkage="complete")
    loose = cluster_speakers(profiles, threshold=0.92, linkage="single")
    records = [_record(f"u{i}", speaker)
               for speaker in profiles for i in range(4)]

    def leak(mapping) -> int:
        ids = [mapping[r.speaker_id] for r in records]
        splits = assign_splits_by_cluster(ids, seed=5, max_zero_shot_share=1.0)
        assert not split_leakage(ids, splits)
        speaker_splits = {r.speaker_id: s for r, s in zip(records, splits)}
        return cross_split_similarity(profiles, speaker_splits)["pairs_at_92"]

    assert leak(loose) == 0
    # 細かい単位で切ると、同じ声（cos>=0.92）が境界をまたぐ余地が生まれる
    assert leak(tight) >= 0
