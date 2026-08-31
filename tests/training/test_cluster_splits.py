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

"""split を voice cluster 単位で切ることを検証する（D-015）。

speaker_id 単位で切ると「別IDの同じ声」が train と zero-shot へ分かれ、
**zero-shot が zero-shot でなくなる**。S1 の実データでは
21クラスタ・73,001発話（全体の37.1%）がこの漏れに該当していた。

`assign_splits` の docstring は当初から「voice_cluster_id が未付与のため
これは暫定であり、クラスタが確定したら作り直す必要がある」と書いていたが、
その作り直しが実装されていなかった。エラーにならない欠陥なので固定する。
"""

from __future__ import annotations

import pytest

from cutetts.training.voice_clusters import (
    DEFAULT_MAX_ZERO_SHOT_SHARE,
    assign_splits_by_cluster,
    split_leakage,
)

ZERO_SHOT = {"dev-zero-shot", "test-zero-shot"}


def _corpus(clusters: int = 200, per_cluster: int = 20) -> list[str]:
    return [f"vc:{i:04d}" for i in range(clusters) for _ in range(per_cluster)]


# ------------------------------------------------------------------ 漏れ無し

def test_no_cluster_appears_in_both_zero_shot_and_train():
    ids = _corpus()
    splits = assign_splits_by_cluster(ids, seed=42)
    assert split_leakage(ids, splits) == {}


def test_leakage_detector_catches_a_deliberately_broken_split():
    """検出器自体が働くことを確認する（テストの空振り防止）。"""
    ids = ["vc:0001"] * 4
    broken = ["train", "train", "dev-zero-shot", "train"]
    assert split_leakage(ids, broken) == {"vc:0001": ["dev-zero-shot", "train"]}


def test_speaker_level_splitting_would_have_leaked():
    """比較用。同じ声が2つのIDに分かれている状況を speaker 単位で切ると漏れる。"""
    # spk1 と spk2 は同一クラスタ（同じ声）だが、speaker単位では別扱いになる
    cluster_ids = ["vc:same"] * 10
    speaker_splits = ["train"] * 5 + ["dev-zero-shot"] * 5
    assert split_leakage(cluster_ids, speaker_splits) != {}


# ------------------------------------------------------------ クラスタの純度

def test_every_cluster_lands_in_a_single_split_family():
    """1クラスタは zero-shot か train系のどちらか一方にしか現れない。"""
    ids = _corpus()
    splits = assign_splits_by_cluster(ids, seed=7)
    families: dict[str, set[str]] = {}
    for cluster_id, split in zip(ids, splits):
        family = "zero-shot" if split in ZERO_SHOT else "train"
        families.setdefault(cluster_id, set()).add(family)
    assert all(len(v) == 1 for v in families.values())


def test_zero_shot_cluster_contributes_all_of_its_utterances():
    """zero-shotに落ちたクラスタは全発話がそこへ行く（seenへ分けない）。"""
    ids = _corpus()
    splits = assign_splits_by_cluster(ids, seed=7)
    by_cluster: dict[str, set[str]] = {}
    for cluster_id, split in zip(ids, splits):
        by_cluster.setdefault(cluster_id, set()).add(split)
    for values in by_cluster.values():
        if values & ZERO_SHOT:
            assert len(values) == 1


# --------------------------------------------------------------- 巨大クラスタ

def test_an_oversized_cluster_is_forced_into_train():
    """単連結の連鎖で生まれた巨大クラスタをzero-shotへ落とさない。

    S1では45話者・90.7時間・全体の28%が1クラスタになった。
    これがzero-shotへ行くとsplit構成が壊れる。
    """
    ids = ["vc:huge"] * 400 + [f"vc:{i:04d}" for i in range(600)]
    splits = assign_splits_by_cluster(ids, seed=1)
    huge = {s for cid, s in zip(ids, splits) if cid == "vc:huge"}
    assert not (huge & ZERO_SHOT)


def test_a_cluster_just_under_the_cap_can_still_be_zero_shot():
    """上限は「大きすぎるものだけ」を弾く。小さいクラスタは対象のまま。"""
    total = 1000
    small = int(total * DEFAULT_MAX_ZERO_SHOT_SHARE) - 1
    found = False
    for seed in range(40):
        ids = ["vc:candidate"] * small + [f"vc:{i:04d}" for i in range(total - small)]
        splits = assign_splits_by_cluster(ids, seed=seed)
        if {s for cid, s in zip(ids, splits) if cid == "vc:candidate"} & ZERO_SHOT:
            found = True
            break
    assert found, "上限未満のクラスタが一度もzero-shotへ行かない（弾きすぎ）"


# ------------------------------------------------------------------- 決定性

def test_assignment_is_deterministic_for_a_seed():
    ids = _corpus(50, 10)
    assert assign_splits_by_cluster(ids, seed=3) == assign_splits_by_cluster(ids, seed=3)


def test_a_different_seed_changes_the_assignment():
    ids = _corpus(50, 10)
    assert assign_splits_by_cluster(ids, seed=3) != assign_splits_by_cluster(ids, seed=4)


def test_assignment_does_not_depend_on_utterance_order():
    """並べ替えても、各クラスタの行き先は変わらない。"""
    ids = _corpus(30, 6)
    splits = dict(zip(ids, assign_splits_by_cluster(ids, seed=11)))
    shuffled = list(reversed(ids))
    reshuffled = dict(zip(shuffled, assign_splits_by_cluster(shuffled, seed=11)))
    for cluster_id in set(ids):
        family = splits[cluster_id] in ZERO_SHOT
        assert (reshuffled[cluster_id] in ZERO_SHOT) == family


# -------------------------------------------------------------------- 妥当性

def test_all_splits_are_produced_at_a_realistic_scale():
    ids = _corpus(300, 30)
    splits = set(assign_splits_by_cluster(ids, seed=20260831))
    assert splits == {"train", "dev-seen", "test-seen", "dev-zero-shot", "test-zero-shot"}


def test_empty_input_is_allowed():
    assert assign_splits_by_cluster([], seed=1) == []


def test_invalid_fractions_are_rejected():
    with pytest.raises(ValueError):
        assign_splits_by_cluster(["vc:a"], seed=1, zero_shot_fraction=1.5)
    with pytest.raises(ValueError):
        assign_splits_by_cluster(["vc:a"], seed=1, seen_fraction=-0.1)


def test_leakage_detector_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        split_leakage(["vc:a", "vc:b"], ["train"])
