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

"""reference/target サンプラの決定的テスト（合成recordのみ。実データ不要）。

R-004（leakage）の防波堤なので、配線を1つ外したら落ちることを重視している。
"""

from __future__ import annotations

import random
from collections import Counter
from itertools import islice

import pytest

from cutetts.training.manifest import Utterance
from cutetts.training.pairing import (
    PairSampler,
    ReferenceTargetPair,
    assert_no_leakage,
    group_size_histogram,
)


def make_utterance(
    utterance_id: str,
    speaker_id: str,
    duration: float = 5.0,
    *,
    voice_cluster_id: str | None = None,
    dataset_id: str = "gol",
) -> Utterance:
    """検証を通る最小構成のrecord。pairingが見るのはID・グループキー・durationだけ。"""
    return Utterance(
        utterance_id=utterance_id,
        dataset_id=dataset_id,
        audio_ref=f"data/raw/{dataset_id}/{utterance_id}.wav",
        text_raw="今日はいい天気ですね。",
        speaker_id=speaker_id,
        duration=duration,
        sample_rate=48000,
        voice_cluster_id=voice_cluster_id,
    )


def make_corpus(
    sizes: dict[str, int],
    *,
    duration: float = 5.0,
    clusters: dict[str, str] | None = None,
) -> list[Utterance]:
    """``{speaker_id: 発話数}`` から合成corpusを作る。"""
    records: list[Utterance] = []
    for speaker_id, count in sizes.items():
        cluster = clusters.get(speaker_id) if clusters else None
        for index in range(count):
            records.append(
                make_utterance(
                    f"{speaker_id}-{index:04d}",
                    speaker_id,
                    duration,
                    voice_cluster_id=cluster,
                )
            )
    return records


def pair_ids(pairs) -> list[tuple[str, tuple[str, ...]]]:
    """比較しやすいIDだけの表現へ落とす。"""
    return [
        (pair.target.utterance_id, tuple(r.utterance_id for r in pair.reference_group))
        for pair in pairs
    ]


# --- leakage -----------------------------------------------------------------


def test_reference_never_contains_the_target_utterance() -> None:
    """最重要。100ペアでtargetがreference側に混ざらないこと（R-004）。"""
    records = make_corpus({f"spk{i:02d}": 6 for i in range(10)})
    sampler = PairSampler(records, seed=7)

    pairs = sampler.sample(100)

    assert len(pairs) == 100
    for pair in pairs:
        reference_ids = [r.utterance_id for r in pair.reference_group]
        assert pair.target.utterance_id not in reference_ids
        assert len(set(reference_ids)) == len(reference_ids)
    assert_no_leakage(pairs)


def test_reference_group_is_never_empty_and_shares_the_group_id() -> None:
    records = make_corpus({f"spk{i:02d}": 4 for i in range(5)})
    sampler = PairSampler(records, seed=3)

    for pair in sampler.sample(50):
        assert len(pair.reference_group) >= 1
        assert pair.target.speaker_id == pair.group_id
        for record in pair.reference_group:
            assert record.speaker_id == pair.group_id


def test_assert_no_leakage_raises_when_target_is_in_its_own_reference() -> None:
    target = make_utterance("spk-0000", "spk")
    other = make_utterance("spk-0001", "spk")
    broken = ReferenceTargetPair(
        target=target,
        reference_group=(other, target),
        group_id="spk",
    )

    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage([broken])


def test_assert_no_leakage_raises_on_duplicate_reference_utterances() -> None:
    target = make_utterance("spk-0000", "spk")
    other = make_utterance("spk-0001", "spk")
    broken = ReferenceTargetPair(
        target=target,
        reference_group=(other, other),
        group_id="spk",
    )

    with pytest.raises(AssertionError, match="repeats utterance"):
        assert_no_leakage([broken])


def test_assert_no_leakage_accepts_clean_pairs() -> None:
    records = make_corpus({"spk_a": 5, "spk_b": 5})
    sampler = PairSampler(records, seed=11)

    assert_no_leakage(sampler.sample(30))


def test_duplicate_utterance_ids_in_input_cannot_leak_into_the_reference() -> None:
    """manifestに重複行があっても reference_group に同じ発話が2度入らない。"""
    records = make_corpus({"spk": 4})
    records = records + list(records)  # 全recordを重複させる

    sampler = PairSampler(records, seed=5)

    assert len(sampler.eligible_groups()["spk"]) == 4
    assert_no_leakage(sampler.sample(50))


def test_empty_reference_group_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="at least one utterance"):
        ReferenceTargetPair(
            target=make_utterance("spk-0000", "spk"),
            reference_group=(),
            group_id="spk",
        )


# --- eligible_groups ---------------------------------------------------------


def test_group_with_a_single_utterance_is_not_eligible() -> None:
    records = make_corpus({"solo": 1, "duo": 2, "many": 5})

    groups = PairSampler(records, seed=0).eligible_groups()

    assert set(groups) == {"duo", "many"}
    assert "solo" not in groups


def test_min_utterances_per_group_filters_small_groups() -> None:
    records = make_corpus({"two": 2, "four": 4, "six": 6})

    groups = PairSampler(records, seed=0, min_utterances_per_group=5).eligible_groups()

    assert set(groups) == {"six"}


def test_exclude_group_ids_removes_those_groups_from_groups_and_samples() -> None:
    records = make_corpus({"keep_a": 4, "drop_me": 40, "keep_b": 4})

    sampler = PairSampler(records, seed=13, exclude_group_ids=frozenset({"drop_me"}))

    assert set(sampler.eligible_groups()) == {"keep_a", "keep_b"}
    sampled_groups = {pair.group_id for pair in sampler.sample(100)}
    assert sampled_groups == {"keep_a", "keep_b"}
    assert "drop_me" not in sampled_groups


def test_eligible_groups_returns_a_defensive_copy() -> None:
    records = make_corpus({"spk_a": 3, "spk_b": 3})
    sampler = PairSampler(records, seed=1)

    groups = sampler.eligible_groups()
    groups["spk_a"].clear()
    del groups["spk_b"]

    assert set(sampler.eligible_groups()) == {"spk_a", "spk_b"}
    assert len(sampler.eligible_groups()["spk_a"]) == 3


def test_groups_and_members_are_sorted_for_determinism() -> None:
    records = make_corpus({"spk_b": 3, "spk_a": 3})
    sampler = PairSampler(records, seed=1)

    groups = sampler.eligible_groups()

    assert list(groups) == ["spk_a", "spk_b"]
    for members in groups.values():
        ids = [record.utterance_id for record in members]
        assert ids == sorted(ids)


# --- determinism -------------------------------------------------------------


def test_same_seed_produces_identical_samples() -> None:
    records = make_corpus({f"spk{i:02d}": 8 for i in range(12)})

    first = PairSampler(records, seed=42).sample(50)
    second = PairSampler(records, seed=42).sample(50)

    assert pair_ids(first) == pair_ids(second)


def test_repeated_sample_calls_on_one_instance_are_identical() -> None:
    records = make_corpus({f"spk{i:02d}": 8 for i in range(12)})
    sampler = PairSampler(records, seed=42)

    assert pair_ids(sampler.sample(50)) == pair_ids(sampler.sample(50))


def test_different_seed_produces_different_samples() -> None:
    records = make_corpus({f"spk{i:02d}": 8 for i in range(12)})

    first = PairSampler(records, seed=42).sample(50)
    second = PairSampler(records, seed=43).sample(50)

    assert pair_ids(first) != pair_ids(second)


def test_sample_matches_the_head_of_iter_pairs() -> None:
    records = make_corpus({f"spk{i:02d}": 6 for i in range(8)})
    sampler = PairSampler(records, seed=9)

    streamed = list(islice(sampler.iter_pairs(speaker_uniform=True), 25))

    assert pair_ids(sampler.sample(25)) == pair_ids(streamed)


def test_input_order_does_not_change_the_result() -> None:
    """入力の並び順に依存しない（manifestの行順が変わっても再現できる）。"""
    records = make_corpus({f"spk{i:02d}": 6 for i in range(8)})
    shuffled = list(records)
    random.Random(1234).shuffle(shuffled)

    assert pair_ids(PairSampler(records, seed=2).sample(40)) == pair_ids(
        PairSampler(shuffled, seed=2).sample(40)
    )


def test_sample_accepts_an_iterator_of_records() -> None:
    records = make_corpus({"spk_a": 4, "spk_b": 4})

    from_list = PairSampler(records, seed=6).sample(20)
    from_iter = PairSampler(iter(records), seed=6).sample(20)

    assert pair_ids(from_list) == pair_ids(from_iter)


def test_sample_zero_returns_empty_list() -> None:
    sampler = PairSampler(make_corpus({"spk": 4}), seed=0)

    assert sampler.sample(0) == []


def test_sample_rejects_negative_n() -> None:
    sampler = PairSampler(make_corpus({"spk": 4}), seed=0)

    with pytest.raises(ValueError, match="non-negative"):
        sampler.sample(-1)


# --- speaker_uniform ---------------------------------------------------------


def test_speaker_uniform_changes_the_speaker_distribution() -> None:
    """発話数が極端に偏ったcorpusで、True と False の分布が実際に変わること。"""
    sizes = {"dominant": 400}
    sizes.update({f"tiny{i:02d}": 2 for i in range(9)})
    records = make_corpus(sizes)
    sampler = PairSampler(records, seed=17)
    total = 400

    uniform = Counter(
        pair.group_id for pair in islice(sampler.iter_pairs(speaker_uniform=True), total)
    )
    by_utterance = Counter(
        pair.group_id for pair in islice(sampler.iter_pairs(speaker_uniform=False), total)
    )

    # グループ一様: 10グループなので dominant はおよそ1/10
    assert uniform["dominant"] / total < 0.3
    # 発話一様: 418発話中400が dominant なのでほぼ独占する
    assert by_utterance["dominant"] / total > 0.8
    assert by_utterance["dominant"] > uniform["dominant"] * 2
    # グループ一様なら小さい話者もきちんと出てくる
    assert len(uniform) == 10


def test_speaker_uniform_true_is_the_default_for_iter_pairs() -> None:
    sizes = {"dominant": 200}
    sizes.update({f"tiny{i:02d}": 2 for i in range(9)})
    records = make_corpus(sizes)
    sampler = PairSampler(records, seed=23)

    default = pair_ids(islice(sampler.iter_pairs(), 40))
    explicit = pair_ids(islice(sampler.iter_pairs(speaker_uniform=True), 40))

    assert default == explicit


# --- reference length（案A: 複数発話の連結） ---------------------------------


def test_short_utterances_are_concatenated_toward_the_target_length() -> None:
    """2.0秒の発話しかないグループで、10秒目標なら5件連結される。"""
    records = make_corpus({"spk": 20}, duration=2.0)
    sampler = PairSampler(records, seed=4, target_reference_seconds=10.0)

    for pair in sampler.sample(20):
        assert len(pair.reference_group) == 5
        assert pair.reference_seconds == pytest.approx(10.0)


def test_a_single_long_utterance_is_enough() -> None:
    """1発話で目標長を超えるなら連結しない。"""
    records = make_corpus({"spk": 10}, duration=30.0)
    sampler = PairSampler(records, seed=4, target_reference_seconds=10.0)

    for pair in sampler.sample(20):
        assert len(pair.reference_group) == 1
        assert pair.reference_seconds == pytest.approx(30.0)


def test_overshooting_utterance_is_dropped_when_that_is_closer_to_the_target() -> None:
    """9.5秒 x N。10秒目標なら19.0秒より9.5秒の方が近いので1件で止まる。"""
    records = make_corpus({"spk": 10}, duration=9.5)
    sampler = PairSampler(records, seed=8, target_reference_seconds=10.0)

    for pair in sampler.sample(20):
        assert len(pair.reference_group) == 1
        assert pair.reference_seconds == pytest.approx(9.5)


def test_max_reference_utterances_is_never_exceeded() -> None:
    """目標長に届かなくても上限で打ち切る。"""
    records = make_corpus({"spk": 40}, duration=1.0)
    sampler = PairSampler(
        records,
        seed=21,
        target_reference_seconds=60.0,
        max_reference_utterances=3,
    )

    for pair in sampler.sample(30):
        assert len(pair.reference_group) == 3
        assert pair.reference_seconds == pytest.approx(3.0)


def test_reference_group_falls_back_to_all_available_utterances() -> None:
    """候補が足りなければ在るだけ使う（2発話グループならreferenceは1件）。"""
    records = make_corpus({"spk": 2}, duration=1.0)
    sampler = PairSampler(records, seed=2, target_reference_seconds=30.0)

    for pair in sampler.sample(10):
        assert len(pair.reference_group) == 1


def test_reference_seconds_sums_the_group_durations() -> None:
    group = (
        make_utterance("spk-0001", "spk", 1.5),
        make_utterance("spk-0002", "spk", 2.25),
    )
    pair = ReferenceTargetPair(
        target=make_utterance("spk-0000", "spk", 4.0),
        reference_group=group,
        group_id="spk",
    )

    assert pair.reference_seconds == pytest.approx(3.75)


def test_to_json_records_pair_provenance() -> None:
    records = make_corpus({"spk": 6}, duration=2.0)
    pair = PairSampler(records, seed=1, target_reference_seconds=4.0).sample(1)[0]

    payload = pair.to_json()

    assert payload["group_id"] == "spk"
    assert payload["target_id"] == pair.target.utterance_id
    assert payload["reference_ids"] == [r.utterance_id for r in pair.reference_group]
    assert payload["target_id"] not in payload["reference_ids"]
    assert payload["reference_seconds"] == pytest.approx(pair.reference_seconds)


# --- group_key の切り替え ----------------------------------------------------


def test_group_key_voice_cluster_id_groups_by_cluster() -> None:
    """同一クラスタの別speaker IDが1グループにまとまる（R-004の本命対策）。"""
    clusters = {"spk_a": "vc-1", "spk_b": "vc-1", "spk_c": "vc-2", "spk_d": None}
    records = make_corpus({"spk_a": 2, "spk_b": 2, "spk_c": 2, "spk_d": 4}, clusters=clusters)

    sampler = PairSampler(records, seed=31, group_key="voice_cluster_id")
    groups = sampler.eligible_groups()

    assert set(groups) == {"vc-1", "vc-2"}
    assert len(groups["vc-1"]) == 4  # spk_a + spk_b
    assert len(groups["vc-2"]) == 2
    assert {r.speaker_id for r in groups["vc-1"]} == {"spk_a", "spk_b"}


def test_records_without_a_voice_cluster_id_are_excluded() -> None:
    records = make_corpus({"spk_a": 4, "spk_b": 4}, clusters={"spk_a": "vc-1"})

    sampler = PairSampler(records, seed=1, group_key="voice_cluster_id")

    assert set(sampler.eligible_groups()) == {"vc-1"}
    assert {pair.group_id for pair in sampler.sample(20)} == {"vc-1"}


def test_voice_cluster_grouping_lets_reference_cross_speaker_ids() -> None:
    """クラスタ単位なので、referenceがtargetと別のspeaker IDから来る。"""
    clusters = {"spk_a": "vc-1", "spk_b": "vc-1"}
    records = make_corpus({"spk_a": 2, "spk_b": 2}, duration=5.0, clusters=clusters)
    sampler = PairSampler(
        records,
        seed=19,
        group_key="voice_cluster_id",
        target_reference_seconds=10.0,
    )

    pairs = sampler.sample(20)

    # 4発話・目標10秒 → reference 2件。同一speakerの候補は1件しか無いので必ず跨ぐ。
    assert all(len(pair.reference_group) == 2 for pair in pairs)
    assert any(
        record.speaker_id != pair.target.speaker_id
        for pair in pairs
        for record in pair.reference_group
    )
    assert_no_leakage(pairs)


def test_speaker_id_and_voice_cluster_id_give_different_groupings() -> None:
    clusters = {"spk_a": "vc-1", "spk_b": "vc-1"}
    records = make_corpus({"spk_a": 3, "spk_b": 3}, clusters=clusters)

    by_speaker = PairSampler(records, seed=1).eligible_groups()
    by_cluster = PairSampler(records, seed=1, group_key="voice_cluster_id").eligible_groups()

    assert set(by_speaker) == {"spk_a", "spk_b"}
    assert set(by_cluster) == {"vc-1"}


# --- 引数の検証 --------------------------------------------------------------


def test_unknown_group_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a field of Utterance"):
        PairSampler(make_corpus({"spk": 4}), seed=0, group_key="speaker")


def test_min_utterances_per_group_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_utterances_per_group"):
        PairSampler(make_corpus({"spk": 4}), seed=0, min_utterances_per_group=1)


def test_non_positive_target_reference_seconds_is_rejected() -> None:
    with pytest.raises(ValueError, match="target_reference_seconds"):
        PairSampler(make_corpus({"spk": 4}), seed=0, target_reference_seconds=0.0)


def test_max_reference_utterances_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_reference_utterances"):
        PairSampler(make_corpus({"spk": 4}), seed=0, max_reference_utterances=0)


def test_sampling_without_eligible_groups_raises() -> None:
    sampler = PairSampler(make_corpus({"solo_a": 1, "solo_b": 1}), seed=0)

    assert sampler.eligible_groups() == {}
    with pytest.raises(ValueError, match="No eligible groups"):
        sampler.sample(1)
    with pytest.raises(ValueError, match="No eligible groups"):
        sampler.iter_pairs()


def test_blank_group_value_is_treated_as_missing() -> None:
    records = make_corpus({"spk_a": 3})
    records += [make_utterance(f"blank-{i}", "   ") for i in range(3)]

    assert set(PairSampler(records, seed=0).eligible_groups()) == {"spk_a"}


# --- group_size_histogram ----------------------------------------------------


def test_group_size_histogram_counts_groups_per_size() -> None:
    records = make_corpus({"a": 1, "b": 1, "c": 3, "d": 5})

    assert group_size_histogram(records) == {1: 2, 3: 1, 5: 1}


def test_group_size_histogram_does_not_apply_sampler_filters() -> None:
    """閾値を決めるための素の分布なので、1発話グループも数える。"""
    records = make_corpus({"solo": 1, "duo": 2})

    assert group_size_histogram(records) == {1: 1, 2: 1}


def test_group_size_histogram_supports_voice_cluster_id() -> None:
    clusters = {"spk_a": "vc-1", "spk_b": "vc-1", "spk_c": "vc-2"}
    records = make_corpus({"spk_a": 2, "spk_b": 2, "spk_c": 3}, clusters=clusters)

    assert group_size_histogram(records, group_key="speaker_id") == {2: 2, 3: 1}
    assert group_size_histogram(records, group_key="voice_cluster_id") == {3: 1, 4: 1}


def test_group_size_histogram_is_sorted_by_size() -> None:
    records = make_corpus({"a": 5, "b": 1, "c": 3})

    assert list(group_size_histogram(records)) == [1, 3, 5]


def test_group_size_histogram_handles_empty_input() -> None:
    assert group_size_histogram([]) == {}
