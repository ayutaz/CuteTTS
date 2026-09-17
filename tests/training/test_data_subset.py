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

"""D1の部分集合づくり（`scripts/build_data_subset.py`）。

**データ量だけを変えたいので、ここが狂うと比較そのものが無意味になる。**
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from build_data_subset import select_clusters, subset_records  # noqa: E402

from cutetts.training.manifest import Utterance


def utterance(uid: str, cluster: str, seconds: float, split: str = "train") -> Utterance:
    return Utterance(
        utterance_id=uid, dataset_id="gol", audio_ref=f"{uid}.wav",
        text_raw="こんにちは", speaker_id=cluster, duration=seconds,
        sample_rate=48000, voice_cluster_id=cluster, split=split,
    )


def sample_records() -> list[Utterance]:
    records = []
    for cluster in range(10):
        for index in range(10):
            records.append(utterance(f"u{cluster}-{index}", f"c{cluster}", 360.0))
    records.append(utterance("dev1", "c0", 100.0, split="dev-seen"))
    records.append(utterance("dev2", "c1", 100.0, split="dev-zero-shot"))
    return records


def test_選んだ時間が要求に届く():
    records = [r for r in sample_records() if r.split == "train"]
    chosen, hours = select_clusters(records, target_hours=3.0, seed=1)
    assert hours >= 3.0
    # 1 clusterは1時間なので、3時間なら3 cluster で足りる
    assert len(chosen) == 3


def test_同じseedなら同じ部分集合になる():
    records = [r for r in sample_records() if r.split == "train"]
    first, _ = select_clusters(records, target_hours=4.0, seed=7)
    second, _ = select_clusters(records, target_hours=4.0, seed=7)
    assert first == second


def test_seedが違えば選ばれるclusterが変わる():
    records = [r for r in sample_records() if r.split == "train"]
    first, _ = select_clusters(records, target_hours=4.0, seed=1)
    second, _ = select_clusters(records, target_hours=4.0, seed=2)
    assert first != second


def test_入力の並び順に依存しない():
    records = [r for r in sample_records() if r.split == "train"]
    forward, _ = select_clusters(records, target_hours=4.0, seed=3)
    backward, _ = select_clusters(list(reversed(records)), target_hours=4.0, seed=3)
    assert forward == backward


def test_devは絞られない():
    """**dev を絞ると学習中の比較が別条件になる。**"""
    kept, chosen, _ = subset_records(sample_records(), target_hours=2.0, seed=5)
    splits = {}
    for record in kept:
        splits[record.split] = splits.get(record.split, 0) + 1
    assert splits["dev-seen"] == 1
    assert splits["dev-zero-shot"] == 1
    assert splits["train"] == 10 * len(chosen)


def test_元の並び順を保つ():
    records = sample_records()
    kept, _, _ = subset_records(records, target_hours=2.0, seed=5)
    order = [r.utterance_id for r in records if r in kept]
    assert [r.utterance_id for r in kept] == order
