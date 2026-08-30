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

"""学習ループが毎stepちがうペアを引くことを保証する。

S0 の最初の学習は、step ごとに :meth:`PairSampler.sample` を呼んでいた。
``sample()`` は仕様どおり毎回 RNG を作り直すので、3000 step すべてが
**同じ4発話** になり、flow loss は 1.02 -> 0.003 まで落ちた。
これは学習ではなく4発話の丸暗記で、別のペアで測ると R2 は負だった。

`sample()` 自体は正しい。誤りは「streamのつもりで毎step呼ぶ」使い方の側。
ここでは仕様と、正しい使い方の両方をテストで固定する。
"""

from __future__ import annotations

from itertools import islice

from cutetts.training.manifest import Utterance
from cutetts.training.pairing import PairSampler

GROUPS = 6
PER_GROUP = 5


def _records() -> list[Utterance]:
    rows = []
    for g in range(GROUPS):
        for i in range(PER_GROUP):
            rows.append(Utterance(
                utterance_id=f"ds:spk{g}:{i}",
                dataset_id="ds",
                speaker_id=f"spk{g}",
                voice_cluster_id=f"vc{g}",
                text_raw=f"テスト文 {g}-{i}",
                text_normalized=f"テスト文 {g}-{i}",
                audio_ref=f"spk{g}/{i}.wav",
                duration=4.0,
                sample_rate=24000,
                language="ja",
                game_id=None,
                split="train",
            ))
    return rows


def _sampler(seed: int = 42) -> PairSampler:
    return PairSampler(_records(), seed=seed, group_key="voice_cluster_id",
                       min_utterances_per_group=2, target_reference_seconds=10.0)


def test_sample_repeats_itself_when_called_again():
    """仕様の確認。``sample()`` は呼ぶたびに同じ列を返す。

    これ自体はバグではないが、step ごとに呼ぶと学習が成立しない。
    """
    sampler = _sampler()
    first = [p.target.utterance_id for p in sampler.sample(4)]
    second = [p.target.utterance_id for p in sampler.sample(4)]
    assert first == second


def test_iter_pairs_keeps_advancing():
    """stream から引けば毎回ちがうペアが出る。"""
    sampler = _sampler()
    stream = sampler.iter_pairs()
    batches = [tuple(p.target.utterance_id for p in islice(stream, 4)) for _ in range(20)]
    assert len(set(batches)) > 1, "stream が進んでいない"


def test_a_training_run_covers_far_more_than_one_batch_of_utterances():
    """学習に相当する回数だけ引くと、データの大半に触れる。

    丸暗記事故の再発検知が目的なので、閾値は「1 batch 分しか見ていない」
    を明確に弾く水準に置く。
    """
    sampler = _sampler()
    stream = sampler.iter_pairs()
    steps, batch_size = 200, 4
    seen = {p.target.utterance_id for p in islice(stream, steps * batch_size)}
    assert len(seen) > batch_size * 2, (
        f"{steps} step で {len(seen)} 発話しか見ていない。"
        "stream ではなく sample() を毎step呼んでいないか確認すること"
    )
    # 200 step もあれば全 30 発話に触れるはず
    assert len(seen) == GROUPS * PER_GROUP


def test_stream_can_be_advanced_for_resume():
    """resume で消費済み分を読み飛ばすと、続きから引ける。"""
    sampler = _sampler()
    full = [p.target.utterance_id for p in islice(sampler.iter_pairs(), 40)]

    resumed = sampler.iter_pairs()
    next(islice(resumed, 20, 20), None)          # 20件を捨てる
    tail = [p.target.utterance_id for p in islice(resumed, 20)]
    assert tail == full[20:]
