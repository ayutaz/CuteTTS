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

"""停止の健全性（G1 / `training.stopping`）。

**実際に観測した3例**を回帰の基準にする（2026-09-17 の試聴）。
"""

from __future__ import annotations

from cutetts.training.stopping import (
    has_extra_tail,
    has_self_repeat,
    stop_health,
    trailing_insertions,
)

# 実測例（参照文、ASR転写）
REAL_EXTRA_TAIL = [
    ("中華料理のお店へ、湊さんと行きませんか。",
     "中華料理のお店へみなとさんと行きませんかぶっ込ませんか"),
    ("切符を拾ったので、案内所へ持って行きました。",
     "切符を拾ったので案内証へ持っていきましたやっとなんて行くました"),
    ("価格は千二百八十円、消費税込みです。",
     "価格は1200円教師税込ですズッコメです"),
]

REAL_CLEAN = [
    ("明日の待ち合わせは、駅の南口で大丈夫ですか。",
     "明日の待ち合わせは駅の南口で大丈夫ですか"),
    ("えっ、本当ですか。それは、すごいじゃないですか。",
     "えっ本当ですかそれはすごいじゃないですか"),
    ("箸を持つ手と、橋を渡る足。", "橋を持つ手と橋を渡る足"),
]


def test_余計な尾を検出する():
    for reference, hypothesis in REAL_EXTRA_TAIL:
        assert has_extra_tail(reference, hypothesis), hypothesis


def test_正しい転写は検出しない():
    """**表記違いや1文字の誤りで誤検出してはいけない。**"""
    for reference, hypothesis in REAL_CLEAN:
        assert not has_extra_tail(reference, hypothesis), hypothesis


def test_長さ超過を数える():
    from cutetts.training.stopping import excess_chars
    assert excess_chars("こんにちは", "こんにちはですよね") == 4
    assert excess_chars("こんにちは", "こんにちは") == 0
    assert excess_chars("こんにちは", "こんにち") == -1


def test_末尾の挿入文字数を数える():
    assert trailing_insertions("こんにちは", "こんにちはですよね") == 4
    assert trailing_insertions("こんにちは", "こんにちは") == 0
    # 途中の挿入は末尾ではないので数えない
    assert trailing_insertions("こんにちは", "こんにですちは") == 0


def test_参照が空なら全部が余計な尾():
    assert trailing_insertions("", "あいうえお") == 5
    assert not has_extra_tail("", "あいうえお")  # 参照が無ければ判定しない


def test_自己反復を検出する():
    assert has_self_repeat("行きませんか行きませんか")
    assert has_self_repeat("そうですねそうですね")
    assert not has_self_repeat("明日の待ち合わせは駅の南口で大丈夫ですか")


def test_短い反復で誤検出しない():
    """`いいえ` の `い` のような1文字は拾わない（min_unit=2）。"""
    assert not has_self_repeat("いいえ")


def test_集計は割合で返す():
    rows = []
    for reference, hypothesis in REAL_EXTRA_TAIL + REAL_CLEAN:
        rows.append({"text": reference, "hypothesis": hypothesis,
                     "status": "ok", "subset": "in_domain", "seconds": 5.0})
    health = stop_health(rows, label="x")
    assert health.n == 6
    assert health.extra_tail_rate == 0.5          # 3/6
    assert health.truncated_rate == 0.0
    assert health.max_excess_chars >= 9


def test_打ち切りは別に数える():
    rows = [{"text": "こんにちは", "hypothesis": "こんにちは", "status": "ok",
             "subset": "in_domain", "seconds": 64.0}]
    health = stop_health(rows)
    assert health.truncated_rate == 1.0
    assert health.extra_tail_rate == 0.0


def test_subsetで絞る():
    rows = [
        {"text": "こんにちは", "hypothesis": "こんにちはですよね", "status": "ok",
         "subset": "in_domain", "seconds": 5.0},
        {"text": "こんにちは", "hypothesis": "こんにちはですよね", "status": "ok",
         "subset": "phonetic", "seconds": 5.0},
    ]
    assert stop_health(rows).n == 1
    assert stop_health(rows, subsets=None).n == 2


def test_失敗した行は数えない():
    rows = [{"text": "こんにちは", "status": "error", "subset": "in_domain"}]
    assert stop_health(rows).n == 0
