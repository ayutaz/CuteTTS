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

"""停止の健全性（G1）。**「文が終わった後に発話を続ける」を数える。**

R-021 では `max_decode_length`（64.0秒）への張り付きだけを別勘定にした。
しかし試聴すると、**打ち切りに至らない軽い形**が頻繁に出る（9文中4文）。

    参照: 中華料理のお店へ、湊さんと行きませんか。
    転写: 中華料理のお店へみなとさんと行きませんか**ぶっ込ませんか**

    参照: 切符を拾ったので、案内所へ持って行きました。
    転写: 切符を拾ったので案内証へ持っていきました**やっとなんて行くました**

**これは発音の誤りではなく停止の失敗**だが、CERには誤りとして入る。
打ち切り勘定には出ないので、いまは指標が無い。

## 数え方

**転写が参照文よりどれだけ長いか**で数える（`excess_chars`）。
**4文字以上、かつ参照長の20%以上**を「余計な尾」とする。

**挿入の位置で数える方法は2回失敗した。**

1. 「末尾に連続する挿入」→ 尾が参照文の末尾と同じ語で終わると拾えない
   （`価格は1200円教師税込です**ズッコメ**です` の `です`）
2. 「終盤で終わる最長の挿入」→ 尾に参照文と一致する文字が混ざると、
   アラインメントが尾を分割する（`っていきましたや` + `となん`）

**最適経路は一意でないので、位置に頼ると不安定になる。**
「喋り続けた」を直接表すのは長さなので、長さ超過で数える。
`insertion_runs` / `extra_tail_chars` は**どこに付いたかを見る診断用**に残す。

`has_self_repeat` は**文字列としての自己反復**を別に見る
（`…行きませんか行きませんか`）。

### この指標が区別できないこと

* **ASR自身の繰り返し・幻聴**。転写が長い原因がTTSかASRかは分けられない
* **表記で長くなる読み間違い**（`1200円` を `せんにひゃく…` と書かれる等）
* したがって**絶対値ではなく、同じ経路で測ったrun間の比較に使う**

**閾値は観測3例（+50% / +55% / +25%）と正常3例（0%）から決めた。**
結果を見てから動かすと、その操作だけで率が動く。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "DEFAULT_MIN_CHARS",
    "DEFAULT_MIN_RATIO",
    "DEFAULT_MIN_EXCESS_RATIO",
    "StopHealth",
    "excess_chars",
    "extra_tail_chars",
    "has_extra_tail",
    "insertion_runs",
    "has_self_repeat",
    "normalize",
    "stop_health",
    "trailing_insertions",
]

_PUNCT = re.compile(r"[\s、。「」『』・…‥！？!?,.\-―ー~〜\"'()（）]")

#: 「余計な尾」と数える最小の文字数。短い挿入はASRの揺れと区別できない。
DEFAULT_MIN_CHARS = 4

#: 同じく、参照文の長さに対する比（**挿入の位置を見る診断用**）。
DEFAULT_MIN_RATIO = 0.15

#: 「余計な尾」と数える最小の長さ超過（参照長に対する比）。
#: 観測3例は +50% / +55% / +25%、正常3例は 0% だった。
DEFAULT_MIN_EXCESS_RATIO = 0.20

#: 「終盤」の定義。仮説の末尾このぶんの位置で終わる挿入を停止の失敗として数える。
DEFAULT_TAIL_WINDOW = 0.20


def normalize(text: str) -> str:
    """CERと同じ正規化（`evaluate_japanese_cer.normalize` と同一の規約）。"""
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text))


def trailing_insertions(reference: str, hypothesis: str) -> int:
    """最適経路で**末尾に連続する挿入**の文字数を返す。

    仮説の末尾から遡り、「参照を使い切った状態」に留まる区間の長さを測る。
    DPの最終行だけを見ればよい: `row[j]` は参照全体と仮説の先頭 j 文字の
    編集距離。末尾の k 文字が純粋な挿入なら `row[m] == row[m-k] + k`。

    Args:
        reference: 参照文（正規化済みでなくてよい）。
        hypothesis: ASR転写。

    Returns:
        末尾の挿入文字数。参照が空なら仮説の長さ。
    """
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return len(hyp)
    if not hyp:
        return 0

    previous = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            current[j] = min(previous[j] + 1,        # 削除
                             current[j - 1] + 1,     # 挿入
                             previous[j - 1] + (rc != hc))
        previous = current

    total = previous[len(hyp)]
    count = 0
    for k in range(1, len(hyp) + 1):
        if previous[len(hyp) - k] + k == total:
            count = k
        else:
            break
    return count


def insertion_runs(reference: str, hypothesis: str) -> list[tuple[int, int]]:
    """最適経路の挿入の連なりを `(開始, 終了)`（仮説の文字位置）で返す。

    最適経路は複数ありうるので、**一致・置換を優先**し、次に挿入、最後に削除を
    選ぶ順で後ろから辿る（決定論的にする）。
    """
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref or not hyp:
        return [(0, len(hyp))] if hyp else []

    table = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        table[i][0] = i
    for j in range(len(hyp) + 1):
        table[0][j] = j
    for i, rc in enumerate(ref, 1):
        for j, hc in enumerate(hyp, 1):
            table[i][j] = min(table[i - 1][j] + 1,
                              table[i][j - 1] + 1,
                              table[i - 1][j - 1] + (rc != hc))

    runs: list[tuple[int, int]] = []
    i, j, run_end = len(ref), len(hyp), None
    while i > 0 or j > 0:
        if (i > 0 and j > 0
                and table[i][j] == table[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])):
            if run_end is not None:
                runs.append((j, run_end))
                run_end = None
            i, j = i - 1, j - 1
        elif j > 0 and table[i][j] == table[i][j - 1] + 1:
            if run_end is None:
                run_end = j
            j -= 1
        else:
            if run_end is not None:
                runs.append((j, run_end))
                run_end = None
            i -= 1
    if run_end is not None:
        runs.append((j, run_end))
    return list(reversed(runs))


def extra_tail_chars(reference: str, hypothesis: str, *,
                     tail_window: float = DEFAULT_TAIL_WINDOW) -> int:
    """**終盤で終わる**挿入の連なりのうち、最長の文字数。

    「終盤」は仮説の末尾 `tail_window` の割合。中盤の挿入は読み間違いで
    ありうるので数えない。
    """
    hyp = normalize(hypothesis)
    if not hyp:
        return 0
    boundary = len(hyp) * (1.0 - tail_window)
    return max((end - start
                for start, end in insertion_runs(reference, hypothesis)
                if end >= boundary), default=0)


def excess_chars(reference: str, hypothesis: str) -> int:
    """転写が参照文より何文字長いか（負なら 0 ではなく負のまま返す）。"""
    return len(normalize(hypothesis)) - len(normalize(reference))


def has_extra_tail(reference: str, hypothesis: str, *,
                   min_chars: int = DEFAULT_MIN_CHARS,
                   min_excess_ratio: float = DEFAULT_MIN_EXCESS_RATIO) -> bool:
    """**喋り続けた**と数えるか。長さ超過で判定する（module docstring参照）。"""
    ref = normalize(reference)
    if not ref:
        return False
    excess = excess_chars(reference, hypothesis)
    return excess >= min_chars and excess >= min_excess_ratio * len(ref)


def has_self_repeat(hypothesis: str, *, min_unit: int = 2,
                    max_unit: int = 12, min_repeats: int = 2) -> bool:
    """転写の末尾が同じ断片の繰り返しで終わっているか。

    `…行きませんか行きませんか` のように、**自分の言ったことを繰り返す**形。
    参照文の一部を繰り返すと挿入として拾いにくいので別に見る。
    """
    hyp = normalize(hypothesis)
    for unit in range(min_unit, max_unit + 1):
        if len(hyp) < unit * min_repeats:
            break
        tail = hyp[-unit:]
        repeats = 1
        while hyp[-unit * (repeats + 1):len(hyp) - unit * repeats] == tail:
            repeats += 1
        if repeats >= min_repeats:
            return True
    return False


@dataclass(frozen=True)
class StopHealth:
    """1実行の停止健全性。割合はすべて 0〜1。"""

    label: str
    n: int
    extra_tail_rate: float
    self_repeat_rate: float
    truncated_rate: float
    mean_excess_chars: float
    max_excess_chars: int

    def as_dict(self) -> dict:
        return {
            "label": self.label, "n": self.n,
            "extra_tail_rate": self.extra_tail_rate,
            "self_repeat_rate": self.self_repeat_rate,
            "truncated_rate": self.truncated_rate,
            "mean_excess_chars": self.mean_excess_chars,
            "max_excess_chars": self.max_excess_chars,
        }


def stop_health(rows: Sequence[dict], *, label: str = "",
                truncation_seconds: float = 63.9,
                subsets: Iterable[str] | None = ("in_domain",),
                **thresholds) -> StopHealth:
    """行から停止健全性を集計する。**生成をやり直さない。**

    Args:
        rows: `text` / `hypothesis` / `status` / `seconds` を持つ行。
        truncation_seconds: これ以上の長さを打ち切りとして数える（R-021）。
        subsets: 数える subset。`None` なら全部。
    """
    wanted = set(subsets) if subsets is not None else None
    extras: list[int] = []
    repeats = truncated = 0
    for row in rows:
        if row.get("status") != "ok" or row.get("hypothesis") is None:
            continue
        if wanted is not None and row.get("subset") not in wanted:
            continue
        extras.append(excess_chars(row["text"], row["hypothesis"]))
        if has_self_repeat(row["hypothesis"]):
            repeats += 1
        if (row.get("seconds") or 0) >= truncation_seconds:
            truncated += 1

    n = len(extras)
    if not n:
        return StopHealth(label, 0, 0.0, 0.0, 0.0, 0.0, 0)
    flagged = sum(
        1 for row, extra in zip(
            (r for r in rows
             if r.get("status") == "ok" and r.get("hypothesis") is not None
             and (wanted is None or r.get("subset") in wanted)), extras)
        if has_extra_tail(row["text"], row["hypothesis"], **thresholds))
    return StopHealth(
        label=label, n=n,
        extra_tail_rate=flagged / n,
        self_repeat_rate=repeats / n,
        truncated_rate=truncated / n,
        mean_excess_chars=sum(extras) / n,
        max_excess_chars=max(extras),
    )
