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

"""抑揚とアクセントの測定（M1）。

**CERはこれらを測れない。** 聴取で残った指摘は読み間違い45% / **抑揚40%** /
**アクセント30%** だが、CERはASRの転写を見るので、`箸` を `橋` のアクセントで
読んでも転写が同じなら誤りにならない。**測る手段が無いと、直ったかどうかを
判定できない。**

このプロジェクトの失敗はすべて測定の失敗だった（R-012 / R-020 / R-021 / R-029）。
**T1（lr探索）にGPU費用を払う前に、判定できる指標を用意する。**

## 測るもの

1. **抑揚の幅** — 有声フレームのF0を半音に直したときのばらつき。
   「棒読み」は幅が小さい。`ProsodyStats.semitone_sd` / `semitone_range`。
2. **輪郭の一致** — 同一話者・同一文の人間音声と比べたF0輪郭の相関。
   `contour_similarity`。
3. **アクセント型** — `pyopenjtalk` が返すアクセント核から、モーラごとの
   高低パターンを作る。`accent_plan`。

## F0推定器の選定（D-040）

`pyworld`（harvest + stonemask）。調波20本のパルス列で **90〜500Hz を誤差0.01%**、
無音・白色雑音を正しく無声と判定する。

`torchaudio.functional.detect_pitch_frequency` は**有声/無声を判定しない**
（全フレームを有声として返す）ので使えない。無音区間のゴミが混ざり、
3秒の発話で32半音というありえない幅が出た。

**ただし `pyworld` にも罠がある（R-033）。** 広い範囲（55〜700Hz）を一度に
探すと**倍音に乗る**。実測で男声（約110Hz）を **392Hz** と報告し、
実音声8件中2件が 3.3〜3.8倍（537.6 / 586.6Hz）になった。
`track_f0` は上限350と700の中央値を比べ、跳んでいれば狭い範囲で推定し直す。
**倍音に乗ったフレームを整数倍で引き戻す方式は棄却した** — どの係数を
選ぶかが中心の推定誤差に強く依存し、実測で 185Hz を 292Hz にしてしまった。

    >>> stats = measure(waveform, 24000)
    >>> stats.semitone_sd > 0
    True
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np

#: F0の解析間隔（ミリ秒）。
FRAME_PERIOD_MS = 10.0

#: 探索するF0の下限・上限（Hz）。女性の高い声まで含める。
#:
#: 下限を 55Hz まで下げると、**白色雑音に 55〜60Hz の偽の周期**を見つける
#: （実測で 8/101 フレームが有声になった）。人の声で 65Hz を下回るのは稀。
F0_FLOOR_HZ = 65.0
F0_CEIL_HZ = 700.0

#: 倍音誤りを見つけるための低い方の上限（Hz）。
#:
#: **広い範囲を一度に探すと倍音に乗る。** 実測で、男声（約110Hz）を
#: 上限700Hzで探すと **392Hz**（約3.5倍）と報告した。実音声でも8件中2件が
#: 3.3〜3.8倍の誤りを出した（537.6Hz / 586.6Hz）。
#: **抑揚の幅が20半音（1.7オクターブ）という不自然な値は、この誤りだった。**
#:
#: 低い上限と高い上限の中央値を比べ、跳んでいれば低い方を採る。
#: **`harvest` の実行時間は上限にも解析間隔にも標本化率にもほとんど依らない**
#: （実測: 10秒の音声で 65-350 も 65-700 も約1.8秒）ので、
#: **段数を減らすことだけが速くなる道**。4段→2段で 7.1秒→3.6秒。
F0_COARSE_CEIL_HZ = 350.0

#: 梯子を上る際、中央値がこの倍率以上に跳んだら倍音誤りと見なす。
#:
#: 上限による切り詰めが解けるときの上がり方は 1.6倍程度（実測 173.6→273.6）、
#: 倍音誤りは 3.3〜3.8倍（実測）。その間に閾値を置く。
F0_OCTAVE_RATIO = 1.8

#: 最終パスで中央値の何倍まで許すか。話者のF0が動く幅を覆いつつ、
#: 3倍（倍音が出る位置）には届かせない。1.9 / 2.2 / 2.6 で中央値は
#: ほぼ変わらず（実測7件で差は最大 6Hz）、有声率が最も高い 2.2 を採った。
F0_SPAN = 2.2

#: 統計を出すのに要る最小の有声フレーム数。これを下回るとNaNを返す。
MIN_VOICED_FRAMES = 10

#: 両端から落とすフレーム数。
#:
#: `harvest` は**最初と最後のフレームだけ値が壊れる**。一定200Hzの合成音で
#: 先頭が -1.13半音、末尾が -3.66半音になり、それだけで sd が
#: 0.00005 → **0.378** に膨らんだ（5〜95パーセンタイル幅は 0.0003 のまま）。
#: 実音声では端が無音なので普段は無声として落ちるが、**落ちない場合に
#: すべての測定を膨らませる**ので明示的に捨てる。
EDGE_FRAMES_DROPPED = 1

#: 小書き仮名。直前のモーラに結合する。
_SMALL_KANA = frozenset("ャュョァィゥェォヮゃゅょぁぃぅぇぉゎ")


@dataclass(frozen=True)
class ProsodyStats:
    """1発話の抑揚の統計。

    `semitone_sd` と `semitone_range` は**話者の平均音高に依存しない**
    （中央値を基準にした半音で測る）ので、男女や話者間で比べられる。
    """

    seconds: float
    n_frames: int
    n_voiced: int
    voiced_ratio: float
    median_hz: float
    semitone_sd: float
    semitone_range: float

    @property
    def is_valid(self) -> bool:
        return self.n_voiced >= MIN_VOICED_FRAMES


def _harvest(samples: np.ndarray, sample_rate: int,
             floor: float, ceil: float) -> np.ndarray:
    import pyworld

    coarse, times = pyworld.harvest(
        samples, sample_rate, f0_floor=floor, f0_ceil=ceil,
        frame_period=FRAME_PERIOD_MS)
    return pyworld.stonemask(samples, coarse, times, sample_rate)


def _median_of(f0: np.ndarray) -> float:
    voiced = f0[f0 > 0]
    if voiced.size < MIN_VOICED_FRAMES:
        return float("nan")
    return float(np.median(voiced))


def estimate_center_hz(samples: np.ndarray, sample_rate: int,
                       wide: np.ndarray | None = None) -> float:
    """話者のF0の中心（中央値）を、**倍音に乗らずに**求める。

    上限 `F0_CEIL_HZ` と `F0_COARSE_CEIL_HZ` の2つで中央値を出し、
    高い方が `F0_OCTAVE_RATIO` 倍以上なら**倍音に乗っている**と見なして
    低い方を採る。判定できなければ NaN。

    `wide` に上限 `F0_CEIL_HZ` の結果を渡すと、その分の計算を省く。
    """
    if wide is None:
        wide = _harvest(samples, sample_rate, F0_FLOOR_HZ, F0_CEIL_HZ)
    top = _median_of(wide)
    coarse = _median_of(
        _harvest(samples, sample_rate, F0_FLOOR_HZ, F0_COARSE_CEIL_HZ))
    if coarse != coarse:
        return top
    if top != top:
        return coarse
    return coarse if top / coarse >= F0_OCTAVE_RATIO else top


#: 局所のオクターブ跳びを直すときに見る窓（フレーム数、前後合わせて）。
#: 10ms刻みなので 41 フレームは約0.4秒＝3〜4モーラ分。
LOCAL_WINDOW_FRAMES = 41

#: 局所中央値からこれ以上離れたフレームを、跳びの候補とする（半音）。
#: 1オクターブ（12半音）より手前、通常の抑揚（±6半音程度）より外に置く。
LOCAL_JUMP_SEMITONES = 7.0


def correct_local_jumps(f0: np.ndarray) -> np.ndarray:
    """**隣と比べて2倍・1/2倍に飛んだフレーム**を引き戻す。

    `track_f0` の中心は発話全体で1つなので、その窓の中に収まってしまう
    跳びは捕まらない。実測で、隣が約100Hzのところに **207Hz**（ちょうど2倍）
    が出て、モーラ単位のアクセント判定を壊していた。

    局所中央値から `LOCAL_JUMP_SEMITONES` 以上離れたフレームについて、
    2倍・1/2倍・3倍・1/3倍を試し、局所中央値に最も近づく候補を採る。
    どれも近づかなければそのまま残す（本物の抑揚かもしれない）。
    """
    values = np.array(f0, dtype=np.float64, copy=True)
    voiced = np.flatnonzero(values > 0)
    if voiced.size < 3:
        return values
    half = LOCAL_WINDOW_FRAMES // 2
    for index in voiced:
        low = max(0, index - half)
        high = min(values.size, index + half + 1)
        window = values[low:high]
        neighbours = window[(window > 0)]
        if neighbours.size < 3:
            continue
        local = float(np.median(neighbours))
        if local <= 0:
            continue
        distance = abs(semitones(values[index] / local))
        if distance < LOCAL_JUMP_SEMITONES:
            continue
        best, best_distance = values[index], distance
        for factor in (2.0, 0.5, 3.0, 1.0 / 3.0):
            candidate = values[index] / factor
            moved = abs(semitones(candidate / local))
            if moved < best_distance:
                best, best_distance = candidate, moved
        values[index] = best
    return values


def track_f0(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """F0の系列（Hz）を返す。**無声フレームは 0**。

    **2段階で推定する。** 先に話者のF0の中心を倍音に乗らずに求め
    （`estimate_center_hz`）、その周り `F0_SPAN` 倍の範囲で本番を推定する。
    一度に広い範囲を探すと倍音に乗る（`F0_COARSE_CEIL_HZ` 参照）。
    倍音に乗っていなければ広域パスをそのまま使い、1回分を省く。
    `F0_SPAN` は 1.9 / 2.2 / 2.6 で中央値がほぼ変わらないことを実測して選んだ。

    `pyworld` が要る（`[prosody]` extra）。
    """
    samples = np.asarray(waveform, dtype=np.float64)
    if samples.ndim > 1:                       # (channel, time) も (time, channel) も
        samples = samples.mean(axis=0 if samples.shape[0] < samples.shape[-1] else 1)
    if samples.size == 0:
        return np.zeros(0, dtype=np.float64)

    # **有声/無声の判定は広い範囲のパスから取る。** 狭い範囲だけで探すと、
    # 白色雑音のような非周期信号にも「その帯域での周期」を見つけてしまう
    # （実測で白色雑音が 1/101 → 62/101 フレーム）。上限350でも 15/101 出る。
    wide = _harvest(samples, sample_rate, F0_FLOOR_HZ, F0_CEIL_HZ)
    center = estimate_center_hz(samples, sample_rate, wide=wide)
    if center != center:
        f0 = np.zeros_like(wide)
    elif abs(semitones(_median_of(wide) / center)) < 1.0:
        # 倍音に乗っていない。**広域パスをそのまま使い、1回分を省く。**
        f0 = correct_local_jumps(wide)
    else:
        f0 = _harvest(samples, sample_rate,
                      max(F0_FLOOR_HZ, center / F0_SPAN),
                      min(F0_CEIL_HZ, center * F0_SPAN))
        size = min(f0.size, wide.size)
        f0 = f0[:size] * (wide[:size] > 0)     # 広域パスで無声なら無声
        f0 = correct_local_jumps(f0)           # 隣と比べた2倍・1/2倍を直す
    if f0.size > 2 * EDGE_FRAMES_DROPPED:     # 端の壊れたフレームを無声にする
        f0[:EDGE_FRAMES_DROPPED] = 0.0
        f0[-EDGE_FRAMES_DROPPED:] = 0.0
    return f0


def semitone_contour(f0: np.ndarray) -> np.ndarray:
    """有声フレームだけを、中央値を0とする半音へ直した系列。

    話者の平均音高を落とすので、**別の話者どうしでも比べられる**。
    """
    voiced = np.asarray(f0)[np.asarray(f0) > 0]
    if voiced.size == 0:
        return np.zeros(0, dtype=np.float64)
    return 12.0 * np.log2(voiced / np.median(voiced))


def measure(waveform: np.ndarray, sample_rate: int, *,
            f0: np.ndarray | None = None) -> ProsodyStats:
    """1発話の抑揚を測る。

    `f0` を渡すと `track_f0` を呼び直さない。**同じ音声に対して何度も
    呼ぶときは必ず渡すこと**（`track_f0` は1発話で約2秒かかる）。
    """
    if f0 is None:
        f0 = track_f0(waveform, sample_rate)
    samples = np.asarray(waveform)
    length = samples.shape[-1] if samples.ndim else 0
    voiced = f0[f0 > 0]
    if voiced.size < MIN_VOICED_FRAMES:
        return ProsodyStats(
            seconds=length / sample_rate, n_frames=int(f0.size),
            n_voiced=int(voiced.size),
            voiced_ratio=float(voiced.size / f0.size) if f0.size else 0.0,
            median_hz=float("nan"), semitone_sd=float("nan"),
            semitone_range=float("nan"))
    contour = semitone_contour(f0)
    low, high = np.percentile(contour, [5, 95])
    return ProsodyStats(
        seconds=length / sample_rate,
        n_frames=int(f0.size),
        n_voiced=int(voiced.size),
        voiced_ratio=float(voiced.size / f0.size),
        median_hz=float(np.median(voiced)),
        semitone_sd=float(contour.std()),
        semitone_range=float(high - low),
    )


def resample_contour(contour: np.ndarray, length: int) -> np.ndarray:
    """輪郭を指定長へ線形補間する。長さの違う2発話を比べるため。"""
    contour = np.asarray(contour, dtype=np.float64)
    if contour.size == 0 or length <= 0:
        return np.zeros(max(length, 0), dtype=np.float64)
    if contour.size == 1:
        return np.full(length, contour[0], dtype=np.float64)
    source = np.linspace(0.0, 1.0, contour.size)
    target = np.linspace(0.0, 1.0, length)
    return np.interp(target, source, contour)


def contour_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """2つのF0輪郭（半音）の相関。同一文の別発話どうしで使う。

    **長さは線形に伸縮して揃える。** DTWは使わない。DTWは時間のずれを
    吸収してしまうが、**間の取り方も抑揚の一部**なので吸収させたくない。
    その代わり「音高は合っているが話速が違う」ケースも低く出る。
    **話速の違いを別に見ること**（`ProsodyStats.seconds`）。

    片方が短すぎる（`MIN_VOICED_FRAMES` 未満）ときは NaN。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < MIN_VOICED_FRAMES or b.size < MIN_VOICED_FRAMES:
        return float("nan")
    length = max(a.size, b.size)
    left, right = resample_contour(a, length), resample_contour(b, length)
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def time_aligned_contour(f0: np.ndarray) -> np.ndarray:
    """**時間軸を保った**半音の輪郭（M4c で追加）。

    `semitone_contour` は**有声フレームだけを詰める**ので、有声・無声の
    判定が少し食い違うだけで系列全体がずれる。実測（人間音声30件）で、
    **同じ発話を VAE で往復させただけで相関が 0.417 まで落ちた**
    （有声判定の一致は 85.6%）。M2 の「天井」+0.382（別テイク同士）と
    ほぼ同じで、**指標が「同じ発話」と「別テイク」を区別できていない**。

    こちらは無声フレームを**前後の有声から線形補間で埋める**ので、
    フレーム番号がそのまま時間に対応する。間（ポーズ）は平らな区間として残る。

    | 比較 | `semitone_contour` | **こちら** |
    |---|---:|---:|
    | 同一音声 | 1.000 | 1.000 |
    | VAE往復（同一発話） | 0.417 | R-050 参照 |
    | 別テイク（同一話者・同一台詞） | 0.382 | R-050 参照 |
    | 別の文（床） | -0.009 | R-050 参照 |

    **既存の値との継続性のため `semitone_contour` は変えない。**
    公表済みの数値はすべてあちらで測ってある。
    """
    array = np.asarray(f0, dtype=np.float64)
    voiced = array > 0
    if not voiced.any():
        return np.zeros(0, dtype=np.float64)
    index = np.arange(array.size, dtype=np.float64)
    filled = np.interp(index, index[voiced], array[voiced])
    return 12.0 * np.log2(filled / np.median(array[voiced]))


def contour_similarity_time(a_f0: np.ndarray, b_f0: np.ndarray) -> float:
    """`time_aligned_contour` で取った輪郭の相関。

    引数は **F0（Hz、無声は 0）** であって半音の輪郭ではない
    （`contour_similarity` は輪郭を受け取るので、間違えないよう名前を分けた）。
    """
    left = time_aligned_contour(a_f0)
    right = time_aligned_contour(b_f0)
    if left.size < MIN_VOICED_FRAMES or right.size < MIN_VOICED_FRAMES:
        return float("nan")
    length = max(left.size, right.size)
    left = resample_contour(left, length)
    right = resample_contour(right, length)
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def split_moras(reading: str) -> list[str]:
    """片仮名の読みをモーラへ分ける。小書き仮名は直前に結合する。

        >>> split_moras("チュウカ")
        ['チュ', 'ウ', 'カ']
    """
    moras: list[str] = []
    for char in reading:
        if moras and char in _SMALL_KANA:
            moras[-1] += char
        else:
            moras.append(char)
    return moras


@dataclass(frozen=True)
class AccentPhrase:
    """1アクセント句。

    `nucleus` は核の位置（1始まり）。**0 は平板型**（核なし）。
    """

    moras: tuple[str, ...]
    nucleus: int

    def pattern(self) -> tuple[bool, ...]:
        """モーラごとに「高いか」を返す（東京式アクセントの規則）。

        * 平板型（核0）: 1モーラ目が低く、以降すべて高い
        * 頭高型（核1）: 1モーラ目が高く、以降すべて低い
        * 中高・尾高（核n>=2）: 1モーラ目が低く、2〜nが高く、以降低い

            >>> AccentPhrase(("ハ", "シ", "ヲ"), 1).pattern()
            (True, False, False)
            >>> AccentPhrase(("ハ", "シ", "ヲ"), 2).pattern()
            (False, True, False)
            >>> AccentPhrase(("ハ", "シ", "ヲ"), 0).pattern()
            (False, True, True)
        """
        size = len(self.moras)
        if size == 0:
            return ()
        if self.nucleus == 0:
            return tuple(index > 0 for index in range(size))
        if self.nucleus == 1:
            return tuple(index == 0 for index in range(size))
        return tuple(0 < index < self.nucleus for index in range(size))


def accent_plan(text: str) -> list[AccentPhrase]:
    """文をアクセント句へ分け、それぞれの核の位置を返す。

    `pyopenjtalk` の `chain_flag` が 1 の語は直前のアクセント句に連なる。
    句の核は**先頭の語の `acc`** を採る（`pyopenjtalk` の規約）。
    記号は落とす。

    **これは辞書の予測であって実測ではない。** 同形異音（`箸`/`橋`）は
    文脈で決まるので、辞書が外すことがある。音声側と突き合わせる基準として使う。
    """
    import pyopenjtalk

    from cutetts.training.yomi import SKIP_POS

    phrases: list[AccentPhrase] = []
    moras: list[str] = []
    nucleus = 0
    for word in pyopenjtalk.run_frontend(text):
        if (word.get("pos") or "") in SKIP_POS:
            continue
        reading = word.get("read") or ""
        if not reading:
            continue
        starts_phrase = word.get("chain_flag", -1) != 1
        if starts_phrase and moras:
            phrases.append(AccentPhrase(tuple(moras), nucleus))
            moras = []
        if starts_phrase:
            nucleus = int(word.get("acc") or 0)
        moras.extend(split_moras(reading))
    if moras:
        phrases.append(AccentPhrase(tuple(moras), nucleus))
    return phrases


def expected_pattern(text: str) -> tuple[bool, ...]:
    """文全体のモーラ高低パターン（アクセント句を連結したもの）。"""
    pattern: list[bool] = []
    for phrase in accent_plan(text):
        pattern.extend(phrase.pattern())
    return tuple(pattern)


def pattern_agreement(left: tuple[bool, ...], right: tuple[bool, ...]) -> float:
    """2つの高低パターンの一致率。長さが違えば短い方に合わせて比べる。"""
    size = min(len(left), len(right))
    if size == 0:
        return float("nan")
    return sum(1 for i in range(size) if left[i] == right[i]) / size


def _rate(hits: int, total: int) -> dict:
    """一致率と分母をまとめる。分母0なら率は None。"""
    return {"rate": (hits / total) if total else None, "n": total}


def summarize_run(rows: list[dict]) -> dict:
    """`evaluate_prosody.py` の行から集計を作る。

    **shardを結合したあとにも同じものを使う。** 並列で測って
    あとで足し合わせるとき、集計を2箇所に書くと必ずずれる。
    """
    from cutetts.training.evalstats import paired_compare

    usable = [r for r in rows if r.get("status") == "ok"
              and r["human"]["semitone_range"] == r["human"]["semitone_range"]
              and r["model"]["semitone_range"] == r["model"]["semitone_range"]]
    similarities = [r["contour_similarity"] for r in usable
                    if r["contour_similarity"] is not None]

    def mean(key: str, side: str) -> float | None:
        values = [r[side][key] for r in usable]
        return statistics.mean(values) if values else None

    summary = {
        "n": len(usable),
        "n_error": sum(1 for r in rows if r.get("status") == "error"),
        "human_semitone_range": mean("semitone_range", "human"),
        "model_semitone_range": mean("semitone_range", "model"),
        "human_semitone_sd": mean("semitone_sd", "human"),
        "model_semitone_sd": mean("semitone_sd", "model"),
        "human_seconds": mean("seconds", "human"),
        "model_seconds": mean("seconds", "model"),
        "contour_similarity_mean": statistics.mean(similarities) if similarities else None,
        "contour_similarity_median": statistics.median(similarities) if similarities else None,
        "n_flatter_than_human": sum(
            1 for r in usable
            if r["model"]["semitone_range"] < r["human"]["semitone_range"]),
    }

    # --- アクセント核 ---
    phrases = [r["accent"] for r in rows if r.get("accent")]
    if phrases:
        def agree(left_key: str, right_key: str) -> tuple[int, int]:
            hits = total = 0
            for entry in phrases:
                for left, right in zip(entry[left_key], entry[right_key]):
                    if left < 0 or right < 0:   # 判定できなかった句は除く
                        continue
                    total += 1
                    hits += left == right
            return hits, total

        counts: dict[int, int] = {}
        for entry in phrases:
            for value in entry["human"]:
                if value >= 0:
                    counts[value] = counts.get(value, 0) + 1
        marginal = sum(counts.values())
        summary["accent"] = {
            "n_utterances": len(phrases),
            "n_phrases": sum(len(e["expected"]) for e in phrases),
            # **本来見たいのはこれ。** 辞書が正しいかに依存しない
            "model_vs_human": _rate(*agree("human", "model")),
            "human_vs_dictionary": _rate(*agree("expected", "human")),
            "model_vs_dictionary": _rate(*agree("expected", "model")),
            # 人間の核の分布から計算した当てずっぽうの水準
            "chance": (sum((c / marginal) ** 2 for c in counts.values())
                       if marginal else None),
            "constant": (max(counts.values()) / marginal) if marginal else None,
        }

    # 床（同一話者・別の文）との対応のある比較。**床を超えていなければ
    # 「抑揚を再現できた」とは言えない。**
    paired = [(r["contour_similarity"], r["floor_similarity"]) for r in usable
              if r["contour_similarity"] is not None
              and r.get("floor_similarity") is not None]
    # **2文以上ないと対応のある検定ができない。** 1文で落とさない
    if len(paired) >= 2:

        # `difference = mean(b) - mean(a)` なので (床, モデル) の順に渡すと
        # **正が「床を上回る」**になる。相関は大きいほど良いので、
        # `better` / `worse` の数え方だけは逆に読むことになる（ここでは使わない）。
        comparison = paired_compare([f for _, f in paired],
                                    [s for s, _ in paired])
        summary["floor_similarity_mean"] = statistics.mean(f for _, f in paired)
        summary["above_floor"] = {
            "difference": comparison.difference,
            "low": comparison.low,
            "high": comparison.high,
            "significant": comparison.significant,
            "n": comparison.n,
        }

    return summary

def semitones(ratio: float) -> float:
    """周波数比を半音へ。"""
    return 12.0 * math.log2(ratio)


#: 核の後で下がったと見なす最小の落差（半音）。
#:
#: **人間の実音声982句で、辞書との一致率が最大になる値を選んだ**（R-034）。
NUCLEUS_DROP_SEMITONES = 0.5

#: 核を探す前にF0を平滑化するモーラ数（一様移動平均）。
#:
#: 平滑しないと 42.5%、一様3で **46.1%**（人間982句、辞書との一致率）。
#: 短い句では峰を鈍らせる副作用があるが、全体では上回る。
NUCLEUS_SMOOTH_MORAS = 3


def mora_pitches(f0, spans) -> list[float]:
    """モーラごとのF0中央値（Hz）。無声のモーラは NaN。

    `spans` は `cutetts.training.alignment.MoraAligner.align` の返り値。
    F0は `FRAME_PERIOD_MS` 間隔なので、区間をフレーム番号へ直して切り出す。
    """
    values: list[float] = []
    frames_per_second = 1000.0 / FRAME_PERIOD_MS
    array = np.asarray(f0)
    for span in spans:
        low = int(span.start * frames_per_second)
        high = max(int(span.end * frames_per_second), low + 1)
        window = array[low:high]
        voiced = window[window > 0]
        values.append(float(np.median(voiced)) if voiced.size else float("nan"))
    return values


def observed_nucleus(pitches, *, threshold: float = NUCLEUS_DROP_SEMITONES,
                     smooth: int = NUCLEUS_SMOOTH_MORAS) -> int:
    """1アクセント句のF0列から、核の位置を読む。**0は「句の中で下がらない」**。

    **核は「F0が最も高いモーラ」。** 峰が末尾なら句の中では下がらない
    （平板か尾高。句の内部では区別できない）。峰の後の最小値との差が
    `threshold` 未満なら、下がっていないと見なす。

    無声のモーラ（NaN）は飛ばす。2つ未満しか残らなければ -1（判定不能）。

    **「隣り合うモーラの落差が最大の位置」ではない。** その規則は人間の
    実音声982句で 34.0% しか当たらず、**固定回答（常に0）の 36.8% に負けた**。
    峰を採ると 46.1% へ上がる（R-034）。日本語のアクセント核は
    「最後に高いモーラ」であって「最も急に下がる位置」ではない。

        >>> observed_nucleus([200.0, 100.0, 100.0])         # 頭高
        1
        >>> observed_nucleus([100.0, 105.0, 110.0])         # 下がらない
        0
        >>> observed_nucleus([float("nan")])                # 判定できない
        -1
    """
    usable = [(index, value) for index, value in enumerate(pitches)
              if value == value and value > 0]
    if len(usable) < 2:
        return -1
    values = np.array([value for _, value in usable], dtype=np.float64)
    if smooth > 1:
        kernel = np.ones(smooth, dtype=np.float64)
        weight = np.convolve(np.ones_like(values), kernel, mode="same")
        values = np.convolve(values, kernel, mode="same") / weight
    position = int(np.argmax(values))
    if position == len(usable) - 1:            # 峰が末尾＝句の中で下がらない
        return 0
    after = float(values[position + 1:].min())
    if semitones(float(values[position]) / after) < threshold:
        return 0
    return usable[position][0] + 1             # 1始まり


def nucleus_agreement(expected, observed) -> float | None:
    """核の位置の一致率。判定できなかった句（-1）は分母から除く。"""
    pairs = [(e, o) for e, o in zip(expected, observed) if o >= 0]
    if not pairs:
        return None
    return sum(1 for e, o in pairs if e == o) / len(pairs)
