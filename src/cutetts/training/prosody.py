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
`track_f0` は中心を先に求めてから狭い範囲で推定し直す。
**倍音に乗ったフレームを整数倍で引き戻す方式は棄却した** — どの係数を
選ぶかが中心の推定誤差に強く依存し、実測で 185Hz を 292Hz にしてしまった。

    >>> stats = measure(waveform, 24000)
    >>> stats.semitone_sd > 0
    True
"""

from __future__ import annotations

import math
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

#: 上限を段階的に上げて倍音誤りを見つけるための梯子（Hz）。
#:
#: **広い範囲を一度に探すと倍音に乗る。** 実測で、男声（約110Hz）を
#: 上限700Hzで探すと **392Hz**（約3.5倍）と報告した。実音声でも8件中2件が
#: 3.3〜3.8倍の誤りを出した（537.6Hz / 586.6Hz）。
#: **抑揚の幅が20半音（1.7オクターブ）という不自然な値は、この誤りだった。**
F0_CEIL_LADDER = (250.0, 350.0, 500.0, 700.0)

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


def estimate_center_hz(samples: np.ndarray, sample_rate: int) -> float:
    """話者のF0の中心（中央値）を、**倍音に乗らずに**求める。

    上限を `F0_CEIL_LADDER` の順に上げ、中央値が `F0_OCTAVE_RATIO` 倍以上
    跳んだところで止める。跳びは倍音誤り、緩やかな上昇は切り詰めが
    解けただけ、と読み分ける。判定できなければ NaN。
    """
    center = float("nan")
    for ceil in F0_CEIL_LADDER:
        f0 = _harvest(samples, sample_rate, F0_FLOOR_HZ, ceil)
        voiced = f0[f0 > 0]
        if voiced.size < MIN_VOICED_FRAMES:
            continue
        median = float(np.median(voiced))
        if center != center:                   # 最初の有効な推定
            center = median
            continue
        if median / center >= F0_OCTAVE_RATIO:  # 倍音へ乗った
            break
        center = median
    return center


def track_f0(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """F0の系列（Hz）を返す。**無声フレームは 0**。

    **2段階で推定する。** 先に話者のF0の中心を倍音に乗らずに求め
    （`estimate_center_hz`）、その周り `F0_SPAN` 倍の範囲で本番を推定する。
    一度に広い範囲を探すと倍音に乗る（`F0_CEIL_LADDER` 参照）。
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
    # （実測で 1/101 → 62/101 フレームが有声になった）。
    wide = _harvest(samples, sample_rate, F0_FLOOR_HZ, F0_CEIL_HZ)
    center = estimate_center_hz(samples, sample_rate)
    if center != center:
        f0 = np.zeros_like(wide)
    else:
        f0 = _harvest(samples, sample_rate,
                      max(F0_FLOOR_HZ, center / F0_SPAN),
                      min(F0_CEIL_HZ, center * F0_SPAN))
        size = min(f0.size, wide.size)
        f0 = f0[:size] * (wide[:size] > 0)     # 広域パスで無声なら無声
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


def measure(waveform: np.ndarray, sample_rate: int) -> ProsodyStats:
    """1発話の抑揚を測る。"""
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


def semitones(ratio: float) -> float:
    """周波数比を半音へ。"""
    return 12.0 * math.log2(ratio)


#: アクセント核と見なすF0の下がり幅（半音）。
#:
#: これ未満なら平板（核なし）と判定する。**`pyopenjtalk` の合成音
#: （辞書どおりのアクセントを持つ）で一致率が最大になる値を選ぶ。**
NUCLEUS_DROP_SEMITONES = 2.0


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


def observed_nucleus(pitches, *, threshold: float = NUCLEUS_DROP_SEMITONES) -> int:
    """1アクセント句のF0列から、核の位置を読む。**0は平板**。

    核は「そのモーラの**後で**F0が落ちる」位置なので、隣り合うモーラの
    差が最大の場所を採る。落差が `threshold` 未満なら平板とする。

    無声のモーラ（NaN）は飛ばして、その前後で比べる。
    全部無声なら判定できないので -1 を返す。

        >>> observed_nucleus([200.0, 100.0, 100.0])   # 頭高
        1
        >>> observed_nucleus([100.0, 200.0, 100.0])   # 中高
        2
        >>> observed_nucleus([100.0, 105.0, 110.0])   # 平板
        0
    """
    usable = [(index, value) for index, value in enumerate(pitches)
              if value == value and value > 0]
    if len(usable) < 2:
        return -1
    best_drop = 0.0
    best_index = 0
    for (left_index, left), (_, right) in zip(usable, usable[1:]):
        drop = semitones(left / right)
        if drop > best_drop:
            best_drop = drop
            best_index = left_index + 1        # 1始まり
    return best_index if best_drop >= threshold else 0


def nucleus_agreement(expected, observed) -> float | None:
    """核の位置の一致率。判定できなかった句（-1）は分母から除く。"""
    pairs = [(e, o) for e, o in zip(expected, observed) if o >= 0]
    if not pairs:
        return None
    return sum(1 for e, o in pairs if e == o) / len(pairs)
