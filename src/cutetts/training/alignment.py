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

"""モーラ単位の強制アラインメント（M1のアクセント測定に要る）。

アクセント核は「どのモーラの後でF0が下がるか」なので、**F0のフレームを
モーラへ対応付けられないと測れない**。テキストは分かっているので、
CTCの強制アラインメントで音声とモーラを対応付ける。

## 構造は full-context label から取る（D-042）

**`run_frontend` の `acc` と `chain_flag` から句を組み立ててはいけない。**
そうやって作ったものは `pyopenjtalk` 自身の合成結果と食い違う。実測した
2つの食い違い:

* **促音でモーラ数がずれる。** `モッテ` は `モ・ッ・テ` の3モーラだが、
  `read` を畳むと2になる。アクセント核の位置は**モーラ番号**なので直接ずれる。
* **核の位置が違う。** `渡る` は `run_frontend` の `acc` が 0（平板）だが、
  full-context は `F:3_3`（尾高）。**合成音が実現しているのは後者**。

`extract_fullcontext` は音素ごとに次を持つ。

| 欄 | 意味 |
|---|---|
| `A:a1+a2+a3` | a2 = アクセント句内のモーラ位置（1始まり） |
| `F:f1_f2@f5_f6` | f1 = 句のモーラ数、f2 = **核の位置**（0は平板）、f5 = 句の位置 |
| `I:i1-i2@i3` | i3 = 呼気段落の位置 |

`(i3, f5)` で句が一意に決まり、`a2` でモーラが決まる。**音素もそこにある**
ので、仮名からローマ字を起こす必要がない。

## 手段の選定（D-041）

`torchaudio.pipelines.MMS_FA` + `torchaudio.functional.forced_align`。
**新しい依存が要らない**（torchaudio 2.5.1 に入っている）。

| 候補 | 依存 | 粒度 | 判断 |
|---|---|---|---|
| **MMS_FA（採用）** | **torchaudio のみ** | 文字 | ラベルが a-z なので音素をそのまま渡せる |
| Julius + segmentation-kit | Julius本体 + perl | 音素 | 日本語の定番だがビルドと配置が重い |
| Montreal Forced Aligner | conda + 数GB | 音素 | 最も正確だが環境が別建てになる |
| Whisper の単語timestamp | 既存 | **単語** | **粒度が足りない**。1モーラ約120msに対し単語境界は±100〜200ms |

MMS_FA はモデル 1.18 GB（初回のみ自動取得、`~/.cache/torch/hub`）。

## 限界

* **促音（`cl`）と撥音（`N`）には固有の音がない。** `cl` は次のモーラの
  子音を重ねて渡す。モーラとしては残るのでモーラ数はずれない。
* **無声化母音（`I` `U`）はF0が取れない。** 区間は返るがF0は NaN になる。
* 句の分け方は `pyopenjtalk` の辞書に依存する。実際の発話が辞書どおりに
  区切られているとは限らない。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

#: OpenJTalk の音素 → MMS_FA のラベル（a-z）。
#:
#: `I` `U` は無声化母音、`N` は撥音、`cl` は促音。
#: `pau` `sil` は音素ではないので落とす。
_PHONEME_TO_LETTERS = {
    "a": "a", "i": "i", "u": "u", "e": "e", "o": "o",
    "I": "i", "U": "u", "N": "n",
    "k": "k", "s": "s", "t": "t", "n": "n", "h": "h", "m": "m",
    "y": "y", "r": "r", "w": "w", "g": "g", "z": "z", "d": "d",
    "b": "b", "p": "p", "f": "f", "j": "j", "v": "v",
    "sh": "sh", "ch": "ch", "ts": "ts",
    "ky": "ky", "gy": "gy", "ny": "ny", "hy": "hy",
    "my": "my", "ry": "ry", "by": "by", "py": "py",
    "dy": "dy", "ty": "ty",
}

#: 音を持たない音素。
_NON_PHONEMES = frozenset({"pau", "sil", "xx"})

_PHONEME = re.compile(r"\-(.*?)\+")
_FIELD_A = re.compile(r"/A:([-\d]+)\+(\d+)\+(\d+)")
_FIELD_F = re.compile(r"/F:(\d+)_(\d+)#\d+_\d+@(\d+)_(\d+)")
_FIELD_I = re.compile(r"/I:\d+-\d+@(\d+)\+")


@dataclass(frozen=True)
class Mora:
    """1モーラ。`letters` は MMS_FA へ渡す文字列。"""

    phonemes: tuple[str, ...]
    letters: str
    position: int
    """アクセント句内のモーラ位置（1始まり）。"""


@dataclass(frozen=True)
class AccentPhrase:
    """1アクセント句。

    `nucleus` は full-context の `f2` をそのまま持つ（**1始まり**）。

    **OpenJTalk は平板（0型）を `f2 = モーラ数` で符号化する。**
    `桜`（0型・3モーラ）は `f1=3 f2=3`、`学校`（0型・4モーラ）は `f1=4 f2=4`。
    `run_frontend` の `acc` は通常の規約（0が平板）なので、**同じものを
    別の数で表している**。`f2 == len(moras)` を `acc == 0` と読み替える。
    """

    nucleus: int
    moras: tuple[Mora, ...]

    @property
    def internal_nucleus(self) -> int:
        """**句の内部で観測できる**核の位置。0は「句の中で下がらない」。

        平板と尾高はどちらも `f2 == モーラ数` になり、**句の中では区別が
        つかない**（下がりは句の後で起きる）。音声から測れるのは
        「句の中のどこで下がるか」だけなので、両者は 0 にまとめる。
        区別したければ次の句まで見る必要がある。
        """
        return self.nucleus if 0 < self.nucleus < len(self.moras) else 0

    def pattern(self) -> tuple[bool, ...]:
        """モーラごとに「高いか」を返す（東京式アクセントの規則）。

            >>> AccentPhrase(1, ()).pattern()
            ()
        """
        size = len(self.moras)
        if size == 0:
            return ()
        nucleus = self.internal_nucleus
        if nucleus == 0:
            return tuple(index > 0 for index in range(size))
        if nucleus == 1:
            return tuple(index == 0 for index in range(size))
        return tuple(0 < index < nucleus for index in range(size))


def phrase_plan(text: str) -> list[AccentPhrase]:
    """文をアクセント句とモーラへ分ける。**full-context label から取る。**

        >>> [p.nucleus for p in phrase_plan("箸を持って橋を渡る")]
        [1, 1, 2, 3]
    """
    import pyopenjtalk

    rows = []
    for label in pyopenjtalk.extract_fullcontext(text):
        phoneme = _PHONEME.search(label)
        if phoneme is None:
            continue
        name = phoneme.group(1)
        if name in _NON_PHONEMES:
            continue
        field_a = _FIELD_A.search(label)
        field_f = _FIELD_F.search(label)
        field_i = _FIELD_I.search(label)
        if not (field_a and field_f):
            continue
        rows.append({
            "phoneme": name,
            "mora": int(field_a.group(2)),
            "nucleus": int(field_f.group(2)),
            "phrase": int(field_f.group(3)),
            "group": int(field_i.group(1)) if field_i else 0,
        })

    phrases: list[AccentPhrase] = []
    current_key = None
    current_nucleus = 0
    moras: list[Mora] = []
    pending: list[str] = []
    pending_position = 0

    def flush_mora() -> None:
        nonlocal pending, pending_position
        if not pending:
            return
        letters = "".join(_PHONEME_TO_LETTERS.get(p, "") for p in pending)
        moras.append(Mora(tuple(pending), letters, pending_position))
        pending = []

    def flush_phrase() -> None:
        nonlocal moras
        flush_mora()
        if moras:
            phrases.append(AccentPhrase(current_nucleus, tuple(moras)))
        moras = []

    for row in rows:
        key = (row["group"], row["phrase"])
        if key != current_key:
            flush_phrase()
            current_key = key
            current_nucleus = row["nucleus"]
        if row["mora"] != pending_position:
            flush_mora()
            pending_position = row["mora"]
        pending.append(row["phoneme"])
    flush_phrase()
    return _fill_geminates(phrases)


def _fill_geminates(phrases: list[AccentPhrase]) -> list[AccentPhrase]:
    """促音に音がないので、**次のモーラの子音を重ねて**渡す。

    そうしないと CTC へ渡す文字が空になり、そのモーラを飛ばすしかなくなる
    （モーラ番号がずれ、アクセント核の位置と突き合わせられなくなる）。
    """
    out: list[AccentPhrase] = []
    flat = [(pi, mi) for pi, phrase in enumerate(phrases)
            for mi in range(len(phrase.moras))]
    letters = {key: phrases[key[0]].moras[key[1]].letters for key in flat}
    for index, key in enumerate(flat):
        if letters[key]:
            continue
        nxt = flat[index + 1] if index + 1 < len(flat) else None
        following = letters.get(nxt, "") if nxt else ""
        letters[key] = following[0] if following else "q"
    for pi, phrase in enumerate(phrases):
        out.append(AccentPhrase(phrase.nucleus, tuple(
            Mora(mora.phonemes, letters[(pi, mi)], mora.position)
            for mi, mora in enumerate(phrase.moras))))
    return out


def mora_sequence(text: str) -> list[Mora]:
    """句をまたいで連結したモーラ列。"""
    return [mora for phrase in phrase_plan(text) for mora in phrase.moras]


@dataclass(frozen=True)
class MoraSpan:
    """1モーラの時間区間。"""

    mora: Mora
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


@lru_cache(maxsize=1)
def _bundle():
    import torchaudio

    return torchaudio.pipelines.MMS_FA


class MoraAligner:
    """音声とテキストをモーラ単位で対応付ける。

    モデルは初回に 1.18 GB を取得する。**使い回すこと**（毎回作ると
    モデルの読み込みが走る）。
    """

    def __init__(self, device: str = "cpu") -> None:
        bundle = _bundle()
        self.bundle = bundle
        self.device = device
        self.model = bundle.get_model().to(device).eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()

    def align(self, waveform, sample_rate: int, text: str) -> list[MoraSpan]:
        """`waveform`（1次元）と `text` を対応付ける。失敗時は空リスト。"""
        import numpy as np
        import torch
        import torchaudio

        moras = mora_sequence(text)
        if not moras:
            return []
        samples = np.asarray(waveform, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=0 if samples.shape[0] < samples.shape[-1] else 1)
        peak = float(np.abs(samples).max()) if samples.size else 0.0
        if peak > 1.0:                          # int16 相当の振幅で来ることがある
            samples = samples / peak
        wave = torch.from_numpy(samples).unsqueeze(0)
        if sample_rate != self.bundle.sample_rate:
            wave = torchaudio.functional.resample(
                wave, sample_rate, self.bundle.sample_rate)
        with torch.inference_mode():
            emission, _ = self.model(wave.to(self.device))
        try:
            spans = self.aligner(emission[0].cpu(),
                                 self.tokenizer([m.letters for m in moras]))
        except (RuntimeError, ValueError):
            # モーラ列が音声より長い等でアラインできないことがある
            return []
        ratio = wave.shape[1] / emission.shape[1] / self.bundle.sample_rate
        return [MoraSpan(mora=mora,
                         start=float(span[0].start) * ratio,
                         end=float(span[-1].end) * ratio)
                for mora, span in zip(moras, spans)]
