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

"""短い reference を学習時の長さまで伸ばす（R-026）。

**同一話者・同一文で人間と聴き比べた結果、reference長で判定が完全に分離した。**

| 話者 | 判定 | reference長 | 声質が違う |
|---|---|---:|---|
| mizuki | 明らかに劣る | 3.4秒 | ✓ |
| shizuru | 明らかに劣る | 3.9秒 | ✓ |
| akiko | 近い | 8.2秒 | — |
| MINATO | 近い | 9.9秒 | — |
| aoi | 近い | 11.0秒 | — |
| CHIHIRO | 近い | 12.7秒 | — |

劣る側の最大3.9秒 < 近い側の最小8.2秒で境界が重ならない。
「声質が違う」は劣る2/2・近い0/4。
学習時の reference は `target_reference_seconds=10.0` で平均9.61秒なので、
**3〜4秒は分布外**であり、話者情報が足りずに声質と抑揚が崩れる。

`runtime.prepare_reference_audio` は2秒未満だけを repeat で伸ばし、
3〜4秒はそのまま通す。**そこは upstream の推論pathなので触らない。**
ここは「公開APIへ渡す前に reference を伸ばす」層である。

    >>> path = ensure_minimum_duration("short.wav", minimum_seconds=8.0)
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_MINIMUM_SECONDS = 8.0
"""既定の下限。speaker encoder が使う先頭8秒に合わせる。

`prepare_reference_audio` は speaker encoder へ先頭8秒、VAE へ先頭30秒を渡す。
8秒あれば speaker 側が埋まり、学習時（平均9.61秒）にも近づく。
"""


def duration_seconds(path: str | Path) -> float:
    with sf.SoundFile(str(path), mode="r") as handle:
        return len(handle) / float(handle.samplerate)


def ensure_minimum_duration(
    path: str | Path,
    *,
    minimum_seconds: float = DEFAULT_MINIMUM_SECONDS,
    cache_dir: str | Path | None = None,
) -> Path:
    """``minimum_seconds`` より短ければ繰り返して伸ばし、そのファイルのpathを返す。

    十分な長さなら元のpathをそのまま返す（余計なコピーを作らない）。
    伸ばす場合は ``cache_dir``（既定は元ファイルと同じ場所の ``.ref-cache``）へ
    書き出す。同じ入力・同じ下限なら同じファイル名になるので使い回せる。

    繰り返しは `runtime._repeat_past_two_seconds` と同じ方針（単純な連結）。
    無音を足すと speaker embedding が薄まるので、波形をそのまま繰り返す。
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"reference audio not found: {source}")

    audio, rate = sf.read(str(source), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"reference audio is empty: {source}")
    seconds = audio.shape[0] / float(rate)
    if seconds >= minimum_seconds:
        return source

    repeats = math.ceil(minimum_seconds * rate / audio.shape[0])
    extended = np.tile(audio, (repeats, 1))

    root = Path(cache_dir) if cache_dir is not None else source.parent / ".ref-cache"
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        f"{source}:{source.stat().st_size}:{minimum_seconds}".encode("utf-8")
    ).hexdigest()[:12]
    target = root / f"{source.stem}-min{minimum_seconds:g}s-{digest}.wav"
    if not target.is_file():
        sf.write(str(target), extended, rate)
    return target
