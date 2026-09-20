#!/usr/bin/env python
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

"""アクセント核が**条件に入っているか**を確かめる（M4f）。

**真の F0 を条件として渡してもアクセント核が動かなかった**（M4c / M4d）。
理由は2つ考えられる。

1. **条件が核を運んでいない**（12.5 Hz では粗すぎる）
2. 運んでいるが**モデルが実現していない**

**生成を通さずに切り分けられる。** 人間の実音声から

* (a) フル解像度の F0（10 ms）で核を読む … 測定系が読む「正解」
* (b) **条件と同じ 12.5 Hz へ畳んでから**核を読む … 条件が運べる情報

を比べればよい。**一致するなら条件は核を運んでいる**（1 は否定され、
モデルを疑う番になる）。一致しないなら条件の表現を変える必要がある。

    uv run --no-sync python scripts/probe_condition_accent.py --limit 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

from cutetts.training import artifacts  # noqa: E402
from cutetts.training.alignment import MoraAligner, phrase_plan  # noqa: E402
from cutetts.training.f0 import F0_FRAMES_PER_LATENT, frame_f0  # noqa: E402
from cutetts.training.latents import LATENT_SAMPLE_RATE  # noqa: E402
from cutetts.training.prosody import (  # noqa: E402
    mora_pitches,
    observed_nucleus,
    track_f0,
)


def read24(path: Path) -> np.ndarray:
    samples, rate = sf.read(str(path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    wave = torch.from_numpy(samples)
    if rate != LATENT_SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, rate, LATENT_SAMPLE_RATE)
    return wave.numpy().astype(np.float64)


def coarse_to_dense(coarse: np.ndarray, length: int) -> np.ndarray:
    """12.5 Hz の F0 を 10 ms 格子へ戻す（**1つを8フレームに複製**）。

    **条件が運べる情報だけ**を残した F0 を作るための操作。
    条件は 12.5 Hz なので、その間の動きは表現できない。
    """
    dense = np.repeat(np.asarray(coarse, dtype=np.float64), F0_FRAMES_PER_LATENT)
    if dense.size < length:
        dense = np.concatenate([dense, np.zeros(length - dense.size)])
    return dense[:length]


def nuclei_from_f0(aligner, text: str, wave: np.ndarray, f0: np.ndarray
                   ) -> list[int] | None:
    """与えた F0 からアクセント句ごとの核を読む（`evaluate_prosody` と同じ手順）。"""
    phrases = phrase_plan(text)
    try:
        spans = aligner.align(wave, LATENT_SAMPLE_RATE, text)
    except Exception:
        return None
    if len(spans) != sum(len(p.moras) for p in phrases):
        return None
    pitches = mora_pitches(f0, spans)
    out, offset = [], 0
    for phrase in phrases:
        size = len(phrase.moras)
        out.append(observed_nucleus(pitches[offset:offset + size]))
        offset += size
    return out


def agreement(left: list[int], right: list[int]) -> tuple[int, int]:
    pairs = [(a, b) for a, b in zip(left, right) if a >= 0 and b >= 0]
    return sum(1 for a, b in pairs if a == b), len(pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default="data/eval/prosody_eval_set_v2.json")
    parser.add_argument("--audio-dir")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    args = parser.parse_args()

    run_dir = artifacts.new_run_dir("m4f-condition-accent", args.artifact_root,
                                    timestamp=args.timestamp)
    payload = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    audio_dir = Path(artifacts.as_local_path(
        args.audio_dir or payload.get("audio_dir", "data/eval/prosody_audio")))
    items = payload["items"][:args.limit]
    aligner = MoraAligner(device=args.device)

    hit_pair = total_pair = 0            # (a) 対 (b)
    hit_dense = total_dense = 0          # (a) 対 辞書
    hit_coarse = total_coarse = 0        # (b) 対 辞書
    unreadable = 0
    rows: list[dict] = []

    print(f"{len(items)} 文  device={args.device}")
    for index, item in enumerate(items):
        text = item["text"]
        try:
            wave = read24(audio_dir / item["human_wav"])
        except Exception as error:
            print(f"  読めない: {error}")
            continue
        dense_f0 = track_f0(wave, LATENT_SAMPLE_RATE)
        coarse = frame_f0(wave, LATENT_SAMPLE_RATE)          # 12.5 Hz
        rebuilt = coarse_to_dense(coarse, dense_f0.size)

        dense_nuclei = nuclei_from_f0(aligner, text, wave, dense_f0)
        coarse_nuclei = nuclei_from_f0(aligner, text, wave, rebuilt)
        if not dense_nuclei or not coarse_nuclei:
            unreadable += 1
            continue
        expected = [p.internal_nucleus for p in phrase_plan(text)]

        a, b = agreement(dense_nuclei, coarse_nuclei)
        hit_pair += a
        total_pair += b
        a, b = agreement(expected, dense_nuclei)
        hit_dense += a
        total_dense += b
        a, b = agreement(expected, coarse_nuclei)
        hit_coarse += a
        total_coarse += b
        rows.append({"text": text, "dense": dense_nuclei, "coarse": coarse_nuclei,
                     "expected": expected})
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(items)}", flush=True)

    def rate(hit: int, total: int) -> float | None:
        return hit / total if total else None

    pair = rate(hit_pair, total_pair)
    dense = rate(hit_dense, total_dense)
    coarse = rate(hit_coarse, total_coarse)
    print()
    print(f"  読めなかった文 {unreadable}")
    print(f"  **12.5 Hz へ畳んでも核が一致する: {pair:.1%}**（n={total_pair} 句）"
          if pair is not None else "  測れない")
    print(f"  フル解像度 対 辞書:   {dense:.1%}（n={total_dense}）"
          if dense is not None else "")
    print(f"  12.5 Hz  対 辞書:    {coarse:.1%}（n={total_coarse}）"
          if coarse is not None else "")
    print()
    print("  **80% 以上なら条件は核を運んでいる**（モデルが実現していない）。")
    print("  低ければ 12.5 Hz では粗すぎる（条件の表現を変える）。")

    artifacts.write_run_metadata(
        run_dir, phase="m4f-condition-accent",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=None,
        inputs={"eval_set": args.eval_set})
    artifacts.write_metrics(run_dir, {
        "phase": "m4f-condition-accent",
        "limit": args.limit,
        "coarse_vs_dense": pair, "coarse_vs_dense_pairs": total_pair,
        "dense_vs_dictionary": dense, "coarse_vs_dictionary": coarse,
        "unreadable": unreadable,
        "rows": rows[:50],
    })
    print(f"  {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
