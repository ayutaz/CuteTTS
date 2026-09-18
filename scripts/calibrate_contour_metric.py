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

"""輪郭の指標を階段で校正する（M4c の前提確認）。

**`semitone_contour` は有声フレームだけを詰めるので時間軸を捨てる。**
有声判定が少し食い違うだけで系列がずれ、実測で**同じ発話を VAE で
往復させただけで 0.417**まで落ちた。M2 の「天井」+0.382（別テイク同士）と
区別が付かない。

そこで4段の階段で両方の指標を測り、**どこまでが指標の限界で、
どこからが本当の差なのか**を切り分ける。

| 段 | 比較 | 意味 |
|---|---|---|
| 1 | 同一音声 | 1.000 でなければ実装が壊れている |
| 2 | **VAE の往復**（同一発話） | **韻律を完璧に写した場合の上限**（codecの限界） |
| 3 | 別テイク（同一話者・同一台詞） | 人間の再現度（M2 の天井） |
| 4 | 別の文（同一話者） | 床 |

    uv run --no-sync python scripts/calibrate_contour_metric.py --limit 40
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

from cutetts.modeling.audio_adapter import AudioAcousticVAEAdapter  # noqa: E402
from cutetts.training import artifacts  # noqa: E402
from cutetts.training.latents import LATENT_SAMPLE_RATE, encode_waveform  # noqa: E402
from cutetts.training.prosody import (  # noqa: E402
    contour_similarity,
    contour_similarity_time,
    semitone_contour,
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


def both(a_f0: np.ndarray, b_f0: np.ndarray) -> tuple[float, float]:
    """（既存の指標, 時間軸を保つ指標）。"""
    old = contour_similarity(semitone_contour(a_f0), semitone_contour(b_f0))
    return old, contour_similarity_time(a_f0, b_f0)


def summarize(name: str, values: list[tuple[float, float]]) -> dict:
    old = np.array([v[0] for v in values])
    new = np.array([v[1] for v in values])
    old, new = old[~np.isnan(old)], new[~np.isnan(new)]
    print(f"  {name:<28} 既存 {old.mean():+.3f}（中央 {np.median(old):+.3f}）"
          f"   **時間軸 {new.mean():+.3f}（中央 {np.median(new):+.3f}）**  n={new.size}")
    return {"name": name, "n": int(new.size),
            "legacy_mean": float(old.mean()) if old.size else None,
            "legacy_median": float(np.median(old)) if old.size else None,
            "time_mean": float(new.mean()) if new.size else None,
            "time_median": float(np.median(new)) if new.size else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default="data/eval/prosody_ceiling_set_v1.json")
    parser.add_argument("--audio-dir", help="eval-set の audio_dir を上書きする")
    parser.add_argument("--model-dir", default="checkpoints/m4a-accent/inference")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    args = parser.parse_args()

    run_dir = artifacts.new_run_dir("m4c-metric", args.artifact_root,
                                    timestamp=args.timestamp)
    payload = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    audio_dir = Path(artifacts.as_local_path(
        args.audio_dir or payload.get("audio_dir", "data/eval/prosody_audio")))
    items = payload["items"][:args.limit]

    vae = AudioAcousticVAEAdapter(
        Path(args.model_dir) / "weights" / "audio_vae").to(args.device).eval()

    same, roundtrip, retake, floor = [], [], [], []
    f0_cache: dict[str, np.ndarray] = {}

    def f0_of(name: str) -> np.ndarray:
        if name not in f0_cache:
            f0_cache[name] = track_f0(read24(audio_dir / name), LATENT_SAMPLE_RATE)
        return f0_cache[name]

    print(f"{len(items)} 組  device={args.device}")
    for index, item in enumerate(items):
        try:
            human = f0_of(item["human_wav"])
            other = f0_of(item["reference_wav"])     # 同一話者の**別の文**
        except Exception as error:
            print(f"  読めない: {error}")
            continue
        if human.size == 0:
            continue

        same.append(both(human, human))
        floor.append(both(human, other))
        # 別テイクは天井set（`prosody_ceiling_set_v1.json`）にだけある
        if item.get("take_b_wav"):
            try:
                retake.append(both(human, f0_of(item["take_b_wav"])))
            except Exception as error:
                print(f"  別テイクが読めない: {error}")

        wave = read24(audio_dir / item["human_wav"])
        with torch.no_grad():
            latent = encode_waveform(vae, torch.from_numpy(wave).float())
            decoded = vae.decode(latent.T.unsqueeze(0).to(args.device))
        decoded = decoded.squeeze().float().cpu().numpy().astype(np.float64)
        roundtrip.append(both(human, track_f0(decoded, LATENT_SAMPLE_RATE)))
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(items)}", flush=True)

    print()
    rows = [summarize("1. 同一音声", same),
            summarize("2. VAE の往復（同一発話）", roundtrip)]
    if retake:
        rows.append(summarize("3. 別テイク（同一台詞）", retake))
    else:
        print("  3. 別テイク                      （このsetには無い）")
    rows.append(summarize("4. 別の文（床）", floor))
    print()
    print("  **2段目が「韻律を完璧に写した場合の上限」。**")
    print("  既存の指標で 2 と 3 が近いなら、天井は指標の限界であって")
    print("  人間の再現度ではない。")

    artifacts.write_run_metadata(
        run_dir, phase="m4c-metric",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=None,
        inputs={"eval_set": args.eval_set, "model_dir": args.model_dir},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "m4c-metric",
        "eval_set": args.eval_set,
        "eval_set_sha256": artifacts.file_checksum(args.eval_set),
        "limit": args.limit,
        "ladder": rows,
    })
    print(f"  {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
