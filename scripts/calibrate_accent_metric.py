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

"""アクセント核の指標に codec 由来の上限があるかを測る（R-050 の続き）。

輪郭については測った。**同じ発話を VAE で往復させただけで +0.367 まで落ちる**
ので、生成音声の実質的な上限はそこだった。

**アクセント核でも同じことが起きているはず**だが、測っていなかった。
核は F0 の峰から読むので、往復で F0 が乱れれば核もずれる。

    往復させた音声 対 元の音声  … **核の実質的な上限**
    人間 対 辞書                … 44.8%（M1 で測った値）
    モデル 対 人間              … 43.6%（現行）

    uv run --no-sync python scripts/calibrate_accent_metric.py --limit 30
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
from cutetts.training.alignment import MoraAligner, phrase_plan  # noqa: E402
from cutetts.training.latents import LATENT_SAMPLE_RATE, encode_waveform  # noqa: E402
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


def nuclei(aligner, text: str, wave: np.ndarray) -> list[int]:
    """アクセント句ごとの核。`evaluate_prosody._nuclei` と同じ手順。"""
    phrases = phrase_plan(text)
    spans = aligner.align(wave, LATENT_SAMPLE_RATE, text)
    if len(spans) != sum(len(p.moras) for p in phrases):
        return []
    pitches = mora_pitches(track_f0(wave, LATENT_SAMPLE_RATE), spans)
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
    parser.add_argument("--model-dir", default="checkpoints/m4a-accent/inference")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    args = parser.parse_args()

    run_dir = artifacts.new_run_dir("m4c-accent-metric", args.artifact_root,
                                    timestamp=args.timestamp)
    payload = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    audio_dir = Path(artifacts.as_local_path(
        args.audio_dir or payload.get("audio_dir", "data/eval/prosody_audio")))
    items = payload["items"][:args.limit]

    vae = AudioAcousticVAEAdapter(
        Path(args.model_dir) / "weights" / "audio_vae").to(args.device).eval()
    aligner = MoraAligner(device=args.device)

    hit_rt = total_rt = hit_dict = total_dict = 0
    unreadable = 0
    print(f"{len(items)} 文  device={args.device}")
    for index, item in enumerate(items):
        text = item["text"]
        try:
            wave = read24(audio_dir / item["human_wav"])
        except Exception as error:
            print(f"  読めない: {error}")
            continue
        with torch.no_grad():
            latent = encode_waveform(vae, torch.from_numpy(wave).float())
            decoded = vae.decode(latent.T.unsqueeze(0).to(args.device))
        decoded = decoded.squeeze().float().cpu().numpy().astype(np.float64)

        human = nuclei(aligner, text, wave)
        roundtrip = nuclei(aligner, text, decoded)
        expected = [p.internal_nucleus for p in phrase_plan(text)]
        if not human or not roundtrip or len(human) != len(roundtrip):
            unreadable += 1
            continue
        a, b = agreement(human, roundtrip)
        hit_rt += a
        total_rt += b
        a, b = agreement(expected, human)
        hit_dict += a
        total_dict += b
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(items)}", flush=True)

    def rate(hit: int, total: int) -> float | None:
        return hit / total if total else None

    rt = rate(hit_rt, total_rt)
    dic = rate(hit_dict, total_dict)
    print()
    print(f"  読めなかった文 {unreadable}")
    print(f"  **往復 対 元の音声: {rt:.1%}**（n={total_rt} 句）"
          if rt is not None else "  往復: 測れない")
    print(f"  人間 対 辞書:      {dic:.1%}（n={total_dict} 句）"
          if dic is not None else "  辞書: 測れない")
    print()
    print("  **往復の値が生成音声の実質的な上限。** 現行のモデル対人間は 43.6%。")

    artifacts.write_run_metadata(
        run_dir, phase="m4c-accent-metric",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=None,
        inputs={"eval_set": args.eval_set, "model_dir": args.model_dir})
    artifacts.write_metrics(run_dir, {
        "phase": "m4c-accent-metric",
        "limit": args.limit,
        "roundtrip_vs_human": rt, "roundtrip_pairs": total_rt,
        "human_vs_dictionary": dic, "dictionary_pairs": total_dict,
        "unreadable": unreadable,
    })
    print(f"  {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
