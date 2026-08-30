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

"""S0 のゴール「reference を入れ替えると speaker identity が追随する」を測る。

同じ text を複数の reference で生成し、生成音声の speaker embedding が
**自分の reference に最も似ているか** を確認する。

    自己類似度  cos(出力(ref_i), ref_i)
    他者類似度  cos(出力(ref_i), ref_j)   j != i

speaker identity が追随していれば 自己 > 他者 になる。
両者に差が無ければ、reference は無視され固定の声が出ている。

reference の音声は **latent cache から VAE decode して作る**。生データの
音声はライセンス上インスタンスへ置かないため（08章「artifactの公開制限」）、
学習に使ったのと同じ latent から再構成する。VAE は freeze なので
再構成音の話者性は保たれる（P1c: speaker cos 0.939）。

既定では **dev-zero-shot**（学習に出てこない話者）から選ぶ。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from cutetts.api import CuteTTS
from cutetts.runtime import prepare_reference_audio
from cutetts.training import artifacts
from cutetts.training.latents import LatentCacheReader
from cutetts.training.manifest import load_manifest

DEFAULT_TEXTS = (
    "今日はいい天気ですね、散歩に行きましょう。",
    "その話は前にも聞いたことがあります。",
    "すみません、もう一度説明してもらえますか。",
)


def pick_utterances(records, *, split: str, count: int, min_seconds: float):
    """voice cluster が重複しないように発話を選ぶ。"""
    chosen, seen = [], set()
    for record in records:
        if record.split != split or record.duration < min_seconds:
            continue
        if record.voice_cluster_id in seen:
            continue
        seen.add(record.voice_cluster_id)
        chosen.append(record)
        if len(chosen) == count:
            break
    return chosen


def decode_reference(tts: CuteTTS, latents: torch.Tensor, path: Path) -> None:
    """cache の生 latent（[T, D]）を波形に戻して書き出す。"""
    vae = tts.runtime.processor.acoustic_vae
    device = tts.runtime.processor.device
    with torch.no_grad():
        # decode は [B, D, T] を期待する
        wave = vae.decode(latents.T.unsqueeze(0).float().to(device))
    wave = wave.squeeze().detach().cpu().float().numpy()
    sf.write(str(path), wave, tts.runtime.sample_rate)


def speaker_embedding(tts: CuteTTS, path: Path) -> torch.Tensor:
    encoder = tts.runtime.speaker_encoder
    _, speaker_wave = prepare_reference_audio(
        path, tts.runtime.sample_rate, int(encoder.sample_rate))
    device = next(encoder.parameters()).device
    with torch.no_grad():
        out = encoder(speaker_wave.to(device), int(encoder.sample_rate))
    return out["embedding"].float().flatten()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="reference追随性の確認（S0）")
    parser.add_argument("--model-dir", default="checkpoints/s0/inference")
    parser.add_argument("--manifest", default="data/manifests/all_clustered.jsonl")
    parser.add_argument("--latent-cache", default="data/cache/latents")
    parser.add_argument("--split", default="dev-zero-shot")
    parser.add_argument("--references", type=int, default=4)
    parser.add_argument("--min-reference-seconds", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decode-length", type=int, default=300)
    parser.add_argument("--label", default="s0-trained")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("s0-refcheck", args.artifact_root,
                                    timestamp=args.timestamp)
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    reader = LatentCacheReader(args.latent_cache)
    records = [r for r in load_manifest(args.manifest) if r.utterance_id in reader]
    picked = pick_utterances(records, split=args.split, count=args.references,
                             min_seconds=args.min_reference_seconds)
    if len(picked) < 2:
        raise SystemExit(f"reference が足りない（{len(picked)}件）。split や長さ条件を見直すこと")
    print(f"reference: {len(picked)}話者（split={args.split}）")

    tts = CuteTTS.from_pretrained(args.model_dir, device=args.device)

    ref_paths, ref_embeds = [], []
    for index, record in enumerate(picked):
        path = audio_dir / f"ref{index}.wav"
        decode_reference(tts, reader.read(record.utterance_id), path)
        ref_paths.append(path)
        ref_embeds.append(speaker_embedding(tts, path))
        print(f"  ref{index}: {record.utterance_id}  cluster={record.voice_cluster_id}")

    rows = []
    for text_index, text in enumerate(DEFAULT_TEXTS):
        for ref_index, ref_path in enumerate(ref_paths):
            out_path = audio_dir / f"t{text_index}_ref{ref_index}.wav"
            result = tts.generate(
                text, mode="voice_clone", reference_audio=str(ref_path),
                seed=args.seed, max_decode_length=args.max_decode_length,
                show_progress=False)
            sf.write(str(out_path), result.waveform.squeeze().cpu().float().numpy(),
                     result.sample_rate)
            embedding = speaker_embedding(tts, out_path)
            sims = [float(torch.nn.functional.cosine_similarity(
                embedding, ref, dim=0)) for ref in ref_embeds]
            rows.append({"text_index": text_index, "reference_index": ref_index,
                         "similarities": sims,
                         "self": sims[ref_index],
                         "others": [s for i, s in enumerate(sims) if i != ref_index],
                         "argmax": int(max(range(len(sims)), key=lambda i: sims[i]))})
            print(f"  text{text_index} x ref{ref_index}: "
                  f"self={sims[ref_index]:.3f} "
                  f"others={[f'{s:.3f}' for i, s in enumerate(sims) if i != ref_index]}")

    self_scores = [r["self"] for r in rows]
    other_scores = [s for r in rows for s in r["others"]]
    correct = sum(1 for r in rows if r["argmax"] == r["reference_index"])
    mean_self = sum(self_scores) / len(self_scores)
    mean_other = sum(other_scores) / len(other_scores)
    margin = mean_self - mean_other
    passed = bool(correct == len(rows) and margin > 0.05)

    print(f"\n自己類似度 mean = {mean_self:.4f}")
    print(f"他者類似度 mean = {mean_other:.4f}")
    print(f"差             = {margin:+.4f}")
    print(f"最も似た reference が自分だった割合: {correct}/{len(rows)}")
    print(f"判定: {'追随している' if passed else '追随が確認できない'}")

    artifacts.write_run_metadata(run_dir, phase="s0-refcheck",
                                 command=["check_reference_following.py"],
                                 seed=args.seed,
                                 inputs={"model_dir": args.model_dir,
                                         "split": args.split})
    artifacts.write_metrics(run_dir, {
        "phase": "s0-refcheck", "label": args.label, "model_dir": args.model_dir,
        "references": [r.utterance_id for r in picked],
        "texts": list(DEFAULT_TEXTS),
        "rows": rows,
        "mean_self_similarity": mean_self,
        "mean_other_similarity": mean_other,
        "margin": margin,
        "argmax_correct": correct, "argmax_total": len(rows),
        "passed": passed,
    })
    print(f"完了: {run_dir}")
    print("注意: audio/ はライセンス上コミット・公開しないこと（08章）")


if __name__ == "__main__":
    main()
