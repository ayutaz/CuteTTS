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

"""flow loss が低い理由を切り分ける。

S0 の学習で flow loss が 1.02 -> 0.003 まで落ちた。これは velocity の
決定係数 0.998 に相当し、flow matching が本来到達できる精度を超えている。
原因が **記憶** なのか **正しい学習** なのかを、次の3点の比較で判定する:

  train         学習に使った発話
  dev-seen      学習話者の未学習発話
  dev-zero-shot 未学習話者の未学習発話

さらに「条件付けを使わない予測器（常に0を出す）」の loss を同時に出す。
clean は正規化済み（分散約1）、noise は N(0,1) なので理論値は約 2.0。
loss の絶対値が何に対して小さいのかは、この値との比で判断する。

  記憶なら       train << dev
  正しい学習なら train ~= dev

`leakage` は別途 tests/training/test_leakage.py で検証済み（条件付けは
厳密に因果的で、target patch i は z[i] に影響しない）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cutetts.training import artifacts
from cutetts.training.collator import build_training_sample, collate
from cutetts.training.dataset import LatentSource
from cutetts.training.forward import training_forward
from cutetts.training.latents import LatentCacheReader
from cutetts.training.manifest import load_manifest
from cutetts.training.pairing import PairSampler, assert_no_leakage
from cutetts.training.prompt import build_voice_clone_prompt
from cutetts.training.speaker_cache import SpeakerEmbeddingCacheReader

REFERENCE_PATCH_CAP = int(30 * 6.25)


def build_batches(records, *, source, speaker_reader, processor, max_length,
                  seed, batch_size, batches, group_key, max_target_patches):
    """train_continual.py と同じ手順で batch を作る。"""
    sampler = PairSampler(records, seed=seed, group_key=group_key,
                          min_utterances_per_group=2, target_reference_seconds=10.0)
    out = []
    for _ in range(batches):
        pairs = sampler.sample(batch_size)
        assert_no_leakage(pairs)
        samples, speakers = [], []
        for pair in pairs:
            target = source.patches(pair.target.utterance_id)[:max_target_patches]
            reference = torch.cat(
                [source.patches(u.utterance_id) for u in pair.reference_group], dim=0
            )[:REFERENCE_PATCH_CAP]
            if target.shape[0] < 2:
                continue
            prompt = build_voice_clone_prompt(processor, pair.target.text_raw)
            budget = max_length - prompt.text_token_count - 1 - int(reference.shape[0])
            if budget < 2:
                continue
            target = target[: min(int(target.shape[0]), budget + 1)]
            samples.append(build_training_sample(
                utterance_id=pair.target.utterance_id, prompt=prompt,
                reference_latents=reference, target_latents=target))
            speakers.append(speaker_reader.read(pair.reference_group[0].utterance_id))
        if samples:
            out.append((collate(samples), torch.stack(speakers)))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="flow loss の記憶/汎化の切り分け")
    parser.add_argument("--model-dir", default="model/CuteTTS",
                        help="比較したいcheckpoint（学習後は checkpoints/s0/inference）")
    parser.add_argument("--base-model-dir", default="model/CuteTTS",
                        help="processor / tokenizer の供給元")
    parser.add_argument("--manifest", default="data/manifests/all_clustered.jsonl")
    parser.add_argument("--latent-cache", default="data/cache/latents")
    parser.add_argument("--speaker-cache", default="data/cache/speaker")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--flow-copies", type=int, default=4)
    parser.add_argument("--max-target-patches", type=int, default=188)
    parser.add_argument("--group-key", default="voice_cluster_id")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--label", default="post-train")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    from train_continual import load_model  # 学習と同じ読み込み経路を使う

    args = build_parser().parse_args()
    device = torch.device(args.device)
    run_dir = artifacts.new_run_dir("s0-diagnose", args.artifact_root, timestamp=args.timestamp)

    model = load_model(Path(args.model_dir), device, args.dtype).eval()

    from cutetts.modeling.processor import CuteTTSProcessor
    from cutetts.modeling.segments import SegmentManagerConfig
    cfg = json.loads((Path(args.base_model_dir) / "config.json").read_text(encoding="utf-8"))
    processor = CuteTTSProcessor(
        acoustic_vae_path=str(Path(args.base_model_dir) / "weights" / "audio_vae"),
        tokenizer=str(Path(args.base_model_dir) / "tokenizer"),
        segment_cfg=SegmentManagerConfig(**cfg["processor"]["segment"]),
        speech_compress_rate=int(cfg["processor"]["speech_compress_rate"]),
    )
    max_length = int(cfg["processor"]["segment"]["max_length"])

    latent_reader = LatentCacheReader(args.latent_cache)
    speaker_reader = SpeakerEmbeddingCacheReader(args.speaker_cache)
    source = LatentSource(reader=latent_reader,
                          scaling=model.speech_scaling_factor.detach().cpu(),
                          bias=model.speech_bias_factor.detach().cpu(),
                          patch_size=model.config.locenc_patch_size)

    records = list(load_manifest(args.manifest))
    results: dict[str, dict] = {}
    for split in ("train", "dev-seen", "dev-zero-shot"):
        rows = [r for r in records
                if r.split == split
                and r.utterance_id in latent_reader and r.utterance_id in speaker_reader]
        if len(rows) < 2:
            results[split] = {"skipped": f"records={len(rows)}"}
            print(f"{split:14s} skip（record {len(rows)}件）")
            continue
        batches = build_batches(
            rows, source=source, speaker_reader=speaker_reader, processor=processor,
            max_length=max_length, seed=args.seed, batch_size=args.batch_size,
            batches=args.batches, group_key=args.group_key,
            max_target_patches=args.max_target_patches)
        if not batches:
            results[split] = {"skipped": "batchが作れない"}
            print(f"{split:14s} skip（batchが作れない）")
            continue
        flow, stop, zero_pred, n = [], [], [], 0
        with torch.no_grad():
            for batch, speaker in batches:
                out = training_forward(model, batch, speaker_embeddings=speaker,
                                       flow_copies=args.flow_copies,
                                       generator=torch.Generator().manual_seed(args.seed))
                flow.append(float(out.flow_loss))
                stop.append(float(out.stop_loss))
                clean = batch.target_patches.float()
                zero_pred.append(float(clean.pow(2).mean()) + 1.0)
                n += out.num_targets
        mean_flow = sum(flow) / len(flow)
        mean_zero = sum(zero_pred) / len(zero_pred)
        results[split] = {
            "records": len(rows), "batches": len(batches), "targets": n,
            "flow_loss": mean_flow,
            "stop_loss": sum(stop) / len(stop),
            "zero_predictor_loss": mean_zero,
            "r2_vs_zero_predictor": 1.0 - mean_flow / mean_zero,
        }
        r = results[split]
        print(f"{split:14s} flow={mean_flow:.4f}  stop={r['stop_loss']:.4f}  "
              f"zero-pred={mean_zero:.4f}  R2={r['r2_vs_zero_predictor']:.4f}  "
              f"({n} targets)")

    train_loss = results.get("train", {}).get("flow_loss")
    dev = results.get("dev-zero-shot", {}) or {}
    dev_loss = dev.get("flow_loss") or results.get("dev-seen", {}).get("flow_loss")
    if train_loss and dev_loss:
        ratio = dev_loss / train_loss
        verdict = (f"記憶が支配的（dev が train の {ratio:.1f} 倍）" if ratio > 3.0
                   else f"汎化している（dev/train = {ratio:.2f}）")
    else:
        verdict = "判定不能（比較できる split がない）"
    print(f"\n判定: {verdict}")

    artifacts.write_run_metadata(run_dir, phase="s0-diagnose",
                                 command=["diagnose_flow_loss.py"], seed=args.seed,
                                 inputs={"model_dir": args.model_dir,
                                         "manifest": args.manifest})
    artifacts.write_metrics(run_dir, {"phase": "s0-diagnose", "label": args.label,
                                      "model_dir": args.model_dir,
                                      "splits": results, "verdict": verdict})
    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
