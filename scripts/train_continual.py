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

"""日本語継続学習のtrainer。S0（10〜30時間 overfit）で使う。

P2 で作った部品を繋ぐだけで、新しい学習semanticsはここに書かない。

* `dataset.LatentSource`   … cache から正規化済み patch を読む
* `pairing.PairSampler`    … reference/target を leakage なく選ぶ
* `collator`               … teacher forcing の sequence を組む
* `forward.training_forward` … loss を計算する
* `checkpointing`          … save/resume（RNG含む）

継続学習の設定は論文のpretraining値をそのまま使わない（04章 第4節）。
S0 では小さめのLRから始め、実測で調整する。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from cutetts.modeling.configuration import CuteTTSConfig
from cutetts.modeling.model import CuteTTSModel
from cutetts.training import artifacts
from cutetts.training.checkpointing import (
    TrainingState,
    export_for_inference,
    load_training_state,
    save_training_state,
)
from cutetts.training.collator import build_training_sample, collate
from cutetts.training.dataset import LatentSource
from cutetts.training.forward import TRAINABLE_MODULES, freeze_all_but, training_forward
from cutetts.training.latents import LatentCacheReader
from cutetts.training.manifest import Utterance, load_manifest
from cutetts.training.objectives import ConditionDropoutConfig
from cutetts.training.pairing import PairSampler, assert_no_leakage
from cutetts.training.prompt import build_voice_clone_prompt
from cutetts.training.speaker_cache import SpeakerEmbeddingCacheReader


def load_model(model_dir: Path, device: torch.device, dtype: str) -> CuteTTSModel:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    architecture = dict(config["architecture"])
    architecture.update(attn_implementation="sdpa", use_pretrained_lm=False,
                        lm_model_name=None, torch_dtype=dtype)
    model = CuteTTSModel(CuteTTSConfig(**architecture))
    missing, unexpected = model.load_state_dict(
        load_file(str(model_dir / "weights" / "tts" / "model.safetensors")), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"weights did not load strictly: {missing} / {unexpected}")
    return model.to(device).train()


def cosine_lr(step: int, *, peak: float, warmup: int, total: int, floor_ratio: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return peak * (floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="日本語継続学習（S0）")
    parser.add_argument("--model-dir", default="model/CuteTTS")
    parser.add_argument("--manifest", default="data/manifests/all_clustered.jsonl")
    parser.add_argument("--latent-cache", default="data/cache/latents")
    parser.add_argument("--speaker-cache", default="data/cache/speaker")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="継続学習なので論文の5e-4より2桁小さくする")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--flow-copies", type=int, default=4)
    parser.add_argument("--stop-weight", type=float, default=1.0)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--group-key", default="voice_cluster_id",
                        choices=("voice_cluster_id", "speaker_id"))
    parser.add_argument("--target-reference-seconds", type=float, default=10.0)
    parser.add_argument("--max-target-patches", type=int, default=188)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--out", default="checkpoints/s0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    run_dir = artifacts.new_run_dir("s0-train", args.artifact_root, timestamp=args.timestamp)
    out_dir = Path(args.out)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    records = [r for r in load_manifest(args.manifest) if r.split == "train"]
    print(f"train records: {len(records):,}")

    model = load_model(Path(args.model_dir), device, args.dtype)
    freeze_all_but(model, TRAINABLE_MODULES)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable)/1e6:.1f}M")

    # prompt は推論と同じ並びを作る必要があるため、実 processor を使う
    from cutetts.modeling.processor import CuteTTSProcessor
    from cutetts.modeling.segments import SegmentManagerConfig
    processor_config = json.loads((Path(args.model_dir) / "config.json").read_text(encoding="utf-8"))
    processor = CuteTTSProcessor(
        acoustic_vae_path=str(Path(args.model_dir) / "weights" / "audio_vae"),
        tokenizer=str(Path(args.model_dir) / "tokenizer"),
        segment_cfg=SegmentManagerConfig(**processor_config["processor"]["segment"]),
        speech_compress_rate=int(processor_config["processor"]["speech_compress_rate"]),
    )
    max_length = int(processor_config["processor"]["segment"]["max_length"])

    latent_reader = LatentCacheReader(args.latent_cache)
    speaker_reader = SpeakerEmbeddingCacheReader(args.speaker_cache)
    source = LatentSource(reader=latent_reader,
                          scaling=model.speech_scaling_factor.detach().cpu(),
                          bias=model.speech_bias_factor.detach().cpu(),
                          patch_size=model.config.locenc_patch_size)

    usable = [r for r in records
              if r.utterance_id in latent_reader and r.utterance_id in speaker_reader]
    print(f"cache にある train records: {len(usable):,}")
    if len(usable) < 2:
        raise SystemExit("学習に足るcacheがありません。先に cache_audio_latents.py を実行してください")

    sampler = PairSampler(usable, seed=args.seed, group_key=args.group_key,
                          min_utterances_per_group=2,
                          target_reference_seconds=args.target_reference_seconds)
    groups = sampler.eligible_groups()
    print(f"eligible groups ({args.group_key}): {len(groups)}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)
    state = TrainingState()
    if args.resume and (out_dir / "training_state.pt").is_file():
        state = load_training_state(out_dir, model=model, optimizer=optimizer,
                                    generator=generator, map_location=str(device))
        print(f"resumed from step {state.step}")

    dropout = (ConditionDropoutConfig(speaker=args.condition_dropout,
                                      reference=args.condition_dropout, joint=True)
               if args.condition_dropout > 0 else None)

    history: list[dict] = []
    started = time.perf_counter()
    for step in range(state.step, args.steps):
        pairs = sampler.sample(args.batch_size)
        assert_no_leakage(pairs)

        samples, speakers = [], []
        for pair in pairs:
            target = source.patches(pair.target.utterance_id)[: args.max_target_patches]
            reference = torch.cat(
                [source.patches(u.utterance_id) for u in pair.reference_group], dim=0
            )[: int(30 * 6.25)]
            if target.shape[0] < 2:
                continue
            prompt = build_voice_clone_prompt(processor, pair.target.text_raw)
            # 系列長が max_length を超えないよう target を切る
            budget = max_length - prompt.text_token_count - 1 - int(reference.shape[0])
            if budget < 2:
                continue
            target = target[: min(int(target.shape[0]), budget + 1)]
            samples.append(build_training_sample(
                utterance_id=pair.target.utterance_id,
                prompt=prompt,
                reference_latents=reference,
                target_latents=target,
            ))
            speakers.append(speaker_reader.read(pair.reference_group[0].utterance_id))
        if not samples:
            continue

        batch = collate(samples)
        speaker_tensor = torch.stack(speakers)

        lr = cosine_lr(step, peak=args.lr, warmup=args.warmup, total=args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        out = training_forward(model, batch, speaker_embeddings=speaker_tensor,
                               flow_copies=args.flow_copies, stop_weight=args.stop_weight,
                               dropout=dropout, generator=generator)
        out.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        if not torch.isfinite(out.loss):
            raise SystemExit(f"step {step}: loss が有限でない（中止・巻き戻し条件）")

        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.perf_counter() - started
            row = {"step": step, "lr": lr, "loss": float(out.loss),
                   "flow": float(out.flow_loss), "stop": float(out.stop_loss),
                   "grad_norm": float(grad_norm), "elapsed": elapsed}
            history.append(row)
            print(f"  step {step:5d}  loss={row['loss']:.4f} flow={row['flow']:.4f} "
                  f"stop={row['stop']:.4f} |g|={row['grad_norm']:.2f} lr={lr:.2e} "
                  f"{elapsed/(step - state.step + 1)*1000:.0f} ms/step")

        if args.save_every and (step + 1) % args.save_every == 0:
            state.step = step + 1
            save_training_state(out_dir, model=model, optimizer=optimizer,
                                state=state, generator=generator)

    state.step = args.steps
    save_training_state(out_dir, model=model, optimizer=optimizer, state=state,
                        generator=generator)
    export_for_inference(out_dir / "inference", model=model,
                         source_model_dir=Path(args.model_dir))

    artifacts.write_run_metadata(
        run_dir, phase="s0-train",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"manifest": args.manifest, "model_dir": args.model_dir},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "s0-train", "settings": vars(args),
        "train_records": len(usable), "eligible_groups": len(groups),
        "history": history,
        "total_seconds": time.perf_counter() - started,
        "checkpoint": str(out_dir), "inference_export": str(out_dir / "inference"),
    })
    print(f"\n完了: {run_dir}\n  checkpoint: {out_dir}\n  推論用: {out_dir / 'inference'}")


if __name__ == "__main__":
    main()
