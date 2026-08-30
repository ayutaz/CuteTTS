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

"""S0 の前段: 学習forwardの peak VRAM と throughput を実測する。

R-007（GPU規模の見積もり誤り）と D-006（full fine-tuning を主案とする）の
再判定材料を作るのが目的。**公開checkpointの実サイズ（228.6M）で測る。**

測るもの:

* microbatch 1 の peak VRAM（forward / backward / optimizer step を含む）
* target patch 数を変えたときの VRAM の伸び
* 1 step あたりの所要時間
* activation checkpointing の有無による差（`--gradient-checkpointing`）
* freeze 構成ごとの差（full / Patch Encoderのみfreeze / LM中心）

結果は `artifacts/s0-memory/<timestamp>/metrics.json` に残す。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from cutetts.modeling.configuration import CuteTTSConfig
from cutetts.modeling.model import CuteTTSModel
from cutetts.training import artifacts
from cutetts.training.collator import build_training_sample, collate
from cutetts.training.forward import TRAINABLE_MODULES, freeze_all_but, training_forward

FREEZE_VARIANTS = {
    # D-005 の主案。Patch Encoder も学習する
    "full": TRAINABLE_MODULES,
    # D-005 の比較案 B。Patch Encoder を freeze
    "freeze_patch_encoder": ("locenc_to_lm_proj", "lm_speaker_linear",
                             "qwen_backbone", "head", "stop_predictor"),
    # D-005 の比較案 C。LM 中心の最小更新
    "lm_only": ("qwen_backbone", "head", "stop_predictor"),
}


def build_model(model_dir: Path, device: torch.device, dtype: str) -> CuteTTSModel:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    architecture = dict(config["architecture"])
    architecture.update(attn_implementation="sdpa", use_pretrained_lm=False,
                        lm_model_name=None, torch_dtype=dtype)
    model = CuteTTSModel(CuteTTSConfig(**architecture))
    weights = load_file(str(model_dir / "weights" / "tts" / "model.safetensors"))
    missing, unexpected = model.load_state_dict(weights, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"weights did not load strictly: {missing} / {unexpected}")
    return model.to(device).train()


def make_batch(*, n_target: int, n_reference: int, n_text: int, dim: int, patch: int,
               device: torch.device, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    sample = build_training_sample(
        utterance_id="bench",
        prefix_ids=torch.randint(0, 16000, (n_text,), generator=generator),
        reference_latents=torch.randn(n_reference, patch, dim, generator=generator),
        target_latents=torch.randn(n_target, patch, dim, generator=generator),
        speaker_slot=True,
    )
    batch = collate([sample])
    speaker = torch.randn(1, 256, generator=generator)
    return batch, speaker.to(device)


def measure(model, batch, speaker, *, flow_copies: int, steps: int,
            optimizer, device: torch.device) -> dict:
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    resident = torch.cuda.memory_allocated(device)

    durations = []
    for i in range(steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = training_forward(model, batch, speaker_embeddings=speaker,
                               flow_copies=flow_copies,
                               generator=torch.Generator().manual_seed(i))
        out.loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)

    warm = durations[1:] or durations
    return {
        "resident_bytes": int(resident),
        "peak_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "seconds_per_step_mean": sum(warm) / len(warm),
        "seconds_per_step_min": min(warm),
        "loss": float(out.loss),
        "flow_loss": float(out.flow_loss),
        "stop_loss": float(out.stop_loss),
        "flow_rows": out.flow_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S0前段: 学習forwardのVRAMとthroughput実測")
    parser.add_argument("--model-dir", default="model/CuteTTS")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--target-patches", type=int, nargs="+", default=[32, 64, 128, 188])
    parser.add_argument("--reference-patches", type=int, default=63)   # 約10秒
    parser.add_argument("--text-tokens", type=int, default=48)
    parser.add_argument("--flow-copies", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--variants", nargs="+", default=["full"], choices=list(FREEZE_VARIANTS))
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("このベンチマークはCUDAが必要です")
    model_dir = Path(args.model_dir)
    run_dir = artifacts.new_run_dir("s0-memory", args.artifact_root, timestamp=args.timestamp)

    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}  VRAM {props.total_memory/1e9:.1f} GB")

    results = []
    for variant in args.variants:
        for n_target in args.target_patches:
            model = build_model(model_dir, device, args.dtype)
            frozen = freeze_all_but(model, FREEZE_VARIANTS[variant])
            if args.gradient_checkpointing:
                model.qwen_backbone.gradient_checkpointing_enable()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            dim = model.config.acoustic_latent_dim
            patch = model.config.locenc_patch_size
            batch, speaker = make_batch(
                n_target=n_target, n_reference=args.reference_patches,
                n_text=args.text_tokens, dim=dim, patch=patch, device=device,
            )
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-5
            )
            try:
                stats = measure(model, batch, speaker, flow_copies=args.flow_copies,
                                steps=args.steps, optimizer=optimizer, device=device)
                status = "ok"
            except torch.cuda.OutOfMemoryError as error:
                stats = {"error": "OOM", "detail": str(error)[:200]}
                status = "oom"
            row = {
                "variant": variant,
                "frozen_children": frozen,
                "trainable_parameters": trainable,
                "target_patches": n_target,
                "target_seconds": n_target / 6.25,
                "reference_patches": args.reference_patches,
                "flow_copies": args.flow_copies,
                "gradient_checkpointing": args.gradient_checkpointing,
                "dtype": args.dtype,
                "status": status,
                **stats,
            }
            results.append(row)
            if status == "ok":
                print(f"  {variant:22s} target={n_target:4d} "
                      f"({row['target_seconds']:5.1f}s)  "
                      f"peak={stats['peak_bytes']/1e9:6.2f} GB  "
                      f"{stats['seconds_per_step_mean']*1000:7.1f} ms/step")
            else:
                print(f"  {variant:22s} target={n_target:4d}  OOM")

            del model, optimizer, batch
            gc.collect()
            torch.cuda.empty_cache()

    metrics = {
        "phase": "s0-memory",
        "gpu": props.name,
        "vram_bytes": int(props.total_memory),
        "settings": vars(args),
        "results": results,
    }
    artifacts.write_run_metadata(
        run_dir, phase="s0-memory",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=None,
        inputs={"model_dir": str(model_dir)},
    )
    artifacts.write_metrics(run_dir, metrics)
    print(f"\n完了: {run_dir}")


if __name__ == "__main__":
    main()
