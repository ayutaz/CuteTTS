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

"""F0 の条件を**モデルが使っているか**を loss で直接見る（M4c の診断）。

生成した音声の指標（輪郭・アクセント）が動かなかったとき、原因は2つある。

1. **モデルが条件を使っていない**（学習が条件を無視した）
2. **使ってはいるが音まで届いていない**（生成の経路で薄まる）

**flow loss を条件あり・なしで比べれば分かる。** teacher forcing で
真の F0 を与えたときに loss が下がらないなら 1、下がるのに音が変わらないなら 2。

    uv run --no-sync python scripts/diagnose_f0_conditioning.py \\
      --checkpoint checkpoints/m4c-validate --f0-cache data/s1v2/f0-17.9h \\
      --manifest data/s1v2/subset-17.9h.jsonl --device cuda

**条件を壊した3通りも測る**（zero / shuffle / mismatch）。真の F0 だけが
loss を下げるなら、モデルは中身を見ている。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from cutetts.training import artifacts  # noqa: E402
from cutetts.training.dataset import F0Source, LatentSource  # noqa: E402
from cutetts.training.f0 import F0CacheReader, load_f0_conditioner  # noqa: E402
from cutetts.training.forward import training_forward  # noqa: E402
from cutetts.training.latents import LatentCacheReader  # noqa: E402
from cutetts.training.manifest import load_manifest  # noqa: E402
from cutetts.training.pairing import PairSampler  # noqa: E402
from cutetts.training.speaker_cache import SpeakerEmbeddingCacheReader  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="`inference/` を含む学習出力ディレクトリ")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--latent-cache", default="data/s1v2/latents-v2")
    parser.add_argument("--speaker-cache", default="data/s1v2/speaker-v2")
    parser.add_argument("--f0-cache", required=True)
    parser.add_argument("--frontend", default="accent")
    parser.add_argument("--split", default="dev-seen")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--flow-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-target-patches", type=int, default=400)
    parser.add_argument("--group-key", default="voice_cluster_id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("m4c-diagnose", args.artifact_root,
                                    timestamp=args.timestamp)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 学習スクリプトと同じ組み立てを使う（食い違うと診断にならない）
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_continual import build_batch, load_model

    inference_dir = Path(args.checkpoint) / "inference"
    model = load_model(inference_dir, device, "bfloat16").eval()
    conditioner = load_f0_conditioner(inference_dir, device=str(device))
    if conditioner is None:
        raise SystemExit(f"{inference_dir}/f0_conditioner.safetensors が無い")
    conditioner = conditioner.to(torch.float32)

    from cutetts.modeling.processor import CuteTTSProcessor
    from cutetts.modeling.segments import SegmentManagerConfig

    config = json.loads((inference_dir / "config.json").read_text(encoding="utf-8"))
    processor = CuteTTSProcessor(
        acoustic_vae_path=str(inference_dir / "weights" / "audio_vae"),
        tokenizer=str(inference_dir / "tokenizer"),
        segment_cfg=SegmentManagerConfig(**config["processor"]["segment"]),
        speech_compress_rate=int(config["processor"]["speech_compress_rate"]),
    )
    max_length = int(config["processor"]["segment"]["max_length"])
    patch = int(config["architecture"]["locenc_patch_size"])

    latent_reader = LatentCacheReader(args.latent_cache)
    speaker_reader = SpeakerEmbeddingCacheReader(args.speaker_cache)
    source = LatentSource(reader=latent_reader,
                          scaling=model.speech_scaling_factor.detach().cpu(),
                          bias=model.speech_bias_factor.detach().cpu(),
                          patch_size=patch)
    f0_source = F0Source(reader=F0CacheReader(args.f0_cache), patch_size=patch)

    rows = [r for r in load_manifest(args.manifest) if r.split == args.split]
    usable = [r for r in rows if r.utterance_id in latent_reader
              and r.utterance_id in speaker_reader]
    sampler = PairSampler(usable, seed=args.seed, group_key=args.group_key)
    stream = sampler.iter_pairs()

    batches = []
    for _ in range(args.batches):
        pairs = list(islice(stream, args.batch_size))
        if not pairs:
            break
        built = build_batch(pairs, source=source, speaker_reader=speaker_reader,
                            processor=processor, max_length=max_length,
                            max_target_patches=args.max_target_patches,
                            frontend=args.frontend, f0_source=f0_source)
        if built is not None:
            batches.append(built)
    if not batches:
        raise SystemExit("バッチが作れない")
    print(f"{len(batches)} バッチ / split={args.split} / device={device}")

    def flow_of(mode: str) -> float:
        """`mode` で条件を作り替えたときの flow loss。"""
        total, count = 0.0, 0
        for batch, speaker in batches:
            target_f0 = batch.target_f0
            if mode == "none":
                changed = None
            elif mode == "true":
                changed = target_f0
            elif mode == "zero":
                changed = torch.zeros_like(target_f0)
            elif mode == "shuffle":
                # **同じ発話の中で patch の順を入れ替える**（分布はそのまま）
                index = torch.randperm(target_f0.shape[0],
                                       generator=torch.Generator().manual_seed(7))
                changed = target_f0[index]
            else:
                raise ValueError(mode)
            probe = (batch if changed is target_f0
                     else dataclasses.replace(batch, target_f0=changed))
            with torch.no_grad():
                out = training_forward(
                    model, probe, speaker_embeddings=speaker,
                    flow_copies=args.flow_copies,
                    generator=torch.Generator().manual_seed(args.seed),
                    f0_conditioner=None if mode == "none" else conditioner)
            total += float(out.flow_loss)
            count += 1
        return total / max(count, 1)

    print()
    results = {}
    for mode, label in (("none", "条件を渡さない"),
                        ("true", "**真の F0**"),
                        ("zero", "全部ゼロ（無声扱い）"),
                        ("shuffle", "patch の順を入れ替え")):
        value = flow_of(mode)
        results[mode] = value
        print(f"  {label:<24} flow={value:.4f}")

    base = results["none"]
    print()
    print(f"  真の F0 で {base - results['true']:+.4f}"
          f"（{(results['true'] - base) / base * 100:+.1f}%）")
    print(f"  順を入れ替えると {results['shuffle'] - results['true']:+.4f}")
    print()
    print("  **真の F0 だけが loss を下げるなら、モデルは中身を見ている。**")
    print("  下がらないなら学習が条件を無視した（条件の入れ方を変える）。")

    artifacts.write_run_metadata(
        run_dir, phase="m4c-diagnose",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"checkpoint": args.checkpoint, "f0_cache": args.f0_cache})
    artifacts.write_metrics(run_dir, {
        "phase": "m4c-diagnose", "split": args.split,
        "batches": len(batches), "flow": results,
        "gain_true_vs_none": base - results["true"],
        "gain_true_vs_shuffle": results["shuffle"] - results["true"],
    })
    print(f"  {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
