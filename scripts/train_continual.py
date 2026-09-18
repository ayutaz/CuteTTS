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
from itertools import islice
from pathlib import Path

import torch
from safetensors.torch import load_file

from cutetts.modeling.configuration import CuteTTSConfig
from cutetts.modeling.model import CuteTTSModel
from cutetts.training import artifacts
from cutetts.training.checkpointing import (
    TrainingState,
    export_for_inference,
    promote_to_float32,
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
from cutetts.training.yomi import FRONTEND_MODES, frontend_text
from cutetts.training.speaker_cache import SpeakerEmbeddingCacheReader


REFERENCE_PATCH_CAP = int(30 * 6.25)
"""referenceに使う最大patch数。推論側の30秒想定に合わせる。"""


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


class ParameterDrift:
    """「重みが実際に動いたか」を追跡する。

    R-020（bf16の丸めでbackboneが凍結する）を二度と見逃さないための計器。
    lossが下がっていても、それがheadだけの適応なら日本語は学習できていない。
    全要素を保持すると重いので、moduleごとに固定indexの標本を持つ。
    """

    SAMPLE = 4096

    def __init__(self, model: CuteTTSModel, *, seed: int, modules=("qwen_backbone", "locenc", "head")):
        self.modules = modules
        self.index: dict[str, torch.Tensor] = {}
        self.before: dict[str, torch.Tensor] = {}
        generator = torch.Generator().manual_seed(seed)
        for name, param in model.named_parameters():
            if not param.requires_grad or param.numel() < self.SAMPLE:
                continue
            picked = torch.randint(0, param.numel(), (self.SAMPLE,), generator=generator)
            self.index[name] = picked
            self.before[name] = param.detach().flatten()[picked].float().cpu().clone()

    def moved(self, model: CuteTTSModel) -> dict[str, float]:
        """moduleごとに「値が変わった標本の割合」を返す。"""
        hit: dict[str, list[int]] = {m: [0, 0] for m in self.modules}
        for name, param in model.named_parameters():
            if name not in self.index:
                continue
            module = next((m for m in self.modules if name.startswith(m)), None)
            if module is None:
                continue
            now = param.detach().flatten()[self.index[name]].float().cpu()
            hit[module][0] += int((now != self.before[name]).sum())
            hit[module][1] += now.numel()
        return {m: (c / t if t else 0.0) for m, (c, t) in hit.items()}


def cosine_lr(step: int, *, peak: float, warmup: int, total: int, floor_ratio: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return peak * (floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def build_batch(pairs, *, source, speaker_reader, processor, max_length,
                max_target_patches, frontend="none", assigner=None):
    """ペア列から `TrainingBatch` と speaker tensor を作る。作れなければ ``None``。"""
    samples, speakers = [], []
    for pair in pairs:
        target = source.patches(pair.target.utterance_id)[:max_target_patches]
        reference = torch.cat(
            [source.patches(u.utterance_id) for u in pair.reference_group], dim=0
        )[:REFERENCE_PATCH_CAP]
        if target.shape[0] < 2:
            continue
        # **学習と推論で同じ frontend を通す**（既定は現状維持の "none"）。
        # 長らく text_raw のままで、推論の J2/J3 と 29.9% の文で食い違っていた
        spoken = frontend_text(pair.target.text_raw, frontend, assigner=assigner)
        prompt = build_voice_clone_prompt(processor, spoken)
        # 系列長が max_length を超えないよう target を切る
        budget = max_length - prompt.text_token_count - 1 - int(reference.shape[0])
        if budget < 2:
            continue
        target = target[: min(int(target.shape[0]), budget + 1)]
        samples.append(build_training_sample(
            utterance_id=pair.target.utterance_id, prompt=prompt,
            reference_latents=reference, target_latents=target))
        speakers.append(speaker_reader.read(pair.reference_group[0].utterance_id))
    if not samples:
        return None
    return collate(samples), torch.stack(speakers)


def build_eval_batches(records, *, seed, batch_size, batches, group_key, **kwargs):
    """dev評価用の **固定** バッチ列を作る。学習中ずっと同じものを使う。

    毎回ちがうバッチで測ると、損失の変化が「学習の進み」なのか
    「バッチの難易度差」なのか分からなくなる。
    """
    sampler = PairSampler(records, seed=seed, group_key=group_key,
                          min_utterances_per_group=2, target_reference_seconds=10.0)
    stream = sampler.iter_pairs()
    out = []
    for _ in range(batches):
        pairs = list(islice(stream, batch_size))
        assert_no_leakage(pairs)
        built = build_batch(pairs, **kwargs)
        if built is not None:
            out.append(built)
    return out


@torch.no_grad()
def evaluate(model, batches, *, flow_copies, stop_weight, seed):
    """固定バッチで flow / stop loss を測る。``model`` の mode は呼び出し側で戻す。"""
    flow, stop = [], []
    for batch, speaker in batches:
        out = training_forward(model, batch, speaker_embeddings=speaker,
                               flow_copies=flow_copies, stop_weight=stop_weight,
                               generator=torch.Generator().manual_seed(seed))
        flow.append(float(out.flow_loss))
        stop.append(float(out.stop_loss))
    if not flow:
        return None
    return {"flow": sum(flow) / len(flow), "stop": sum(stop) / len(stop)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="日本語継続学習（S0）")
    parser.add_argument("--model-dir", default="model/CuteTTS")
    parser.add_argument("--manifest", default="data/manifests/all_clustered.jsonl")
    parser.add_argument("--latent-cache", default="data/cache/latents")
    parser.add_argument("--speaker-cache", default="data/cache/speaker")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--frontend", default="none", choices=FRONTEND_MODES,
                        help="学習テキストに掛ける frontend（M4a）。"
                             "**既定は none で、学習21回すべてこれだった**。"
                             "推論側と同じ値を使うこと（食い違うと条件が変わる）。"
                             "accent は全文を片仮名にしてアクセント核に記号を置く")
    parser.add_argument("--trainable", default=",".join(TRAINABLE_MODULES),
                        help="学習する子moduleをカンマ区切りで指定する（T2）。"
                             "既定は6 module全部。**R-020以前は実質 head だけが"
                             "学習されていた**ので、全部動かす今が最適とは限らない。"
                             "head を凍結する例: locenc,locenc_to_lm_proj,"
                             "lm_speaker_linear,qwen_backbone,stop_predictor")
    parser.add_argument("--param-dtype", default="float32",
                        choices=("float32", "checkpoint"),
                        help="学習中のパラメータdtype。checkpoint は bf16 のまま更新する"
                             "（R-020を再現するためだけの選択肢）")
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
    parser.add_argument("--eval-every", type=int, default=0,
                        help="このstep間隔でdevのflow/stop lossを測る。0で無効。"
                             "長い学習では必ず有効にすること（D-025）")
    parser.add_argument("--eval-batches", type=int, default=16,
                        help="dev評価に使う固定バッチ数")
    parser.add_argument("--export-every-save", action="store_true",
                        help="--save-every のたびに推論用exportも残す。"
                             "長い学習でCERの推移を測るために使う（R-015）")
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
    # **公開checkpointは backbone / locenc が bf16。**AdamW がそれを直接更新すると
    # lr=2e-5 の更新量が bf16 の丸め幅を下回り、backboneの91% / locencの84%が
    # 1stepも動かない（R-020）。fp32 に上げてから学習し、export で元のdtypeへ戻す。
    export_dtypes = promote_to_float32(model) if args.param_dtype == "float32" else None
    # **学習対象は選べる**（T2）。R-020以前は実質 head だけが学習されていたので、
    # 6 module全部を動かす今が最適とは限らない。
    trainable_names = tuple(
        name.strip() for name in args.trainable.split(",") if name.strip())
    if not trainable_names:
        raise SystemExit("--trainable が空")
    frozen = freeze_all_but(model, trainable_names)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable)/1e6:.1f}M"
          f"  ({', '.join(trainable_names)})")
    if frozen:
        print(f"frozen: {', '.join(frozen)}")
    print(f"param dtype: {args.param_dtype}"
          + ("" if export_dtypes else "  ← bf16のまま（R-020の再現用）"))
    drift = ParameterDrift(model, seed=args.seed)

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

    # `sample()` は呼ぶたびに RNG を作り直すため、step ごとに呼ぶと
    # 毎回まったく同じペアが返る（= 数発話だけを丸暗記する）。
    # stream を1本持って、そこから順に引くこと。
    pair_stream = sampler.iter_pairs()

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

    # dev の固定バッチ。学習ループの損失だけでは成否を判断できない（D-025）。
    # **frontend は学習と推論で同じものを通す**（M4a）。`yomi` だけ語彙が要る
    frontend_assigner = None
    if args.frontend == "yomi":
        from cutetts.training.yomi import ReadingAssigner

        frontend_assigner = ReadingAssigner.from_model_dir(args.model_dir)
    if args.frontend != "none":
        sample = frontend_text(records[0].text_raw, args.frontend,
                              assigner=frontend_assigner)
        print(f"frontend={args.frontend}  例: {records[0].text_raw} → {sample}")
    build_kwargs = dict(source=source, speaker_reader=speaker_reader,
                        processor=processor, max_length=max_length,
                        max_target_patches=args.max_target_patches,
                        frontend=args.frontend, assigner=frontend_assigner)
    eval_sets: dict[str, list] = {}
    if args.eval_every:
        for split in ("dev-seen", "dev-zero-shot"):
            rows = [r for r in load_manifest(args.manifest)
                    if r.split == split
                    and r.utterance_id in latent_reader and r.utterance_id in speaker_reader]
            if len(rows) < 2:
                print(f"  dev評価: {split} は record {len(rows)}件のためスキップ")
                continue
            eval_sets[split] = build_eval_batches(
                rows, seed=args.seed + 1, batch_size=args.batch_size,
                batches=args.eval_batches, group_key=args.group_key, **build_kwargs)
            print(f"  dev評価: {split} {len(eval_sets[split])} バッチ")

    history: list[dict] = []
    evaluations: list[dict] = []

    def run_eval(step: int) -> None:
        """固定バッチでdevを測って `evaluations` へ積む。"""
        if not eval_sets:
            return
        model.eval()
        row = {"step": step}
        for split, batches in eval_sets.items():
            scores = evaluate(model, batches, flow_copies=args.flow_copies,
                              stop_weight=args.stop_weight, seed=args.seed)
            if scores:
                row[split] = scores
        model.train()
        evaluations.append(row)
        summary = "  ".join(f"{split}: flow={v['flow']:.4f} stop={v['stop']:.4f}"
                            for split, v in row.items() if isinstance(v, dict))
        print(f"  [dev] step {step:6d}  {summary}")

    # **学習前に1度測る。** これが無いと、途中のdev値が改善なのか悪化なのか
    # 判定できない（比較対象が無い）。
    if args.eval_every and not state.step:
        run_eval(0)

    started = time.perf_counter()
    first_step = state.step   # state.step は save のたびに進むので、開始点は別に持つ
    if first_step:
        # resume 時は消費済みの分だけ stream を進めて、同じペアを繰り返さない
        next(islice(pair_stream, first_step * args.batch_size, first_step * args.batch_size), None)
    for step in range(state.step, args.steps):
        pairs = list(islice(pair_stream, args.batch_size))
        assert_no_leakage(pairs)

        built = build_batch(pairs, **build_kwargs)
        if built is None:
            continue
        batch, speaker_tensor = built

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
            ms_per_step = elapsed / (step - first_step + 1) * 1000
            row = {"step": step, "lr": lr, "loss": float(out.loss),
                   "flow": float(out.flow_loss), "stop": float(out.stop_loss),
                   "grad_norm": float(grad_norm), "elapsed": elapsed,
                   "ms_per_step": ms_per_step}
            history.append(row)
            print(f"  step {step:5d}  loss={row['loss']:.4f} flow={row['flow']:.4f} "
                  f"stop={row['stop']:.4f} |g|={row['grad_norm']:.2f} lr={lr:.2e} "
                  f"{ms_per_step:.0f} ms/step")

        if args.eval_every and (step + 1) % args.eval_every == 0:
            run_eval(step + 1)

        if args.save_every and (step + 1) % args.save_every == 0:
            state.step = step + 1
            save_training_state(out_dir, model=model, optimizer=optimizer,
                                state=state, generator=generator)
            if args.export_every_save:
                # 推論用exportも残す。学習中にCERを測るには推論可能な形が要る。
                # flow/stop loss では生成の崩壊を検知できない（R-015）。
                export_for_inference(out_dir / f"inference-{step + 1}", model=model,
                                     source_model_dir=Path(args.model_dir),
                                     dtypes=export_dtypes)

    state.step = args.steps
    save_training_state(out_dir, model=model, optimizer=optimizer, state=state,
                        generator=generator)
    export_for_inference(out_dir / "inference", model=model,
                         source_model_dir=Path(args.model_dir),
                         dtypes=export_dtypes)

    artifacts.write_run_metadata(
        run_dir, phase="s0-train",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"manifest": args.manifest, "model_dir": args.model_dir},
    )
    # **重みが実際に動いたかを必ず記録する。** loss が下がっても、それが head だけの
    # 適応なら日本語は学習できていない（R-020）。
    moved = drift.moved(model)
    print("\n重みが動いた標本の割合:")
    for module, ratio in moved.items():
        print(f"  {module:16s} {ratio*100:6.2f}%"
              + ("" if ratio > 0.5 else "  ← ほぼ動いていない"))

    artifacts.write_metrics(run_dir, {
        "phase": "s0-train", "settings": vars(args),
        "frozen_modules": frozen,
        "train_records": len(usable), "eligible_groups": len(groups),
        "history": history,
        "evaluations": evaluations,
        "parameter_moved_ratio": moved,
        "total_seconds": time.perf_counter() - started,
        "checkpoint": str(out_dir), "inference_export": str(out_dir / "inference"),
    })
    print(f"\n完了: {run_dir}\n  checkpoint: {out_dir}\n  推論用: {out_dir / 'inference'}")


if __name__ == "__main__":
    main()
