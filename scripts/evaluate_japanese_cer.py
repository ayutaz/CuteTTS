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

"""固定評価setで日本語CERを測る。S0のゲート判定に使う。

同じ評価set・同じASR・同じ推論設定で checkpoint を比較する
（06章「同じtext、reference、seed、推論設定でcheckpointを比較する」）。

ASR は `kotoba-tech/kotoba-whisper-v2.0` に固定（D-019）。
subset ごと（in_domain / out_of_domain / phonetic）に分けて集計する。
R-010 のとおり out_of_domain は学習分布の外なので、
**in_domain と同列に平均しない**。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from cutetts import CuteTTS
from cutetts.training import artifacts

ASR_MODEL = "kotoba-tech/kotoba-whisper-v2.0"
_PUNCT = re.compile(r"[\s、。「」『』・…‥！？!?,.\-―ー~〜\"'()（）]")


def normalize(text: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text))


def cer(reference: str, hypothesis: str) -> float | None:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return None
    distances = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        previous, distances[0] = distances[0], i
        for j, hc in enumerate(hyp, 1):
            current = distances[j]
            distances[j] = min(distances[j] + 1, distances[j - 1] + 1,
                               previous + (rc != hc))
            previous = current
    return distances[len(hyp)] / len(ref)


class Transcriber:
    def __init__(self, device: torch.device):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(ASR_MODEL)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            ASR_MODEL, torch_dtype=dtype).to(device).eval()
        self.device = device
        self.dtype = dtype

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> str:
        wave16 = torchaudio.functional.resample(waveform, sample_rate, 16000)
        features = self.processor(wave16.squeeze(0).cpu().numpy(),
                                  sampling_rate=16000, return_tensors="pt")
        inputs = features.input_features.to(self.device, self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(inputs, language="ja", task="transcribe",
                                      max_new_tokens=128)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="固定評価setで日本語CERを測る")
    parser.add_argument("--model-dir", default="model/CuteTTS")
    parser.add_argument("--eval-set", default="data/eval/s0_eval_set.json")
    parser.add_argument("--reference-audio", default="assets/default_reference.wav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="voice_clone", choices=("tts", "voice_clone"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decode-length", type=int, default=400)
    parser.add_argument("--label", default="baseline", help="artifactに残す識別名")
    parser.add_argument("--save-samples", type=int, default=6,
                        help="保存する音声の数。artifacts配下（公開禁止）")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    run_dir = artifacts.new_run_dir("s0-cer", args.artifact_root, timestamp=args.timestamp)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    model = CuteTTS.from_pretrained(args.model_dir, device=str(device))
    asr = Transcriber(device)
    print(f"model: {args.model_dir} (variant={model.variant})")
    print(f"eval set: {args.eval_set}  checksum {artifacts.file_checksum(args.eval_set)[:16]}...")

    rows: list[dict] = []
    saved = 0
    for subset, items in payload["subsets"].items():
        for index, item in enumerate(items):
            text = item["text"]
            try:
                result = model.generate(
                    text, mode=args.mode,
                    reference_audio=args.reference_audio if args.mode == "voice_clone" else None,
                    seed=args.seed, max_decode_length=args.max_decode_length,
                    show_progress=False,
                )
            except Exception as error:  # 生成失敗も記録して先へ進む
                rows.append({"subset": subset, "index": index, "text": text,
                             "status": "error", "detail": f"{type(error).__name__}: {error}"[:200]})
                continue
            waveform = result.waveform
            hypothesis = asr(waveform.to(device), result.sample_rate)
            value = cer(text, hypothesis)
            rows.append({
                "subset": subset, "index": index, "text": text,
                "hypothesis": hypothesis, "cer": value,
                "seconds": waveform.shape[-1] / result.sample_rate, "status": "ok",
            })
            if saved < args.save_samples:
                sf.write(samples_dir / f"{subset}_{index:02d}.wav",
                         waveform.squeeze(0).float().numpy(), result.sample_rate)
                saved += 1
            print(f"  [{subset}/{index:02d}] CER={value*100 if value is not None else -1:5.1f}%")

    summary: dict = {}
    for subset in payload["subsets"]:
        values = [r["cer"] for r in rows
                  if r["subset"] == subset and r.get("status") == "ok" and r.get("cer") is not None]
        if not values:
            summary[subset] = {"n": 0}
            continue
        values_sorted = sorted(values)
        summary[subset] = {
            "n": len(values),
            "cer_mean": statistics.mean(values),
            "cer_median": statistics.median(values),
            "cer_p90": values_sorted[int(len(values) * 0.9) - 1] if len(values) >= 10 else None,
            "cer_min": values_sorted[0],
            "cer_max": values_sorted[-1],
        }

    metrics = {
        "phase": "s0-cer",
        "label": args.label,
        "model_dir": str(args.model_dir),
        "asr_model": ASR_MODEL,
        "eval_set": str(args.eval_set),
        "eval_set_sha256": artifacts.file_checksum(args.eval_set),
        "settings": {"mode": args.mode, "seed": args.seed,
                     "max_decode_length": args.max_decode_length},
        "summary": summary,
        "rows": rows,
    }
    artifacts.write_run_metadata(
        run_dir, phase="s0-cer",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"model_dir": str(args.model_dir), "eval_set": str(args.eval_set)},
    )
    artifacts.write_metrics(run_dir, metrics)

    print("\n=== subset別 CER ===")
    for subset, stats in summary.items():
        if stats.get("n"):
            print(f"  {subset:14s} n={stats['n']:3d}  "
                  f"mean={stats['cer_mean']*100:5.1f}%  median={stats['cer_median']*100:5.1f}%")
    print(f"\n完了: {run_dir}")


if __name__ == "__main__":
    main()
