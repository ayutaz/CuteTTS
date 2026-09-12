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

"""英語・中国語の忘却（catastrophic forgetting）を固定subsetで測る（R-005）。

**この測定が意味を持つようになったのは R-020 の修正以降である。**
それまでは `qwen_backbone` の91%が bf16 の丸めで凍結しており、
100%日本語で学習しても既存言語の能力は構造的に劣化しようがなかった。
backbone が実際に動くようになった以上、忘却は現実のリスクになる。

base と学習後を同じ経路で測り、**差**を見る。絶対値は評価文と
reference条件に依存するので、base との差だけを主張する。

    python scripts/evaluate_forgetting.py --model-dir model/CuteTTS --label base
    python scripts/evaluate_forgetting.py \\
        --model-dir checkpoints/s1v2-fp32-long/inference --label fp32-12000
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutetts import CuteTTS  # noqa: E402
from cutetts.training import artifacts  # noqa: E402

ASR_MODEL = "openai/whisper-large-v3"
"""多言語ASR。日本語CERの `kotoba-whisper-v2.0` は日本語専用なので使えない。"""

# 一般的な語彙・構文で、固有名詞と数字を避けた文。TTSの素の能力を見る。
ENGLISH = [
    "The weather is quite pleasant this afternoon.",
    "She opened the window to let the fresh air in.",
    "I would like a cup of coffee with milk, please.",
    "They decided to walk home instead of taking the bus.",
    "The library closes early on weekends during winter.",
    "He forgot to bring his umbrella again this morning.",
    "We should probably leave before the traffic gets worse.",
    "The garden looks beautiful after the recent rain.",
    "Could you explain that part one more time, slowly?",
    "My brother works as a teacher at a small school.",
    "The book was much longer than I had expected.",
    "Please remember to turn off the lights when you leave.",
    "There is a quiet park just behind the train station.",
    "She learned to play the piano when she was young.",
    "The meeting was postponed until the following week.",
    "I have never seen such a bright and clear sky.",
    "He speaks slowly, but every word is very clear.",
    "The children were playing happily in the back yard.",
    "We ran out of milk, so I went to the shop.",
    "It takes about twenty minutes to walk from here.",
]

CHINESE = [
    "今天的天气非常好，适合出去散步。",
    "她打开窗户，让新鲜空气进来。",
    "请给我一杯加牛奶的咖啡。",
    "他们决定走路回家，不坐公共汽车。",
    "图书馆在冬天的周末关门比较早。",
    "他今天早上又忘记带雨伞了。",
    "我们最好在堵车之前出发。",
    "下过雨之后，花园看起来很漂亮。",
    "你能再慢慢地解释一遍那个部分吗？",
    "我哥哥在一所小学校当老师。",
    "这本书比我原来想象的要长得多。",
    "离开的时候请记得关灯。",
    "火车站后面有一个安静的公园。",
    "她小时候学过弹钢琴。",
    "会议被推迟到了下个星期。",
    "我从来没有见过这么明亮的天空。",
    "他说话很慢，但是每个字都很清楚。",
    "孩子们在后院里快乐地玩耍。",
    "牛奶喝完了，所以我去了商店。",
    "从这里走过去大概需要二十分钟。",
]

_PUNCT = re.compile(r"[\s、。「」『』・…‥！？!?,.\-―ー~〜\"'()（），：；、。！？]")


def _edit_distance(reference: list, hypothesis: list) -> int:
    distances = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        previous, distances[0] = distances[0], i
        for j, h in enumerate(hypothesis, 1):
            current = distances[j]
            distances[j] = min(distances[j] + 1, distances[j - 1] + 1,
                               previous + (r != h))
            previous = current
    return distances[len(hypothesis)]


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """英語は語単位。大文字小文字と句読点は落とす。"""
    ref = re.sub(r"[^\w\s']", " ", reference.lower()).split()
    hyp = re.sub(r"[^\w\s']", " ", hypothesis.lower()).split()
    if not ref:
        return None
    return _edit_distance(ref, hyp) / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    """中国語は文字単位。日本語CERと同じ正規化を使う。"""
    ref = _PUNCT.sub("", unicodedata.normalize("NFKC", reference))
    hyp = _PUNCT.sub("", unicodedata.normalize("NFKC", hypothesis))
    if not ref:
        return None
    return _edit_distance(list(ref), list(hyp)) / len(ref)


class MultilingualTranscriber:
    def __init__(self, device: torch.device, model_name: str = ASR_MODEL):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_name)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=dtype).to(device).eval()
        self.device = device
        self.dtype = dtype

    def __call__(self, waveform: torch.Tensor, sample_rate: int, language: str) -> str:
        wave16 = torchaudio.functional.resample(waveform, sample_rate, 16000)
        features = self.processor(wave16.squeeze(0).cpu().numpy(),
                                  sampling_rate=16000, return_tensors="pt")
        inputs = features.input_features.to(self.device, self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(inputs, language=language, task="transcribe",
                                      max_new_tokens=128)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


SUBSETS = {
    "english": (ENGLISH, "en", word_error_rate, "WER"),
    "chinese": (CHINESE, "zh", character_error_rate, "CER"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="英語・中国語の忘却を測る（R-005）")
    parser.add_argument("--model-dir", default="model/CuteTTS")
    parser.add_argument("--reference-audio", default="assets/default_reference.wav",
                        help="日本語CER評価と同じ条件にするため既定を揃える")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="voice_clone", choices=("tts", "voice_clone"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decode-length", type=int, default=400)
    parser.add_argument("--asr-model", default=ASR_MODEL)
    parser.add_argument("--label", default="base")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    run_dir = artifacts.new_run_dir("forgetting", args.artifact_root, timestamp=args.timestamp)

    model = CuteTTS.from_pretrained(args.model_dir, device=str(device))
    asr = MultilingualTranscriber(device, args.asr_model)
    print(f"model: {args.model_dir} (variant={model.variant})  label={args.label}")

    rows: list[dict] = []
    for subset, (texts, language, metric, metric_name) in SUBSETS.items():
        for index, text in enumerate(texts):
            try:
                result = model.generate(
                    text, mode=args.mode,
                    reference_audio=args.reference_audio if args.mode == "voice_clone" else None,
                    seed=args.seed, max_decode_length=args.max_decode_length,
                    show_progress=False,
                )
            except Exception as error:  # 生成失敗も記録して先へ進む
                rows.append({"subset": subset, "index": index, "text": text,
                             "status": "error",
                             "detail": f"{type(error).__name__}: {error}"[:200]})
                continue
            hypothesis = asr(result.waveform.to(device), result.sample_rate, language)
            value = metric(text, hypothesis)
            rows.append({
                "subset": subset, "index": index, "text": text,
                "hypothesis": hypothesis, "cer": value, "metric": metric_name,
                "seconds": result.waveform.shape[-1] / result.sample_rate,
                "status": "ok",
            })
            print(f"  [{subset}/{index:02d}] {metric_name}="
                  f"{value * 100 if value is not None else -1:5.1f}%")

    summary: dict = {}
    for subset, (_, _, _, metric_name) in SUBSETS.items():
        values = [r["cer"] for r in rows
                  if r["subset"] == subset and r.get("status") == "ok" and r.get("cer") is not None]
        summary[subset] = {"n": len(values), "metric": metric_name} if not values else {
            "n": len(values),
            "metric": metric_name,
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }

    print("\n=== subset別 ===")
    for subset, value in summary.items():
        if value["n"]:
            print(f"  {subset:10s} n={value['n']:3d}  {value['metric']} "
                  f"mean={value['mean'] * 100:5.1f}%  median={value['median'] * 100:5.1f}%")

    artifacts.write_run_metadata(
        run_dir, phase="forgetting",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"model_dir": args.model_dir, "asr_model": args.asr_model},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "forgetting", "label": args.label,
        "settings": {"mode": args.mode, "seed": args.seed,
                     "max_decode_length": args.max_decode_length,
                     "asr_model": args.asr_model,
                     "reference_audio": args.reference_audio},
        "summary": summary, "rows": rows,
    })
    print(f"\n完了: {run_dir}")


if __name__ == "__main__":
    main()
