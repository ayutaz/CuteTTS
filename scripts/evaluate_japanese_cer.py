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
from cutetts.training.evalstats import summarize_subsets
from cutetts.training.reading import expand_kanji_numerals, to_arabic_numerals
from cutetts.training.yomi import ReadingAssigner, reading_form

ASR_MODEL = "kotoba-tech/kotoba-whisper-v2.0"
_PUNCT = re.compile(r"[\s、。「」『』・…‥！？!?,.\-―ー~〜\"'()（）]")


def normalize(text: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text))


def reading_cer(reference: str, hypothesis: str) -> float | None:
    """**読み**に直してから測るCER（R-029）。

    仮名で入力すると ASR も仮名で書き戻すので、漢字の参照文に対する素のCERは
    「発音は正しいのに表記が違う」を誤りと数える。J3 の評価300文では
    悪化とされた80文のうち **24文がこれ**だった。

    同音異義の誤りは見えなくなるので、**素のCERと併記する**。
    `pyopenjtalk` が要る（`[ja]` extra）。入っていなければ None。
    """
    try:
        ref, hyp = reading_form(reference), reading_form(hypothesis)
    except ImportError:
        return None
    return cer(ref, hyp) if ref else None


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
    parser.add_argument("--assign-yomi", action="store_true",
                        help="生成前に byte-fallback を含む語を読みへ置き換える"
                             "（J3 / D-034）。CERは元のtextに対して測る")
    parser.add_argument("--expand-numerals", action="store_true",
                        help="生成前に漢数字を読み（仮名）へ展開する（J2 / D-008）。CERは元のtextに対して測るので、比較はそのまま成立する")
    parser.add_argument("--save-samples", type=int, default=6,
                        help="保存する音声の数。artifacts配下（公開禁止）")
    parser.add_argument("--shard", metavar="K/N",
                        help="評価setを N 分割して K 番目だけを測る（1始まり）。"
                             "**生成はbatch=1でGPUが埋まらないので、分割して"
                             "同時に走らせると速い**")
    parser.add_argument("--merge", metavar="PATHS",
                        help="shardの metrics.json をカンマ区切りで渡すと、"
                             "生成をやり直さず行を結合して集計だけ作る（GPU不要）")
    parser.add_argument("--no-warmup", action="store_true",
                        help="捨て生成を省く。**プロセス内の最初の生成だけ結果が"
                             "違う**ので、既定では1回捨てて揃える")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def parse_shard(text: str, total: int) -> set[int]:
    """`K/N` を、担当する通し番号の集合へ。**飛び飛びに取る**（負荷を均す）。"""
    part, _, count = text.partition("/")
    index, size = int(part), int(count)
    if not (1 <= index <= size):
        raise SystemExit(f"--shard の値が範囲外: {text}")
    return set(range(index - 1, total, size))


def merge_rows(paths: str) -> list[dict]:
    """shardの metrics.json から行を集める。**(subset, index) で重複を弾く。**"""
    seen: dict[tuple, dict] = {}
    for path in paths.split(","):
        payload = json.loads(Path(path.strip()).read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            key = (row.get("subset"), row.get("index"))
            if key in seen:
                raise SystemExit(f"{key} が複数のshardにある: {path}")
            seen[key] = row
    return [seen[key] for key in sorted(seen, key=lambda k: (str(k[0]), k[1]))]


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    run_dir = artifacts.new_run_dir("s0-cer", args.artifact_root, timestamp=args.timestamp)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))

    if args.merge:
        # **生成をやり直さない。** 行を結合して集計だけ作る
        rows = merge_rows(args.merge)
        summary = summarize_subsets(rows, payload["subsets"])
        print(f"{len(rows)} 行を結合（shard {len(args.merge.split(','))} 個）")
        write_output(run_dir, args, summary, rows)
        return

    model = CuteTTS.from_pretrained(args.model_dir, device=str(device))
    asr = Transcriber(device)
    # J3 は tokenizer の語彙を見るので、評価対象の checkpoint から読む
    yomi = ReadingAssigner.from_model_dir(args.model_dir) if args.assign_yomi else None
    print(f"model: {args.model_dir} (variant={model.variant})")
    print(f"eval set: {args.eval_set}  checksum {artifacts.file_checksum(args.eval_set)[:16]}...")

    # 全subsetを通した番号で分割する（subsetごとに分けると偏る）
    flat = [(subset, index, item)
            for subset, items in payload["subsets"].items()
            for index, item in enumerate(items)]
    picked = (parse_shard(args.shard, len(flat)) if args.shard
              else set(range(len(flat))))
    if args.shard:
        print(f"shard {args.shard}: {len(picked)}/{len(flat)} 文")

    # **プロセス内の最初の生成だけ結果が違う**（seedを毎回設定していても）。
    # 初回の遅延初期化が乱数列をずらすため。捨て生成で揃える。
    # これをしないと --shard の結合結果が一括と一致しない。
    if not args.no_warmup:
        first = next(item for order, (_, _, item) in enumerate(flat)
                     if order in picked)
        model.generate(first["text"], mode=args.mode,
                       reference_audio=(args.reference_audio
                                        if args.mode == "voice_clone" else None),
                       seed=args.seed, max_decode_length=args.max_decode_length,
                       show_progress=False)

    rows: list[dict] = []
    saved = 0
    order = -1
    for subset, items in payload["subsets"].items():
        for index, item in enumerate(items):
            order += 1
            if order not in picked:
                continue
            text = item["text"]
            # **CERは元のtextに対して測る。** 展開するのは生成への入力だけなので、
            # 展開なしの実行とそのまま比較できる。
            spoken = expand_kanji_numerals(text) if args.expand_numerals else text
            if yomi is not None:
                spoken = yomi.apply(spoken)
            try:
                result = model.generate(
                    spoken, mode=args.mode,
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
            # **数字の表記に依存しないCERも残す。** ASRは音声を聞いて `1280円` と
            # 書くが、参照は `千二百八十円`。正しく読めているほど素のCERは
            # 上がってしまい、J2（読み展開）の効果が測れない。
            numeric = cer(to_arabic_numerals(text), to_arabic_numerals(hypothesis))
            # **表記に依存しないCERも残す。** 仮名で入力すると ASR も仮名で
            # 書き戻すので、素のCERは正しい発音を誤りと数える（R-029）。
            reading = reading_cer(text, hypothesis)
            rows.append({
                "subset": subset, "index": index, "text": text,
                "hypothesis": hypothesis, "cer": value,
                "cer_numeric": numeric, "cer_reading": reading,
                "spoken": None if spoken == text else spoken,
                "seconds": waveform.shape[-1] / result.sample_rate, "status": "ok",
            })
            if saved < args.save_samples:
                sf.write(samples_dir / f"{subset}_{index:02d}.wav",
                         waveform.squeeze(0).float().numpy(), result.sample_rate)
                saved += 1
            print(f"  [{subset}/{index:02d}] CER={value*100 if value is not None else -1:5.1f}%")

    summary = summarize_subsets(rows, payload["subsets"])

    write_output(run_dir, args, summary, rows)


def write_output(run_dir, args, summary: dict, rows: list[dict]) -> None:
    """metrics.json を書いて結果を表示する。**shardでも結合でも同じ形にする。**"""
    metrics = {
        "phase": "s0-cer",
        "label": args.label,
        "model_dir": str(args.model_dir),
        "asr_model": ASR_MODEL,
        "eval_set": str(args.eval_set),
        "eval_set_sha256": artifacts.file_checksum(args.eval_set),
        "settings": {"mode": args.mode, "seed": args.seed,
                     "max_decode_length": args.max_decode_length},
        "shard": args.shard, "merged_from": args.merge,
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
            line = (f"  {subset:14s} n={stats['n']:3d}  "
                    f"mean={stats['cer_mean']*100:5.1f}%  "
                    f"median={stats['cer_median']*100:5.1f}%")
            if stats.get("cer_reading_mean") is not None:
                line += f"  読み={stats['cer_reading_mean']*100:5.1f}%"
            print(line)
    print(f"\n完了: {run_dir}")


if __name__ == "__main__":
    main()
