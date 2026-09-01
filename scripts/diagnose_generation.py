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

"""自己回帰生成が壊れる条件を切り分ける。

S1では **dev flow loss が改善し続けるのに CER が悪化した**
（35.8% -> 31.2% @2000step -> 51.6% @8000step）。
teacher forcing の1ステップ予測は良くなっているので、壊れているのは
自己回帰そのもの。原因の候補を条件を変えて測る。

測る軸:

``reference_seconds``
    学習時のreferenceは中央値9.4秒だが、CER評価は
    `assets/default_reference.wav`（3.7秒）を使っている。
    この分布ずれが原因なら、長いreferenceでCERが改善する。

``max_decode_length``
    暴走生成が上限に張り付いているだけなのか、短く切れば妥当なのか。

同じ text・同じ seed で、条件だけを変えて生成長とCERを比べる。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import soundfile as sf
import torch

from cutetts import CuteTTS
from cutetts.training import artifacts
from cutetts.training.latents import LatentCacheReader
from cutetts.training.manifest import load_manifest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_japanese_cer import ASR_MODEL, Transcriber, cer  # noqa: E402


def build_reference(tts: CuteTTS, reader, records, *, seconds: float, out_path: Path):
    """cache の latent を連結して指定秒数の reference 音声を作る。

    生音声はライセンス上インスタンスへ置かないので、学習に使ったのと同じ
    latent から復元する（VAEはfreezeなので話者性は保たれる。P1c cos 0.939）。
    """
    frames_needed = int(seconds * 12.5)
    chunks, used = [], []
    for record in records:
        if record.utterance_id not in reader:
            continue
        latent = reader.read(record.utterance_id)
        chunks.append(latent)
        used.append(record.utterance_id)
        if sum(c.shape[0] for c in chunks) >= frames_needed:
            break
    if not chunks:
        raise SystemExit("referenceに使える latent が無い")
    latents = torch.cat([c if isinstance(c, torch.Tensor) else torch.as_tensor(c)
                         for c in chunks], dim=0)[:frames_needed]
    vae = tts.runtime.processor.acoustic_vae
    with torch.no_grad():
        wave = vae.decode(latents.T.unsqueeze(0).float().to(tts.runtime.processor.device))
    sf.write(str(out_path), wave.squeeze().detach().cpu().float().numpy(),
             tts.runtime.sample_rate)
    return used


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="自己回帰生成の破綻条件を切り分ける")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--eval-set", default="data/eval/s0_eval_set.json")
    parser.add_argument("--manifest", default="data/manifests/all_clustered.jsonl")
    parser.add_argument("--latent-cache", default="data/cache/latents")
    parser.add_argument("--subset", default="in_domain")
    parser.add_argument("--limit", type=int, default=15, help="評価する文数")
    parser.add_argument("--reference-seconds", type=float, nargs="+",
                        default=[3.7, 10.0],
                        help="試すreference長。3.7はassets/default_reference.wav相当")
    parser.add_argument("--max-decode-lengths", type=int, nargs="+", default=[400],
                        help="試すmax_decode_length")
    parser.add_argument("--reference-split", default="dev-seen")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="probe")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("s1-genprobe", args.artifact_root,
                                    timestamp=args.timestamp)
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    items = payload["subsets"][args.subset][: args.limit]
    print(f"{args.subset}: {len(items)}文")

    tts = CuteTTS.from_pretrained(args.model_dir, device=args.device)
    transcriber = Transcriber(torch.device(args.device))

    reader = LatentCacheReader(args.latent_cache)
    records = [r for r in load_manifest(args.manifest) if r.split == args.reference_split]

    references: dict[float, Path] = {}
    for seconds in args.reference_seconds:
        path = audio_dir / f"ref_{seconds:.1f}s.wav"
        if abs(seconds - 3.7) < 0.05 and Path("assets/default_reference.wav").is_file():
            # CER評価と同じ既定referenceをそのまま使う
            data, rate = sf.read("assets/default_reference.wav", always_2d=True)
            sf.write(str(path), data.mean(axis=1), rate)
        else:
            build_reference(tts, reader, records, seconds=seconds, out_path=path)
        actual = sf.info(str(path)).duration
        references[seconds] = path
        print(f"  reference {seconds:.1f}s → 実測 {actual:.1f}s")

    rows = []
    for seconds, ref_path in references.items():
        for cap in args.max_decode_lengths:
            values, lengths = [], []
            for index, item in enumerate(items):
                result = tts.generate(item["text"], mode="voice_clone",
                                      reference_audio=str(ref_path), seed=args.seed,
                                      max_decode_length=cap, show_progress=False)
                wave = result.waveform.squeeze().cpu().float()
                duration = wave.shape[-1] / result.sample_rate
                hypothesis = transcriber(wave.unsqueeze(0), result.sample_rate)
                value = cer(item["text"], hypothesis)
                rows.append({"reference_seconds": seconds, "max_decode_length": cap,
                             "index": index, "text": item["text"],
                             "hypothesis": hypothesis, "cer": value,
                             "generated_seconds": duration})
                if value is not None:
                    values.append(value)
                lengths.append(duration)
            print(f"  ref={seconds:4.1f}s cap={cap:4d}  "
                  f"CER mean={statistics.mean(values)*100:5.1f}% "
                  f"median={statistics.median(values)*100:5.1f}%  "
                  f"生成長 median={statistics.median(lengths):4.1f}s "
                  f"max={max(lengths):5.1f}s")

    artifacts.write_run_metadata(run_dir, phase="s1-genprobe",
                                 command=["diagnose_generation.py"], seed=args.seed,
                                 inputs={"model_dir": args.model_dir})
    artifacts.write_metrics(run_dir, {"phase": "s1-genprobe", "label": args.label,
                                      "model_dir": args.model_dir,
                                      "asr_model": ASR_MODEL, "rows": rows})
    print(f"\n完了: {run_dir}")


if __name__ == "__main__":
    main()
