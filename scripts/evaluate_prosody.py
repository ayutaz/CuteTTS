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

"""抑揚を測る（M1）。**CERでは測れないものを測る。**

聴取で残った指摘は読み間違い45% / **抑揚40%** / アクセント30%。
CERはASRの転写を見るので、抑揚が平坦でも転写が合えば誤りにならない。

同一文・同一話者の**人間の実音声**を基準に、次を測る。

* **抑揚の幅**（`semitone_range`）— 「棒読み」は小さい
* **輪郭の一致**（`contour_similarity`）— 人間と同じ上下をしているか

**アクセント核の位置は測れていない。** モーラ単位の強制アラインメントが
要るため。`cutetts.training.prosody.accent_plan` で辞書側の予測は取れるので、
アラインメントを入れれば繋がる。**指摘30%はまだ測れない**ことを明記しておく。

    python scripts/evaluate_prosody.py \\
      --model-dir checkpoints/s1v2-fp32-30000 \\
      --eval-set data/eval/prosody_eval_set.json --label trained --device cuda
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from cutetts import CuteTTS  # noqa: E402
from cutetts.training import artifacts  # noqa: E402
from cutetts.training.prosody import (  # noqa: E402
    contour_similarity,
    measure,
    semitone_contour,
    track_f0,
)
from cutetts.training.reading import expand_kanji_numerals  # noqa: E402


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(str(path), dtype="float64")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sample_rate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抑揚を測る（M1）")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--eval-set", default="data/eval/prosody_eval_set.json")
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decode-length", type=int, default=400)
    parser.add_argument("--expand-numerals", action="store_true",
                        help="J2（漢数字の読み展開）を掛けてから合成する")
    parser.add_argument("--assign-yomi", action="store_true",
                        help="J3（語の読み付与）を掛けてから合成する")
    parser.add_argument("--save-samples", type=int, default=0,
                        help="生成音声を残す件数。**artifacts配下の音声は公開しない**")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("prosody", args.artifact_root,
                                    timestamp=args.timestamp)
    device = resolve_device(args.device)

    payload = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    audio_dir = Path(payload.get("audio_dir", "data/eval/asr_floor"))
    items = payload["items"]
    print(f"{len(items)} 文 / {len({i['speaker'] for i in items})} 話者  device={device}")

    model = CuteTTS.from_pretrained(args.model_dir, device=str(device))
    assigner = None
    if args.assign_yomi:
        from cutetts.training.yomi import ReadingAssigner

        assigner = ReadingAssigner.from_model_dir(args.model_dir)

    samples_dir = run_dir / "samples"
    if args.save_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    saved = 0
    for index, item in enumerate(items):
        text = item["text"]
        spoken = expand_kanji_numerals(text) if args.expand_numerals else text
        if assigner is not None:
            spoken = assigner.apply(spoken)
        reference = audio_dir / item["reference_wav"]
        try:
            result = model.generate(
                spoken, mode="voice_clone", reference_audio=str(reference),
                seed=args.seed, max_decode_length=args.max_decode_length,
                show_progress=False,
            )
        except Exception as error:                 # 失敗も記録して先へ進む
            rows.append({"index": index, "text": text, "status": "error",
                         "detail": f"{type(error).__name__}: {error}"[:200]})
            continue

        model_wave = result.waveform.squeeze(0).float().numpy().astype(np.float64)
        human_wave, human_rate = read_audio(audio_dir / item["human_wav"])
        reference_wave, reference_rate = read_audio(reference)

        model_stats = measure(model_wave, result.sample_rate)
        human_stats = measure(human_wave, human_rate)
        human_contour = semitone_contour(track_f0(human_wave, human_rate))
        similarity = contour_similarity(
            human_contour, semitone_contour(track_f0(model_wave, result.sample_rate)))
        # **床を同じ文ごとに測る。** reference は同一話者の**別の文**なので、
        # 内容を共有しないときの相関になる（全体で中央値 +0.08）。
        # モデルがこれを有意に上回らなければ、抑揚を再現できていない。
        floor = contour_similarity(
            human_contour,
            semitone_contour(track_f0(reference_wave, reference_rate)))
        rows.append({
            "index": index, "text": text, "speaker": item["speaker"],
            "group": item.get("group"), "status": "ok",
            "human": vars(human_stats), "model": vars(model_stats),
            "contour_similarity": None if np.isnan(similarity) else float(similarity),
            "floor_similarity": None if np.isnan(floor) else float(floor),
            "spoken": None if spoken == text else spoken,
        })
        if saved < args.save_samples:
            sf.write(samples_dir / f"{index:03d}.wav",
                     result.waveform.squeeze(0).float().numpy(), result.sample_rate)
            saved += 1
        print(f"  [{index:3d}] 幅 人{human_stats.semitone_range:5.1f} "
              f"/ 模{model_stats.semitone_range:5.1f}  相関 {similarity:5.2f}")

    usable = [r for r in rows if r.get("status") == "ok"
              and r["human"]["semitone_range"] == r["human"]["semitone_range"]
              and r["model"]["semitone_range"] == r["model"]["semitone_range"]]
    similarities = [r["contour_similarity"] for r in usable
                    if r["contour_similarity"] is not None]

    def mean(key: str, side: str) -> float | None:
        values = [r[side][key] for r in usable]
        return statistics.mean(values) if values else None

    summary = {
        "n": len(usable),
        "n_error": sum(1 for r in rows if r.get("status") == "error"),
        "human_semitone_range": mean("semitone_range", "human"),
        "model_semitone_range": mean("semitone_range", "model"),
        "human_semitone_sd": mean("semitone_sd", "human"),
        "model_semitone_sd": mean("semitone_sd", "model"),
        "human_seconds": mean("seconds", "human"),
        "model_seconds": mean("seconds", "model"),
        "contour_similarity_mean": statistics.mean(similarities) if similarities else None,
        "contour_similarity_median": statistics.median(similarities) if similarities else None,
        "n_flatter_than_human": sum(
            1 for r in usable
            if r["model"]["semitone_range"] < r["human"]["semitone_range"]),
    }

    # 床（同一話者・別の文）との対応のある比較。**床を超えていなければ
    # 「抑揚を再現できた」とは言えない。**
    paired = [(r["contour_similarity"], r["floor_similarity"]) for r in usable
              if r["contour_similarity"] is not None
              and r.get("floor_similarity") is not None]
    if paired:
        from cutetts.training.evalstats import paired_compare

        # `difference = mean(b) - mean(a)` なので (床, モデル) の順に渡すと
        # **正が「床を上回る」**になる。相関は大きいほど良いので、
        # `better` / `worse` の数え方だけは逆に読むことになる（ここでは使わない）。
        comparison = paired_compare([f for _, f in paired],
                                    [s for s, _ in paired])
        summary["floor_similarity_mean"] = statistics.mean(f for _, f in paired)
        summary["above_floor"] = {
            "difference": comparison.difference,
            "low": comparison.low,
            "high": comparison.high,
            "significant": comparison.significant,
            "n": comparison.n,
        }

    artifacts.write_run_metadata(
        run_dir, phase="prosody",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"model_dir": args.model_dir, "eval_set": args.eval_set},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "prosody", "label": args.label, "model_dir": str(args.model_dir),
        "eval_set": str(args.eval_set),
        "eval_set_sha256": artifacts.file_checksum(args.eval_set),
        "settings": {"seed": args.seed, "expand_numerals": args.expand_numerals,
                     "assign_yomi": args.assign_yomi},
        "summary": summary, "rows": rows,
    })

    print("\n=== 抑揚 ===")
    print(f"  n={summary['n']}  生成失敗 {summary['n_error']}")
    if not usable:
        print("  **測定できた発話が無い**（有声フレームが足りないか、生成が全滅）")
        print(f"\n完了: {run_dir}")
        return
    print(f"  半音の幅（5〜95%）  人間 {summary['human_semitone_range']:5.2f}  "
          f"→ モデル {summary['model_semitone_range']:5.2f}")
    print(f"  半音sd              人間 {summary['human_semitone_sd']:5.2f}  "
          f"→ モデル {summary['model_semitone_sd']:5.2f}")
    print(f"  長さ(秒)            人間 {summary['human_seconds']:5.2f}  "
          f"→ モデル {summary['model_seconds']:5.2f}")
    if summary["contour_similarity_mean"] is not None:
        print(f"  輪郭の相関          平均 {summary['contour_similarity_mean']:5.2f}  "
              f"中央値 {summary['contour_similarity_median']:5.2f}")
    print(f"  人間より平坦だった文 {summary['n_flatter_than_human']}/{summary['n']}")
    if "above_floor" in summary:
        above = summary["above_floor"]
        print(f"\n  床（同一話者・別の文）平均 {summary['floor_similarity_mean']:+.3f}")
        print(f"  床との差 {above['difference']:+.3f} "
              f"95%CI [{above['low']:+.3f}, {above['high']:+.3f}]  "
              f"{'**有意に上回る**' if above['significant'] else '**床と区別できない**'}")
    print("\n  **アクセント核の位置は測れていない**（強制アラインメントが要る）")
    print(f"\n完了: {run_dir}")


if __name__ == "__main__":
    main()
