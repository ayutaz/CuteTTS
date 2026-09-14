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

* **アクセント核の位置**（`observed_nucleus`）— `箸` と `橋` を読み分けているか

アクセントは **モーラ単位の強制アラインメント**（`MMS_FA`）で音声とテキストを
対応付けてから測る。基準は2つ置く。

* **辞書**（`pyopenjtalk` の full-context label）との一致率。測定器が信号を
  拾えているかの確認。人間の実音声で **46.1%**（固定回答36.8% / 当てずっぽう25.1%）
* **人間の実音声**との一致率。**こちらが本来見たい値。** モデルが人間と
  同じところで下げているか。辞書が正しいかどうかに依存しない

    python scripts/evaluate_prosody.py \\
      --model-dir checkpoints/s1v2-fp32-30000 \\
      --eval-set data/eval/prosody_eval_set_v2.json --label trained --device cuda
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
from cutetts.training.alignment import MoraAligner, phrase_plan  # noqa: E402
from cutetts.training.prosody import (  # noqa: E402
    contour_similarity,
    measure,
    mora_pitches,
    observed_nucleus,
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


def _rate(hits: int, total: int) -> dict:
    """一致率と分母をまとめる。"""
    return {"rate": (hits / total) if total else None, "n": total}


def _nuclei(aligner, text, waveform, sample_rate):
    """アクセント句ごとの核の位置を読む。失敗したら空リスト。"""
    if aligner is None:
        return []
    phrases = phrase_plan(text)
    spans = aligner.align(waveform, sample_rate, text)
    if len(spans) != sum(len(p.moras) for p in phrases):
        return []
    pitches = mora_pitches(track_f0(waveform, sample_rate), spans)
    out = []
    offset = 0
    for phrase in phrases:
        size = len(phrase.moras)
        out.append(observed_nucleus(pitches[offset:offset + size]))
        offset += size
    return out


def _accent(aligner, text, human_wave, human_rate, model_wave, model_rate):
    """辞書・人間・モデルの核を並べて返す。"""
    if aligner is None:
        return {}
    expected = [p.internal_nucleus for p in phrase_plan(text)]
    human = _nuclei(aligner, text, human_wave, human_rate)
    model = _nuclei(aligner, text, model_wave, model_rate)
    if len(human) != len(expected) or len(model) != len(expected):
        return {"accent": None}
    return {"accent": {"expected": expected, "human": human, "model": model}}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抑揚を測る（M1）")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--eval-set", default="data/eval/prosody_eval_set_v2.json")
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decode-length", type=int, default=400)
    parser.add_argument("--expand-numerals", action="store_true",
                        help="J2（漢数字の読み展開）を掛けてから合成する")
    parser.add_argument("--assign-yomi", action="store_true",
                        help="J3（語の読み付与）を掛けてから合成する")
    parser.add_argument("--no-accent", action="store_true",
                        help="アクセント核の測定を行わない（強制アラインメントを"
                             "省く）。初回は 1.18 GB のモデルを取得する")
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
    # **話者は speaker_key（game×speaker）で数える。** golの話者IDは表示名の
    # SHA-256 なので、game をまたぐと別人でも同じIDになりうる。
    keys = {i.get("speaker_key") or i["speaker"] for i in items}
    print(f"{len(items)} 文 / {len(keys)} 話者  device={device}")

    model = CuteTTS.from_pretrained(args.model_dir, device=str(device))
    assigner = None
    if args.assign_yomi:
        from cutetts.training.yomi import ReadingAssigner

        assigner = ReadingAssigner.from_model_dir(args.model_dir)

    aligner = None
    if not args.no_accent:
        # アラインメントのモデルは初回のみ 1.18 GB を取得する
        aligner = MoraAligner(device=str(device))

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
        # 内容を共有しないときの相関になる（240文で平均 -0.001 / sd 0.152。
        # **真の床はほぼゼロ**。n=67 のときの +0.070 はノイズだった）。
        # モデルがこれを有意に上回らなければ、抑揚を再現できていない。
        floor = contour_similarity(
            human_contour,
            semitone_contour(track_f0(reference_wave, reference_rate)))
        accent = _accent(aligner, text, human_wave, human_rate,
                         model_wave, result.sample_rate)
        rows.append({
            "index": index, "text": text, "speaker": item["speaker"],
            "speaker_key": item.get("speaker_key") or item["speaker"],
            "group": item.get("group"), "status": "ok",
            "human": vars(human_stats), "model": vars(model_stats),
            "contour_similarity": None if np.isnan(similarity) else float(similarity),
            "floor_similarity": None if np.isnan(floor) else float(floor),
            "spoken": None if spoken == text else spoken,
            **accent,
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

    # --- アクセント核 ---
    phrases = [r["accent"] for r in rows if r.get("accent")]
    if phrases:
        def agree(left_key: str, right_key: str) -> tuple[int, int]:
            hits = total = 0
            for entry in phrases:
                for left, right in zip(entry[left_key], entry[right_key]):
                    if left < 0 or right < 0:   # 判定できなかった句は除く
                        continue
                    total += 1
                    hits += left == right
            return hits, total

        counts: dict[int, int] = {}
        for entry in phrases:
            for value in entry["human"]:
                if value >= 0:
                    counts[value] = counts.get(value, 0) + 1
        marginal = sum(counts.values())
        summary["accent"] = {
            "n_utterances": len(phrases),
            "n_phrases": sum(len(e["expected"]) for e in phrases),
            # **本来見たいのはこれ。** 辞書が正しいかに依存しない
            "model_vs_human": _rate(*agree("human", "model")),
            "human_vs_dictionary": _rate(*agree("expected", "human")),
            "model_vs_dictionary": _rate(*agree("expected", "model")),
            # 人間の核の分布から計算した当てずっぽうの水準
            "chance": (sum((c / marginal) ** 2 for c in counts.values())
                       if marginal else None),
            "constant": (max(counts.values()) / marginal) if marginal else None,
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
    accent = summary.get("accent")
    if accent:
        print("\n=== アクセント核 ===")
        print(f"  {accent['n_utterances']} 発話 / {accent['n_phrases']} アクセント句")
        for key, label in (("model_vs_human", "**モデル 対 人間**"),
                           ("human_vs_dictionary", "人間 対 辞書"),
                           ("model_vs_dictionary", "モデル 対 辞書")):
            entry = accent[key]
            if entry["rate"] is None:
                continue
            print(f"  {label:18s} {entry['rate']:6.1%}  (n={entry['n']})")
        print(f"  当てずっぽう {accent['chance']:6.1%}   "
              f"固定回答 {accent['constant']:6.1%}")
    else:
        print("\n  アクセント核は測っていない（--no-accent）")
    print(f"\n完了: {run_dir}")


if __name__ == "__main__":
    main()
