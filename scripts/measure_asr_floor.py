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

"""ASR の誤り床を測る。**人間の実音声**を文字起こしして正解textと比べる。

これが答えるのは2つ:

1. **CER指標の床。** S0の学習後CERは 28.4% だが、これを「0%が理想」として
   読んでよいかは分からない。人間の実音声ですらASRが15%誤るなら、
   28.4% の意味は変わる。TTS由来の誤りとASR由来の誤りを分離する。
2. **ASR文字起こしを学習ラベルに使えるか。** moe-speech-plus のtextは
   ASR出力なので、ASRが誤る分だけ学習ラベルが汚れる。
   特に **数字を含む文** で誤り方が違うかを分けて測る。

gol-dataset を使う。textがゲームスクリプト（正解）で、音声と1:1に対応するため。

`scripts/evaluate_japanese_cer.py` と **同じASR・同じ正規化・同じCER実装**を
再利用する。両者の数値を直接比較できないと意味がないため。

    # 標本を作る（CPU、tarから抽出）
    python scripts/measure_asr_floor.py --build --tar-dir data/raw/gol/tars

    # 測る（GPU）
    python scripts/measure_asr_floor.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import statistics
import sys
import tarfile
from pathlib import Path

import soundfile as sf
import torch

from cutetts.training import artifacts

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_japanese_cer import ASR_MODEL, Transcriber, cer  # noqa: E402

DIGIT = re.compile(r"[0-9０-９]")
SAMPLE_DIR = "data/eval/asr_floor"


def has_lexical_content(text: str) -> bool:
    """`build_eval_set.py` と同じ基準。語彙的内容の無い発話を除く。"""
    stripped = re.sub(r"[\s、。「」『』・…‥！？!?,.\-―ー~〜\"'()（）]", "", text)
    if len(stripped) < 8:
        return False
    if re.search(r"(.)\1{3,}", stripped):
        return False
    if len(set(stripped)) / len(stripped) < 0.45:
        return False
    return bool(re.search(r"[ァ-ヶ一-龥]", stripped))


def build_sample(*, metadata: Path, tar_dir: Path, out_dir: Path,
                 per_group: int, seed: int) -> list[dict]:
    """手元のtarにあるゲームから、通常文と数字を含む文を同数ずつ選ぶ。"""
    available = {p.stem: p for p in sorted(tar_dir.glob("*.tar"))}
    if not available:
        raise SystemExit(f"tarが無い: {tar_dir}")
    print(f"手元のtar: {len(available)}本")

    plain, digits = [], []
    csv.field_size_limit(10**7)
    with metadata.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["game_id"] not in available:
                continue
            text = (row["text"] or "").strip()
            if not has_lexical_content(text):
                continue
            (digits if DIGIT.search(text) else plain).append(row)
    print(f"候補: 通常 {len(plain):,} / 数字入り {len(digits):,}")
    if len(digits) < per_group:
        print(f"警告: 数字入りが {len(digits)} 件しかない（要求 {per_group}）")

    rng = random.Random(seed)
    chosen = ([("plain", r) for r in rng.sample(plain, min(per_group, len(plain)))]
              + [("digit", r) for r in rng.sample(digits, min(per_group, len(digits)))])

    wanted: dict[str, list[tuple[str, dict]]] = {}
    for group, row in chosen:
        wanted.setdefault(row["game_id"], []).append((group, row))

    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for game_id, entries in wanted.items():
        members = {row["file_path"]: (group, row) for group, row in entries}
        with tarfile.open(available[game_id]) as tar:
            for member in tar:
                key = next((k for k in members if member.name.endswith(k.split("/")[-1])), None)
                if key is None:
                    continue
                group, row = members.pop(key)
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                data, rate = sf.read(io.BytesIO(extracted.read()), dtype="float32",
                                     always_2d=True)
                name = f"{game_id[:8]}_{Path(row['file_path']).stem}.wav"
                sf.write(str(out_dir / name), data.mean(axis=1), rate)
                items.append({"group": group, "game_id": game_id, "wav": name,
                              "text": row["text"].strip(), "sample_rate": int(rate),
                              "seconds": len(data) / rate})
                if not members:
                    break
    (out_dir / "items.json").write_text(
        json.dumps({"seed": seed, "items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"標本を書き出した: {out_dir}（{len(items)}件）")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASRの誤り床を測る")
    parser.add_argument("--build", action="store_true", help="tarから標本を作る（CPU）")
    parser.add_argument("--metadata", default="data/raw/gol/metadata.tsv")
    parser.add_argument("--tar-dir", default="data/raw/gol/tars")
    parser.add_argument("--sample-dir", default=SAMPLE_DIR)
    parser.add_argument("--per-group", type=int, default=40,
                        help="通常文・数字入りそれぞれの件数")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sample_dir = Path(args.sample_dir)

    if args.build:
        build_sample(metadata=Path(args.metadata), tar_dir=Path(args.tar_dir),
                     out_dir=sample_dir, per_group=args.per_group, seed=args.seed)
        return

    payload = json.loads((sample_dir / "items.json").read_text(encoding="utf-8"))
    items = payload["items"]
    print(f"標本 {len(items)}件を測る（ASR={ASR_MODEL}）")

    run_dir = artifacts.new_run_dir("asr-floor", args.artifact_root,
                                    timestamp=args.timestamp)
    transcriber = Transcriber(torch.device(args.device))

    rows = []
    for index, item in enumerate(items):
        data, rate = sf.read(str(sample_dir / item["wav"]), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
        hypothesis = transcriber(waveform, rate)
        value = cer(item["text"], hypothesis)
        rows.append({**{k: item[k] for k in ("group", "game_id", "wav", "text", "seconds")},
                     "hypothesis": hypothesis, "cer": value})
        if index % 20 == 0:
            print(f"  {index}/{len(items)}")

    summary = {}
    for group in ("plain", "digit"):
        values = [r["cer"] for r in rows if r["group"] == group and r["cer"] is not None]
        if not values:
            continue
        ordered = sorted(values)
        summary[group] = {
            "n": len(values),
            "cer_mean": statistics.mean(values),
            "cer_median": statistics.median(values),
            "cer_p90": ordered[int(len(ordered) * 0.9) - 1] if len(ordered) >= 10 else None,
        }

    artifacts.write_run_metadata(run_dir, phase="asr-floor",
                                 command=["measure_asr_floor.py"], seed=args.seed,
                                 inputs={"sample_dir": str(sample_dir)})
    artifacts.write_metrics(run_dir, {"phase": "asr-floor", "asr_model": ASR_MODEL,
                                      "summary": summary, "rows": rows})

    print("\n=== ASRの誤り床（人間の実音声 vs ゲームスクリプト） ===")
    for group, stats in summary.items():
        label = "通常文" if group == "plain" else "数字入り"
        p90 = f"{stats['cer_p90']*100:5.1f}%" if stats["cer_p90"] is not None else "  n/a"
        print(f"  {label:8s} n={stats['n']:3d}  mean={stats['cer_mean']*100:5.1f}%  "
              f"median={stats['cer_median']*100:5.1f}%  p90={p90}")
    print(f"\n完了: {run_dir}")


if __name__ == "__main__":
    main()
