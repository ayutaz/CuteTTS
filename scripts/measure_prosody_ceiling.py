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

"""抑揚・アクセントの**天井**を測る（M2）。**生成しない。**

`evaluate_prosody.py` が「人間 対 モデル」を測るのに対し、こちらは
**人間 対 人間（同じ台詞の別テイク）** を測る。行の形も集計も
`evaluate_prosody.py` と同じにしてあるので、**そのまま比較できる**
（テイクAが `human`、テイクBが `model` の位置に入る）。

* 輪郭の相関: テイクA 対 テイクB
* 床: テイクA 対 **同一話者の別の文**（`evaluate_prosody.py` と同じ定義）
* アクセント核: 辞書 / テイクA / テイクB

**出るのは天井の下限。** 別テイクは「同じ読み方をしようとした2回」ではなく、
感情や文脈が違いうる。

    python scripts/measure_prosody_ceiling.py --label ceiling --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from cutetts.training import artifacts  # noqa: E402
from cutetts.training.alignment import MoraAligner, phrase_plan  # noqa: E402
from cutetts.training.prosody import (  # noqa: E402
    contour_similarity,
    measure,
    semitone_contour,
    summarize_run,
    track_f0,
)

# 生成をしないので、評価scriptの読み込み系だけを使い回す
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_prosody import (  # noqa: E402
    _accent,
    _cached_f0,
    merge_rows,
    parse_shard,
    read_audio,
    report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抑揚・アクセントの天井を測る（M2）")
    parser.add_argument("--eval-set", default="data/eval/prosody_ceiling_set_v1.json")
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-accent", action="store_true",
                        help="アクセント核の測定を行わない（強制アラインメントを省く）")
    parser.add_argument("--shard", metavar="K/N",
                        help="setを N 分割して K 番目だけを測る（1始まり）")
    parser.add_argument("--merge", metavar="PATHS",
                        help="shardの metrics.json をカンマ区切りで渡すと、"
                             "行を結合して集計だけ作る")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def write_metrics(run_dir, args, summary: dict, rows: list[dict]) -> None:
    artifacts.write_run_metadata(
        run_dir, phase="prosody",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"eval_set": args.eval_set},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "prosody", "label": args.label, "model_dir": "human-retake",
        "eval_set": str(args.eval_set),
        "eval_set_sha256": artifacts.file_checksum(args.eval_set),
        "settings": {"seed": args.seed, "kind": "ceiling"},
        "shard": args.shard, "merged_from": args.merge,
        "summary": summary, "rows": rows,
    })


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("prosody", args.artifact_root,
                                    timestamp=args.timestamp)

    payload = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    audio_dir = artifacts.as_local_path(
        payload.get("audio_dir", "data/eval/prosody_ceiling_audio"))
    items = payload["items"]

    if args.merge:
        rows = merge_rows(args.merge)
        summary = summarize_run(rows)
        print(f"{len(rows)} 行を結合（shard {len(args.merge.split(','))} 個）")
        write_metrics(run_dir, args, summary, rows)
        report(summary, run_dir)
        return

    device = resolve_device(args.device)
    indices = (parse_shard(args.shard, len(items)) if args.shard
               else list(range(len(items))))
    keys = {items[i].get("speaker_key") or items[i]["speaker"] for i in indices}
    print(f"{len(indices)}/{len(items)} 組 / {len(keys)} 話者  device={device}")

    aligner = MoraAligner(device=device) if not args.no_accent else None

    rows: list[dict] = []
    f0_cache: dict[str, np.ndarray] = {}
    for index in indices:
        item = items[index]
        text = item["text"]
        try:
            take_a, rate_a = read_audio(audio_dir / item["human_wav"])
            take_b, rate_b = read_audio(audio_dir / item["take_b_wav"])
            floor_wave, floor_rate = read_audio(audio_dir / item["reference_wav"])
        except Exception as error:
            rows.append({"index": index, "text": text, "status": "error",
                         "detail": f"{type(error).__name__}: {error}"[:200]})
            continue

        f0_a = _cached_f0(f0_cache, item["human_wav"], take_a, rate_a)
        f0_b = _cached_f0(f0_cache, item["take_b_wav"], take_b, rate_b)
        f0_floor = _cached_f0(f0_cache, item["reference_wav"], floor_wave, floor_rate)

        stats_a = measure(take_a, rate_a, f0=f0_a)
        stats_b = measure(take_b, rate_b, f0=f0_b)
        contour_a = semitone_contour(f0_a)
        similarity = contour_similarity(contour_a, semitone_contour(f0_b))
        floor = contour_similarity(contour_a, semitone_contour(f0_floor))
        accent = _accent(aligner, text, take_a, rate_a, f0_a,
                         take_b, rate_b, f0_b)
        rows.append({
            "index": index, "text": text, "speaker": item["speaker"],
            "speaker_key": item.get("speaker_key") or item["speaker"],
            "status": "ok",
            "human": vars(stats_a), "model": vars(stats_b),
            "contour_similarity": None if np.isnan(similarity) else float(similarity),
            "floor_similarity": None if np.isnan(floor) else float(floor),
            **accent,
        })
        print(f"  [{index:3d}] 幅 A{stats_a.semitone_range:5.1f} "
              f"/ B{stats_b.semitone_range:5.1f}  相関 {similarity:5.2f}")

    summary = summarize_run(rows)
    write_metrics(run_dir, args, summary, rows)
    report(summary, run_dir)
    print("\n**これは天井の下限**（別テイクは感情や文脈が違いうる）。")


if __name__ == "__main__":
    main()
