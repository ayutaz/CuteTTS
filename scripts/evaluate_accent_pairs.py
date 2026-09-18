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

"""アクセント最小対でモデルの核の実現を測る（A1）。

`scripts/build_accent_pair_set.py` が作った対は、**片仮名が完全に同一で
核の位置だけが違う**。

    その神を見ていました。 → ソノカ'ミオミテイマシタ。   核 1
    その髪を見ていました。 → ソノカミ'オミテイマシタ。   核 2

**モデルの音声がこの区別を作れているか**を測る。frontend ごとに条件が変わる。

* ``none``   入力は漢字。**モデルが語を知っていれば**区別できる
* ``accent`` 入力に核記号がある。**記号に従えば**区別できる

主指標は3つ。

``nucleus_match``
    辞書の核と一致した変種の割合。
``pair_differentiated``
    対の2文で**実現した核が違った**割合。区別できていないと 0 に落ちる。
``pair_correct``
    2文とも辞書どおりだった割合。**これが本命。**

核の読み取りは1回では揺れるので、**seed を変えて複数回引いて多数決**を取る
（`--repeats`）。

    python scripts/evaluate_accent_pairs.py --model-dir checkpoints/m4a-accent/inference \\
        --label a1-accent --frontend accent --device cuda
"""

from __future__ import annotations

import argparse
import collections
import json
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
    mora_pitches,
    observed_nucleus,
    track_f0,
)
from cutetts.training.yomi import FRONTEND_MODES, frontend_text  # noqa: E402

#: 判定できなかったことを表す値（`observed_nucleus` と同じ規約）。
UNDETERMINED = -1


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_reference() -> str | None:
    """既定の reference。**対の両側で同じ声にする**ために1つに固定する。"""
    root = artifacts.as_local_path("data/eval/prosody_audio")
    if not Path(root).is_dir():
        return None
    files = sorted(Path(root).glob("*.wav"))
    return str(files[0]) if files else None


def phrase_span(text: str, index: int) -> tuple[int, int]:
    """`index` 番目のアクセント句が占めるモーラ範囲（0始まり・半開区間）。"""
    plan = phrase_plan(text)
    offset = 0
    for position, phrase in enumerate(plan):
        size = len(phrase.moras)
        if position == index:
            return offset, offset + size
        offset += size
    raise IndexError(f"句 {index} が無い（{len(plan)} 句）")


def measure_nucleus(aligner, text: str, waveform, sample_rate: int,
                    index: int) -> int:
    """対象のアクセント句で実現した核の位置。読めなければ ``UNDETERMINED``。

    アラインメントには**原文**を渡す（frontend を掛けた文ではない）。
    対の両側で仮名が同一なので、モーラ列も同一になる。
    """
    try:
        spans = aligner.align(waveform, sample_rate, text)
    except Exception:
        return UNDETERMINED
    plan = phrase_plan(text)
    if len(spans) != sum(len(p.moras) for p in plan):
        return UNDETERMINED               # アラインメントが合わない
    low, high = phrase_span(text, index)
    f0 = track_f0(waveform, sample_rate)
    pitches = mora_pitches(f0, spans)
    return observed_nucleus(pitches[low:high])


def majority(values: list[int]) -> int:
    """多数決。判定できた値だけで取る。同数なら小さい方。"""
    usable = [value for value in values if value != UNDETERMINED]
    if not usable:
        return UNDETERMINED
    counts = collections.Counter(usable)
    best = max(counts.values())
    return min(value for value, count in counts.items() if count == best)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="アクセント最小対を測る（A1）")
    parser.add_argument("--model-dir")
    parser.add_argument("--eval-set", default="data/eval/accent_pair_set_v1.json")
    parser.add_argument("--label", required=True)
    parser.add_argument("--frontend", choices=FRONTEND_MODES, default="none",
                        help="入力テキストの作り方。**学習と揃える**")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3,
                        help="1文あたりの生成回数（seedを変えて多数決）")
    parser.add_argument("--max-decode-length", type=int, default=400)
    parser.add_argument("--reference-audio", default=None,
                        help="既定は prosody_audio の先頭。**対の両側で同じ声**")
    parser.add_argument("--save-samples", type=int, default=0,
                        help="生成音声を残す件数。**artifacts配下の音声は公開しない**")
    parser.add_argument("--shard", metavar="K/N",
                        help="対を N 等分して K 番目だけ測る（1始まり）")
    parser.add_argument("--merge", metavar="PATHS",
                        help="shard の metrics.json をカンマ区切りで結合する")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def parse_shard(text: str, total: int) -> list[int]:
    part, count = (int(value) for value in text.split("/"))
    if not 1 <= part <= count:
        raise SystemExit(f"--shard が不正: {text}")
    return [index for index in range(total) if index % count == part - 1]


def merge_rows(paths: str) -> list[dict]:
    rows: list[dict] = []
    for path in paths.split(","):
        payload = json.loads(Path(path.strip()).read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    seen: dict[str, dict] = {}
    for row in rows:
        seen[f"{row['pair_id']}:{row['surface']}"] = row
    return list(seen.values())


def summarize(rows: list[dict]) -> dict:
    """変種単位と対単位の両方で集計する。"""
    determined = [row for row in rows if row["observed"] != UNDETERMINED]
    match = sum(1 for row in determined if row["observed"] == row["expected"])

    by_pair: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)

    pairs = differentiated = correct = incomplete = 0
    per_pair: list[dict] = []
    for pair_id, group in by_pair.items():
        if len(group) != 2 or any(r["observed"] == UNDETERMINED for r in group):
            incomplete += 1
            continue
        pairs += 1
        left, right = group
        is_diff = left["observed"] != right["observed"]
        is_correct = (left["observed"] == left["expected"]
                      and right["observed"] == right["expected"])
        differentiated += int(is_diff)
        correct += int(is_correct)
        per_pair.append({"pair_id": pair_id, "differentiated": int(is_diff),
                         "correct": int(is_correct),
                         "kana": left.get("kana", "")})

    def rate(hits: int, total: int) -> float | None:
        return hits / total if total else None

    return {
        "variants": len(rows),
        "determined": len(determined),
        "undetermined": len(rows) - len(determined),
        "nucleus_match": rate(match, len(determined)),
        "pairs": pairs,
        "pairs_incomplete": incomplete,
        "pair_differentiated": rate(differentiated, pairs),
        "pair_correct": rate(correct, pairs),
        "per_pair": per_pair,
    }


def write_metrics(run_dir, args, summary: dict, rows: list[dict]) -> None:
    """metrics.json を書く。**shardでも結合でも同じ形にする。**"""
    artifacts.write_run_metadata(
        run_dir, phase="a1-accent-pairs",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"model_dir": args.model_dir, "eval_set": args.eval_set},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "a1-accent-pairs",
        "label": args.label,
        "model_dir": str(args.model_dir),
        "frontend": args.frontend,
        "eval_set": str(args.eval_set),
        "eval_set_sha256": artifacts.file_checksum(args.eval_set),
        "settings": {"seed": args.seed, "repeats": args.repeats,
                     "reference_audio": args.reference_audio},
        "shard": args.shard, "merged_from": args.merge,
        "summary": summary,
        "rows": rows,
    })


def report(summary: dict, run_dir) -> None:
    def percent(value: float | None) -> str:
        return "—" if value is None else f"{value:.1%}"

    print()
    print(f"  変種 {summary['variants']} / 判定できた {summary['determined']}"
          f"（読めなかった {summary['undetermined']}）")
    print(f"  辞書の核と一致            {percent(summary['nucleus_match'])}")
    print(f"  対 {summary['pairs']} 組（片方が読めず外した {summary['pairs_incomplete']} 組）")
    print(f"  **対の2文で核が違った**   {percent(summary['pair_differentiated'])}")
    print(f"  **2文とも辞書どおり**     {percent(summary['pair_correct'])}")
    print()
    print("  区別できていなければ `核が違った` が 0 に落ちる。")
    print(f"  {run_dir}")


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("a1-accent-pairs", args.artifact_root,
                                    timestamp=args.timestamp)

    payload = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    items = payload["items"]

    if args.merge:
        rows = merge_rows(args.merge)
        summary = summarize(rows)
        print(f"{len(rows)} 変種を結合（shard {len(args.merge.split(','))} 個）")
        write_metrics(run_dir, args, summary, rows)
        report(summary, run_dir)
        return

    if args.model_dir is None:
        raise SystemExit("--model-dir が要る（--merge のときは不要）")

    device = resolve_device(args.device)
    indices = (parse_shard(args.shard, len(items)) if args.shard
               else list(range(len(items))))
    reference = args.reference_audio or default_reference()
    mode = "voice_clone" if reference else "tts"
    shard_note = f"  shard {args.shard}" if args.shard else ""
    print(f"{len(indices)}/{len(items)} 対  frontend={args.frontend}  "
          f"mode={mode}  repeats={args.repeats}  device={device}{shard_note}")

    model = CuteTTS.from_pretrained(args.model_dir, device=str(device))
    assigner = None
    if args.frontend == "yomi":
        from cutetts.training.yomi import ReadingAssigner

        assigner = ReadingAssigner.from_model_dir(args.model_dir)
    aligner = MoraAligner(device=str(device))

    samples_dir = run_dir / "samples"
    if args.save_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)

    # **プロセス内の最初の生成だけ結果が違う**（`evaluate_prosody.py` と同じ）
    first = items[indices[0]]["variants"][0]["text"]
    kwargs = {"reference_audio": reference} if reference else {}
    model.generate(first, mode=mode, seed=args.seed,
                   max_decode_length=args.max_decode_length,
                   show_progress=False, **kwargs)

    rows: list[dict] = []
    saved = 0
    for index in indices:
        item = items[index]
        for variant in item["variants"]:
            text = variant["text"]
            spoken = frontend_text(text, args.frontend, assigner=assigner)
            observations: list[int] = []
            for repeat in range(args.repeats):
                try:
                    result = model.generate(
                        spoken, mode=mode, seed=args.seed + repeat,
                        max_decode_length=args.max_decode_length,
                        show_progress=False, **kwargs)
                except Exception as error:
                    print(f"  生成に失敗: {text} / {error}")
                    observations.append(UNDETERMINED)
                    continue
                # **`.audio` ではない。** `GenerationResult` は `.waveform`
                # （torch tensor）と `.sample_rate` を持つ
                waveform = result.waveform.squeeze(0).float().cpu().numpy(
                    ).astype(np.float64)
                observations.append(measure_nucleus(
                    aligner, text, waveform, result.sample_rate,
                    item["phrase_index"]))
                if saved < args.save_samples:
                    sf.write(str(samples_dir / f"{item['pair_id']}-"
                                 f"{variant['surface']}-{repeat}.wav"),
                             waveform, result.sample_rate)   # 公開しない
                    saved += 1
            rows.append({
                "pair_id": item["pair_id"],
                "kana": item["kana"],
                "surface": variant["surface"],
                "text": text,
                "spoken": spoken,
                "phrase_index": item["phrase_index"],
                "expected": variant["expected_nucleus"],
                "observed": majority(observations),
                "observations": observations,
            })
        done = len(rows) // 2
        if done % 10 == 0:
            print(f"  {done}/{len(indices)} 対")

    summary = summarize(rows)
    write_metrics(run_dir, args, summary, rows)
    report(summary, run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
