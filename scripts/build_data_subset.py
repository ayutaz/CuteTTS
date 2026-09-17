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

"""学習manifestから指定時間の部分集合を作る（D1）。

**データ量だけを変えるための道具。** 2026-09-02 のデータ量比較と同じ規約で作る。

* **クラスタ単位で無作為に選ぶ。** 大きいクラスタから採ると1話者あたりの
  学習量（密度）が上がり、**量と密度が交絡する**（R-017 / R-018）。
* **`split` が `train` の record だけを絞る。** `dev-seen` / `dev-zero-shot` は
  そのまま残すので、**dev の比較は同条件**のままになる。
* **latent cache は作り直さない。** 部分集合は manifest を絞るだけなので、
  既存の cache をそのまま使える（前処理が要らない）。

D1 で答えたい問いは「**30,000 step でもデータ量は弱いのか**」。
以前の比較は 3,000 step で、17h はほぼ1 epoch・325.9h は1 epochの5%しか
見ていなかった（**大きいデータ側に不利な条件**）。

    python scripts/build_data_subset.py \\
      --manifest data/s1v2/manifests-v2/all_clustered.jsonl \\
      --target-hours 17 --out data/s1v2/manifests-v2/subset_17h.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutetts.training import artifacts  # noqa: E402
from cutetts.training.manifest import (  # noqa: E402
    Utterance,
    load_manifest,
    summarize,
    write_manifest,
)

#: 学習に使う split の名前。これ以外の record は**絞らずに全部残す**。
TRAIN_SPLIT = "train"


def cluster_key(record: Utterance) -> str:
    """クラスタの識別子。`voice_cluster_id` が無ければ話者IDで代用する。"""
    return record.voice_cluster_id or record.split_group_id or record.speaker_id


def select_clusters(records, target_hours: float, seed: int) -> tuple[set[str], float]:
    """クラスタ単位で無作為に選び、`target_hours` に届くまで採る。

    **順序は seed とクラスタIDだけで決まる**（入力の並び順に依存しない）ので、
    同じ manifest と同じ seed なら必ず同じ部分集合になる。

    Returns:
        選んだクラスタID集合と、その合計時間（時間）。
    """
    seconds: dict[str, float] = defaultdict(float)
    for record in records:
        seconds[cluster_key(record)] += float(record.duration)

    def order(key: str) -> str:
        return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()

    chosen: set[str] = set()
    total = 0.0
    for key in sorted(seconds, key=order):
        if total >= target_hours * 3600.0:
            break
        chosen.add(key)
        total += seconds[key]
    return chosen, total / 3600.0


def subset_records(records: list[Utterance], target_hours: float, seed: int):
    """`train` だけを絞った record 列を、**元の並び順のまま**返す。"""
    train = [r for r in records if r.split == TRAIN_SPLIT]
    if not train:
        raise SystemExit("split=train の record が無い")
    chosen, hours = select_clusters(train, target_hours, seed)
    kept = [r for r in records
            if r.split != TRAIN_SPLIT or cluster_key(r) in chosen]
    return kept, chosen, hours


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="学習manifestの部分集合を作る（D1）")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-hours", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("data-subset", args.artifact_root,
                                    timestamp=args.timestamp)

    records = list(load_manifest(args.manifest))
    kept, chosen, hours = subset_records(records, args.target_hours, args.seed)

    before = summarize(records)
    after = summarize(kept)
    written = write_manifest(args.out, kept)
    print(f"元: {before['total']:,} 発話 / {before['hours']:.1f} h "
          f"/ cluster {before['voice_clusters']:,}")
    print(f"後: {after['total']:,} 発話 / {after['hours']:.1f} h "
          f"/ cluster {after['voice_clusters']:,}")
    print(f"  train を {len(chosen):,} cluster に絞った（{hours:.2f} h）")
    print(f"  split別: {after['by_split']}")
    print(f"{written:,} 行を書いた: {args.out}")

    artifacts.write_run_metadata(
        run_dir, phase="data-subset",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"manifest": args.manifest},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "data-subset", "target_hours": args.target_hours,
        "seed": args.seed, "out": str(args.out),
        "train_clusters": len(chosen), "train_hours": round(hours, 4),
        "before": before, "after": after,
        "sha256": artifacts.file_checksum(args.out),
    })
    print(f"完了: {run_dir}  sha256 {artifacts.file_checksum(args.out)[:16]}...")


if __name__ == "__main__":
    main()
