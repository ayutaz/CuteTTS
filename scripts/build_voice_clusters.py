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

"""P1d の中核: speaker embedding から voice クラスタを作り、manifest へ付与する。

両datasetの speaker ID は声の識別子ではない（R-004）。

* gol-dataset: ``SHA-256(キャラクター表示名)[:32]``。同名異キャラが同一IDへ統合され、
  総称ラベルは複数の声が1 IDに混在する
* moe-speech-plus: ``uuid4().hex[:8]``。同一声優でも別IDとREADMEに明記

そのため speaker-disjoint split は voice-actor-disjoint を保証しない。
frozen の公式 Speaker Encoder が作った 256 次元 embedding をクラスタリングし、
**split単位をIDからvoiceクラスタへ移す**（D-015）。
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter
from pathlib import Path

from cutetts.training import artifacts, text_rules
from cutetts.training.manifest import load_manifest, manifest_checksum, summarize, write_manifest
from cutetts.training.speaker_cache import SpeakerEmbeddingCacheReader
from cutetts.training.voice_clusters import (
    DEFAULT_MAX_ZERO_SHOT_SHARE,
    DEFAULT_SEEN_FRACTION,
    DEFAULT_ZERO_SHOT_FRACTION,
    assign_clusters,
    assign_splits_by_cluster,
    build_profiles,
    cluster_members,
    cluster_speakers,
    cluster_summary,
    split_leakage,
    suspicious_speakers,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P1d: voiceクラスタの構築とmanifestへの付与")
    parser.add_argument("--manifest", default="data/manifests/all.jsonl")
    parser.add_argument("--speaker-cache", default="data/cache/speaker")
    parser.add_argument("--out", default="data/manifests/all_clustered.jsonl")
    parser.add_argument("--zero-shot-fraction", type=float, default=DEFAULT_ZERO_SHOT_FRACTION,
                        help="zero-shotへ回すクラスタの割合（dev/testで折半）")
    parser.add_argument("--seen-fraction", type=float, default=DEFAULT_SEEN_FRACTION,
                        help="trainクラスタ内でseen評価へ回す発話の割合")
    parser.add_argument("--max-zero-shot-share", type=float, default=DEFAULT_MAX_ZERO_SHOT_SHARE,
                        help="1クラスタがzero-shotへ持ち込める発話数の上限（全体比）")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--dispersion-threshold", type=float, default=0.35)
    parser.add_argument("--max-samples-per-speaker", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("p1d-clusters", args.artifact_root, timestamp=args.timestamp)
    records = list(load_manifest(args.manifest))
    print(f"manifest: {len(records):,} 発話")

    with SpeakerEmbeddingCacheReader(args.speaker_cache) as reader:
        print(f"speaker cache: {len(reader):,} embedding")
        profiles = build_profiles(
            reader, records,
            max_samples_per_speaker=args.max_samples_per_speaker, seed=args.seed,
        )
        print(f"profiles: {len(profiles):,} 話者")

        mapping = cluster_speakers(profiles, threshold=args.threshold)
        summary = cluster_summary(mapping)
        members = cluster_members(mapping)
        suspicious = suspicious_speakers(profiles, dispersion_threshold=args.dispersion_threshold)

        clustered = list(assign_clusters(records, mapping))

        # split を **クラスタ単位で切り直す**（D-015）。
        # manifest 段階の split は speaker_id 単位の暫定値で、
        # 「別IDの同じ声」が train と zero-shot へ分かれて漏れる。
        cluster_ids = [r.voice_cluster_id for r in clustered]
        new_splits = assign_splits_by_cluster(
            cluster_ids, seed=args.seed,
            zero_shot_fraction=args.zero_shot_fraction,
            seen_fraction=args.seen_fraction,
            max_zero_shot_share=args.max_zero_shot_share)
        before = Counter(r.split for r in clustered)
        clustered = [dataclasses.replace(r, split=s)
                     for r, s in zip(clustered, new_splits)]
        after = Counter(r.split for r in clustered)

        leaked = split_leakage(cluster_ids, new_splits)
        if leaked:
            raise SystemExit(
                f"クラスタ単位のsplitに漏れがある: {len(leaked)} 件 "
                f"（例 {list(leaked.items())[:3]}）")
        print("split を voice cluster 単位へ切り直した（漏れ 0 件）")
        for name in sorted(set(before) | set(after)):
            print(f"  {name:16s} {before.get(name, 0):8,} -> {after.get(name, 0):8,}")

        written = write_manifest(args.out, clustered)

    # 総称ラベル（既知）と dispersion 検出の突き合わせ
    generic_ids = text_rules.generic_speaker_ids()
    known_generic = [s for s in profiles if s in generic_ids]
    detected_generic = [s for s in suspicious if s in generic_ids]

    merged = {cid: ids for cid, ids in members.items() if len(ids) > 1}
    metrics = {
        "phase": "p1d-clusters",
        "threshold": args.threshold,
        "dispersion_threshold": args.dispersion_threshold,
        "speakers_profiled": len(profiles),
        "cluster_summary": summary,
        "merged_clusters": len(merged),
        "merged_examples": {cid: ids for cid, ids in list(merged.items())[:20]},
        "suspicious_speakers": len(suspicious),
        "known_generic_in_profiles": len(known_generic),
        "known_generic_flagged_by_dispersion": len(detected_generic),
        "dispersion_stats": {
            "max": max((p.dispersion for p in profiles.values()), default=None),
            "mean": (sum(p.dispersion for p in profiles.values()) / len(profiles)
                     if profiles else None),
        },
        "output": {
            "path": args.out, "records": written, "sha256": manifest_checksum(args.out),
        },
        "clustered_summary": summarize(clustered),
        "note": (
            "speaker IDは声の識別子ではないため、splitはこのvoice_cluster_id単位で行う（D-015）。"
            "手元のPass A規模では話者数が限られるため、閾値はPass B後に再調整すること。"
        ),
    }

    artifacts.write_run_metadata(
        run_dir, phase="p1d-clusters",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"manifest": args.manifest, "speaker_cache": args.speaker_cache},
    )
    artifacts.write_metrics(run_dir, metrics)

    print(f"クラスタ: {summary}")
    print(f"複数話者を含むクラスタ: {len(merged)} 件")
    for cid, ids in list(merged.items())[:5]:
        print(f"   {cid} <- {ids}")
    print(f"dispersion が高い話者: {len(suspicious)} 件 "
          f"(既知の総称ラベル {len(known_generic)} 件のうち {len(detected_generic)} 件を検出)")
    print(f"\n完了: {run_dir}")


if __name__ == "__main__":
    main()
