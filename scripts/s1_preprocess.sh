#!/usr/bin/env bash
# Copyright 2026 OPPO and Fudan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# S1（100〜500時間）の前処理を vast.ai 上で完結させる。
#
# ローカルへ 215 GB を落とさない。インスタンス上で gol を直接取得し、
# 永続化するのは latent cache 約 1.8 GB だけ。以降のインスタンスは
# それを取得するだけで学習できる。
#
#   HF_TOKEN=<write権限> bash scripts/s1_preprocess.sh
#
# 環境変数:
#   HF_TOKEN     必須。gated dataset の読み取りと成果物の書き込みに使う
#   UPLOAD_REPO  既定 tts-dataset/cutetts-ja-latents
#   SKIP_UPLOAD  1 ならアップロードしない（動作確認用）
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
UPLOAD_REPO="${UPLOAD_REPO:-tts-dataset/cutetts-ja-latents}"
GOL_REPO="midralab/gol-dataset"

# S1 の選定 5 game（326.0 h / 1,197話者ID / 214.8 GB）。
# 大きい game は _partN に分割されているので実ファイルは 7 本。
# 選定根拠は docs/japanese-training/03-data-and-frontend.md を参照。
GOL_FILES=(
  "9381931FAB68786161D5A740F32C5A33_part1.tar"   # 93.0h / 249話者
  "9381931FAB68786161D5A740F32C5A33_part2.tar"
  "AA538DEF78C6ED34DE73227273218CF1_part1.tar"   # 76.9h / 215話者
  "AA538DEF78C6ED34DE73227273218CF1_part2.tar"
  "F1C9239826100C845CC7FB342E0C8374.tar"         # 61.7h / 140話者
  "2A8DB5A7035796A4BEA940C82E530521.tar"         # 52.2h / 166話者
  "F4736E42EB295C75542EE2BFBCEEADA0.tar"         # 42.2h / 427話者
)

export PYTHONIOENCODING=utf-8
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$WORKDIR"

: "${HF_TOKEN:?HF_TOKEN を設定すること（gated datasetの読み取りに必要）}"

step() { echo; echo "=== $* ==="; date -u +"    %Y-%m-%dT%H:%M:%SZ"; }

step "0/6 事前確認"
df -h . | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
from huggingface_hub import HfApi
print("  HF:", HfApi().whoami()["name"])
PY

step "1/6 metadata.tsv（1.68 GB）"
if [ ! -f data/raw/gol/metadata.tsv ]; then
  hf download "$GOL_REPO" metadata.tsv --repo-type dataset --local-dir data/raw/gol
fi
wc -l data/raw/gol/metadata.tsv

step "2/6 音声 tar（214.8 GB / 7ファイル）"
mkdir -p data/raw/gol/tars
for f in "${GOL_FILES[@]}"; do
  if [ -f "data/raw/gol/tars/$f" ]; then
    echo "  skip（取得済み）: $f"
    continue
  fi
  echo "  取得: $f"
  hf download "$GOL_REPO" "$f" --repo-type dataset --local-dir data/raw/gol/tars
  df -h . | tail -1
done
du -sh data/raw/gol/tars

step "3/6 checkpoint（VAE と Speaker Encoder を使う）"
if [ ! -f model/CuteTTS/config.json ]; then
  hf download OPPOer/CuteTTS --local-dir ./model/CuteTTS
fi

step "4/6 manifest"
# 分割tarを game 単位へまとめる処理は prepare_japanese_manifest 側で対応済み
python -u scripts/prepare_japanese_manifest.py --skip-full-accounting
wc -l data/manifests/*.jsonl

step "5/6 latent cache + speaker embedding（約7.3 GPU時間）"
python -u scripts/cache_audio_latents.py --manifest data/manifests/all.jsonl
du -sh data/cache/latents data/cache/speaker

step "6/6 voice cluster（閾値0.92。既定0.70はこの埋め込み空間で破綻する）"
python -u scripts/build_voice_clusters.py --threshold 0.92
python - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("data/manifests/all_clustered.jsonl", encoding="utf-8")]
n = collections.Counter(r["split"] for r in rows)
h = collections.Counter()
cl = collections.defaultdict(set)
for r in rows:
    h[r["split"]] += r["duration"]
    cl[r["split"]].add(r["voice_cluster_id"])
print(f"{'split':16s} {'発話':>8s} {'時間':>8s} {'cluster':>8s}")
for s in sorted(n):
    print(f"{s:16s} {n[s]:8,} {h[s]/3600:7.1f}h {len(cl[s]):8d}")
PY

if [ "${SKIP_UPLOAD:-0}" = "1" ]; then
  echo; echo "SKIP_UPLOAD=1 のためアップロードしない"; exit 0
fi

step "アップロード（latent cache と manifest のみ。音声は含まない）"
hf upload "$UPLOAD_REPO" data/cache/latents latents --repo-type dataset \
  --commit-message "latent cache: gol 5 game / 326h"
hf upload "$UPLOAD_REPO" data/cache/speaker speaker --repo-type dataset \
  --commit-message "speaker embeddings: gol 5 game / 326h"
hf upload "$UPLOAD_REPO" data/manifests manifests --repo-type dataset \
  --commit-message "manifests: voice cluster t=0.92"

echo
echo "完了。以降のインスタンスは次で復元できる:"
echo "  hf download $UPLOAD_REPO --repo-type dataset --local-dir data/restored"
