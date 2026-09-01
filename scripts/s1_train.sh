#!/usr/bin/env bash
# Copyright 2026 OPPO and Fudan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# S1（100〜500時間 PoC）の学習を vast.ai 上で回す。
#
# 前処理済みのデータを Hugging Face から取るので、音声215 GBは不要。
# 取得するのは latent cache と manifest の約2 GBだけ。
#
#   HF_TOKEN=<read権限> bash scripts/s1_train.sh
#
# 環境変数:
#   HF_TOKEN    必須。gated dataset の読み取りに使う
#   STEPS       既定 40000（159,964発話 / batch 4 で約1 epoch）
#   LR          既定 5e-5
#   BATCH       既定 4（S0と同じ。RTX 3090で約9.8 GB）
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
DATA_REPO="${DATA_REPO:-tts-dataset/cutetts-ja-latents}"
STEPS="${STEPS:-40000}"
LR="${LR:-5e-5}"
BATCH="${BATCH:-4}"

export PYTHONIOENCODING=utf-8
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$WORKDIR"

: "${HF_TOKEN:?HF_TOKEN を設定すること（gated datasetの読み取りに必要）}"

step() { echo; echo "=== $* ==="; date -u +"    %Y-%m-%dT%H:%M:%SZ"; }

step "1/4 checkpoint"
if [ ! -f model/CuteTTS/config.json ]; then
  hf download OPPOer/CuteTTS --local-dir ./model/CuteTTS >/dev/null
fi
du -sh model/CuteTTS

step "2/4 前処理済みデータ（約2 GB）"
mkdir -p data/cache data/manifests
if [ ! -f data/manifests/all_clustered.jsonl ]; then
  hf download "$DATA_REPO" --repo-type dataset --local-dir data/_hf
  mv data/_hf/latents data/cache/latents
  mv data/_hf/speaker data/cache/speaker
  mv data/_hf/manifests/*.jsonl data/manifests/
fi
du -sh data/cache/latents data/cache/speaker
python - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("data/manifests/all_clustered.jsonl", encoding="utf-8")]
n = collections.Counter(); h = collections.Counter(); c = collections.defaultdict(set)
for r in rows:
    n[r["split"]] += 1; h[r["split"]] += r["duration"]; c[r["split"]].add(r["voice_cluster_id"])
for s in sorted(n):
    print(f"  {s:16s} {n[s]:8,} {h[s]/3600:7.1f}h  cluster {len(c[s]):4d}")
PY

step "3/4 学習（steps=$STEPS lr=$LR batch=$BATCH）"
# --eval-every は必須。学習ループの損失だけでは成否を判断できない（D-025）
python -u scripts/train_continual.py \
  --steps "$STEPS" --batch-size "$BATCH" --lr "$LR" \
  --warmup 500 --save-every 5000 --eval-every 2000 --eval-batches 16 \
  --group-key voice_cluster_id --condition-dropout 0.1 \
  --out checkpoints/s1 --device cuda

step "4/4 学習後の診断（base と同じ経路で比較する）"
python -u scripts/diagnose_flow_loss.py --model-dir checkpoints/s1/inference \
  --label s1-trained --batches 20 --device cuda
python -u scripts/diagnose_flow_loss.py --model-dir model/CuteTTS \
  --label base --batches 20 --device cuda

echo
echo "学習完了。次は evaluate_japanese_cer.py と check_reference_following.py。"
