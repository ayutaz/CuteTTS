#!/usr/bin/env bash
# Copyright 2026 OPPO and Fudan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# vast.ai インスタンス上で CuteTTS 日本語学習の実行環境を作る。
#
#   bash vastai_bootstrap.sh            # 環境構築 + checkpoint取得
#   bash vastai_bootstrap.sh --bench    # 続けてVRAMベンチマークを実行
#
# HF_TOKEN を環境変数で渡すと gated dataset にもアクセスできる。
# 公開checkpoint (OPPOer/CuteTTS) は token 不要。
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ayutaz/CuteTTS.git}"
BRANCH="${BRANCH:-feat/japanese-training}"
WORKDIR="${WORKDIR:-/workspace/CuteTTS}"

echo "=== 1/5 システム依存 ==="
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq git libsndfile1 >/dev/null 2>&1 || true

echo "=== 2/5 リポジトリ ==="
if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" fetch --all --quiet && git -C "$WORKDIR" checkout "$BRANCH" --quiet
  git -C "$WORKDIR" pull --quiet
else
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
echo "  commit: $(git rev-parse --short HEAD)"

echo "=== 3/5 Python依存 ==="
# ベースイメージに torch 2.5.1 + cu121 が入っている前提。無ければ入れる。
python -c "import torch; assert torch.__version__.startswith('2.5.1')" 2>/dev/null || \
  pip install -q torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -q -e . 2>&1 | tail -2
pip install -q pytest pyyaml huggingface_hub accelerate 2>&1 | tail -1
python - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU: {p.name}  VRAM {p.total_memory/1e9:.1f} GB")
PY

echo "=== 4/5 checkpoint ==="
if [ ! -f model/CuteTTS/config.json ]; then
  mkdir -p model
  hf download OPPOer/CuteTTS --local-dir ./model/CuteTTS >/dev/null
fi
du -sh model/CuteTTS

echo "=== 5/5 テスト（環境の健全性確認） ==="
python -m pytest tests/training -q -p no:warnings -m "not slow" --tb=line 2>&1 | tail -3

if [ "${1:-}" = "--bench" ]; then
  echo
  echo "=== VRAM / throughput ベンチマーク ==="
  python scripts/benchmark_training_memory.py \
    --variants full freeze_patch_encoder lm_only \
    --target-patches 32 64 128 188 \
    --dtype bfloat16 --steps 4
fi

echo
echo "bootstrap 完了。作業ディレクトリ: $WORKDIR"
