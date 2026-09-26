#!/usr/bin/env bash
# Copyright 2026 ayutaz
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

REPO_URL="${REPO_URL:-https://github.com/ayutaz/CuteTTS-jp.git}"
BRANCH="${BRANCH:-main}"
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
# **uv だけで揃える。`pip` は使わない。** torch を PyTorch の cu121 index から
# 取る設定は pyproject.toml に宣言してあるので、`uv sync` が cu121 版を入れる。
# ベースイメージの Python が 3.12 でなくても uv が 3.12 を用意する。
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
uv sync --all-extras
export PATH="$WORKDIR/.venv/bin:$PATH"
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
  uv run hf download OPPOer/CuteTTS --local-dir ./model/CuteTTS >/dev/null
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
