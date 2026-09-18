#!/usr/bin/env bash
# Copyright 2026 ayutaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# M4c: F0 の条件づけの**経路が通っているか**を安く確かめる（vast.ai上で完結）。
#
# **本番の前に必ずこれを通す。** データ作り（325.9h の decode + F0）だけで
# 10時間かかるので、経路が死んでいたら丸ごと無駄になる。
# ここは 17.9h の部分集合で回すので約$0.5で済む。
#
#   HF_TOKEN=<read権限> bash scripts/m4c_validate.sh
#
# 環境変数: STEPS（既定 10000）/ HOURS（既定 17.9）/ WORKERS（既定 16）
#           FRONTEND（既定 accent）/ SHARDS（既定 2）
#
# **判定**（結果を見る前に決めてある。R-050）:
#
#   **主指標はアクセント核**（R-050）。輪郭は codec の上限 +0.367 と
#   人間の天井 +0.382 が区別できないが、アクセントは上限 75.0% が
#   天井 64.5% を上回るので伸びしろを解像できる。
#
#   oracle の**アクセント対人間が 55% 以上**（現行 43.6%）
#   → 経路は通っている。本番（325.9h）へ進む
#   副指標: 輪郭 +0.20 以上（現行 +0.091、上限 +0.367）
#
#   mismatch（**別の文**のF0）が oracle と同じだけ上がる
#   → 効果は特異的でない。条件が効いているのではなく何かが変わっただけ
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
STEPS="${STEPS:-10000}"
HOURS="${HOURS:-17.9}"
WORKERS="${WORKERS:-16}"
FRONTEND="${FRONTEND:-accent}"
SHARDS="${SHARDS:-2}"
MODEL="${MODEL:-model/CuteTTS}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN が要る" >&2
  exit 1
fi
cd "$WORKDIR"
export PATH="$WORKDIR/.venv/bin:/root/.local/bin:$PATH"

LOCK="${LOCK:-/workspace/m4c-driver.lock}"
lock_holder() { cat "$LOCK" 2>/dev/null; }
if [ -f "$LOCK" ] && kill -0 "$(lock_holder)" 2>/dev/null; then
  echo "別の駆動が動いている（PID $(lock_holder)）" >&2
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

MANIFEST="data/s1v2/manifests-v2/all_clustered.jsonl"
LATENTS="data/s1v2/latents-v2"
SPEAKERS="data/s1v2/speaker-v2"
SUBSET="data/s1v2/subset-${HOURS}h.jsonl"
F0="data/s1v2/f0-${HOURS}h"
OUT="checkpoints/m4c-validate"

# ---------------------------------------------------------------- 1. データ
if [ ! -f "$MANIFEST" ]; then
  echo "=== 1/6 latent cache を取得 ==="
  hf download tts-dataset/cutetts-ja-latents --repo-type dataset \
    --local-dir data/s1v2 >/dev/null
fi
LINES="$(wc -l < "$MANIFEST")"
[ "$LINES" = "286864" ] || {
  echo "manifestの行数が違う（期待 286864、実際 $LINES）" >&2; exit 1; }
echo "  manifest $LINES 行"

[ -d data/eval/prosody_audio ] || python -u scripts/fetch_prosody_audio.py

# ---------------------------------------------------------------- 2. 部分集合
if [ ! -f "$SUBSET" ]; then
  echo "=== 2/6 ${HOURS}h の部分集合 ==="
  python -u scripts/build_data_subset.py --manifest "$MANIFEST" \
    --out "$SUBSET" --target-hours "$HOURS"
fi
echo "  部分集合 $(wc -l < "$SUBSET") 行"

# ---------------------------------------------------------------- 3. F0 cache
# **ここで decode の速度が分かる。** 本番の見積もりはこの実測で更新する
echo "=== 3/6 F0 cache（decode + harvest）==="
python -u scripts/cache_f0_targets.py --latent-cache "$LATENTS" --out "$F0" \
  --model-dir "$MODEL" --manifest "$SUBSET" --device cuda --workers "$WORKERS"

# ---------------------------------------------------------------- 4. 学習
if [ -d "$OUT/inference" ]; then
  echo "=== 4/6 学習は済み ==="
else
  echo "=== 4/6 学習（${STEPS} step / frontend=${FRONTEND}）==="
  python -u scripts/train_continual.py \
    --manifest "$SUBSET" --latent-cache "$LATENTS" --speaker-cache "$SPEAKERS" \
    --model-dir "$MODEL" --param-dtype float32 --frontend "$FRONTEND" \
    --f0-cache "$F0" --f0-lr 2e-4 \
    --steps "$STEPS" --batch-size 4 --lr 2e-5 --seed 42 \
    --save-every "$STEPS" --export-every-save --eval-every 1000 \
    --out "$OUT"
fi
[ -f "$OUT/inference/f0_conditioner.safetensors" ] || {
  echo "**条件の重みが出ていない**（`--f0-cache` が効いていない）" >&2; exit 1; }

# ---------------------------------------------------------------- 5. 評価
eval_sharded() {
  local label="$1" extra="$2"
  local dirs=() run_dir
  for k in $(seq 1 "$SHARDS"); do
    python -u scripts/evaluate_prosody.py --model-dir "$OUT/inference" \
      --label "${label}-s${k}" --shard "${k}/${SHARDS}" --device cuda \
      --save-samples 0 --eval-set data/eval/prosody_eval_set_v2.json \
      --frontend "$FRONTEND" $extra > "/tmp/${label}-s${k}.log" 2>&1 &
  done
  wait
  for k in $(seq 1 "$SHARDS"); do
    run_dir="$(grep -ao 'artifacts/[A-Za-z0-9_-]*/[0-9T:-]*' "/tmp/${label}-s${k}.log" | tail -1)"
    if [ -z "$run_dir" ] || [ ! -f "$run_dir/metrics.json" ]; then
      echo "shard ${k} が落ちた（/tmp/${label}-s${k}.log）" >&2
      tail -5 "/tmp/${label}-s${k}.log" >&2
      return 1
    fi
    dirs+=("$run_dir/metrics.json")
  done
  local joined
  joined="$(IFS=,; echo "${dirs[*]}")"
  python -u scripts/evaluate_prosody.py --merge "$joined" --label "$label" \
    --save-samples 0 --eval-set data/eval/prosody_eval_set_v2.json \
    --frontend "$FRONTEND" $extra
}

echo "=== 5/6 評価（3条件。**同じ重み**で条件だけ変える）==="
for spec in "none| " "oracle|--f0-source oracle" "mismatch|--f0-source mismatch"; do
  IFS='|' read -r name extra <<< "$spec"
  [ -f "/workspace/done-m4c-${name}" ] && { echo "  済み: ${name}"; continue; }
  echo "--- ${name} ---"
  eval_sharded "m4c-${name}-prosody" "$extra"
  touch "/workspace/done-m4c-${name}"
done

# ---------------------------------------------------------------- 6. 比較
echo
echo "=== 6/6 比較 ==="
python -u scripts/compare_prosody_runs.py m4c-none-prosody m4c-oracle-prosody || true
python -u scripts/compare_prosody_runs.py m4c-none-prosody m4c-mismatch-prosody || true

echo
echo '判定: oracle の輪郭が +0.20 以上なら経路は通っている（上限は +0.367）。'
echo 'mismatch が同じだけ上がるなら、効いているのは条件ではない。'
