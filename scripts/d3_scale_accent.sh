#!/usr/bin/env bash
# Copyright 2026 ayutaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# D3: **7% の水準でデータ量の傾きが残っているか**を測る（vast.ai上で完結）。
#
# **S2（1,000時間）へ進むかを $2 で決めるための実験。**
#
# D2（R-042）は 17.9h / 80.1h / 325.9h の3点で **-3.0pt/10倍** の対数直線を
# 得たが、**すべて frontend 無し**（読みCER 17.20 / 15.22 / 13.38%）だった。
# **C2 で「計算量は 7.58% の水準では効かない」と分かっている**（13.38% では
# -1.40pt 有意 → 7.58% では -0.33pt 有意差なし。R-055）ので、
# **データ量も同じく飽和している可能性がある。**
#
#   HF_TOKEN=<read権限> bash scripts/d3_scale_accent.sh
#
# 環境変数: HOURS_LIST（既定 "17 80"）/ STEPS（既定 30000）/ SHARDS（既定 3）
#           FRONTEND（既定 accent）/ SKIP_TRAIN（1 で評価だけ）
#
# **325.9h の点は測らない。** `m4a-accent`（読みCER 7.58%）が
# 同じ条件（30,000 step / batch 4 / lr 2e-5 / seed 42 / frontend accent）
# なのでそのまま基準線に使う。D1 / D2 も既存の基準線を再利用している。
#
# **判定**（結果を見る前に決めてある）:
#
#   17.9h → 325.9h（18倍）の差が **有意** → 傾きは残っている。S2 へ進む
#   **有意差なし** → **読みは完了と宣言し、S2 を取り下げる**
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
HOURS_LIST="${HOURS_LIST:-17 80}"
STEPS="${STEPS:-30000}"
SHARDS="${SHARDS:-3}"
FRONTEND="${FRONTEND:-accent}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
MODEL="${MODEL:-model/CuteTTS}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN が要る" >&2
  exit 1
fi
cd "$WORKDIR"
export PATH="$WORKDIR/.venv/bin:/root/.local/bin:$PATH"
export PYTHONIOENCODING=utf-8

python -c "import accelerate, transformers, soundfile, pyopenjtalk" || {
  echo "依存が足りない。uv sync --all-extras" >&2; exit 1; }

MANIFEST="data/s1v2/manifests-v2/all_clustered.jsonl"
LATENTS="data/s1v2/latents-v2"
SPEAKERS="data/s1v2/speaker-v2"

# ---------------------------------------------------------------- 1. データ
if [ ! -f "$MANIFEST" ]; then
  echo "=== 1/5 latent cache を取得 ==="
  hf download tts-dataset/cutetts-ja-latents --repo-type dataset \
    --local-dir data/s1v2 >/dev/null
fi
LINES="$(wc -l < "$MANIFEST")"
[ "$LINES" = "286864" ] || {
  echo "manifestの行数が違う（期待 286864、実際 $LINES）" >&2; exit 1; }
echo "  manifest $LINES 行"

# ---------------------------------------------------------------- 2. 部分集合
# **seed は既定（20260916）のまま。** D1 / D2 と同じ部分集合になる
# （`build_data_subset.py` は manifest と seed だけで決まる）。
echo "=== 2/5 部分集合 ==="
for h in $HOURS_LIST; do
  subset="data/s1v2/manifests-v2/subset_${h}h.jsonl"
  if [ ! -f "$subset" ]; then
    python -u scripts/build_data_subset.py \
      --manifest "$MANIFEST" --out "$subset" --target-hours "$h"
  fi
  echo "  ${h}h: $(wc -l < "$subset") 行"
done

# ---------------------------------------------------------------- 3. 学習
echo "=== 3/5 学習（${STEPS} step / batch 4 / lr 2e-5 / frontend ${FRONTEND}）==="
for h in $HOURS_LIST; do
  name="d3-${h}h-${FRONTEND}"
  out="checkpoints/${name}"
  if [ -d "$out/inference" ]; then
    echo "  済み: ${name}"
    continue
  fi
  echo "--- ${name} ---"
  python -u scripts/train_continual.py \
    --manifest "data/s1v2/manifests-v2/subset_${h}h.jsonl" \
    --latent-cache "$LATENTS" --speaker-cache "$SPEAKERS" \
    --model-dir "$MODEL" --param-dtype float32 --frontend "$FRONTEND" \
    --steps "$STEPS" --batch-size 4 --lr 2e-5 --seed 42 \
    --save-every "$STEPS" --export-every-save --eval-every 1000 \
    --out "$out"
  touch "/workspace/done-d3-train-${h}"
done

# **重みが動いた割合を必ず確認する**（R-020）。
echo "=== 4/5 ParameterDrift ==="
python - <<'PYEOF'
import glob
import json

for path in sorted(glob.glob("artifacts/s0-train/*/metrics.json")):
    payload = json.loads(open(path, encoding="utf-8").read())
    moved = payload.get("parameter_moved_ratio")
    if not moved:
        continue
    settings = payload.get("settings") or {}
    out = str(settings.get("out", ""))
    if "d3-" not in out:
        continue
    worst = min(moved.values())
    mark = "" if worst > 0.99 else "   ← **凍結している。比較に使えない**"
    text = "  ".join(f"{k}={v:.4f}" for k, v in moved.items())
    print(f"  {out.split('/')[-1]:18s} steps={settings.get('steps')}  {text}{mark}")
PYEOF

# ---------------------------------------------------------------- 5. 評価
eval_sharded() {
  local model="$1" label="$2"
  local dirs=() run_dir
  for k in $(seq 1 "$SHARDS"); do
    python -u scripts/evaluate_japanese_cer.py --model-dir "$model" \
      --label "${label}-s${k}" --shard "${k}/${SHARDS}" --device cuda \
      --save-samples 0 --frontend "$FRONTEND" \
      --eval-set data/eval/eval_set_v3.json \
      > "/tmp/${label}-s${k}.log" 2>&1 &
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
  python -u scripts/evaluate_japanese_cer.py --merge "$joined" --label "$label" \
    --save-samples 0 --frontend "$FRONTEND" --eval-set data/eval/eval_set_v3.json
}

echo "=== 5/5 読みCER（v3 600文 / frontend ${FRONTEND}）==="
for h in $HOURS_LIST; do
  name="d3-${h}h-${FRONTEND}"
  [ -f "/workspace/done-d3-cer-${h}" ] && { echo "  済み: ${name}"; continue; }
  echo "--- ${name} ---"
  eval_sharded "checkpoints/${name}/inference" "${name}-cer"
  touch "/workspace/done-d3-cer-${h}"
done

echo
echo "=== 集計 ==="
python -u scripts/summarize_eval_runs.py --metric cer_reading 2>/dev/null \
  | grep -aE "label|d3-" || true

echo
echo '基準線は `m4a-accent` の 7.58%（325.9h / 同条件）。'
echo '**17.9h → 325.9h（18倍）が有意なら傾きは残っている → S2 へ。**'
echo '**有意差なしなら読みは完了と宣言し、S2 を取り下げる。**'
echo 'D2（frontend 無し）の傾きは -3.0pt/10倍 だった。'
