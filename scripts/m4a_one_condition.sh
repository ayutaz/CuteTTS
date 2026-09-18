#!/usr/bin/env bash
# Copyright 2026 ayutaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# frontend 1条件を学習して測る（vast.ai上で完結）。
#
# M4a-split で条件ごとの bespoke なスクリプトが増えたので1本にまとめた。
#
#   NAME=kanafull-clean FRONTEND=kana_full_clean \
#     HF_TOKEN=<read権限> bash scripts/m4a_one_condition.sh
#
# 環境変数:
#   NAME       必須。checkpoint は `checkpoints/m4a-${NAME}`
#   FRONTEND   必須。`yomi.FRONTEND_MODES` のいずれか
#   STEPS      既定 30000 / BATCH 既定 4 / LR 既定 2e-5 / SEED 既定 42
#   CER_SHARDS 既定 3（アライナを使わないので3で問題ない）
#   PROSODY_SHARDS 既定 2（**3ではCUDA OOMで文が落ちる**）
#   WAIT_FOR   このプロセス名が消えるまで待つ（省略可）
#
# **学習と評価で同じ frontend を通す。** 食い違うと条件が変わる（F1 の教訓）。
set -uo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
NAME="${NAME:?NAME が要る}"
FRONTEND="${FRONTEND:?FRONTEND が要る}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-4}"
LR="${LR:-2e-5}"
SEED="${SEED:-42}"
CER_SHARDS="${CER_SHARDS:-3}"
PROSODY_SHARDS="${PROSODY_SHARDS:-2}"
WAIT_FOR="${WAIT_FOR:-}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN が要る" >&2
  exit 1
fi
cd "$WORKDIR"

# ---------------------------------------------------------------- 直列化
# **待機中の駆動スクリプトはGPUを使わない**ので、GPUやジョブの有無では
# 生存を判定できない（それで同じ学習を二重に走らせた）。
LOCK="${LOCK:-/workspace/m4a-driver.lock}"
lock_holder() { cat "$LOCK" 2>/dev/null; }
lock_alive() { [ -f "$LOCK" ] && kill -0 "$(lock_holder)" 2>/dev/null; }

if [ -n "$WAIT_FOR" ]; then
  # 括弧は grep 自身のコマンド行に当たらないようにするため
  pattern="[${WAIT_FOR:0:1}]${WAIT_FOR:1}"
  waited=0
  while ps -eo args | grep -q "$pattern"; do
    sleep 60
    waited=$((waited + 1))
    [ "$waited" -ge 600 ] && { echo "10時間待っても終わらない。やめる" >&2; exit 1; }
  done
  echo "${WAIT_FOR} を ${waited} 分待った"
fi

waited=0
while lock_alive; do
  sleep 60
  waited=$((waited + 1))
  [ "$waited" -ge 600 ] && { echo "ロックが空かない。やめる" >&2; exit 1; }
done
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

MANIFEST="data/s1v2/manifests-v2/all_clustered.jsonl"
LINES="$(wc -l < "$MANIFEST")"
[ "$LINES" = "286864" ] || {
  echo "manifestの行数が違う（期待 286864、実際 $LINES）" >&2; exit 1; }

eval_sharded() {
  local script="$1" model="$2" label="$3" shards="$4" extra="${5:-}"
  local dirs=() run_dir
  for k in $(seq 1 "$shards"); do
    python -u "scripts/${script}" --model-dir "$model" --label "${label}-s${k}" \
      --shard "${k}/${shards}" --device cuda --save-samples 0 $extra \
      > "/tmp/${label}-s${k}.log" 2>&1 &
  done
  wait
  for k in $(seq 1 "$shards"); do
    run_dir="$(grep -ao 'artifacts/[A-Za-z0-9_-]*/[0-9T:-]*' "/tmp/${label}-s${k}.log" \
               | tail -1)"
    if [ -z "$run_dir" ] || [ ! -f "$run_dir/metrics.json" ]; then
      echo "shard ${k} の run dir が取れない（/tmp/${label}-s${k}.log）" >&2
      tail -5 "/tmp/${label}-s${k}.log" >&2
      return 1
    fi
    dirs+=("$run_dir/metrics.json")
  done
  local joined
  joined="$(IFS=,; echo "${dirs[*]}")"
  python -u "scripts/${script}" --merge "$joined" --label "$label" \
    --save-samples 0 $extra
}

OUT="checkpoints/m4a-${NAME}"
if [ -d "$OUT/inference" ]; then
  echo "=== 学習は済み: $OUT ==="
else
  echo "=== 学習 m4a-${NAME}（frontend=${FRONTEND} / ${STEPS} step / batch ${BATCH}）==="
  python -u scripts/train_continual.py \
    --manifest "$MANIFEST" --latent-cache data/s1v2/latents-v2 \
    --speaker-cache data/s1v2/speaker-v2 \
    --param-dtype float32 --frontend "$FRONTEND" \
    --steps "$STEPS" --batch-size "$BATCH" --lr "$LR" --seed "$SEED" \
    --save-every "$STEPS" --export-every-save --eval-every 1000 \
    --out "$OUT" || { echo "学習が落ちた" >&2; exit 1; }
fi

echo "=== 読みCER m4a-${NAME}（${CER_SHARDS} shard）==="
eval_sharded evaluate_japanese_cer.py "$OUT/inference" "m4a-${NAME}-cer" \
  "$CER_SHARDS" "--eval-set data/eval/eval_set_v3.json --frontend ${FRONTEND}"

echo "=== 抑揚 m4a-${NAME}（${PROSODY_SHARDS} shard）==="
eval_sharded evaluate_prosody.py "$OUT/inference" "m4a-${NAME}-prosody" \
  "$PROSODY_SHARDS" \
  "--eval-set data/eval/prosody_eval_set_v2.json --frontend ${FRONTEND}"

echo "=== ParameterDrift ==="
python - <<'PYEOF'
import glob
import json

for path in sorted(glob.glob("artifacts/s0-train/*/metrics.json")):
    payload = json.loads(open(path, encoding="utf-8").read())
    moved = payload.get("parameter_moved_ratio")
    settings = payload.get("settings") or {}
    out = str(settings.get("out", ""))
    if not moved or "m4a-" not in out:
        continue
    worst = min(moved.values())
    mark = "" if worst > 0.99 else "   ← **凍結している。比較に使えない**"
    text = "  ".join(f"{k}={v:.4f}" for k, v in moved.items())
    print(f"  {out.split('/')[-1]:16s} frontend={settings.get('frontend')} "
          f"steps={settings.get('steps')}  {text}{mark}")
PYEOF

echo
echo "=== 比較 ==="
python scripts/summarize_eval_runs.py --metric cer_reading 2>/dev/null \
  | grep -aE "label|m4a-" || true
for base in m4a-kanafull-cer m4a-accent-cer; do
  python scripts/summarize_eval_runs.py --metric cer_reading \
    --compare "$base" "m4a-${NAME}-cer" 2>/dev/null | tail -9 || true
done
python scripts/compare_prosody_runs.py m4a-kanafull-prosody \
  "m4a-${NAME}-prosody" 2>/dev/null || true
echo "完了"
