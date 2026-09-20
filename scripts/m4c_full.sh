#!/usr/bin/env bash
# Copyright 2026 ayutaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# M4c 本番: 325.9h で F0 の条件を学習する（vast.ai上で完結）。
#
# **`scripts/m4c_validate.sh` が通ってから回す。** データ作りだけで
# 10時間かかるので、経路が死んでいたら丸ごと無駄になる。
#
#   HF_TOKEN=<read権限> bash scripts/m4c_full.sh
#
# 環境変数: STEPS（既定 30000）/ WORKERS（既定 16）/ FRONTEND（既定 accent）
#           SHARDS（既定 2。アライナを使う評価は3本で OOM する）
#           CER_SHARDS（既定 3。読みCERはアライナを使わない）
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
STEPS="${STEPS:-30000}"
WORKERS="${WORKERS:-16}"
LOOKAHEAD="${LOOKAHEAD:-4}"
INJECT="${INJECT:-head}"
POSITION="${POSITION:-1}"
DROPOUT="${DROPOUT:-0.1}"
MLP="${MLP:-0}"
F0LR="${F0LR:-2e-4}"
FRONTEND="${FRONTEND:-accent}"
SHARDS="${SHARDS:-2}"
CER_SHARDS="${CER_SHARDS:-3}"
MODEL="${MODEL:-model/CuteTTS}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN が要る" >&2
  exit 1
fi
cd "$WORKDIR"
export PATH="$WORKDIR/.venv/bin:/root/.local/bin:$PATH"

LOCK="${LOCK:-/workspace/m4c-driver.lock}"
lock_holder() { cat "$LOCK" 2>/dev/null; }
waited=0
while [ -f "$LOCK" ] && kill -0 "$(lock_holder)" 2>/dev/null; do
  sleep 60
  waited=$((waited + 1))
  [ "$waited" -ge 720 ] && { echo "ロックが空かない" >&2; exit 1; }
done
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
echo "待機 ${waited} 分"

MANIFEST="data/s1v2/manifests-v2/all_clustered.jsonl"
LATENTS="data/s1v2/latents-v2"
SPEAKERS="data/s1v2/speaker-v2"
F0="${F0_CACHE:-data/s1v2/f0-v2}"   # **検証で作った分を引き継ぐ**
OUT="checkpoints/m4c-full"

LINES="$(wc -l < "$MANIFEST")"
[ "$LINES" = "286864" ] || {
  echo "manifestの行数が違う（期待 286864、実際 $LINES）" >&2; exit 1; }

# ---------------------------------------------------------------- 1. F0 cache
# **一番重い段。** 律速は CPU の harvest なので、コア数で決まる
# （16 workers で 50× / 12 workers で 23× の実測）。
# **train と dev だけ作る**（test は学習にも dev 評価にも使わない）。
# 既にある id は飛ばすので、途中で落ちても同じコマンドで再開できる。
F0_MANIFEST="${MANIFEST%.jsonl}-f0.jsonl"
if [ ! -f "$F0_MANIFEST" ]; then
  python - "$MANIFEST" "$F0_MANIFEST" <<'PYEOF'
import json
import sys

source, target = sys.argv[1], sys.argv[2]
kept = 0
with open(source, encoding="utf-8") as src, open(target, "w", encoding="utf-8") as dst:
    for line in src:
        row = json.loads(line)
        if str(row.get("split", "")).startswith(("train", "dev")):
            dst.write(line)
            kept += 1
print(f"  F0 を作る対象: {kept:,} 行（test は除く）")
PYEOF
fi

echo "=== 1/4 F0 cache（325.9h）==="
python -u scripts/cache_f0_targets.py --latent-cache "$LATENTS" --out "$F0" \
  --model-dir "$MODEL" --manifest "$F0_MANIFEST" --device cuda \
  --workers "$WORKERS" --report-every 2000

# ---------------------------------------------------------------- 2. 学習
if [ -d "$OUT/inference" ]; then
  echo "=== 2/4 学習は済み ==="
else
  echo "=== 2/4 学習（${STEPS} step / ${FRONTEND} / 先読み ${LOOKAHEAD} / 差込 ${INJECT} / dropout ${DROPOUT}）==="
  python -u scripts/train_continual.py \
    --manifest "$MANIFEST" --latent-cache "$LATENTS" --speaker-cache "$SPEAKERS" \
    --model-dir "$MODEL" --param-dtype float32 --frontend "$FRONTEND" \
    --f0-cache "$F0" --f0-lr "$F0LR" --f0-lookahead "$LOOKAHEAD" --f0-inject "$INJECT" --f0-dropout "$DROPOUT" --f0-mlp "$MLP" ${POSITION:+--f0-position} \
    --steps "$STEPS" --batch-size 4 --lr 2e-5 --seed 42 \
    --save-every "$STEPS" --export-every-save --eval-every 1000 \
    --out "$OUT"
fi
[ -f "$OUT/inference/f0_conditioner.safetensors" ] || {
  echo "**条件の重みが出ていない**" >&2; exit 1; }

# ---------------------------------------------------------------- 3. 評価
eval_sharded() {
  local script="$1" label="$2" shards="$3" extra="$4"
  local dirs=() run_dir
  for k in $(seq 1 "$shards"); do
    python -u "scripts/${script}" --model-dir "$OUT/inference" \
      --label "${label}-s${k}" --shard "${k}/${shards}" --device cuda \
      --save-samples 0 --frontend "$FRONTEND" $extra \
      > "/tmp/${label}-s${k}.log" 2>&1 &
  done
  wait
  for k in $(seq 1 "$shards"); do
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
  python -u "scripts/${script}" --merge "$joined" --label "$label" \
    --save-samples 0 --frontend "$FRONTEND" $extra
}

echo "=== 3/4 評価 ==="
PROS="--eval-set data/eval/prosody_eval_set_v2.json"
for spec in "none| " "oracle|--f0-source oracle" "mismatch|--f0-source mismatch"; do
  IFS='|' read -r name extra <<< "$spec"
  [ -f "/workspace/done-full-${name}" ] && { echo "  済み: ${name}"; continue; }
  echo "--- 抑揚 ${name} ---"
  eval_sharded evaluate_prosody.py "m4cfull-${name}-prosody" "$SHARDS" "$PROS $extra"
  touch "/workspace/done-full-${name}"
done

# **読みを壊していないことも確かめる**（+1pt 以上悪化したら差し戻す）
if [ ! -f /workspace/done-full-cer ]; then
  echo "--- 読みCER（条件なし）---"
  eval_sharded evaluate_japanese_cer.py "m4cfull-cer" "$CER_SHARDS" \
    "--eval-set data/eval/eval_set_v3.json"
  touch /workspace/done-full-cer
fi

# ---------------------------------------------------------------- 4. 比較
echo
echo "=== 4/4 比較 ==="
python -u scripts/compare_prosody_runs.py m4cfull-none-prosody m4cfull-oracle-prosody || true
python -u scripts/compare_prosody_runs.py m4cfull-none-prosody m4cfull-mismatch-prosody || true
python -u scripts/summarize_eval_runs.py --metric cer_reading 2>/dev/null | grep -aE "label|m4cfull" || true

echo
echo '判定: oracle の輪郭が +0.20 以上（上限 +0.367）なら韻律の経路は有効。'
echo 'mismatch が同じだけ上がるなら条件が効いているのではない。'
echo '読みCER が 7.58% から +1pt 以上悪化していたら差し戻す。'
