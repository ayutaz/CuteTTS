#!/usr/bin/env bash
# Copyright 2026 ayutaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# アライナを使う評価だけを**2 shard**で回す（M4a-split / A1）。
#
# **3 shard では CUDA OOM で文が落ちる。** MMS_FA のアライナが1プロセス
# 1.18 GB を取るので、3本並べると 24 GB に収まらない。実測で
# `m4a-shuffled` の抑揚評価が 240文中 41文（17%）失敗した
# （shard 3 で 31/80、shard 2 で 9、shard 1 で 1）。
# **落ちた文は長い文に偏るので、絶対値が他の run と比べられなくなる。**
# 対応のある比較（共通文だけ）は成立するが、公表値には使えない。
#
# 読みCER（`evaluate_japanese_cer.py`）はアライナを使わないので 3 shard で
# 問題ない（実測で失敗 0）。
#
#   HF_TOKEN=<read権限> bash scripts/m4a_aligner_evals.sh
#
# 先に走っている駆動スクリプトのロックが空くのを待つ。
set -uo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
SHARDS="${SHARDS:-2}"

cd "$WORKDIR"


# ---------------------------------------------------------------- 二重起動の防止
# **待機中の駆動スクリプトはGPUを使わない。** そのため `nvidia-smi` や
# `train_continual` の有無では生存を判定できず、実際に**同じ checkpoint を
# 2プロセスで学習した**（`m4a-kanafull` を44分と18分、GPUを分け合っていた）。
# **駆動スクリプトはロックで直列化する。**
LOCK="${LOCK:-/workspace/m4a-driver.lock}"
lock_holder() { cat "$LOCK" 2>/dev/null; }
lock_alive() { [ -f "$LOCK" ] && kill -0 "$(lock_holder)" 2>/dev/null; }

# ---------------------------------------------------------------- 前の run を待つ
# **ロックが空くのを待つ。** プロセス名の grep では待機中の駆動を見落とす
# （それで同じ学習を二重に走らせた）。
waited=0
while lock_alive; do
  sleep 60
  waited=$((waited + 1))
  [ "$waited" -ge 480 ] && { echo "8時間待っても空かない。やめる" >&2; exit 1; }
done
echo "待機 ${waited} 分"

if lock_alive; then
  echo "別の駆動スクリプトが動いている（PID $(lock_holder)）。何もしない" >&2
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

eval_sharded() {
  local script="$1" model="$2" label="$3" extra="${4:-}"
  local dirs=() run_dir
  for k in $(seq 1 "$SHARDS"); do
    python -u "scripts/${script}" --model-dir "$model" --label "${label}-s${k}" \
      --shard "${k}/${SHARDS}" --device cuda --save-samples 0 $extra \
      > "/tmp/${label}-s${k}.log" 2>&1 &
  done
  wait
  for k in $(seq 1 "$SHARDS"); do
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

# ---------------------------------------------------------------- 抑揚
# `shuffled` は3 shardでOOMしたので**測り直す**（240文を揃える）
for spec in "shuffled|accent_shuffled" "kanafull|kana_full" "clean|accent_clean"; do
  IFS='|' read -r name frontend <<< "$spec"
  model="checkpoints/m4a-${name}/inference"
  [ -d "$model" ] || { echo "  checkpointが無い: $model"; continue; }
  [ -f "/workspace/done2-${name}-prosody" ] && { echo "=== 抑揚は済み: ${name} ==="; continue; }
  echo "=== 抑揚 m4a-${name}（frontend=${frontend} / ${SHARDS} shard）==="
  if eval_sharded evaluate_prosody.py "$model" "m4a-${name}-prosody" \
      "--eval-set data/eval/prosody_eval_set_v2.json --frontend ${frontend}"; then
    touch "/workspace/done2-${name}-prosody"
  fi
done

# ---------------------------------------------------------------- A1
# **`kanafull` は対の2文の入力が完全に同一**なので、区別が 0% に落ちるのが
# 期待値。落ちなければ測定側の欠陥で、他の値も読めない。
for spec in "accent|accent" "shuffled|accent_shuffled" "kanafull|kana_full" "kana|yomi"; do
  IFS='|' read -r name frontend <<< "$spec"
  model="checkpoints/m4a-${name}/inference"
  [ -d "$model" ] || { echo "  checkpointが無い: $model"; continue; }
  [ -f "/workspace/done2-${name}-a1" ] && { echo "=== A1は済み: ${name} ==="; continue; }
  echo "=== A1 m4a-${name}（frontend=${frontend} / ${SHARDS} shard）==="
  if eval_sharded evaluate_accent_pairs.py "$model" "a1-${name}" \
      "--eval-set data/eval/accent_pair_set_v1.json --frontend ${frontend}"; then
    touch "/workspace/done2-${name}-a1"
  fi
done

# ---------------------------------------------------------------- 比較
echo
echo "=== 抑揚の比較 ==="
for pair in "m4a-accent-prosody m4a-shuffled-prosody" \
            "m4a-accent-prosody m4a-kanafull-prosody" \
            "m4a-accent-prosody m4a-clean-prosody"; do
  set -- $pair
  python scripts/compare_prosody_runs.py "$1" "$2" 2>/dev/null || true
done

echo
echo '完了。A1 は pair_differentiated と pair_correct を読む。'
