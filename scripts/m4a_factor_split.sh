#!/usr/bin/env bash
# Copyright 2026 ayutaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# M4a の -5.80pt を要因に分ける（vast.ai上で完結）。
#
# `m4a-kana` の frontend は `yomi` で**全文仮名化ではない**（漢字 18.16% →
# 14.65%。`accent` は 0%）。そのため「記号だけの効果 -4.06pt」は誤った帰属で、
# 実際は**全文仮名化 + 記号**の合計だった。分けるには対照が2本要る。
#
#   kanafull  全文片仮名・記号なし          → 仮名化だけの効果
#   shuffled  核を偽の位置へ（記号数は同一） → 核の位置の情報の効果
#   clean     語境界をまたぐ長音化を止める   → frontend の欠陥の効果
#
# A1（アクセント最小対）も同じ checkpoint で測る。**`kanafull` は対の2文の
# 入力が完全に同一になるので、区別が 0% に落ちなければ測定側がおかしい。**
#
#   HF_TOKEN=<read権限> bash scripts/m4a_factor_split.sh
#
# 環境変数: STEPS（既定 30000）/ SHARDS（既定 3）/ SKIP_A1 / SKIP_PROSODY
#
# **実行中にこのファイルを上書きしないこと。** bash はスクリプトをバイト位置で
# 逐次読むので、行数が変わると途中から別の位置を実行して壊れる
# （実際に `line 79: y: command not found` で評価を落とした）。
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
STEPS="${STEPS:-30000}"
SHARDS="${SHARDS:-3}"
SKIP_A1="${SKIP_A1:-}"
SKIP_PROSODY="${SKIP_PROSODY:-}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN が要る" >&2
  exit 1
fi
cd "$WORKDIR"

MANIFEST="data/s1v2/manifests-v2/all_clustered.jsonl"
LATENTS="data/s1v2/latents-v2"
SPEAKERS="data/s1v2/speaker-v2"
LINES="$(wc -l < "$MANIFEST")"
[ "$LINES" = "286864" ] || {
  echo "manifestの行数が違う（期待 286864、実際 $LINES）" >&2; exit 1; }

[ -d data/eval/prosody_audio ] || HF_TOKEN="$HF_TOKEN" python -u scripts/fetch_prosody_audio.py

# 条件: 名前|frontend
RUNS=("shuffled|accent_shuffled" "kanafull|kana_full" "clean|accent_clean")

# ---------------------------------------------------------------- 評価の共通部
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

# 学習と評価を1条件ずつ通す。**途中で落ちても残りが進むように、条件ごとに
# 済みかどうかを見て飛ばす。**
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name frontend <<< "$spec"
  out="checkpoints/m4a-${name}"

  if [ -d "$out/inference" ]; then
    echo "=== 学習は済み: $out ==="
  else
    echo "=== 学習 m4a-${name}（frontend=${frontend} / ${STEPS} step / batch 4）==="
    python -u scripts/train_continual.py \
      --manifest "$MANIFEST" --latent-cache "$LATENTS" --speaker-cache "$SPEAKERS" \
      --param-dtype float32 --frontend "$frontend" \
      --steps "$STEPS" --batch-size 4 --lr 2e-5 --seed 42 \
      --save-every "$STEPS" --export-every-save --eval-every 1000 \
      --out "$out"
  fi

  if [ -f "/workspace/done-${name}-cer" ]; then
    echo "=== 読みCERは済み: m4a-${name} ==="
  else
    echo "=== 評価 m4a-${name}: 読みCER（frontend=${frontend}）==="
    eval_sharded evaluate_japanese_cer.py "$out/inference" "m4a-${name}-cer" \
      "--eval-set data/eval/eval_set_v3.json --frontend ${frontend}"
    touch "/workspace/done-${name}-cer"
  fi

  if [ -n "$SKIP_PROSODY" ] || [ -f "/workspace/done-${name}-prosody" ]; then
    echo "=== 抑揚は飛ばす/済み: m4a-${name} ==="
  else
    echo "=== 評価 m4a-${name}: 抑揚・アクセント（frontend=${frontend}）==="
    eval_sharded evaluate_prosody.py "$out/inference" "m4a-${name}-prosody" \
      "--eval-set data/eval/prosody_eval_set_v2.json --frontend ${frontend}"
    touch "/workspace/done-${name}-prosody"
  fi
done

# ---------------------------------------------------------------- A1
# **`kanafull` は入力が同一になるので、区別が 0% に落ちるのが期待値。**
# 落ちなければ測定側の欠陥（そのときは他の値も読めない）。
if [ -z "$SKIP_A1" ]; then
  A1=("accent|accent" "shuffled|accent_shuffled" "kanafull|kana_full" "kana|yomi")
  for spec in "${A1[@]}"; do
    IFS='|' read -r name frontend <<< "$spec"
    model="checkpoints/m4a-${name}/inference"
    [ -d "$model" ] || { echo "  checkpointが無い: $model"; continue; }
    [ -f "/workspace/done-${name}-a1" ] && { echo "=== A1は済み: ${name} ==="; continue; }
    echo "=== A1 m4a-${name}（frontend=${frontend}）==="
    eval_sharded evaluate_accent_pairs.py "$model" "a1-${name}" \
      "--eval-set data/eval/accent_pair_set_v1.json --frontend ${frontend}"
    touch "/workspace/done-${name}-a1"
  done
fi

# **重みが動いた割合を必ず確認する**（R-020）
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
    print(f"  {out.split('/')[-1]:14s} frontend={settings.get('frontend')} "
          f"steps={settings.get('steps')}  {text}{mark}")
PYEOF

echo
echo "完了。基準線は `m4a-accent` 7.58% と `m4a-kana` 11.64%。"
echo "  shuffled が 7.6% 付近 → 効いたのは区切り / 11.6% 付近 → 核の内容"
echo "  kanafull と accent の差が**記号の真の効果**"
