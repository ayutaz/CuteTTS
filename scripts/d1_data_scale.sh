#!/usr/bin/env bash
# Copyright 2026 ayutaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# D1: データ量を **30,000 step** で測り直す（vast.ai上で完結）。
#
# **S2（1,000時間）へ進むかを決めるための実験。**
#
# データ量の比較は 3,000 step でしか行っていない（17h → 325.9h で -1.90pt）。
# その条件では 17h がほぼ1 epoch・325.9h は **1 epochの5%** しか見ておらず、
# **大きいデータ側に不利**だった。T2 で「効いているのは計算量」と分かったので
# （R-036）、**計算量を現行最良と同じ 30,000 step にして測り直す。**
#
#   HF_TOKEN=<read権限> bash scripts/d1_data_scale.sh
#
# 環境変数:
#   HF_TOKEN     必須
#   HOURS        部分集合の目標時間。既定 17
#   STEPS        step数。既定 30000（**現行最良と同じ計算量にする**）
#   SHARDS       評価の並列数。既定 3
#   SKIP_TRAIN   1 なら学習を飛ばして評価だけ（再開用）
#
# **基準線は既存の 325.9h / 30,000 step を使う**（読みCER 13.38% / 素 20.10%）。
# 同じ条件なので測り直さない。
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
HOURS="${HOURS:-17}"
STEPS="${STEPS:-30000}"
SHARDS="${SHARDS:-3}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
NAME="d1-${HOURS}h"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN が要る" >&2
  exit 1
fi
cd "$WORKDIR"

python -c "import accelerate, transformers, soundfile, pyworld, pyopenjtalk" || {
  echo "依存が足りない。pip install -e '.[ja,prosody,eval]'" >&2; exit 1; }

# **抑揚setの音声を先に揃える。** ここを忘れると学習とCER評価が終わった後に
# `soundfile.LibsndfileError` で落ちる（実測で3.9時間インスタンスを遊ばせた）。
[ -d data/eval/prosody_audio ] || HF_TOKEN="$HF_TOKEN" python scripts/fetch_prosody_audio.py

MANIFEST="data/s1v2/manifests-v2/all_clustered.jsonl"
LATENTS="data/s1v2/latents-v2"
SPEAKERS="data/s1v2/speaker-v2"
LINES="$(wc -l < "$MANIFEST")"
[ "$LINES" = "286864" ] || {
  echo "manifestの行数が違う（期待 286864、実際 $LINES）" >&2; exit 1; }

# ---------------------------------------------------------------- 部分集合
# **クラスタ単位で無作為に選ぶ**（大きいクラスタから採ると密度が交絡する）。
# **train だけを絞る**（dev を絞ると学習中の比較が別条件になる）。
SUBSET="data/s1v2/manifests-v2/subset_${HOURS}h.jsonl"
if [ ! -f "$SUBSET" ]; then
  echo "=== 部分集合を作る（${HOURS}h）==="
  python -u scripts/build_data_subset.py \
    --manifest "$MANIFEST" --out "$SUBSET" --target-hours "$HOURS"
fi

# ---------------------------------------------------------------- 学習
out="checkpoints/${NAME}"
if [ "$SKIP_TRAIN" = "1" ] && [ -d "$out" ]; then
  echo "  既にある: $out"
else
  echo "=== 学習 ${NAME}（${STEPS} step / batch 4 / lr 2e-5）==="
  python -u scripts/train_continual.py \
    --manifest "$SUBSET" --latent-cache "$LATENTS" --speaker-cache "$SPEAKERS" \
    --param-dtype float32 \
    --steps "$STEPS" --batch-size 4 --lr 2e-5 --seed 42 \
    --save-every "$STEPS" --export-every-save --eval-every 1000 \
    --out "$out"
fi

# **重みが動いた割合を必ず確認する**（R-020）。
echo "=== ParameterDrift ==="
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
    if "d1-" not in out:
        continue
    worst = min(moved.values())
    mark = "" if worst > 0.99 else "   ← **凍結している。比較に使えない**"
    text = "  ".join(f"{k}={v:.4f}" for k, v in moved.items())
    print(f"  {out.split('/')[-1]:14s} steps={settings.get('steps')}  {text}{mark}")
PYEOF

# ---------------------------------------------------------------- 評価
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

echo "=== 評価 ${NAME}: 読みCER ==="
eval_sharded evaluate_japanese_cer.py "checkpoints/${NAME}/inference" \
  "${NAME}-cer" "--eval-set data/eval/eval_set_v3.json"
echo "=== 評価 ${NAME}: 抑揚・アクセント ==="
eval_sharded evaluate_prosody.py "checkpoints/${NAME}/inference" \
  "${NAME}-prosody" "--eval-set data/eval/prosody_eval_set_v2.json"

echo
echo "完了。基準線は既存の 325.9h / 30,000 step（読みCER 13.38% / 素 20.10%）。"
echo "**有意かつ2pt以上なら S2 へ進む。有意でなければ S2 は見送り。**"
