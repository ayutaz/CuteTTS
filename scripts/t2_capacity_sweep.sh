#!/usr/bin/env bash
# Copyright 2026 ayutaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# T2: batch size と学習対象を確かめる。
#
# T1 で lr は梃子でないと分かった（R-035）。学習21回すべてで
# `batch_size=4` / `flow_copies=4` / `condition_dropout=0.1` /
# **6 module全部を学習** が固定のまま。
#
# **学習対象がとくに怪しい。** R-020 以前は実質 fp32 の head（70.5M）だけが
# 学習されていた。いま全部を動かしているが、**head を凍結した方が良い**
# 可能性は一度も試していない。
#
#   HF_TOKEN=<read権限> bash scripts/t2_capacity_sweep.sh
#
# 環境変数:
#   HF_TOKEN    必須
#   SHARDS      評価の並列数。既定 3
#   SKIP_TRAIN  1 なら学習を飛ばして評価だけ（再開用）
#
# **基準線は T1 の lr=2e-5（batch 4 / 全module / 10,000 step）を使う。**
# 同じ条件なので測り直さない。
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
SHARDS="${SHARDS:-3}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# head を凍結した場合の学習対象（6 module から head だけを外す）
NO_HEAD="locenc,locenc_to_lm_proj,lm_speaker_linear,qwen_backbone,stop_predictor"

# 条件: 名前|steps|batch|trainable
#
# **batch を変えると2つのものが同時に動く。**
#   同じ step 数 → batch 16 は 4倍のデータを見る
#   データ量を揃える → batch 16 は更新回数が 1/4
# 片方では分離できないので、**両方を走らせて挟み撃ちにする**。
RUNS=(
  "nohead|10000|4|$NO_HEAD"
  "batch16-samesteps|10000|16|"
  "batch16-samesamples|2500|16|"
)

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN が要る" >&2
  exit 1
fi
cd "$WORKDIR"

# ---------------------------------------------------------------- 準備
python -c "import accelerate, transformers, soundfile, pyworld, pyopenjtalk" || {
  echo "依存が足りない。uv sync --all-extras" >&2
  exit 1
}

MANIFEST="data/s1v2/manifests-v2/all_clustered.jsonl"
LATENTS="data/s1v2/latents-v2"
SPEAKERS="data/s1v2/speaker-v2"
LINES="$(wc -l < "$MANIFEST")"
[ "$LINES" = "286864" ] || {
  echo "manifestの行数が違う（期待 286864、実際 $LINES）" >&2; exit 1; }
for f in data/eval/eval_set_v3.json data/eval/prosody_eval_set_v2.json; do
  [ -f "$f" ] || { echo "無い: $f" >&2; exit 1; }
done
[ -d data/eval/prosody_audio ] || HF_TOKEN="$HF_TOKEN" python scripts/fetch_prosody_audio.py

# ---------------------------------------------------------------- 学習
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name steps batch trainable <<< "$spec"
  out="checkpoints/t2-${name}"
  if [ "$SKIP_TRAIN" = "1" ] && [ -d "$out" ]; then
    echo "  既にある: $out"
    continue
  fi
  echo "=== 学習 ${name}（steps=${steps} batch=${batch}${trainable:+ trainable=head凍結}）==="
  python -u scripts/train_continual.py \
    --manifest "$MANIFEST" --latent-cache "$LATENTS" --speaker-cache "$SPEAKERS" \
    --param-dtype float32 \
    --steps "$steps" --batch-size "$batch" --lr 2e-5 --seed 42 \
    ${trainable:+--trainable "$trainable"} \
    --save-every "$steps" --export-every-save --eval-every 500 \
    --out "$out"
done

# **重みが動いた割合を必ず確認する**（R-020）。
# **head凍結のrunでは head が 0% になるのが正しい。**
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
    if "t2-" not in out:
        continue
    frozen = payload.get("frozen_modules") or []
    text = "  ".join(f"{k}={v:.4f}" for k, v in moved.items())
    print(f"  {out.split('/')[-1]:22s} batch={settings.get('batch_size')} "
          f"steps={settings.get('steps')}  {text}"
          + (f"  凍結: {','.join(frozen)}" if frozen else ""))
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
  # **`$extra` も渡す**（`--eval-set` を落とすと既定値を読んで落ちる）
  python -u "scripts/${script}" --merge "$joined" --label "$label" \
    --save-samples 0 $extra
}

for spec in "${RUNS[@]}"; do
  IFS='|' read -r name _ _ _ <<< "$spec"
  echo "=== 評価 ${name}: 読みCER ==="
  eval_sharded evaluate_japanese_cer.py "checkpoints/t2-${name}/inference" \
    "t2-${name}-cer" "--eval-set data/eval/eval_set_v3.json"
  echo "=== 評価 ${name}: 抑揚・アクセント ==="
  eval_sharded evaluate_prosody.py "checkpoints/t2-${name}/inference" \
    "t2-${name}-prosody" "--eval-set data/eval/prosody_eval_set_v2.json"
done

echo
echo "完了。**主指標は読みCER**（T1と同じ扱い）。抑揚とアクセントは"
echo "「壊れていないかの確認」として読む（3指標×3比較で偶然の有意が出るため）。"
echo "基準線は T1 の lr=2e-5（batch 4 / 全module / 10,000 step）。"
