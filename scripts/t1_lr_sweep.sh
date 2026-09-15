#!/usr/bin/env bash
# Copyright 2026 ayutaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# T1: 学習率の探索を vast.ai 上で完結させる。
#
# **学習17回すべてが lr=2e-5 固定で、しかも backbone が凍結していた時期に
# 選んだ値**（R-020 / R-025）。いま 228.6M 全部が動くので、最適な lr が
# 同じである理由がない。「データ量は弱い、step数は頭打ち」という結論も、
# その3軸の中での話にすぎない。
#
#   HF_TOKEN=<read権限> bash scripts/t1_lr_sweep.sh
#
# 環境変数:
#   HF_TOKEN      必須。gated dataset（cutetts-ja-latents）の読み取りに使う
#   LRS           探索する学習率。既定 "1e-5 2e-5 5e-5 1e-4"
#   STEPS         各水準のstep数。既定 10000
#   SHARDS        評価の並列数。既定 3（**生成はbatch=1でGPUが埋まらない**）
#   EVAL_TIER     1 なら CER で篩ってから上位2水準だけ抑揚を測る（既定 0）
#   SKIP_TRAIN    1 なら学習を飛ばして評価だけ（再開用）
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
LRS="${LRS:-1e-5 2e-5 5e-5 1e-4}"
STEPS="${STEPS:-10000}"
SHARDS="${SHARDS:-3}"
EVAL_TIER="${EVAL_TIER:-0}"
DATA_REPO="tts-dataset/cutetts-ja-latents"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN が要る（gated dataset の読み取り）" >&2
  exit 1
fi

cd "$WORKDIR"

# ---------------------------------------------------------------- 準備
echo "=== 準備 ==="
python -m pip install -q torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -q -e ".[ja,prosody]"
python -m pip install -q "huggingface_hub[cli]" transformers soundfile

mkdir -p model checkpoints data
[ -d model/CuteTTS ] || hf download OPPOer/CuteTTS --local-dir ./model/CuteTTS

# 学習データ（latent cache 約2GB）。**音声そのものは置いていない**
if [ ! -d data/s1v2 ]; then
  HF_TOKEN="$HF_TOKEN" hf download "$DATA_REPO" --repo-type dataset \
    --local-dir ./data/s1v2
fi

MANIFEST="$(find data/s1v2 -name 'all_clustered.jsonl' | head -1)"
LATENTS="$(find data/s1v2 -maxdepth 3 -type d -name 'latents*' | head -1)"
SPEAKERS="$(find data/s1v2 -maxdepth 3 -type d -name 'speaker*' | head -1)"
[ -n "$MANIFEST" ] || { echo "manifest が見つからない" >&2; exit 1; }
echo "manifest: $MANIFEST"
echo "latents:  $LATENTS"

# 評価setは結果を見てから作らない（凍結済みのものを取得する）
python scripts/build_eval_set.py --out data/eval/eval_set_v3.json 2>/dev/null || \
  echo "  ※ eval_set_v3.json は別途配置すること（結果を見てから作らない）"

# ---------------------------------------------------------------- 学習
train_one() {
  local lr="$1" out="checkpoints/t1-lr${1}"
  if [ "${SKIP_TRAIN}" = "1" ] && [ -d "$out" ]; then
    echo "  既にある: $out"
    return
  fi
  echo "=== 学習 lr=$lr ($STEPS step) ==="
  python -u scripts/train_continual.py \
    --manifest "$MANIFEST" --latent-cache "$LATENTS" \
    ${SPEAKERS:+--speaker-cache "$SPEAKERS"} \
    --param-dtype float32 \
    --steps "$STEPS" --lr "$lr" --batch-size 4 --seed 42 \
    --save-every "$STEPS" --export-every-save \
    --eval-every 500 \
    --out "$out"
}

for lr in $LRS; do train_one "$lr"; done

# **重みが動いた割合を必ず確認する。** 100%から外れていたら比較は無意味（R-020）
echo "=== ParameterDrift ==="
for lr in $LRS; do
  python - "checkpoints/t1-lr${lr}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for path in sorted(root.glob("**/metrics.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    moved = payload.get("parameter_moved_ratio")
    if moved:
        print(f"  {root.name}: {moved}")
        break
else:
    print(f"  {root.name}: parameter_moved_ratio が無い")
PY
done

# ---------------------------------------------------------------- 評価
# **生成は batch=1 の自己回帰なのでGPUが埋まらない**（使用率15〜71%、
# 消費電力45〜82W / TDP285W の実測）。分割して同時に走らせる。
eval_sharded() {
  local script="$1" model="$2" label="$3" extra="${4:-}"
  local dirs=()
  for k in $(seq 1 "$SHARDS"); do
    python -u "scripts/${script}" --model-dir "$model" --label "${label}-s${k}" \
      --shard "${k}/${SHARDS}" --device cuda --save-samples 0 $extra \
      > "/tmp/${label}-s${k}.log" 2>&1 &
  done
  wait
  for k in $(seq 1 "$SHARDS"); do
    dirs+=("$(grep -ao 'artifacts[^ ]*metrics.json' "/tmp/${label}-s${k}.log" | tail -1)")
  done
  local joined
  joined="$(IFS=,; echo "${dirs[*]}")"
  python -u "scripts/${script}" --merge "$joined" --label "$label" --save-samples 0
}

cer_for() {
  eval_sharded evaluate_japanese_cer.py "checkpoints/t1-lr${1}/inference" \
    "t1-lr${1}-cer" "--eval-set data/eval/eval_set_v3.json"
}
prosody_for() {
  eval_sharded evaluate_prosody.py "checkpoints/t1-lr${1}/inference" \
    "t1-lr${1}-prosody" "--eval-set data/eval/prosody_eval_set_v2.json"
}

echo "=== 評価: 読みCER（4水準）==="
for lr in $LRS; do cer_for "$lr"; done

if [ "$EVAL_TIER" = "1" ]; then
  echo "=== 評価: 抑揚・アクセント（CER上位2水準のみ）==="
  TOP="$(python - <<'PY'
import glob, json
best = []
for path in glob.glob("artifacts/s0-cer/*/metrics.json"):
    payload = json.loads(open(path, encoding="utf-8").read())
    label = str(payload.get("label", ""))
    if not label.startswith("t1-lr") or label.endswith(("-s1", "-s2", "-s3")):
        continue
    stats = payload["summary"].get("in_domain") or {}
    value = stats.get("cer_reading_mean") or stats.get("cer_mean")
    if value is not None:
        best.append((value, label.split("-")[1][2:]))
print(" ".join(lr for _, lr in sorted(best)[:2]))
PY
)"
  echo "  上位2水準: $TOP"
  for lr in $TOP; do prosody_for "$lr"; done
else
  echo "=== 評価: 抑揚・アクセント（全水準）==="
  for lr in $LRS; do prosody_for "$lr"; done
fi

# ---------------------------------------------------------------- まとめ
echo "=== 横断集計 ==="
python scripts/summarize_eval_runs.py --subset in_domain --metric cer_reading || true

echo
echo "完了。**CERだけで判定しないこと**（読みが直っても抑揚が壊れる場合がある）。"
echo "artifacts/ を持ち帰って scripts/summarize_eval_runs.py --compare で"
echo "信頼区間を出す。ParameterDrift が 100% から外れている水準は比較に使えない。"
