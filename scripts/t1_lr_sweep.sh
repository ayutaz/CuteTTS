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
SKIP_TRAIN="${SKIP_TRAIN:-0}"
DATA_REPO="tts-dataset/cutetts-ja-latents"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN が要る（gated dataset の読み取り）" >&2
  exit 1
fi

cd "$WORKDIR"

# ---------------------------------------------------------------- 準備
echo "=== 準備 ==="
# **vast.ai の pytorch イメージは Python 3.11 で、`pyproject.toml` は 3.12 固定。**
# uv で 3.12 の venv を作る（`uv` のvenvには pip が入らないので `uv pip` を使う）。
if python -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)' 2>/dev/null \
   && python -c 'import torch, pyopenjtalk, pyworld' 2>/dev/null; then
  echo "  既に整っている（$(python -V 2>&1)）"
else
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  [ -d .venv ] || uv venv --python 3.12 .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
  uv pip install -q torch==2.5.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
  uv pip install -q -e ".[ja,prosody,eval]"
  uv pip install -q "huggingface_hub[cli]"
fi
# **ASRの読み込みに accelerate が要る。** 無いと transformers が
# `NameError: init_empty_weights` で落ちる（vast.aiの素の環境で実測）
python -c "import accelerate, transformers, soundfile" || {
  echo "評価に要る依存が無い（accelerate / transformers / soundfile）" >&2
  exit 1
}
python -c "import sys, torch; print('  ', sys.version.split()[0], torch.__version__, torch.cuda.is_available())"

mkdir -p model checkpoints data
[ -d model/CuteTTS ] || hf download OPPOer/CuteTTS --local-dir ./model/CuteTTS

# 学習データ（latent cache 約2GB）。**音声そのものは置いていない**
if [ ! -d data/s1v2 ]; then
  HF_TOKEN="$HF_TOKEN" hf download "$DATA_REPO" --repo-type dataset \
    --local-dir ./data/s1v2
fi

# **-v2 を明示する。** repoには旧版（manifests / latents / speaker）も
# 入っており、`find | head -1` では旧版を拾うことがある。
# S1v2 の 30,000 step は **manifests-v2（286,864発話）** で学習した。
# 旧版は 232,941発話なので、取り違えると lr の比較そのものが無意味になる。
MANIFEST="data/s1v2/manifests-v2/all_clustered.jsonl"
LATENTS="data/s1v2/latents-v2"
SPEAKERS="data/s1v2/speaker-v2"
for path in "$MANIFEST" "$LATENTS" "$SPEAKERS"; do
  [ -e "$path" ] || { echo "無い: $path" >&2; exit 1; }
done
LINES="$(wc -l < "$MANIFEST")"
[ "$LINES" = "286864" ] || {
  echo "manifestの行数が違う（期待 286864、実際 $LINES）。データ版を確認せよ" >&2
  exit 1
}
echo "manifest: $MANIFEST ($LINES 発話)"

# **評価setは作り直さない**（結果を見てから変えると基準線が動く）。
# JSONはローカルから転送済みのはず。音声だけ gol から取り出す。
for f in data/eval/eval_set_v3.json data/eval/prosody_eval_set_v2.json; do
  [ -f "$f" ] || { echo "無い: $f（ローカルから転送すること）" >&2; exit 1; }
done
HF_TOKEN="$HF_TOKEN" python scripts/fetch_prosody_audio.py

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

# **重みが動いた割合を必ず確認する。** 100%から外れていたら比較は無意味（R-020）。
# metricsは `checkpoints/` ではなく `artifacts/s0-train/` に出る。
echo "=== ParameterDrift（100%から外れた水準は比較に使えない）==="
python - <<'PYEOF'
import glob
import json

rows = []
for path in sorted(glob.glob("artifacts/s0-train/*/metrics.json")):
    payload = json.loads(open(path, encoding="utf-8").read())
    moved = payload.get("parameter_moved_ratio")
    if not moved:
        continue
    settings = payload.get("settings") or {}
    rows.append((settings.get("lr"), settings.get("steps"), moved))
if not rows:
    print("  **parameter_moved_ratio が1件も無い。** 学習metricsを確認せよ")
for lr, steps, moved in rows:
    worst = min(moved.values())
    mark = "" if worst > 0.99 else "   ← **凍結している。この水準は使えない**"
    text = "  ".join(f"{k}={v:.4f}" for k, v in moved.items())
    print(f"  lr={lr} steps={steps}  {text}{mark}")
PYEOF

# ---------------------------------------------------------------- 評価
# **生成は batch=1 の自己回帰なのでGPUが埋まらない**（使用率15〜71%、
# 消費電力45〜82W / TDP285W の実測）。分割して同時に走らせる。
eval_sharded() {
  local script="$1" model="$2" label="$3" extra="${4:-}"
  local dirs=() run_dir
  for k in $(seq 1 "$SHARDS"); do
    python -u "scripts/${script}" --model-dir "$model" --label "${label}-s${k}" \
      --shard "${k}/${SHARDS}" --device cuda --save-samples 0 $extra \
      > "/tmp/${label}-s${k}.log" 2>&1 &
  done
  wait
  # **ログ末尾は `完了: artifacts/<phase>/<timestamp>` で、`metrics.json` は
  # 出ない。** run dir を取って自分で付ける（付けずに grep すると空になり、
  # `--merge ",,"` で死ぬ。実測でここで止まった）。
  for k in $(seq 1 "$SHARDS"); do
    run_dir="$(grep -ao 'artifacts/[A-Za-z0-9_-]*/[0-9T:-]*' "/tmp/${label}-s${k}.log" \
               | tail -1)"
    if [ -z "$run_dir" ] || [ ! -f "$run_dir/metrics.json" ]; then
      echo "shard ${k} の run dir が取れない（/tmp/${label}-s${k}.log を見よ）" >&2
      tail -5 "/tmp/${label}-s${k}.log" >&2
      return 1
    fi
    dirs+=("$run_dir/metrics.json")
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
