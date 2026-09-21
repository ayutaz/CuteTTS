#!/usr/bin/env bash
# Copyright 2026 ayutaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# F2 → M3b → D2 → C1 → T3 を1インスタンスで順に回す（約21時間 / 約$4.8）。
#
#   HF_TOKEN=<read権限> bash scripts/batch_f2_m3b_d2_c1_t3.sh
#
# 環境変数:
#   HF_TOKEN   必須
#   ONLY       "f2 m3b d2 c1 t3" のうち回すものだけを空白区切りで指定（既定は全部）
#   SHARDS     評価の並列数。既定 3
#
# **段は独立に再開できる。** 途中で落ちても ONLY で続きだけ回せる。
# **GPUを使う処理は同時に走らせない**（学習時間が測れなくなる。execution-logの規約）。
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
SHARDS="${SHARDS:-3}"
ONLY="${ONLY:-f2 m3b d2 c1 t3}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN が要る" >&2
  exit 1
fi
cd "$WORKDIR"

python -c "import accelerate, transformers, soundfile, pyworld, pyopenjtalk" || {
  echo "依存が足りない。uv sync --all-extras" >&2; exit 1; }

MANIFEST="data/s1v2/manifests-v2/all_clustered.jsonl"
LATENTS="data/s1v2/latents-v2"
SPEAKERS="data/s1v2/speaker-v2"
LINES="$(wc -l < "$MANIFEST")"
[ "$LINES" = "286864" ] || {
  echo "manifestの行数が違う（期待 286864、実際 $LINES）" >&2; exit 1; }

# **現行最良は HF に無い**のでローカルから転送する（1.2 GB）。
# `inference/` 下ではなく直下に config.json がある形。
BEST="${BEST:-checkpoints/s1v2-fp32-30000}"
[ -f "$BEST/config.json" ] || {
  echo "現行最良の checkpoint が無い: $BEST（ローカルから転送する）" >&2; exit 1; }

wants() { case " $ONLY " in *" $1 "*) return 0;; *) return 1;; esac; }

# 評価setの音声を揃える（**忘れると学習の後で落ちる**。D1 の失敗）
[ -d data/eval/prosody_audio ] || HF_TOKEN="$HF_TOKEN" python -u scripts/fetch_prosody_audio.py
if wants f2 || wants m3b; then
  [ -d data/eval/prosody_ceiling_audio ] || \
    HF_TOKEN="$HF_TOKEN" python -u scripts/fetch_prosody_audio.py \
      --eval-set data/eval/prosody_ceiling_set_v1.json
fi

# ---------------------------------------------------------------- 共通
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

train_one() {
  local name="$1" steps="$2" batch="$3" extra="${4:-}"
  local out="checkpoints/${name}"
  if [ -d "$out/inference" ]; then
    echo "  既にある: $out"
    return
  fi
  echo "=== 学習 ${name}（steps=${steps} batch=${batch} ${extra}）==="
  python -u scripts/train_continual.py \
    --manifest "${MANIFEST_OVERRIDE:-$MANIFEST}" --latent-cache "$LATENTS" \
    --speaker-cache "$SPEAKERS" --param-dtype float32 \
    --steps "$steps" --batch-size "$batch" --lr 2e-5 --seed 42 \
    --save-every "$steps" --export-every-save --eval-every 1000 \
    $extra --out "$out"
}

drift() {
  python - "$1" <<'PYEOF'
import glob
import json
import sys

pattern = sys.argv[1]
for path in sorted(glob.glob("artifacts/s0-train/*/metrics.json")):
    payload = json.loads(open(path, encoding="utf-8").read())
    moved = payload.get("parameter_moved_ratio")
    settings = payload.get("settings") or {}
    out = str(settings.get("out", ""))
    if not moved or pattern not in out:
        continue
    worst = min(moved.values())
    mark = "" if worst > 0.99 else "   ← **凍結している。比較に使えない**"
    text = "  ".join(f"{k}={v:.4f}" for k, v in moved.items())
    print(f"  {out.split('/')[-1]:24s} steps={settings.get('steps')} "
          f"batch={settings.get('batch_size')}  {text}{mark}")
PYEOF
}

# ---------------------------------------------------------------- F2
if wants f2; then
  echo "########## F2: frontend の音への効果 ##########"
  # (1) 数詞set: frontend あり（修正後の順序）と無し
  eval_sharded evaluate_japanese_cer.py "$BEST" "f2-numeral-frontend" \
    "--eval-set data/eval/numeral_eval_set.json --expand-numerals --assign-yomi"
  eval_sharded evaluate_japanese_cer.py "$BEST" "f2-numeral-plain" \
    "--eval-set data/eval/numeral_eval_set.json"
  # (2) 抑揚set を frontend 込みで
  eval_sharded evaluate_prosody.py "$BEST" "f2-prosody-frontend" \
    "--eval-set data/eval/prosody_eval_set_v2.json --expand-numerals --assign-yomi"
  # (3) 参照音声を gol の日本語話者に替えて600文（喋り続けの条件差を分ける）
  JA_REF="data/eval/prosody_audio/$(python - <<'PYEOF'
import json
items = json.load(open("data/eval/prosody_eval_set_v2.json", encoding="utf-8"))["items"]
print(max(items, key=lambda i: i["reference_seconds"])["reference_wav"])
PYEOF
)"
  echo "  日本語参照: $JA_REF"
  eval_sharded evaluate_japanese_cer.py "$BEST" "f2-jaref" \
    "--eval-set data/eval/eval_set_v3.json --reference-audio $JA_REF"
fi

# ---------------------------------------------------------------- M3b
if wants m3b; then
  echo "########## M3b: 韻律転写の診断（参照＝同じ台詞の別テイク）##########"
  eval_sharded evaluate_prosody.py "$BEST" "m3b-transfer" \
    "--eval-set data/eval/prosody_transfer_set_v1.json"
fi

# ---------------------------------------------------------------- D2
if wants d2; then
  echo "########## D2: データ量の3点目（80h）##########"
  SUBSET="data/s1v2/manifests-v2/subset_80h.jsonl"
  [ -f "$SUBSET" ] || python -u scripts/build_data_subset.py \
    --manifest "$MANIFEST" --out "$SUBSET" --target-hours 80
  MANIFEST_OVERRIDE="$SUBSET" train_one "d2-80h" 30000 4
  drift "d2-"
  eval_sharded evaluate_japanese_cer.py "checkpoints/d2-80h/inference" \
    "d2-80h-cer" "--eval-set data/eval/eval_set_v3.json"
  eval_sharded evaluate_prosody.py "checkpoints/d2-80h/inference" \
    "d2-80h-prosody" "--eval-set data/eval/prosody_eval_set_v2.json"
fi

# ---------------------------------------------------------------- C1
if wants c1; then
  echo "########## C1: 計算量4倍（batch16 × 30,000 step）##########"
  train_one "c1-batch16-30k" 30000 16
  drift "c1-"
  eval_sharded evaluate_japanese_cer.py "checkpoints/c1-batch16-30k/inference" \
    "c1-cer" "--eval-set data/eval/eval_set_v3.json"
  eval_sharded evaluate_prosody.py "checkpoints/c1-batch16-30k/inference" \
    "c1-prosody" "--eval-set data/eval/prosody_eval_set_v2.json"
fi

# ---------------------------------------------------------------- T3
if wants t3; then
  echo "########## T3: flow_copies / condition_dropout ##########"
  # 基準線は T1 の lr=2e-5（10,000 step / batch 4 / flow_copies 4 / dropout 0.1）
  train_one "t3-flow2"  10000 4 "--flow-copies 2"
  train_one "t3-flow8"  10000 4 "--flow-copies 8"
  train_one "t3-drop00" 10000 4 "--condition-dropout 0.0"
  train_one "t3-drop03" 10000 4 "--condition-dropout 0.3"
  drift "t3-"
  for name in t3-flow2 t3-flow8 t3-drop00 t3-drop03; do
    eval_sharded evaluate_japanese_cer.py "checkpoints/${name}/inference" \
      "${name}-cer" "--eval-set data/eval/eval_set_v3.json"
  done
  for name in t3-flow2 t3-flow8 t3-drop00 t3-drop03; do
    eval_sharded evaluate_prosody.py "checkpoints/${name}/inference" \
      "${name}-prosody" "--eval-set data/eval/prosody_eval_set_v2.json"
  done
fi

echo
echo "完了: $ONLY"
echo "**主指標は読みCER。** 抑揚とアクセントは壊れていないかの確認として読む。"
echo "停止の健全性は scripts/summarize_stop_health.py で別に出す（G1）。"
