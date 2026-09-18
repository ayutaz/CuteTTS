#!/usr/bin/env bash
# Copyright 2026 ayutaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# M4a: アクセント核をテキストに明示して学習する（vast.ai上で完結）。
#
# 抑揚は7軸すべてで動かず（lr / batch / 学習対象 / データ量19倍 / 計算量4倍 /
# flow_copies / condition_dropout）、**同じ台詞の別テイクを参照に渡しても
# 写さなかった**（R-041 / D-047）。モデルには韻律を受け取る経路が無いので、
# 入力に明示する。
#
#   HF_TOKEN=<read権限> bash scripts/m4a_accent_marks.sh
#
# 環境変数:
#   HF_TOKEN   必須
#   STEPS      既定 30000（現行最良と同じ計算量）
#   SHARDS     評価の並列数。既定 3
#   ONLY       "kana accent" のうち回すものだけ
#
# **2本を比べる。** (a) 仮名化のみ / (b) 仮名化+核記号。
# (b) だけだと「仮名化の効果」と「記号の効果」が混ざる。
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
STEPS="${STEPS:-30000}"
SHARDS="${SHARDS:-3}"
ONLY="${ONLY:-kana accent}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN が要る" >&2
  exit 1
fi
cd "$WORKDIR"

python -c "import accelerate, transformers, soundfile, pyworld, pyopenjtalk" || {
  echo "依存が足りない。pip install -e '.[ja,prosody,eval]'" >&2; exit 1; }

MANIFEST="data/s1v2/manifests-v2/all_clustered.jsonl"
LATENTS="data/s1v2/latents-v2"
SPEAKERS="data/s1v2/speaker-v2"
LINES="$(wc -l < "$MANIFEST")"
[ "$LINES" = "286864" ] || {
  echo "manifestの行数が違う（期待 286864、実際 $LINES）" >&2; exit 1; }

# **評価setの音声を先に揃える**（忘れると学習の後で落ちる。D1 の失敗）
[ -d data/eval/prosody_audio ] || HF_TOKEN="$HF_TOKEN" python -u scripts/fetch_prosody_audio.py

wants() { case " $ONLY " in *" $1 "*) return 0;; *) return 1;; esac; }

# 条件: 名前|frontend
RUNS=("kana|yomi" "accent|accent")

# ---------------------------------------------------------------- 学習
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name frontend <<< "$spec"
  wants "$name" || continue
  out="checkpoints/m4a-${name}"
  if [ -d "$out/inference" ]; then
    echo "  既にある: $out"
    continue
  fi
  echo "=== 学習 m4a-${name}（frontend=${frontend} / ${STEPS} step / batch 4）==="
  python -u scripts/train_continual.py \
    --manifest "$MANIFEST" --latent-cache "$LATENTS" --speaker-cache "$SPEAKERS" \
    --param-dtype float32 --frontend "$frontend" \
    --steps "$STEPS" --batch-size 4 --lr 2e-5 --seed 42 \
    --save-every "$STEPS" --export-every-save --eval-every 1000 \
    --out "$out"
done

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

# **学習と同じ frontend で評価する。** 食い違うと条件が変わる
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name frontend <<< "$spec"
  wants "$name" || continue
  echo "=== 評価 m4a-${name}: 読みCER（frontend=${frontend}）==="
  eval_sharded evaluate_japanese_cer.py "checkpoints/m4a-${name}/inference" \
    "m4a-${name}-cer" "--eval-set data/eval/eval_set_v3.json --frontend ${frontend}"
  echo "=== 評価 m4a-${name}: 抑揚・アクセント（frontend=${frontend}）==="
  eval_sharded evaluate_prosody.py "checkpoints/m4a-${name}/inference" \
    "m4a-${name}-prosody" \
    "--eval-set data/eval/prosody_eval_set_v2.json --frontend ${frontend}"
done

echo
echo "完了。**主指標はアクセント（句単位）**。基準線は 30,000 step / batch 4"
echo "（読みCER 13.38% / 輪郭 +0.125 / アクセント 45.3%）。"
echo "**アクセントが +5pt 以上 有意に動けば、韻律の条件づけは有効**（D-047 の次へ）。"
echo "読みCER が +1pt 以上悪化したら、kana と accent の差で原因を分ける。"
