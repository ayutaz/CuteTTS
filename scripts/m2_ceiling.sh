#!/usr/bin/env bash
# Copyright 2026 ayutaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# M2: 抑揚・アクセントの**天井**を測る（vast.ai上で完結）。
#
# M1 では「同一文・同一話者の別テイクが無いので天井が測れない」と書いたが、
# gol の metadata には別テイクが 27,367組ある。人間 対 人間で測れば天井になる。
#
#   HF_TOKEN=<read権限> bash scripts/m2_ceiling.sh
#
# **setはローカルで作って転送する**（`build_retake_set.py` は metadata.tsv
# 1.68 GB を要るので、インスタンス上では作り直さない）。
# 音声は `fetch_prosody_audio.py` が **tarを1本ずつ落として消す**（約22 GB）。
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
SET="${SET:-data/eval/prosody_ceiling_set_v1.json}"
LABEL="${LABEL:-m2-ceiling}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN が要る" >&2
  exit 1
fi
cd "$WORKDIR"

[ -f "$SET" ] || {
  echo "無い: $SET（ローカルの build_retake_set.py で作って転送する）" >&2; exit 1; }

python -c "import pyworld, pyopenjtalk, torchaudio, soundfile" || {
  echo "依存が足りない。pip install -e '.[ja,prosody,eval]'" >&2; exit 1; }

# ---------------------------------------------------------------- 音声
echo "=== 音声を取り出す（tarは1本ずつ落として消す）==="
HF_TOKEN="$HF_TOKEN" python -u scripts/fetch_prosody_audio.py --eval-set "$SET"
df -h /workspace | tail -1

# ---------------------------------------------------------------- 測定
echo "=== 天井: 人間（テイクA）対 人間（テイクB）==="
python -u scripts/measure_prosody_ceiling.py \
  --eval-set "$SET" --label "$LABEL" --device cuda

echo
echo "完了。**出るのは天井の下限**（別テイクは感情や文脈が違いうる）。"
echo "現行モデルは 輪郭 +0.122 / アクセント（句単位）45.3%、床は -0.009。"
