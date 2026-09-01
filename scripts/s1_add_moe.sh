#!/usr/bin/env bash
# Copyright 2026 OPPO and Fudan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# S1に moe-speech-plus を混ぜて前処理する。
#
# S0（7.15時間・moe 30%）は CER 28.4% を達成したが、S1（265.7時間・golのみ）は
# 34.1% にとどまる。**量ではなく質の問題**を疑い、S0で効いていた可能性のある
# moe を大きなスケールで入れて検証する（R-015 の残件）。
#
# moe を選ぶ理由:
#   - スタジオ収録、BGM/SEなしと明記
#   - NISQA MOS と Silero VAD で品質フィルタ済み
#   - 話者あたり下限が保証されている（最小14.3分・121ファイル）
#
# 話者は **f0で層化サンプリング** した77話者（`data/s1_moe_speakers.json`）。
# 大きい順に取ると21話者で100時間に届くが、話者多様性と低音話者が犠牲になる
# （低音3人 vs 層化28人）。zero-shot voice cloning の評価に効く。
#
#   HF_TOKEN=<write権限> bash scripts/s1_add_moe.sh
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
MOE_REPO="ayousanz/moe-speech-plus"
UPLOAD_REPO="${UPLOAD_REPO:-tts-dataset/cutetts-ja-latents}"

export PYTHONIOENCODING=utf-8
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$WORKDIR"

: "${HF_TOKEN:?HF_TOKEN を設定すること}"

step() { echo; echo "=== $* ==="; date -u +"    %Y-%m-%dT%H:%M:%SZ"; }

step "0/5 事前確認"
df -h . | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
test -f data/s1_moe_speakers.json || { echo "話者リストが無い"; exit 1; }
python -c "import json;print('  選定話者:', len(json.load(open('data/s1_moe_speakers.json'))))"

step "1/5 moe zip（約25 GB / 77話者）"
mkdir -p data/raw/moe
python - <<'PY'
import json, os
from huggingface_hub import hf_hub_download
names = json.load(open("data/s1_moe_speakers.json"))
for index, name in enumerate(names, 1):
    target = f"data/raw/moe/{name}.zip"
    if os.path.exists(target):
        continue
    hf_hub_download("ayousanz/moe-speech-plus", f"{name}.zip",
                    repo_type="dataset", local_dir="data/raw/moe")
    if index % 10 == 0:
        print(f"  {index}/{len(names)}", flush=True)
print(f"  完了 {len(names)} 話者")
PY
du -sh data/raw/moe

step "2/5 moe だけの manifest"
# **golのtar（215 GB）は再取得しない。** gol_records は tar が無ければ
# 空を返すので、ここでは moe だけが manifest に入る。
# gol の latent は既に HF から取得済みで、後で結合する。
python -u scripts/prepare_japanese_manifest.py --skip-full-accounting --gol-tars /nonexistent --out-dir data/manifests_moe
wc -l data/manifests_moe/all.jsonl

step "3/5 moe の latent cache（既存のgol分はそのまま残る）"
python -u scripts/cache_audio_latents.py --manifest data/manifests_moe/all.jsonl
du -sh data/cache/latents data/cache/speaker

step "3.5/5 gol と moe の manifest を結合"
python - <<'PY'
import json
from pathlib import Path
gol = [json.loads(l) for l in open("data/manifests/all_clean.jsonl", encoding="utf-8")]
moe = [json.loads(l) for l in open("data/manifests_moe/all.jsonl", encoding="utf-8")]
merged = gol + moe
out = Path("data/manifests/all.jsonl")
with out.open("w", encoding="utf-8") as handle:
    for record in merged:
        print(json.dumps(record, ensure_ascii=False), file=handle)
print(f"  gol {len(gol):,} + moe {len(moe):,} = {len(merged):,}")
PY

step "4/5 voice cluster"
python -u scripts/build_voice_clusters.py --threshold 0.92
python - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("data/manifests/all_clustered.jsonl", encoding="utf-8")]
n = collections.Counter(); h = collections.Counter(); c = collections.defaultdict(set)
ds = collections.Counter()
for r in rows:
    n[r["split"]] += 1; h[r["split"]] += r["duration"]; c[r["split"]].add(r["voice_cluster_id"])
    if r["split"] == "train":
        ds[r["utterance_id"].split(":")[0]] += 1
for s in sorted(n):
    print(f"  {s:16s} {n[s]:8,} {h[s]/3600:7.1f}h  cluster {len(c[s]):4d}")
total = sum(ds.values())
print(f"  train の内訳: " + "  ".join(f"{k} {v/total*100:.0f}%" for k, v in ds.items()))
PY

step "5/5 アップロード（latentとmanifestのみ。音声は含まない）"
hf upload "$UPLOAD_REPO" data/cache/latents latents --repo-type dataset \
  --commit-message "latent cache: gol + moe 混合"
hf upload "$UPLOAD_REPO" data/cache/speaker speaker --repo-type dataset \
  --commit-message "speaker embeddings: gol + moe 混合"
hf upload "$UPLOAD_REPO" data/manifests manifests --repo-type dataset \
  --commit-message "manifests: gol + moe 混合"

echo
echo "完了。次は train_continual.py で 2000 step 学習し、CERを 34.1% と比べる。"
