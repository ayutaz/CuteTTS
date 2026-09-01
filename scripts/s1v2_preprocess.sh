#!/usr/bin/env bash
# Copyright 2026 OPPO and Fudan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# S1やり直し（v2）の前処理。**密度基準で選び直した8 game** を取得する。
#
# S1(v1)は「時間の長いgame」を選んで失敗した。305時間で学習しても
# S0(7.15時間)の 28.4% に届かず、最良は 30.8%。
# 原因は `PairSampler` のクラスタ密度依存（R-018）:
#
#   S1 v1 の 5 game: 発話/話者 median 239（66〜313）
#   S1 v2 の 8 game: 発話/話者 median 443
#
# 密なクラスタに絞ると 36.8% -> 30.8% に改善したので、
# 最初から密度の高い game を選べば S0 を超えられるはず、という設計。
#
#   HF_TOKEN=<write権限> bash scripts/s1v2_preprocess.sh
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/CuteTTS}"
UPLOAD_REPO="${UPLOAD_REPO:-tts-dataset/cutetts-ja-latents}"
GOL_REPO="midralab/gol-dataset"

export PYTHONIOENCODING=utf-8
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$WORKDIR"

: "${HF_TOKEN:?HF_TOKEN を設定すること}"

step() { echo; echo "=== $* ==="; date -u +"    %Y-%m-%dT%H:%M:%SZ"; }

step "0/6 事前確認"
df -h . | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
test -f data/s1v2_games.json || { echo "game リストが無い"; exit 1; }
python -c "import json;print('  選定game:', len(json.load(open('data/s1v2_games.json'))))"

step "1/6 metadata.tsv"
if [ ! -f data/raw/gol/metadata.tsv ]; then
  hf download "$GOL_REPO" metadata.tsv --repo-type dataset --local-dir data/raw/gol
fi

step "2/6 音声 tar（約279 GB / 8 game）"
# 大きい game は _partN に分割されている。実ファイル名をrepoから解決する
mkdir -p data/raw/gol/tars
python - <<'PY'
import json, os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
files = [f for f in api.list_repo_files("midralab/gol-dataset", repo_type="dataset")
         if f.endswith(".tar")]
want = json.load(open("data/s1v2_games.json"))
targets = [f for f in files if any(f.startswith(g) for g in want)]
print(f"  実ファイル {len(targets)} 本（分割込み）", flush=True)
for index, name in enumerate(targets, 1):
    if os.path.exists(f"data/raw/gol/tars/{name}"):
        print(f"  skip {name}", flush=True); continue
    print(f"  [{index}/{len(targets)}] {name}", flush=True)
    hf_hub_download("midralab/gol-dataset", name, repo_type="dataset",
                    local_dir="data/raw/gol/tars")
    os.system("df -h . | tail -1")
PY
du -sh data/raw/gol/tars

step "3/6 checkpoint"
if [ ! -f model/CuteTTS/config.json ]; then
  hf download OPPOer/CuteTTS --local-dir ./model/CuteTTS >/dev/null
fi

step "4/6 manifest"
python -u scripts/prepare_japanese_manifest.py --skip-full-accounting
wc -l data/manifests/all.jsonl

step "5/6 latent cache"
python -u scripts/cache_audio_latents.py --manifest data/manifests/all.jsonl
du -sh data/cache/latents data/cache/speaker

step "6/6 voice cluster + 密度の確認"
python -u scripts/build_voice_clusters.py --threshold 0.92
python - <<'PY'
import json, collections, statistics
rows = [json.loads(l) for l in open("data/manifests/all_clustered.jsonl", encoding="utf-8")]
n = collections.Counter(); h = collections.Counter(); c = collections.defaultdict(set)
for r in rows:
    n[r["split"]] += 1; h[r["split"]] += r["duration"]; c[r["split"]].add(r["voice_cluster_id"])
for s in sorted(n):
    print(f"  {s:16s} {n[s]:8,} {h[s]/3600:7.1f}h  cluster {len(c[s]):4d}")
tr = [r for r in rows if r["split"] == "train"]
cl = collections.Counter(r["voice_cluster_id"] for r in tr)
sizes = sorted(cl.values())
print(f"\n  クラスタ内発話数: median {statistics.median(sizes):.0f}  "
      f"p25 {sizes[len(sizes)//4]}  p75 {sizes[len(sizes)*3//4]}")
print(f"  >=40発話のクラスタ: {sum(1 for v in sizes if v >= 40)}/{len(sizes)}")
print(f"  参考: S0 median 44 / S1v1 median 5")
PY

step "アップロード（latentとmanifestのみ。音声は含まない）"
hf upload "$UPLOAD_REPO" data/cache/latents latents-v2 --repo-type dataset \
  --commit-message "S1 v2: 密度基準で選んだ8 game の latent cache"
hf upload "$UPLOAD_REPO" data/cache/speaker speaker-v2 --repo-type dataset \
  --commit-message "S1 v2: speaker embeddings"
hf upload "$UPLOAD_REPO" data/manifests manifests-v2 --repo-type dataset \
  --commit-message "S1 v2: manifests"

echo
echo "完了。次は train_continual.py を 3000 step で回し、CER を 28.4% と比べる。"
