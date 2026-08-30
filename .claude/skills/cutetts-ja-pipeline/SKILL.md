---
name: cutetts-ja-pipeline
description: Use when running, resuming, or debugging any CuteTTS Japanese continual-training phase (P0 baseline, P1b tokenizer, P1c VAE, P1d manifest, P1e latent cache) in this repository — covers setup, the venv, GPU rules, exact commands with their inputs and outputs, and the environment traps that make these scripts fail.
---

# CuteTTS 日本語学習パイプラインの実行

P0/P1 スクリプトを実際に完走させるためのリファレンス。
実測値は [`docs/japanese-training/RESULTS.md`](../../../docs/japanese-training/RESULTS.md)、
フェーズ定義は `docs/japanese-training/08-execution-plan.md`。

## 実行前に必ず守る2点

1. **Python は必ず `.venv/Scripts/python.exe`。** リポジトリルートから実行する。
   システム既定は3.14で torch 2.5.1 が動かない（対応は3.9〜3.12）。
2. **GPUを使う処理は実行前にユーザーへ確認する。** バックグラウンドでも同じ。
   並列起動しない（1本ずつ直列）。所要時間とVRAMの見積もりを伝える。
   `.claude/settings.json` の PreToolUse フックが検出して確認を促す。

コマンド例はbash記法。PowerShellで実行するなら行継続 `\` は使えない（1行にする）。

## セットアップ（未構築のとき）

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/Scripts/python.exe -e .
uv pip install --python .venv/Scripts/python.exe pytest pyyaml triton-windows
```

`uv` が無ければ `py -3.12 -m venv .venv` で作り、以降は
`.venv/Scripts/python.exe -m pip install ...` で代用できる。

checkpointとデータの取得には Hugging Face CLI が要る（`pip install -U huggingface_hub`、
`hf auth login` で認証）。両datasetは gated なのでアクセス権が必要。

```bash
hf download OPPOer/CuteTTS --local-dir ./model/CuteTTS
hf download OPPOer/CuteTTS-distill --local-dir ./model/CuteTTS-distill
```

データ（`data/` は gitignore。既定パスは各スクリプトの引数既定値）:

```text
data/raw/gol/metadata.tsv     # midralab/gol-dataset の metadata.tsv（1.68 GB）
data/raw/gol/tars/*.tar       # 同上の音声書庫（全602本で7 TB。必要な分だけ落とす）
data/raw/moe/*.zip            # ayousanz/moe-speech-plus の話者zip
data/raw/moe/info.csv         # 同上の話者一覧
```

入手元は HF の `midralab/gol-dataset` と `ayousanz/moe-speech-plus`。
どちらも gated で、全量は 7 TB + 152 GB あるため必要な分だけ取得する。
選び方や実測値は [`docs/japanese-training/data-inventory.md`](../../../docs/japanese-training/data-inventory.md)。

動作確認: `.venv/Scripts/python.exe -m pytest tests/training -q -p no:warnings`

## スクリプト一覧（入力 → 出力）

| phase | スクリプト | GPU | 入力 | 出力 |
|---|---|---|---|---|
| p0 | `reproduce_baseline.py` | **要** | `model/` | `artifacts/p0/<ts>/` |
| p1b | `analyze_japanese_tokenizer.py` | 不要 | `model/CuteTTS/tokenizer`, gol metadata | `artifacts/p1b/<ts>/` |
| p1c | `evaluate_japanese_vae.py` | **要** | VAE weight, `data/raw/moe/*.zip` | `artifacts/p1c/<ts>/` |
| p1d | `prepare_japanese_manifest.py` | 不要 | gol metadata + tars, moe zips | `data/manifests/{gol,moe,all}.jsonl`, `artifacts/p1d/<ts>/` |
| p1e | `cache_audio_latents.py` | **要** | `data/manifests/all.jsonl` + `model/CuteTTS`（VAE と Speaker Encoder） | `data/cache/{latents,speaker}/`, `artifacts/p1e/<ts>/` |
| p1d | `build_voice_clusters.py` | 不要 | `data/cache/speaker/`, manifest | `data/manifests/all_clustered.jsonl`, `artifacts/p1d-clusters/<ts>/` |

**依存順序**: `prepare_japanese_manifest` → `cache_audio_latents` → `build_voice_clusters`。
P0/P1b/P1c は互いに独立（P1cは音声zipだけあればよい）。

`--config` を取るのは **`analyze_japanese_tokenizer.py` だけ**。他はすべて個別フラグ。

## コマンド

```bash
# P0（GPU）。gate_passed は artifacts/p0/<ts>/metrics.json の summary.gate_passed
.venv/Scripts/python.exe scripts/reproduce_baseline.py --sampler-compile-mode auto
.venv/Scripts/python.exe scripts/reproduce_baseline.py --checkpoint CuteTTS-distill
jq '.summary' artifacts/p0/*/metrics.json   # gate_passed は true/false。error_cases が 0 で全ケース ok なら true

# P1b（CPU）
.venv/Scripts/python.exe scripts/analyze_japanese_tokenizer.py \
  --config configs/japanese/tokenizer-coverage.yaml

# P1c（GPU）。--config は無い
.venv/Scripts/python.exe scripts/evaluate_japanese_vae.py --max-speakers 10

# P1d manifest（CPU）。--skip-full-accounting で7.4M行集計を省略
.venv/Scripts/python.exe scripts/prepare_japanese_manifest.py

# P1e（GPU）。--limit は manifest の先頭N件。既存cacheはスキップするので再開可能
.venv/Scripts/python.exe scripts/cache_audio_latents.py --limit 300   # パイロット
.venv/Scripts/python.exe scripts/cache_audio_latents.py               # 本実行

# voiceクラスタ（CPU）。既定0.70ではなく 0.92
.venv/Scripts/python.exe scripts/build_voice_clusters.py --threshold 0.92
```

`--sampler-compile-mode` の有効値: `auto`（triton有無で判定）/ `eager` / `euler-only` / `full-sampler`。
tritonを入れられないなら `eager` で回避できる（RTFは悪化する）。

**ユーザーへ伝える見積もり（実測値）**

| スクリプト | スループット | peak VRAM |
|---|---|---:|
| `reproduce_baseline.py` | 1 checkpointあたり数分（model load 44〜56秒） | 2.05 GB |
| `cache_audio_latents.py` | **44.5× realtime**（音声8.13時間を658秒） | 2.48 GB |
| `evaluate_japanese_vae.py` | 10話者80発話で数分 | 1.5 GB |

gol全体（10,654時間）をP1eに通すと **約239 GPU時間 / latent cache 65.3 GB**。
これはローカルで回すには重すぎるので vast.ai を検討する。

パイロットと本実行は同じcacheへ書く。**衝突ではなく再開**として扱われる（既存IDはスキップ）。

## 環境の罠

| 症状 | 原因と対処 |
|---|---|
| `BackendCompilerFailed: Cannot find a working triton installation` | WindowsのPyTorchにtritonが同梱されない。`uv pip install --python .venv/Scripts/python.exe triton-windows`。検証は `.venv/Scripts/python.exe -c "import triton;print(triton.__version__)"`。未対処だと**distillが全ケース失敗する**。`--sampler-compile-mode eager` でも回避可 |
| `UnicodeEncodeError: 'cp932'` | stdoutがcp932。`PYTHONIOENCODING=utf-8` を付けるか、ファイルへUTF-8明示で書く |
| `ModuleNotFoundError` | システムPythonで実行している。`.venv/Scripts/python.exe` を使う。venv側で出るなら上のセットアップを実行 |
| `No usable checkpoint under ...` | `--checkpoint` にパスを渡した。**ディレクトリ名**（`CuteTTS-distill`）を渡す |
| manifestの件数が倍 | moe zipの `.bak.json` を除外していない（本体と同数ある） |
| クラスタが1つに潰れる | 閾値0.70はこの埋め込み空間で破綻する。0.92を使う |
| 途中で落ちた | `artifacts/<phase>/<ts>/` に metrics.json が無ければ未完。消してよい。cacheは再開されるので消さない |

## データの前提（推測で埋めない）

- **gol の sample_rate は 44.1/48 kHz 混在。** dataset単位で仮定せず実ファイルから読む
- **speaker ID は声の識別子ではない。** gol は `SHA-256(表示名)[:32]`、moe は `uuid4()`。
  split は voice クラスタ単位で行う（実データで gol×moe 跨ぎの同一声 cos 0.93 を検出済み）
- 総称ラベル話者（`？？？`『女の子』等91件）は複数の声が1 IDに混在する。除外する

閾値0.92の根拠: 話者内cos P5=0.608 が話者間cos P95=0.837 を下回り分布が重なる。
0.70では77話者中62が1クラスタになった。妥当性は `cluster_summary` の
`largest_cluster_size` が数個以内に収まるかで確認する。

## artifact

```text
artifacts/<phase>/<timestamp>/
├─ run.json / env.json / inputs.json / metrics.json
└─ samples/     # 音声
```

`artifacts/` と `data/` は gitignore 済み。
**`samples/` の音声はコミットも公開もしてはならない**
（MoeSpeech LICENSE: 音声ファイルを1つでも公開すれば再配布とみなす）。
provenance を問わず全音声に適用する — P0のサンプルも例外にしない。
ユーザーへローカルで見せるのは可。数値は `metrics.json` と `RESULTS.md` に残す。
