---
name: cutetts-ja-pipeline
description: Use when running, resuming, or debugging any CuteTTS Japanese continual-training phase (P0 baseline, P1b tokenizer, P1c VAE, P1d manifest, P1e latent cache, S0 training and CER evaluation, S1 preprocessing on vast.ai) in this repository — covers setup, the venv, GPU rules, running jobs on vast.ai, publishing preprocessed data to Hugging Face, exact commands with their inputs and outputs, and the traps that make these scripts silently produce wrong results.
---

# CuteTTS 日本語学習パイプラインの実行

P0/P1/S0/S1 スクリプトを実際に完走させるためのリファレンス。
実測値は [`docs/japanese-training/RESULTS.md`](../../../docs/japanese-training/RESULTS.md)、
フェーズ定義は `docs/japanese-training/08-execution-plan.md`。

## 実行前に必ず守る2点

1. **Python は必ず `.venv/Scripts/python.exe`。** リポジトリルートから実行する。
   システム既定は3.14で torch 2.5.1 が動かない（対応は3.9〜3.12）。
2. **ローカルGPUを使う処理は実行前にユーザーへ確認する。** バックグラウンドでも同じ。
   並列起動しない（1本ずつ直列）。所要時間とVRAMの見積もりを伝える。
   `.claude/settings.json` の PreToolUse フックが検出して確認を促す
   （`python <script>.py`、`--device cuda|auto`、`cutetts` CLI、`pytest -m gpu` が対象。
   ssh/scp/rsync/vastai の遠隔実行と git commit のメッセージ本文は除外）。
3. **重いGPU処理は vast.ai へ回す。** 起動前に費用見積もりを提示する。
   終わったらインスタンスを破棄する。**自分が作っていないインスタンスには触らない。**

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
| s0 | `build_eval_set.py` | 不要 | manifest | `data/eval/s0_eval_set.json` |
| s0 | `evaluate_japanese_cer.py` | **要** | eval set, model dir | `artifacts/s0-cer/<ts>/`（音声つき） |
| s0 | `train_continual.py` | **要** | `all_clustered.jsonl`, 両cache, `model/CuteTTS` | `checkpoints/s0/`, `artifacts/s0-train/<ts>/` |
| s0 | `diagnose_flow_loss.py` | **要** | 同上 + 比較したいcheckpoint | `artifacts/s0-diagnose/<ts>/` |
| s0 | `check_reference_following.py` | **要** | latent cache, checkpoint | `artifacts/s0-refcheck/<ts>/`（音声つき） |
| — | `benchmark_training_memory.py` | **要** | `model/CuteTTS` | VRAM/throughput の実測 |
| s1 | `measure_asr_floor.py` | **要**（`--build` は不要） | gol metadata + tars | `artifacts/asr-floor/<ts>/` |
| s1 | `s1_preprocess.sh` | **要** | HF（gol）+ `HF_TOKEN` | latent cache を HF へ upload |

**依存順序**: `prepare_japanese_manifest` → `cache_audio_latents` → `build_voice_clusters`
→ `train_continual` → `diagnose_flow_loss` / `evaluate_japanese_cer` / `check_reference_following`。
P0/P1b/P1c は互いに独立（P1cは音声zipだけあればよい）。
`build_eval_set` は manifest だけで動く（学習前に基準線を測るため先に作る）。

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

# voiceクラスタ（CPU）。既定0.70ではなく 0.92。linkage既定は complete
# voice_cluster_id（完全連結）と split_group_id（単連結）の両方を作り、
# split は split_group_id 単位で切り直す。漏れがあれば異常終了する
.venv/Scripts/python.exe scripts/build_voice_clusters.py --threshold 0.92

# --- S0 ---
# 評価set（CPU）。学習前に作り、基準線を測ってからゲート値を固定する
.venv/Scripts/python.exe scripts/build_eval_set.py

# 基準線CER（GPU）。学習前に必ず測る
python scripts/evaluate_japanese_cer.py --label baseline --device cuda

# 学習（GPU）。S0の通過実績がある設定
python scripts/train_continual.py --steps 3000 --batch-size 4 --lr 2e-5   --warmup 100 --save-every 1000 --group-key voice_cluster_id   --condition-dropout 0.1 --out checkpoints/s0 --device cuda

# 学習後（GPU）。この3本を必ず回す。1本ずつ直列
python scripts/diagnose_flow_loss.py --model-dir checkpoints/s0/inference --device cuda
python scripts/diagnose_flow_loss.py --model-dir model/CuteTTS --label base --device cuda
python scripts/evaluate_japanese_cer.py --model-dir checkpoints/s0/inference   --label trained --device cuda
python scripts/check_reference_following.py --model-dir checkpoints/s0/inference   --split dev-seen --references 4 --device cuda
```

**学習後の判定は `diagnose_flow_loss` を base と学習後の両方で回して比較する。**
学習ループが出す損失だけでは成否が判定できない（下の「静かに壊れる罠」を読むこと）。

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

## S1の前処理（vast.ai上で完結させる）

**215 GBをローカルへ落とさない。** インスタンス上でgolを直接取得し、
永続化するのは latent cache 約1.8 GB だけ。

```bash
# disk 300GB以上のインスタンスを立てる
vastai create instance <offer_id> --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel   --disk 320 --ssh --direct --label cutetts-s1-preprocess

# HFトークンを渡す（gated dataset の読み取りと成果物の書き込みに要る）
ssh ... 'mkdir -p ~/.cache/huggingface && cat > ~/.cache/huggingface/token'  # ローカルから流し込む

# DL → manifest → latent cache → cluster → upload を1本で
ssh ... "cd /workspace/CuteTTS && export HF_TOKEN=\$(cat ~/.cache/huggingface/token) && \n  setsid nohup bash scripts/s1_preprocess.sh > s1.log 2>&1 < /dev/null &"
```

所要 約9時間・**$2.8**（gol 5 game / 326時間 / 215 GB）。

**成果物**: [tts-dataset/cutetts-ja-latents](https://huggingface.co/datasets/tts-dataset/cutetts-ja-latents)
（public / **gated: manual**）。以降のインスタンスは復元するだけでよい。

```bash
hf download tts-dataset/cutetts-ja-latents --repo-type dataset --local-dir data/restored
```

**音声そのものは絶対に上げない。** latentは公開VAEで復元できる（往復CER中央値0.00%）
ので、音声と同じ扱いにする。上げてよいのは latent / speaker embedding / manifest だけ。

## vast.ai で回す

```bash
# bootstrap（リポジトリ取得 + 依存 + checkpoint + テスト）
ssh -p <port> root@<host> 'bash -s' < scripts/vastai_bootstrap.sh

# データ転送（55 MB。scp よりtar over sshが速い）
tar czf - data/cache/latents data/cache/speaker data/manifests/all_clustered.jsonl   | ssh -p <port> root@<host> 'cd /workspace/CuteTTS && tar xzf -'

# 実行は setsid + nohup で切り離し、ログファイルへ落とす
ssh -p <port> root@<host> "cd /workspace/CuteTTS && printf '%s
'   'export PYTHONIOENCODING=utf-8'   'python -u scripts/train_continual.py ... --device cuda' > job.sh   && setsid nohup bash job.sh > job.log 2>&1 < /dev/null & echo launched"

# 進捗は remote の tail -f を監視する
ssh -p <port> root@<host> 'tail -f -n +1 /workspace/CuteTTS/job.log'

# 終わったら artifact を回収（音声も含めるなら --exclude を外す）
ssh -p <port> root@<host> 'cd /workspace/CuteTTS && tar czf - artifacts/s0-*' | tar xzf -

# 破棄（確認プロンプトが出るので yes を渡す）
yes | vastai destroy instance <id>
```

**リモート実行で必ず守ること**

- `python -u` を使い、**リモート側で `| tail` や `| grep` に通さない**。
  パイプがバッファするため、プロセスが終わるまで出力が1行も来ない。
- `pkill -f <pattern>` を使わない。ssh のコマンド全文や親の `bash -c` に
  同じ文字列が含まれ、**自分自身を殺す**。実際に2回踏んだ。
- 転送速度は安定しない。570 MB が39 KB/s まで落ちた実績がある。
  **大きいcheckpointの回収を当てにしない。** 必要なら早めに落とす。

## 静かに壊れる罠（結果が出るのに間違っている）

| 症状 | 原因と対処 |
|---|---|
| **flow loss が 0.01 を下回る** | ほぼ確実に異常。flow matching は velocity を完全には当てられない。「常に0を出す予測器」の loss が約2.0なので、0.003 は決定係数0.998に相当し原理的に到達できない。`diagnose_flow_loss.py` で train / dev / 未学習base を同じ経路で測る |
| 学習は進むのにモデルが悪化する | `PairSampler.sample()` を step ごとに呼んでいる。**呼ぶたびにRNGを作り直す仕様**なので毎回同じペアが返る。`iter_pairs()` の stream を1本持って `islice` で引く。`tests/training/test_pair_stream.py` が検知する |
| stop loss が 0.0000 になる | 上と同じ原因の可能性が高い。少数sampleの丸暗記 |
| 評価CERが特定subsetだけ84%前後に張り付く | 評価文に語彙的内容が無い（`ふあぁぁぁ…` のような感情表現）。`build_eval_set.py` の `has_lexical_content()` が除外する。**CERを見て文を選び直さないこと** |
| CERを0%基準で読んでしまう | **人間の実音声でも同じ経路で10.4%出る**（`measure_asr_floor.py`）。TTS由来の誤りは実測値からこの床を引いて考える |
| golのgameが丸ごとmanifestに出ない | 大きいgameは `<game>_part1.tar` / `_part2.tar` に分割されている。tarのファイル名をそのまま game_id に使うと落ちる（S1では170時間・52%が消えていた） |
| reference追随が学習で悪化する | voiceクラスタに別の声が混ざっている。単連結は連鎖で巨大クラスタを作る。`--linkage complete`（既定）を使い、`build_voice_clusters.py` が出す「クラスタ内の最小cos」が閾値以上かを見る（R-014） |
| zero-shotがzero-shotでない | splitを `voice_cluster_id`（細かい）で切っている。**`split_group_id`（単連結・粗い）で切る**。`build_voice_clusters.py` は漏れがあれば異常終了する |
| `ms/step` が異常な値 | 表示のみのバグ。save のたびに `state.step` が進むため分母が壊れる（修正済み） |

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
| `AttributeError: 'GenerationResult' object has no attribute ...` | `tts.generate()` は tensor ではなく `GenerationResult` を返す。`.waveform` と `.sample_rate` を取る |
| CUDA generator エラー | CPU generator を CUDA device で使った。`objectives._randn` が吸収するが、新しい乱数経路を足すときは同じ扱いにする |

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

S0で生成した音声は `artifacts/s0-cer/*/samples/` と `artifacts/s0-refcheck/*/audio/` にある
（学習データそのものではなく生成物だが、同じ扱いにする）。
