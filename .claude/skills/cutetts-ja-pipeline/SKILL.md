---
name: cutetts-ja-pipeline
description: Use when running, resuming, or debugging any CuteTTS Japanese continual-training phase in this repository (P0 baseline, P1b tokenizer, P1c VAE, P1d manifest, P1e latent cache, S0/S1 training, CER evaluation with the v3 600-sentence set, forgetting, streaming, listening kits, numeral reading J2, the completed J3 reading assignment, M1 prosody and accent measurement, the completed T1 learning-rate sweep and T2 batch-size/trainable-module sweep, the completed F1 frontend alignment and M2 prosody-ceiling measurement, plus the planned D1 / C1 phases) — covers setup, the venv, GPU rules, running jobs on vast.ai, publishing preprocessed data to Hugging Face, exact commands with their inputs and outputs, the fp32 master-weight requirement that made training work at all, and the measurement defects and silent failures that repeatedly produced wrong conclusions.
---

# CuteTTS-jp 学習パイプラインの実行

P0/P1/S0/S1 スクリプトを実際に完走させるためのリファレンス。
実測値は [`docs/japanese-training/RESULTS.md`](../../../docs/japanese-training/RESULTS.md)、
フェーズ定義は `docs/japanese-training/execution-log.md`。

## 学習と評価で必ず守ること（これを外すと結論が壊れる）

1. **`--param-dtype float32` を必ず付ける（既定）。** 公開checkpointは
   `qwen_backbone` / `locenc` が bf16 で、`AdamW` が直接更新すると lr=2e-5 の
   更新量が丸め幅を下回り、**3,000 step 回しても backbone は 3.68% しか動かない**（R-020）。
   S0/S1 の19回はすべてこの状態で、**実質 head だけを学習していた**。
   毎runの metrics にある `parameter_moved_ratio` が 100% 付近かを確認する。
2. **評価は v3（`data/eval/eval_set_v3.json`、in_domain 600文）を使う。**
   旧 v2 は30文で、検出できる最小差が **6.9pt**。S1で比較した2〜3ptの差は
   すべてその下にあり、**step数の順位すら取り違えていた**。
   差は必ず `scripts/summarize_eval_runs.py --compare A B` で信頼区間を出す。
3. **CERの読み方を間違えない。** 素のCERは
   (a) 打ち切り生成を発音誤りとして数え（R-021）、
   (b) `1280円` と `千二百八十円` を不一致とし（R-023）、
   (c) **ASRが選んだ表記の違いを誤りと数える**（R-029。`なにも` と話しても
   ASRは `何も` と書く）。
   打ち切りは `mean_excluding_truncated`、**表記は `cer_reading`（読みCER）** を見る。
   読みCERは数字も吸収するので `cer_numeric` を包含する。
   **仮名を入力に含む比較（J2 / J3）は必ず読みCERで見ること。**
   `summarize_eval_runs.py --metric cer_reading --compare A B`。
   人間の実音声の床すら 10.42% → **5.59%** と半分近くが表記だった。

### 評価は分割して並列に回す

**生成は `batch=1` の自己回帰なのでGPUが埋まらない**（単独で使用率15〜71%、
消費電力45〜82W / TDP285W。3並列で98%）。1プロセスのVRAMは抑揚4.2GiB /
CER5.4GiB なので3並列が載る。

    for k in 1 2 3; do
      python scripts/evaluate_prosody.py --shard $k/3 --label x-s$k ... &
    done; wait
    python scripts/evaluate_prosody.py --merge <s1>,<s2>,<s3> --label x --eval-set ...

**分割結果が一括と一致することは検証済み**（抑揚9文で行9/9、CER8文で
集計10/10・行8/8）。集計は `prosody.summarize_run` /
`evalstats.summarize_subsets` の1箇所にある。

**`--no-warmup` を付けないこと。** プロセス内の最初の生成だけ結果が違う
（同じ文が1件目だと幅21.99、2件目以降だと13.60）。捨て生成で揃えないと
結合が一括と合わない。

### 現在の最良checkpoint

**`checkpoints/m4a-accent/inference/`**（M4a。全文片仮名 + アクセント核の記号で
30,000 step。ローカル退避済み、`strict=True` でロード確認済み）。

読みCER **7.58%**（従来の `s1v2-fp32-30000` は 13.38%。-5.80pt、有意）。
**人間の床 5.59% まで 2.0pt。**

**推論時も同じ frontend が要る**（`--frontend accent`）。素の漢字テキストを
渡すと分布外になる。

**記号を外した `kana_full`（7.24%）とは読みCERで区別できない**
（+0.34pt、95%CI [-0.25, +1.00]）。それでも `accent` を採るのは、
**記号を外すと同音異義のアクセントを一切作り分けられなくなる**から
（A1 / R-048。対の区別 75.0% → **0.0%**、対辞書 45.5% → 37.5%）。
**読みCERは原理的にこの誤りを見ない**（`箸` も `橋` も読みは `ハシ`）。

### 韻律転写（M4h）

**`checkpoints/m4h-prosody/inference/`**（325.9h / 30,000 step / 運用点）。
**TTS の重みと F0 条件づけが揃っているので、そのまま使える。**

* 設定: frontend `accent` / 差込 `head` / 先読み 4 patch / 位置あり /
  `--f0-dropout 0.1` / 条件づけは線形・`--f0-lr 2e-4`
* **同じ台詞の人間の読みから F0 を取り出して渡すと輪郭が写る**
  （利得 **+0.104**、有意。R-060）
* **条件を渡さなくても読みは壊れない**（読みCER **7.12%**。
  `m4a-accent` の 7.58% と有意差なし）
* **アクセント核は写らない**（oracle 対 条件なしで +0.37pt、有意差なし）。
  **間違った F0 を渡すと逆に壊れる**（-3.78pt、有意）ので、
  **参照は必ずその文の読みにする**

**条件の効果は `oracle` 対 `none` で見る。** `mismatch` 比では
**mismatch が壊れるだけで「改善」に見えてしまう**（M4g の基準の失敗。R-060）。

ロード確認済み（CPU / `strict=True`）。条件づけは metadata から復元される:
`inject=head` / `lookahead=4` / `position=True` / 17 → 256（4,608パラメータ）。

**使い方**:

```bash
python scripts/synthesize_japanese.py   --model-dir checkpoints/m4h-prosody/inference   --text "..."   --reference-audio <声質の参照>   --prosody-reference <**同じ台詞を読んだ**音声>   --output out.wav
```

`--reference-audio`（声質）と `--prosody-reference`（韻律）は**別物**で、
両方渡せる。条件の組み立ては `training.f0.prosody_generate_kwargs` にあり、
**`evaluate_prosody.py` と同じ実装を使う**（先読み・位置・差込先は
conditioner の metadata から取るので、2箇所に書くと学習・評価・合成で
別の条件になる）。

**参照は必ずその文の読みにする。** 別の文の F0 を渡すとアクセントが
**-3.78pt 壊れる**（有意。R-060）。

**条件づけの重みが無い checkpoint に `--prosody-reference` を渡すと落ちる。**
黙って素通りさせると、写っていないのに写ったつもりになる。

**本番の F0 cache は `data/s1v2/f0-v2/` にローカル退避してある**（79 MB /
252,415発話）。**作るのに CPU で 7.5時間かかる**（45.5× 実時間）ので、
325.9h で条件づけを学習し直すときは**必ずこれを使う**。
`cache_f0_targets.py` は既存の id を飛ばすので、そのまま `--out` に渡せばよい。

**M4c 期の条件の重みは `checkpoints/m4c-conditioners/` にある**
（`head4.safetensors`: 17.9h / 10,000 step）。**単体では使えない** —
一緒に学習した TTS の重みを持ち帰っていない。実験の記録として残してある。

他に残してある checkpoint:

| | 読みCER | A1 対の区別 | frontend | 用途 |
|---|---:|---:|---|---|
| `c2-accent/inference/` | **7.25%** | — | `accent` | **計算量4倍**（48万サンプル）。`m4a-accent` と**有意差なし**（-0.33pt、95%CI [-1.04, +0.35]。R-055）。点推定は最良。ローカル退避済み・ロード確認済み |
| `m4a-kanafull/inference/` | **7.24%** | 0.0% | `kana_full` | **依存最小**（アクセント辞書が要らない）。自己反復も 0.3% で最小。ローカル退避済み・ロード確認済み |
| `m4a-kanafull-clean` | 7.29% | — | `kana_full_clean` | 語境界の長音化を直した版。**効果なし**（R-049） |
| `m4a-shuffled` | 7.40% | 61.7% | `accent_shuffled` | 偽の核の対照（R-046） |
| `m4a-kana` | 11.64% | 51.7% | `yomi` | J3+J2 だけの対照 |
| `c1-batch16-30k/inference/` | 11.98% | — | なし | 計算量4倍（R-043） |
| `s1v2-fp32-30000/` | 13.38% | — | なし | 3指標の基準線（M1 / M2 / T1 / T2 / D1 / D2） |

| 指標 | base | **現行**（`m4a-accent`） | 人間 / 上限 |
|---|---:|---:|---:|
| 素CER（v3 600文） | 35.86% | **16.39%** | 10.42% |
| **読みCER（`--frontend accent`）** | — | **7.58%** | 床 **5.59%** |
| 読みCER（frontend無し・比較用） | 30.94% | 13.38% | 5.59% |
| 輪郭の相関（240文） | +0.024 | **+0.091** | **上限 +0.367**（codec） |
| アクセント核（対人間） | 35.2% | **43.6%** | 辞書 44.8% / **上限 75.0%** |

盲検A/Bで 15/18（83%、p=0.0038）と知覚できる差がある。
**3指標すべてで学習が有意に効いているが、どれも人間に届いていない。**

**評価は既定で frontend を適用しない。** 現行最良の値が要るときは
`evaluate_japanese_cer.py --frontend accent` を付ける。
**frontend 込みの値と frontend 無しの値を混同しない。**
**checkpoint どうしの比較では付けずに揃える**（過去の値と比較するため）。

**frontend は `yomi.apply_frontend` 経由で掛ける（J3 → J2 の順）。**
順序を逆にすると J3 が J2 の仮名列を再解釈して漢数字を復活させる
（`千二百八十円` → `せんにひゃく八ジュウエン`。R-039）。
小書き仮名 `ゅ` が byte-fallback なのが引き金。

## 実行環境とGPUの規約

0. **`uv sync --all-extras` だけで環境が揃う。`pip` / `uv pip` は使わない**
   （ユーザー指示。2026-09-21）。`.venv`（Python 3.12）の作成、**cu121 版の
   torch 2.5.1**、ja / prosody / eval / dev のすべてが入る。
   **torch を PyTorch の index から取る設定は `pyproject.toml` に宣言済み**
   （`[[tool.uv.index]] pytorch-cu121` + `[tool.uv.sources]`）なので、
   `uv sync` が PyPI の CPU 版で上書きすることはない。
   vast.ai のイメージが Python 3.11 でも uv が 3.12 を用意する。
   依存を足すときは **`uv add <package>`**。
1. **Python は `uv run python`**（ユーザー指示。2026-09-19）。リポジトリルートから実行する。
   ~~`--no-sync` を外さない~~ → **不要になった**（2026-09-21）。index を宣言したので
   同期し直しても cu121 版のまま。中身は `.venv/Scripts/python.exe` で同じ。
   システム既定は3.14で torch 2.5.1 が動かない（対応は3.9〜3.12）。
2. **GPUはすべて vast.ai を使う。ローカルGPUは使わない**（D-023 / D-024、2026-09-15）。
   起動前に**費用見積もりを提示する**。終わったらインスタンスを破棄する。
   **自分が作っていないインスタンスには触らない。**
   APIキーは `~/.config/vastai/vast_api_key` にあるが**残高0**。入金はユーザーが行う。
   事前確認の PreToolUse フックは削除した（毎回の承認が作業を止めるため）。
3. **同じインスタンス上で評価を分割並列するのは可**（上の「評価は分割して
   並列に回す」を参照）。T1 はこれで 4水準を約6.5時間 / $0.80 で回した。T2 は3条件を約6時間で回した。

コマンド例はbash記法。PowerShellで実行するなら行継続 `\` は使えない（1行にする）。

## セットアップ（未構築のとき）

```bash
uv sync --all-extras
```

**これだけ。** `.venv`（Python 3.12）・cu121 版 torch 2.5.1・`[ja]`（読み付与）・
`[prosody]`（抑揚）・`[eval]`（CER）・`[dev]`（テスト）・`triton-windows`（Windowsのみ）
がすべて入る。**`uv.lock` があるので毎回同じ版が入る。**

`[ja]` は `pyopenjtalk-plus`（+ 必須依存の sudachipy / sudachidict-core、計332MB）。
4候補を実測比較して選んだ（D-035）。**`[onnxruntime]` extra は入れない**
— 有効化しても14語すべて結果が同一で、架空の人名も直らなかった。
`pyopenjtalk` 本家は Windows wheel が無くビルドが失敗する。**fork の `-plus` を使う。**

checkpointとデータの取得には Hugging Face CLI が要る。**`uv sync` で入る**ので、
`uv run hf auth login` で認証するだけでよい。両datasetは gated なのでアクセス権が必要。

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
| s1 | `summarize_eval_runs.py` | 不要 | `artifacts/**/metrics.json` | 横断集計・対応のある検定（信頼区間つき） |
| g1 | `summarize_stop_health.py` | 不要 | `artifacts/**/s0-cer/*/metrics.json` | **喋り続け・自己反復・打切**の率（生成をやり直さない）。`--compare A B` で検定 |
| s1 | `evaluate_forgetting.py` | **要** | checkpoint | `artifacts/forgetting/<ts>/`（英語WER / 中国語CER） |
| j2 | `build_numeral_eval_set.py` | 不要 | — | `data/eval/numeral_eval_set.json`（200文・桁1〜7） |
| j2 | `synthesize_japanese.py` | **要** | checkpoint, text | wav。**J2（読み展開）が既定で有効** |
| m1 | `build_listening_kit.py` | **要**（`--html-only` は不要） | checkpoint, eval set | `artifacts/listen-kit/`（盲検A/B + アンカー + 書き出し） |
| j3 | `build_yomi_eval_set.py` | 不要 | gol metadata, 学習manifest | `data/eval/yomi_eval_set.json`（300文。J3が置換する語を含む文だけ） |
| m1 | `build_prosody_set.py` | 不要 | gol metadata + tars, 学習manifest | `data/eval/prosody_eval_set_v2.json`（240文/53話者）+ 音声 |
| m1 | `fetch_prosody_audio.py` | 不要 | 凍結済みの評価set + `HF_TOKEN` | 評価setに必要な音声だけを gol から取り出す（**setは作り直さない**） |
| m1 | `evaluate_prosody.py` | **要** | checkpoint, 評価set | `artifacts/prosody/<ts>/`（抑揚の幅・輪郭の相関・アクセント核） |
| m2 | `build_retake_set.py` | 不要 | gol metadata + game_id | `data/eval/prosody_ceiling_set_v1.json`（別テイク200組/70話者）。**同一長の組は落とす** |
| m2 | `measure_prosody_ceiling.py` | **要** | 別テイクset + 音声 | 人間 対 人間の天井（**生成しない**）。行の形は `evaluate_prosody.py` と同じ |
| m2 | `m2_ceiling.sh` | **要** | 凍結set + `HF_TOKEN` | 音声取得から天井測定までを vast.ai 上で完結 |
| d1 | `build_data_subset.py` | 不要 | 学習manifest | 指定時間の部分集合（**trainだけをクラスタ単位で絞る**。devは触らない） |
| d1 | `d1_data_scale.sh` | **要** | HF（latent cache）+ `HF_TOKEN` | 部分集合づくり → 30,000 step 学習 → CER・抑揚評価 |
| t1 | `t1_lr_sweep.sh` | **要** | HF（latent cache）+ `HF_TOKEN` | vast.ai上で学習4水準 + 評価を完結（評価は `--shard` 並列） |
| t2 | `t2_capacity_sweep.sh` | **要** | 同上 | batch size / 学習対象の3条件を学習 + 評価（`train_continual.py --trainable` で head凍結）。`SKIP_TRAIN=1` で評価だけ再開 |
| m4a | `m4a_one_condition.sh` | **要** | HF（latent cache）+ `HF_TOKEN` | **frontend 1条件を学習して測る汎用ランナー**。`NAME` / `FRONTEND` / `BATCH` / `STEPS` を環境変数で渡す |
| m4a | `m4a_factor_split.sh` | **要** | 同上 | 要因分解（`kana_full` / `accent_shuffled` / `accent_clean`）をまとめて回す |
| m4a | `m4a_aligner_evals.sh` | **要** | 同上 | **アライナを使う評価だけを2 shard**で回す（3本は CUDA OOM で文が落ちる） |
| a1 | `build_accent_pair_set.py` | 不要 | gol metadata | `data/eval/accent_pair_set_v1.json`（**片仮名は同一で核だけ違う**60組） |
| a1 | `evaluate_accent_pairs.py` | **要** | 最小対set, checkpoint | 対の区別（`pair_differentiated`）と対辞書の正解（`pair_correct`）。**seed 3回の多数決** |
| m4c | `cache_f0_targets.py` | **要** | latent cache + VAE | latent を decode して F0 を作る（**音声は置いていない**ので decode が要る）。GPU と CPU を並べる |
| m4c | `m4c_validate.sh` | **要** | 同上 + `HF_TOKEN` | **経路の検証**（17.9h / 10,000 step）。`LOOKAHEAD` / `INJECT` / `POSITION` / `DROPOUT` を環境変数で渡す |
| m4c | `m4c_full.sh` | **要** | 同上 | 本番（325.9h）。**検証が通ってから回す** |
| m4c | `diagnose_f0_conditioning.py` | **要** | checkpoint + F0 cache | **モデルが条件を使っているか**を flow loss で見る（真の F0 / ゼロ / 順を入れ替え） |
| m4c | `calibrate_contour_metric.py` | **要** | 評価set + 音声 | **輪郭の指標の上限**を4段の階段で測る（同一音声 / VAE往復 / 別テイク / 別の文） |
| m4c | `calibrate_accent_metric.py` | **要** | 同上 | アクセント核の指標の上限（**往復 75.0% > 天井 64.5%** なので指標として使える） |
| m4c | `compare_prosody_runs.py` | 不要 | `artifacts/prosody/*/metrics.json` | 抑揚の**対応のある比較**（テキストで対応付け・信頼区間つき） |
| m4b | `train_f0_predictor.py` | 不要（CPUで足りる） | F0 cache + manifest | テキスト → 長さを正規化した輪郭。**辞書・床と比べる** |

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

# --- 評価set ---
# in_domain 600文（CPU、数分）。**学習manifestをテキストでも照合して漏洩0を確認する**
.venv/Scripts/python.exe scripts/build_eval_set.py   --train-manifest data/manifests/all_clustered.jsonl   --in-domain-count 600 --scan-limit 4000000   --out data/eval/eval_set_v3.json --seed 20260903

# 数詞専用（CPU、即時）。J2の効果を測るため
.venv/Scripts/python.exe scripts/build_numeral_eval_set.py --count 200

# --- 学習 ---
# 基準線CER（GPU）。学習前に必ず測る
python scripts/evaluate_japanese_cer.py --model-dir model/CuteTTS   --eval-set data/eval/eval_set_v3.json --label v3-base --device cuda

# 学習（GPU）。**--param-dtype float32 が必須**（既定。R-020）
# 30,000 step で v3 20.10%。収穫逓減あり（20,000→30,000 は -0.96pt）
python scripts/train_continual.py --steps 30000 --batch-size 4 --lr 2e-5   --warmup 100 --group-key voice_cluster_id --condition-dropout 0.1   --param-dtype float32 --save-every 10000 --export-every-save   --eval-every 10000 --out checkpoints/run --device cuda

# 旧挙動を再現したいときだけ（A/B検証用）
#   --param-dtype checkpoint

# --- 学習後（GPU）。1本ずつ直列 ---
python scripts/evaluate_japanese_cer.py --model-dir checkpoints/run/inference   --eval-set data/eval/eval_set_v3.json --label v3-trained --device cuda
python scripts/evaluate_japanese_cer.py --model-dir checkpoints/run/inference   --eval-set data/eval/numeral_eval_set.json --label numerals --expand-numerals
python scripts/check_reference_following.py --model-dir checkpoints/run/inference   --split dev-zero-shot --references 4 --device cuda
python scripts/evaluate_forgetting.py --model-dir checkpoints/run/inference --label trained
python scripts/reproduce_baseline.py --model-root checkpoints/run --checkpoint inference

# --- 集計（CPU）。**点推定の順位ではなく信頼区間で判断する** ---
.venv/Scripts/python.exe scripts/summarize_eval_runs.py
.venv/Scripts/python.exe scripts/summarize_eval_runs.py --compare v3-base v3-trained

# --- 合成（GPU）。J2が既定で有効 ---
python scripts/synthesize_japanese.py --model-dir checkpoints/s1v2-fp32-30000   --text "価格は千二百八十円、消費税込みです。" --output out.wav

# --- 聴取キット（GPU。HTMLだけ作り直すなら --html-only でCPU） ---
python scripts/build_listening_kit.py --model-dir checkpoints/run/inference   --out artifacts/listen-kit
.venv/Scripts/python.exe scripts/build_listening_kit.py --html-only --out artifacts/listen-kit
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

# **評価setを転送する**（`data/` は gitignore なので bootstrap では入らない）。
# 忘れると `fetch_prosody_audio.py` が
# `FileNotFoundError: data/eval/prosody_eval_set_v2.json` で落ちる（実際に踏んだ）
tar czf - data/eval/*.json | ssh -p <port> root@<host> 'cd /workspace/CuteTTS && tar xzf -'

# **未pushのコミットがあるときは作業ツリーごと送る**（bootstrap は clone するだけ）
git archive --format=tar HEAD | gzip | ssh -p <port> root@<host> 'cd /workspace/CuteTTS && tar xzf -'

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

**`git archive` は `core.autocrlf=true` だと CRLF に変換する。**
shellスクリプトを送ると `set: pipefail: invalid option name` で落ちる
（bash が `pipefail
` を読む）。**`git -c core.autocrlf=false archive` を使う。**
Pythonは CRLF でも動くので、**shellスクリプトだけが静かに壊れる。**

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
| **条件づけがまったく効かない** | teacher forcing では **patch i-1 の真の latent が入力にある**ので、その patch の F0 は履歴から予測できる。真の F0 を渡しても flow loss が 0.2% しか動かず、**順を入れ替えたものと区別が付かない**（R-051）。**先読み（この先 K patch）を渡す**と効く |
| **条件の効果を `none` と比べて過大評価する** | 条件づけを入れるとモデルが依存し、**条件が外れると素より悪くなる**。`none` と比べると「条件があること」の効果が混ざる。**正しい基準線は `mismatch`**（別の文の条件を与えたもの） |
| **条件を落として学習したら効果ごと消えた** | 依存と能力は同じもの。`--f0-dropout 0.5` では条件を使わなくなる（+0.018、有意差なし）。**0.1 なら効果を91%保つ**（R-057） |
| **輪郭の相関が 0.4 程度で頭打ちになる** | **指標の上限**。同じ発話を VAE で往復させるだけで +0.367 まで落ちる（R-050）。生成音声は必ず VAE を通るので**それ以上は出ない**。M2 の「天井 +0.382」は人間の再現度ではなく上限そのもの |
| **`speaker_adaln` 経由の条件に勾配が流れない** | zero-init のままだと gate が 0 で枝ごと消え、`W_gate = 0` なので speaker へ勾配が来ない。公開checkpointでは学習済み（\|w\| 平均 2.06e-02）なので実機では問題ないが、**tiny model のテストでは明示的に学習済みへ見立てる**必要がある |
| **flow loss が 0.01 を下回る** | ほぼ確実に異常。flow matching は velocity を完全には当てられない。「常に0を出す予測器」の loss が約2.0なので、0.003 は決定係数0.998に相当し原理的に到達できない。`diagnose_flow_loss.py` で train / dev / 未学習base を同じ経路で測る |
| 学習は進むのにモデルが悪化する | `PairSampler.sample()` を step ごとに呼んでいる。**呼ぶたびにRNGを作り直す仕様**なので毎回同じペアが返る。`iter_pairs()` の stream を1本持って `islice` で引く。`tests/training/test_pair_stream.py` が検知する |
| stop loss が 0.0000 になる | 上と同じ原因の可能性が高い。少数sampleの丸暗記 |
| 評価CERが特定subsetだけ84%前後に張り付く | 評価文に語彙的内容が無い（`ふあぁぁぁ…` のような感情表現）。`build_eval_set.py` の `has_lexical_content()` が除外する。**CERを見て文を選び直さないこと** |
| CERを0%基準で読んでしまう | **人間の実音声でも同じ経路で10.4%出る**（`measure_asr_floor.py`）。TTS由来の誤りは実測値からこの床を引いて考える |
| golのgameが丸ごとmanifestに出ない | 大きいgameは `<game>_part1.tar` / `_part2.tar` に分割されている。tarのファイル名をそのまま game_id に使うと落ちる（S1では170時間・52%が消えていた） |
| reference追随が学習で悪化する | voiceクラスタに別の声が混ざっている。単連結は連鎖で巨大クラスタを作る。`--linkage complete`（既定）を使い、`build_voice_clusters.py` が出す「クラスタ内の最小cos」が閾値以上かを見る（R-014） |
| zero-shotがzero-shotでない | splitを `voice_cluster_id`（細かい）で切っている。**`split_group_id`（単連結・粗い）で切る**。`build_voice_clusters.py` は漏れがあれば異常終了する |
| `ms/step` が異常な値 | 表示のみのバグ。save のたびに `state.step` が進むため分母が壊れる（修正済み） |
| **loss は下がるのにデータを増やしても効かない** | bf16パラメータを `AdamW` が直接更新している。lr=2e-5 の更新量が丸め幅を下回り、**backboneが3.68%しか動かない**（R-020）。`--param-dtype float32`。metricsの `parameter_moved_ratio` が100%付近かを見る |
| **2〜3ptの差で方針を決めてしまう** | v2（30文）の検出限界は6.9pt。**step数の順位を取り違えた実績がある**。v3（600文）を使い、`summarize_eval_runs.py --compare` で信頼区間を出す |
| **打ち切り生成をCERに数えてしまう** | `max_decode_length`（400 patch = 64.0秒）張り付きは停止の失敗で、発音誤りではない。S0系は0件、S1系は1〜7件あり、除外すると差がほぼ消えた（R-021）。`mean_excluding_truncated` を見る |
| **数詞のCERが改善を隠す** | ASRは `1280円` と書くが参照は `千二百八十円`。**正しく読めるほど素のCERは悪化する**（R-023）。`cer_numeric`（数字正規化CER）を見る。J2の効果は素のCERで +3.1pt、正規化CERで -11.80pt と符号が逆になる |
| **仮名を入力すると素のCERが悪化する** | 仮名で入力すると **ASRも仮名で書き戻す**ので、漢字の参照文に対する素のCERは発音が正しくても誤りと数える（R-029）。J3の効果は素CERで -2.64pt、**読みCERで -4.23pt**。悪化とされた80文のうち**24文はこれ**だった。`--metric cer_reading` を使う |
| **dev flow の分離を過適合と読む** | 30,000 step で dev-zero-shot flow は base より悪化するが、CERは改善し話者追随も保たれた。**flow lossは品質の指標にならない**（R-015の3例目） |
| 誤読が直らない | 主因は byte-fallback。`華` は単独pieceを持たず3つのバイト断片になる（R-027）。**J3（`--assign-yomi` / `synthesize_japanese.py` は既定で有効）が機械化済み**。読みCERで -4.23pt [-5.44, -3.05]。作品固有名（`藤宮高邦`）は一般語辞書では直らない |
| 中国語が壊れている | **仕様**。日本語学習で漢字の読みが上書きされ、CER 11.5% → 77.2%（R-022）。D-032で日本語特化と決定。英語は無傷（WER 1.7%）。中国語CERは回帰の監視指標としてのみ使う |
| **vast.aiで評価が即死する** | `accelerate` が無いと `transformers` のモデル読み込みが `NameError: init_empty_weights` で落ちる。**ローカルには偶然入っていることがある**。`uv sync --all-extras` |
| **shard結合が静かに壊れる** | ログ末尾は `完了: artifacts/<phase>/<timestamp>` で`metrics.json` は出ない。pathを組み立て直すこと。`--merge` にも `--eval-set` を渡す（抑揚側は既定値が存在するので**偶然通ってしまう**） |
| **データ版を取り違える** | HF repoに旧版（232,941発話）と v2（286,864発話）が両方ある。`find \| head -1` では旧版を拾う。**取り違えると比較そのものが無意味**。行数で検証する |
| **評価setJSONのpathがOS依存** | Windowsで書いた `data\\eval\\prosody_audio` はLinuxで**1つのファイル名**になる。`artifacts.as_local_path` で正規化する。**読み側と書き側が同じ間違いをするので動いてしまう** |
| **単体で通ったから大丈夫と考える** | 上の3件は非shard経路では通っていた。**経路ごとに確かめる**（shard → 結合まで小さく1回通す） |
| **交絡を確かめずに因果と判断する** | 6点が reference長で完全分離したので原因と考えたが、**対象文の長さと r=0.947 で交絡**しており直接検証も一貫しなかった（R-026は棄却）。完全分離は交絡を確かめるまで証拠にならない |

## 環境の罠

| 症状 | 原因と対処 |
|---|---|
| `BackendCompilerFailed: Cannot find a working triton installation` | WindowsのPyTorchにtritonが同梱されない。`uv sync --all-extras`（`triton-windows` は Windows で自動的に入る）。検証は `.venv/Scripts/python.exe -c "import triton;print(triton.__version__)"`。未対処だと**distillが全ケース失敗する**。`--sampler-compile-mode eager` でも回避可 |
| `UnicodeEncodeError: 'cp932'` | stdoutがcp932。`PYTHONIOENCODING=utf-8` を付けるか、ファイルへUTF-8明示で書く |
| `ModuleNotFoundError` | システムPythonで実行している。`.venv/Scripts/python.exe` を使う。venv側で出るなら上のセットアップを実行 |
| `No usable checkpoint under ...` | `--checkpoint` にパスを渡した。**ディレクトリ名**（`CuteTTS-distill`）を渡す |
| manifestの件数が倍 | moe zipの `.bak.json` を除外していない（本体と同数ある） |
| クラスタが1つに潰れる | 閾値0.70はこの埋め込み空間で破綻する。0.92を使う |
| 途中で落ちた | `artifacts/<phase>/<ts>/` に metrics.json が無ければ未完。消してよい。cacheは再開されるので消さない |
| `AttributeError: 'GenerationResult' object has no attribute ...` | `tts.generate()` は tensor ではなく `GenerationResult` を返す。`.waveform` と `.sample_rate` を取る |
| CUDA generator エラー | CPU generator を CUDA device で使った。`objectives._randn` が吸収するが、新しい乱数経路を足すときは同じ扱いにする |
| **生成したコードが無言で壊れる** | bash heredoc（`<<'PY'` でも）経由でPythonへ渡すと**バックスラッシュが1段落ちる**。`\1` が 0x01 になり正規表現が死に、`'\n'` が生の改行になってJavaScriptが構文エラーになった（**2回踏んだ**）。**生成コードはWriteツールで `.py` に書いてから実行する。** 書き出したら括弧の対応と文字列リテラルを検査する |
| **待機中の駆動スクリプトを「死んだ」と判定する** | 次のジョブを待っている駆動スクリプトは**GPUもCPUも使わない**。`nvidia-smi` や `train_continual` の有無で生きているかを判定すると見落とし、**同じ checkpoint を2プロセスで学習する**（実際にやった）。**駆動スクリプトはロックファイルで直列化する**（`m4a_factor_split.sh` の `lock_alive`）。確認は `ps -eo pid,args \| grep "[b]ash /workspace"` のように**スクリプト名**で行う |
| **リモートジョブを二重起動する** | `timeout` で ssh が切れても**リモートプロセスは生き続ける**。死んだと判断して再起動し、GPUを並列で使った（規約違反）。**再起動の前に必ず `ps -eo pid,args \| grep <script>` で生存を確認する** |
| **実行中のシェルスクリプトを上書きする** | bash はスクリプトを**バイト位置で逐次読む**。行を足すと、次に読む位置が別の行の途中になって壊れる（実測: `line 79: y: command not found` で、学習の後の評価だけが落ちた。学習は完了していたので気づきにくい）。**転送先は別名にして、実行は別名の複製から行う**（`cp run.sh run.running.sh && bash run.running.sh`） |
| `tail`/`grep` を通した進捗が出ない | パイプがバッファするため、プロセスが終わるまで1行も来ない。ファイルへ落としてから読む |

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
