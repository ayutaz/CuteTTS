# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## このリポジトリの位置づけ

`OPPO-Mente-Lab/CuteTTS` のfork（`origin`: `https://github.com/ayutaz/CuteTTS.git`）。
upstreamのコードは **推論専用** であり、学習コード（trainer / dataset / loss / packing）は一切含まれない。

このforkの目的は `docs/japanese-training/` にある通り、公開base checkpoint `OPPOer/CuteTTS` からの
**日本語継続学習** を段階的に進めること。作業ブランチは `feat/japanese-training`。

`src/cutetts/` 配下はupstream由来のinference実装で、`modeling/model.py` と `modeling/processor.py` は
明示的にinference-onlyと宣言されている（公開modelの構成以外はコンストラクタで `ValueError` を投げる）。
日本語学習の作業は基本的に **新規追加**（`src/cutetts/training/`、`scripts/`、`configs/`）であり、
既存推論pathを壊さないことが前提。

## コマンド

セットアップ（**Python 3.12 固定**。`requires-python = ">=3.12,<3.13"`）:

```bash
pip install torch==2.5.1 torchaudio==2.5.1  # CUDA 12.1なら --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install -e ".[ja]"        # J3（読み付与）。pyopenjtalk-plus（D-035）
pip install -e ".[prosody]"   # M1（抑揚・アクセント）。pyworld（D-040）
pip install -e ".[eval]"      # CER評価。**accelerate が無いとASRが読めない**
```

weightの取得（`model/` は .gitignore 済み）:

```bash
mkdir -p ./model
hf download OPPOer/CuteTTS --local-dir ./model/CuteTTS
hf download OPPOer/CuteTTS-distill --local-dir ./model/CuteTTS-distill
```

実行:

```bash
# CLI（= python infer.py と同一のentrypoint）
cutetts --model-dir ./model/CuteTTS --mode tts --text "..." --output tts.wav
cutetts --model-dir ./model/CuteTTS-distill --mode voice_clone \
  --reference-audio assets/default_reference.wav --text "..." --output clone.wav

# Web demo（FastAPI + WebSocket streaming、model-dirは各checkpointの親ディレクトリを渡す）
cutetts-demo --model-dir ./model --device auto --host 127.0.0.1 --port 7860
```

### このマシンでの実行環境（重要）

システム既定のPythonは3.14で **torch 2.5.1 が動かない**（対応は3.9〜3.12）。
日本語学習側の作業はリポジトリ直下の `.venv`（Python 3.12 + torch 2.5.1+cu121）で行う。

**Python は uv から実行する**（2026-09-19、ユーザー指示）。
**`--no-sync` を付ける。** 付けないと `uv run` が pyproject から環境を
同期し直すので、別途入れた torch 2.5.1+cu121 を入れ替えてしまう。

```bash
uv run --no-sync python -m pytest tests/training -v   # テスト
uv run --no-sync python scripts/<name>.py --config configs/japanese/<name>.yaml
```

`uv run --no-sync python -c "import sys; print(sys.executable)"` で
`.venv/Scripts/python.exe` を指していることを確認できる。

GPUは RTX 4070 Ti SUPER 16 GB（05章が想定した4090 24GBより小さい。R-007参照）。

### テスト

`pyproject.toml` に `[project.optional-dependencies] dev`（pytest / pyyaml）と
`[tool.pytest.ini_options]` を追加済み。`.gitignore` の `tests/` 除外も解除済みなので、
`tests/training/` はgit管理される。lint設定は引き続き未整備。

`model/`、`artifacts/`、`data/` はgitignore。**`artifacts/` 配下の音声は学習データの
ライセンス上、公開・コミットしてはならない**（08章「artifactの公開制限」）。

## アーキテクチャ

### 生成パイプライン

```
text ─┐
      ├─ SegmentManager でprefix組み立て（text / speaker slot / reference speech）
ref  ─┤   → Speaker Encoder (ECAPA student, 16 kHz → 256-dim)
audio└─   → Audio VAE encoder (24 kHz → 12.5 Hz, 64-dim latentのposterior mean)
      ▼
Patch Encoder (locenc, patch=2 latent frame) → locenc_to_lm_proj
      ▼
Qwen3系 Causal Backbone（7層に切り詰め）→ hidden state
      ▼
AudioDiTHead（flow matching, Euler + sway sampling）→ 次の連続latent patch
      ▼
Causal VAE Decoder（streaming時はcausal convのstateを保持して逐次decode）→ waveform
```

LMのtoken rateは `12.5 / 2 = 6.25 patch/s`。`--max-decode-length 750` は約120秒に相当。

### 主要ファイルの責務

| ファイル | 役割 |
|---|---|
| `runtime.py` | `config.json` からmodel/processor/speaker encoderを組み立て、safetensorsを `strict=True` でload。deviceとattention実装の解決もここ。 |
| `api.py` | 公開API `CuteTTS`。variant別のパラメータ検証、reference前処理、CFG branch構築、`generate` / `generate_stream`。 |
| `modeling/model.py` | `CuteTTSModel`（locenc / lm_speaker_linear / qwen_backbone / head / stop_predictor）と `prepare_input_embeds`。 |
| `modeling/segments.py` | `CuteTTSSegment` と `SegmentManager`。text / speech / speaker slotのmask付き結合。 |
| `modeling/processor.py` | Tokenizer + Audio VAE adapter + prompt文字列の組み立て。 |
| `modeling/diffusion_head.py` | `AudioLocEnc`（Patch Encoder）と `AudioDiTHead`（DiT + adaLN-Zero speaker条件 + Euler sampler）。 |
| `inference/generation.py` | 自己回帰ループ本体（KV cache、LM-level CFGの2 branch、stop判定、streaming/offline decode）。最大のファイル。 |
| `inference/conditioning.py` | mode（tts / voice_clone）とcfg_modeから `GuidancePlan` を決め、cond/uncond prefixを作る。 |
| `demo/` | FastAPI app（`/api/generate` WebSocketでPCMチャンクを送出）、model reload、TTFA/RTF計測。 |
| `audio_codec/` | Audio VAE本体（DAC由来のcausal conv）とSpeaker Encoder。 |

### 設計上の重要な性質（変更時に壊しやすい箇所）

- **checkpoint駆動の構成**: architecture値は `config.json` から来る。`variant` が `base` / `distill` 以外、
  `model_type != "cutetts"`、weightのmissing/unexpectedはすべて即エラー。configとコードは対で動く。
- **Qwen3 backboneの層切り詰め**: `lm_keep_num_hidden_layers=7` で公開configの28層を7層に上書きしてから
  `AutoModel.from_config` する。embeddingは `extended_vocab_size=16385` にresizeされる。
- **dtypeの混在**: backboneとlocencはcheckpoint dtype（bf16想定）、`head`（DiT）だけ **fp32固定**。
- **acoustic latentの正規化**: `speech_scaling_factor` / `speech_bias_factor` はcheckpointのbufferで、
  未設定（NaN）だとforward時にエラー。学習側でもこの正規化を再現する必要がある。
- **base と distill の非対称**: baseはLM-level CFG（cond/uncondの2 branch × 10 NFE）+ sway sampling。
  distillはCFG強度とstep数をDiT側の条件に埋め込み、`diffusion_steps ∈ {1,2,4}`、sway不可。
  この分岐は `api.generate` と `conditioning.build_guidance_plan` の両方に散っている。
- **prompt textは英語のinstruction固定**: `processor._text_only_prompt` /
  `_reference_prompt_segments` に埋め込まれた英語文と `<|im_start|>` / `<|im_end|>` /
  `<|endofprompt|>` が入力sequenceの一部。日本語学習でこのテンプレートを変える場合、
  推論側と学習側で必ず一致させる。
- **streaming decode**: `AudioStreamingVAEDecoder` がdecoderのcausal Conv1d / ConvTranspose1d の
  `forward` を差し替えてstateを保持する（`_CausalConv1d__padding` 等のname-mangled属性に依存）。
  VAE実装を触るとここが静かに壊れる。
- **MPS対策**: deviceがmpsのときprocessor / speaker encoderはCPUに置き、samplerのcompile modeも
  `eager` に落とす。device依存の分岐が `runtime.py` / `api.py` / `generation.py` に点在する。
- **reference前処理の既定値**: VAE用は先頭30秒、speaker encoder用は先頭8秒、2秒未満はrepeatで延長
  （`prepare_reference_audio`）。学習のreference samplingでもこの規約を意識する。

## 日本語継続学習プロジェクト（docs/japanese-training/）

**まず読むべきは [`docs/japanese-training/RESULTS.md`](docs/japanese-training/RESULTS.md)**（P0/P1の実測値一覧）と
[`08-execution-plan.md`](docs/japanese-training/08-execution-plan.md)（フェーズ定義とゴール）。
背景は README → 01〜07、データは data-inventory.md。

文書は情報を **確認済み / 決定済み / 提案 / 未確定** の4状態で区別する規約がある。
「実装した」と「日本語学習が成功した」を混同しないこと。
07章の意思決定表（D-001〜D-044）は項目を削除せず、状態と理由を追記して更新する。

### 進捗（2026-09-20）

**P0 / P1 / P2 / S0 / S1 / J2 / J3 / M1 / T1 / T2 / F1 / M2 / D1 / G1 / M3 /
F2 / M3b / D2 / C1 / T3 完了。M4a / M4a-split / A1 / C2 / M4b / M4c / M4d /
M4e も完了（2026-09-19〜20）。**

#### 読み: **投資を止める段階に来た**

**最良checkpointは `m4a-accent`（読みCER 7.58%、`--frontend accent`）。**
点推定だけなら `c2-accent`（7.25%）と `m4a-kanafull`（7.24%）が上だが、
**どれも互いに有意差なし**。人間の床 5.59% に対して TTS由来の誤りは約 2pt。

| 手段 | 効果 | 状態 |
|---|---:|---|
| **全文片仮名化**（M4a-split） | **-4.40pt** | 採用 |
| 学習と推論の表記を揃える（M4a） | -1.74pt | 採用 |
| アクセント記号 | 読みは +0.34pt（**効果なし**） | **採用**（同音異義の区別が消えるため。A1） |
| 核を正しい位置に置く | -0.18pt（効果なし） | — |
| 語境界の長音化の修正 | +0.05pt（効果なし） | — |
| **計算量4倍**（C2） | **-0.33pt（有意差なし）** | **頭打ち**（R-055 / D-053） |
| データ量 | -3.0pt/10倍 | **1,000hでも -1.5pt で床に届かない** |

**frontend も計算量も出尽くした。** 残るのはデータ量だけで、
外挿では床に届かない（D-053）。

#### 抑揚: **経路はできた。供給側が無い**

**輪郭は条件で動かせる**（M4c / R-052 / R-054）。要点は2つ。

* **条件は「先読み」でなければ使われない。** その patch の F0 だけだと
  teacher forcing の履歴から予測できてしまい、**モデルが無視する**
  （真の F0 で flow loss が 0.2% しか動かない。R-051）
* **差し込む場所は DiT head の adaLN。** LM 入力の約2倍効く
  （対照比 +0.054 → **+0.118**）

**`--f0-dropout 0.1` が運用点**（R-057）。条件を落として学習すると
「条件が外れると素より悪い」問題が消えるが、**落としすぎると条件の効果も
消える**（p=0.5 で +0.018、有意差なし）。**依存と能力は同じもので、
連続的に取引できる。** p=0.1 なら効果を 91% 保ったまま劣化の大半が消える。

**供給側が足りない。** テキストからの予測は実測と **+0.113** しか相関せず、
**辞書は輪郭の形を一切持っていない**（-0.001。R-053）。
辞書はアクセント核を約45%当てるのに、連続的な動きは持たない。

**アクセント核は真の F0 を与えても動かない。** 位置を足しても変わらず
（M4d / R-056）、原因は未特定。**次は「条件に核が入っているか」を
生成を通さずに確かめる**（M4f。CPU のみ）。

**韻律転写は機能として成立した**（D-052）。同じ台詞の人間の読みから F0 を
取り出して head へ渡せば写る。M3b で「参照音声からは写らない」と分かって
いたので、**F0 を取り出す一手間**が答えだった。

#### 測定器を5回直した

R-032 / R-033 / R-034（M1）に加え、**R-050 で輪郭の指標に codec 由来の
上限があると分かった**。同じ発話を VAE で往復させるだけで +0.367 まで落ちる。
**M2 の「天井 +0.382」は人間の再現度ではなく上限そのものだった。**
アクセント核は上限 75.0% > 天井 64.5% なので、**そちらは指標として使える**。

3指標での位置:

| 指標 | base | **現行** | 人間 / 上限 |
|---|---:|---:|---:|
| 読みCER（600文・frontend無し） | 30.94% | 13.38% | 5.59% |
| **読みCER（`--frontend accent`）** | — | **7.58%** | 5.59% |
| phonetic（読み） | 46.9% | **2.43%** | — |
| 輪郭の相関（240文） | +0.024 | **+0.091** | 天井 +0.382 / **上限 +0.367** |
| アクセント核（対人間） | 35.2% | **43.6%** | 天井 64.5% / **上限 75.0%** |

**学習は3指標すべてを有意に改善している。** ただし**どれも人間に届いていない**。

**S1が失敗していた原因は データではなく学習の実装だった（R-020）。**
公開checkpointの `qwen_backbone` / `locenc` は bf16 で、`AdamW` がそれを直接
更新すると lr=2e-5 の更新量が bf16 の丸め幅を下回り、**backboneの91%が
1stepも動いていなかった**。学習されていたのは fp32 の DiT head だけ。
19回の試行はすべてこの条件下の観測なので、S1の結論群は再検証を要する。

修正後（同一データ・同一step・同一seed、dtypeのみ変更）:

| 実行 | in_domain mean / median | 打切 |
|---|---:|---:|
| base | 35.78 / 30.25 | 0/30 |
| S0（7.15h） | 28.36 / 25.53 | 0/30 |
| S1v2 bf16 対照（325.9h） | 32.43 / 29.06 | 1/30 |
| S1v2 fp32 3,000 step | 24.36 / 23.23 | 0/30 |
| S1v2 fp32 5,000 step | 22.25 / 21.75 | 0/30 |
| S1v2 fp32 12,000 step | 22.76 / 16.99 | 0/30 |

**確定値は評価set v3（600文）で測る。** v2 は step 数の順位を取り違えていた。

| 実行（v3・600文） | in_domain mean / median | phonetic |
|---|---:|---:|
| base | 35.86 / 31.91 | 46.9 / 42.3 |
| bf16 対照 3,000 step | 31.21 / 28.00 | 41.8 / 51.9 |
| fp32 3,000 step | 25.98 / 22.86 | 36.2 / 33.8 |
| fp32 10,000 step | 22.93 / 20.00 | 26.2 / 22.5 |
| fp32 20,000 step | 21.05 / 17.86 | 26.3 / 33.0 |
| **fp32 30,000 step** | **20.10 / 16.67** | **22.2 / 19.2** |

base → 30,000 step は **-15.77pt**（95%CI [-17.26, -14.35]、600文中457文で改善）。
ASR床10.4%に対し、TTS由来の誤りは **25.5pt → 9.7pt（62%削減）**。
**収穫逓減が見える**（10,000→20,000 で -1.88pt、20,000→30,000 で -0.96pt）。

**話者追随は step 数を増やしても壊れない。** dev flow で seen/zero-shot が
分離するので過適合を疑ったが、reference追随の実測では否定された
（margin は 10,000〜30,000 で +0.24〜+0.25 で安定、正答 11〜12/12）。
**dev flow の分離を過適合と読んではいけない**（R-015 の3例目）。

bf16対照との差 **-8.07pt**（95%CI [-13.22, -3.02]、有意。改善19文/悪化9文）。
base比 -11.42pt。`ParameterDrift` の実測で、bf16では3,000 step後も backbone が
**3.68%** しか動いていなかった（fp32は100%）。

**修正後は step 数も効く**（凍結下では効かなかった）。5,000 step で
**S0 を有意に上回った**（-6.11pt、95%CI [-9.28, -2.99]）。
`phonetic` は 46.9% → 24.8%。全fp32実行で打ち切り 0/30。

| フェーズ | 状態 | 主要な結論 |
|---|---|---|
| P0 | 完了 | `gate_passed: true`。base/distill 各7/7 |
| P1a | 完了 | accepted **10,466.4 h** / 18,279話者ID（除外1.8%） |
| P1b | 完了 | `<unk>` 0%だが byte-fallback が token 9.66% / 文45.6% → 既存Tokenizerで開始可（D-018） |
| P1c | 完了 | CER差 +0.58pt、往復CER中央値0.00% → **VAE freeze確定**（D-003）、Stage 4見送り |
| P1d | 完了 | voiceクラスタ **t=0.92**（既定0.70は破綻）、leakage 0件 |
| P1e | Pass A完了 | 44.5× realtime、外挿 65.3 GB / **239 GPU時間**。Pass BはS2直前 |
| P2 | 完了 | ゴール7件達成。変異テスト9/9検出。すべてCPUで検証 |
| S0 | 完了 | **in_domain CER 35.8% → 28.4%**。reference追随 12/12。7.15hで通過 |
| S1 | 完了 | bf16でbackboneが凍結していた（R-020）。修正後 **20.10%**（v3・600文。base比 -15.77pt） |
| J2 | 完了 | 漢数字の読み展開。**-11.80pt / 再学習不要** |
| J3 | 完了 | 語の読み付与。専用set 読みCER **-4.23pt**、会話文 -1.26pt。**再学習不要**（D-036 / D-037） |
| J4 | **見送り** | J3が内容語のfallbackをほぼ吸収した（R-030 / D-039） |
| M1 | 完了 | 抑揚とアクセントを測れるようにした。**測定器を3回直した**（R-032 / R-033 / R-034）。学習が両方を有意に改善 |
| T1 | 完了 | **学習率は梃子ではなかった**（R-035 / D-043）。2e-5 がほぼ底 |
| T2 | 完了 | **batch size も学習対象も梃子ではなかった**（R-036 / D-044）。同じ計算量なら batch 4 と 16 は区別できない。head凍結は有意に悪い |
| M2 | 完了 | **抑揚の天井は +0.38 / アクセントは 64.5%**（R-037 / D-045）。現行は輪郭で隔たりの約32%。**伸びしろが残っている** |
| D1 | 完了 | **データ量は 30,000 step では梃子だった**（R-038 / D-046）。読みCER **-3.82pt**。ただし**抑揚は動かない**（+0.01、有意差なし） |
| M4a | 完了 | 全文片仮名 + 核記号で **読みCER 7.58%**（R-045）。抑揚は動かず |
| M4a-split | 完了 | **効いていたのは全文片仮名化だけ**（-4.40pt）。記号・核の正しさ・長音修正はいずれも読みでは有意差なし（R-047 / R-049） |
| A1 | 完了 | **記号はアクセントの区別を与えていた**（対の区別 0.0% → **75.0%**）。**読みCERは原理的にこの誤りを見ない**（R-048 / D-051） |
| C2 | 完了 | **計算量は頭打ちになった**（-0.33pt、有意差なし。R-055 / D-053）。C1（13.38%の水準）では -1.40pt 有意だったので、**床に近づくほど効かない** |
| M4b | 完了 | **辞書は輪郭の形を一切持っていない**（-0.001、床 +0.006）。学習した予測器も +0.113（R-053） |
| M4c | 完了 | **輪郭は条件で動かせる**（対照比 +0.118）。**先読み**が要り、**差し込む場所は DiT head の adaLN**。既定にはしない（R-051〜R-054 / D-052） |
| M4d | 完了 | **位置を足しても話速は合わない**（長さ 4.99 → 4.97秒）。アクセントも動かない（R-056） |
| M4e | 完了 | **依存と能力は連続的に取引できる**。**`--f0-dropout 0.1` が運用点**（効果を91%保ち劣化の大半が消える。R-057） |

### 実装済み

```text
src/cutetts/training/   P1: artifacts, manifest, text_rules, pairing,
                            latents, speaker_cache, voice_clusters
                        P2: objectives, collator, dataset, forward,
                            packing, checkpointing, prompt
                        S1: evalstats（対応のある検定・打ち切り勘定）,
                            reading（漢数字の読み展開 = J2）,
                            reference（短いreferenceの延長。R-026は棄却済み）,
                            listening_page（聴取評価ページ）
                        J3: yomi（語の読み付与 + 読みレベルCER）
                        M1: prosody（F0・抑揚の幅・輪郭の相関・アクセント核）,
                            alignment（MMS_FAでモーラ単位の強制アラインメント）
                        G1: stopping（喋り続け・自己反復・打切の勘定）
                        M4a: accent（片仮名 + アクセント核の記号。
                             `shuffle` と `merge_across_words` の対照つき）
                        M4c: f0（F0 を 12.5 Hz で取る / 先読み / 位置 /
                             zero-init の `F0Conditioner` / F0 cache）
                        M4b: f0_predictor（テキスト → 長さを正規化した輪郭）
scripts/                reproduce_baseline, analyze_japanese_tokenizer,
                        evaluate_japanese_vae, prepare_japanese_manifest,
                        cache_audio_latents, build_voice_clusters,
                        summarize_eval_runs（CER横断集計・信頼区間）,
                        evaluate_forgetting, build_numeral_eval_set,
                        synthesize_japanese（J2つき合成entrypoint）
                        S0: train_continual, diagnose_flow_loss,
                            check_reference_following, build_eval_set,
                            evaluate_japanese_cer
                        S1: measure_asr_floor, s1_preprocess.sh
                        J3: build_yomi_eval_set
                        M1: build_prosody_set, fetch_prosody_audio,
                            evaluate_prosody
                        T1: t1_lr_sweep.sh（vast.ai上で完結。評価は --shard 並列）
                        T2: t2_capacity_sweep.sh（同上。--trainable で学習対象を選ぶ）
                        G1: summarize_stop_health
                        M4a: m4a_accent_marks.sh, m4a_factor_split.sh,
                             m4a_aligner_evals.sh, m4a_one_condition.sh（汎用）
                        A1:  build_accent_pair_set, evaluate_accent_pairs
                        M4c: cache_f0_targets（latentをdecodeしてF0）,
                             m4c_validate.sh（経路の検証）, m4c_full.sh,
                             diagnose_f0_conditioning（条件を使っているか）,
                             calibrate_contour_metric / calibrate_accent_metric
                             （**指標の上限を測る**）,
                             compare_prosody_runs（抑揚の対応のある比較）
                        M4b: train_f0_predictor
tools/                  mutation_check（テストが実際に効くかの検証）
tests/training/         全件PASS（slowマーカーは実checkpointを要する）
```

S1のデータは [tts-dataset/cutetts-ja-latents](https://huggingface.co/datasets/tts-dataset/cutetts-ja-latents)
（public / **gated: manual**）にある。約2 GBの取得だけで学習を開始できる。
**音声そのものは置いていない**（latentは復元可能なので同じ扱い）。

実行手順は `.claude/skills/cutetts-ja-pipeline/SKILL.md` にまとめてある。

### 絶対に守ること

- **GPUはすべて vast.ai を使う。ローカルGPUは使わない**（D-023 / D-024、2026-09-15）。
  起動前に**費用見積もりを提示する**。APIキーは設定済みだが**残高0**なので、
  入金はユーザーが行う。事前確認のフックは削除した（毎回の承認が作業を止めるため）。
  **同じインスタンス上で評価を分割並列するのは可**（`--shard`。生成は
  `batch=1` の自己回帰でGPUが2〜3割しか埋まらない）。
- **`artifacts/` 配下の音声をコミット・公開しない**。MoeSpeech LICENSEは
  「音声ファイルを1つであっても公開することは再配布とみなす」と規定している。
- **話者IDを匿名化として扱わない**。golのIDは `SHA-256(表示名)[:32]` で辞書攻撃可能。

### P2で自分で決めた事項（決定済み。変更するなら理由を残す）

| 項目 | 決定 |
|---|---|
| padding patchのloss除外 | 分子からも分母からも除く |
| stopラベルの位置 | 位置iのhiddenが「patch iが最終patchか」。`STOP_STOP=1` 固定 |
| stopのclass imbalance | `positive_weight`（重み付き平均） |
| flow/stopの重み | `stop_weight=1.0` 既定。Stage 0で調整 |
| condition dropoutの対象 | speaker + reference、既定はjoint |

### S0で判明した落とし穴（再発させない）

- **`PairSampler.sample()` を step ごとに呼ばない**。呼ぶたびにRNGを作り直すので
  毎回同じペアが返る。1回目のS0は3000step全部が同じ4発話で、
  flow loss 0.003 は丸暗記だった（R-012）。`iter_pairs()` の stream から引く。
- **学習ループの損失だけで成否を判断しない**（D-025）。train / dev / 未学習base を
  同じ経路で測る（`scripts/diagnose_flow_loss.py`）。flow loss は
  「常に0を出す予測器」が約2.0なので、それより十分小さいかで絶対値を判断する。
- **CERには床がある**。人間の実音声を同じ経路で測ると素CER 10.4%。
  現在の 20.10% のうちTTS由来は約9.7pt（S0の28.4%では約18pt）。
- **素のCERは表記の違いを誤りと数える**（R-029）。ASRは `なにも` と話しても
  `何も` と書く。**床の半分近くは発音でなく表記だった**（10.42% → 読みCER 5.59%）。
  **仮名を入力に含む比較（J2 / J3）では `cer_reading` を主に読む**（D-038）。
  同音異義が見えなくなるので素CERと併記する。
  **CERは会話文では知覚と一致する**（盲検A/Bで13/14、p=0.0009）が、
  **抑揚とアクセントは測れない**（聴取での指摘は40%と30%）。
- **評価スクリプトは既定で frontend を適用しない。** `--expand-numerals`（J2）と
  `--assign-yomi`（J3）は任意フラグで、T1 / T2 / step数sweep はすべて frontend 無し。
  **frontend 込みの値と frontend 無しの値を混同しない**（F1）。
  2026-09-20 時点の実運用値は **7.58%**（`--frontend accent`）で、
  frontend 無しの比較用の値は 13.38%。
- **J2 と J3 の順序を逆にしない（R-039）。** J3 を**先**に掛ける。逆にすると
  J3 が J2 の仮名列を再解釈して漢数字を復活させる（`せんにひゃく八ジュウ`）。
  **`yomi.apply_frontend` を使う**（順序が固定されている）。
- **文末の繰り返しは打ち切りとは別の失敗**（G1 / R-040）。
  `training.stopping` で数える（`scripts/summarize_stop_health.py`）。
  評価setでは base 3.2% → 現行 1.2%（-2.00pt、有意）。
  **試聴9文中4文（44%）は一般化してはいけない**。参照音声か文の違いで、
  評価setでは桁が違う。**絶対値ではなくrun間の比較に使う。**
- ~~zero-shot split の話者不足（R-013）~~ → S1前処理で解消（119 cluster）。

### 探索し終えた手段（2026-09-19 時点）

**学習設定は出尽くした**（dtype / step数 / データ量 / 学習率 / batch size /
学習対象 / `flow_copies` / `condition_dropout`）。**frontend も出尽くした**
（表記の一致 / 全文片仮名化 / アクセント記号 / 核の正しさ / 長音の修正）。

**読みで残るのはデータ量だけ**（計算量は C2 で頭打ちになった。R-055）。
**抑揚は12通りの手段で動かなかった**が、**13通り目で動いた** —
**F0 を先読みで DiT head の adaLN へ渡すと輪郭が動く**（M4c / R-054）。
ただし**供給側が無い**ので既定にはしない（D-052）。

効いた順（実測）:

| 手段 | 効果 | 学習の要否 |
|---|---:|---|
| dtype修正（bf16→fp32） | **-15.77pt** | 要（済） |
| **J2 読み展開（数詞）** | **-11.80pt** | **不要** |
| **J3 読み付与（語）** | **-4.23pt**（読みCER） | **不要** |
| step数 3,000 → 30,000 | -4.20pt | 要（済） |
| **データ量 17.9h → 325.9h（19倍・30,000 step）** | **-3.82pt**（読みCER） | 要 |
| **学習率（T1で4水準）** | **効果なし**（2e-5が最良） | 要（済） |
| **batch size（T2で 4 / 16）** | **同じ計算量なら効果なし** | 要（済） |
| **head凍結（T2）** | **+0.78pt（悪化）** | 要（済） |
| **計算量4倍（C1）** | **-1.40pt** | 要（済） |
| **flow_copies / condition_dropout（T3）** | **効果なし**（既定値が最良） | 要（済） |
| **全文片仮名化（M4a-split）** | **-4.40pt**（新記録 7.58%） | 要（済） |
| **学習と推論の表記を揃える（M4a）** | **-1.74pt** | 要（済） |
| アクセント核の記号（M4a-split） | **読みは +0.34pt（効果なし）**。ただし**同音異義の区別は記号だけが与える**（A1: 0.0% → 75.0%） | 要（済） |
| 核を正しい位置に置く（M4a-split） | **効果なし**（-0.18pt。偽の核でも同じ） | 要（済） |
| 語境界の長音化の修正（M4a-split） | **効果なし**（+0.05pt） | 要（済） |
| **計算量4倍（C2）** | **効果なし**（-0.33pt、有意差なし。**頭打ち**） | 要（済） |
| **F0 の条件づけ（M4c）** | **輪郭 +0.118**（対照比、有意）。**読みには無関係** | 要（済） |
| 位置を条件に足す（M4d） | **効果なし**（長さが動かない） | 要（済） |
| 条件の dropout（M4e） | **p=0.1 が運用点**（効果を91%保ち劣化を消す） | 要（済） |

~~**データ量は測った中で最も弱い。**~~ → **訂正**（D1 / R-038）。
3,000 step での -1.90pt は**大きいデータに不利な条件**での観測で、
**30,000 step で測り直すと -3.82pt** だった。
**ただし抑揚はデータ量では動かない**（17.9h でも +0.12、19倍にして +0.01）。
**規模は読みを直すが抑揚は直さない。**

聴取で残った指摘は読み間違い45% / 抑揚40% / アクセント30%。
M2 で天井を測った結果、**輪郭は隔たりの68%が未達**で、
そこが最大の伸びしろだと分かっている。

| # | フェーズ | 内容 | 状態 |
|---|---|---|---|
| ~~J3~~ | 読み付与 frontend | byte-fallback を含む語を読みへ置換。**再学習不要**。専用set 読みCER **-4.23pt**、会話文でも **-1.26pt**（どちらも有意） | **完了** |
| ~~J4~~ | Tokenizer 互換拡張 | J3が内容語のfallbackをほぼ吸収したので**見送り**（R-030 / D-039） | 見送り |
| ~~M1~~ | 抑揚・アクセントの測定 | **完了**（240文/53話者）。学習で輪郭の相関 +0.024→**+0.122**、アクセント対人間 35.2%→**43.6%**（どちらも有意）。base は床と区別できない＝**抑揚は学習が与えている** | 完了 |
| ~~T1~~ | 学習率の探索 | **完了。梃子ではなかった**（R-035 / D-043）。半分と5倍は読みCERが有意に悪く（+2.06 / +3.00pt）、2.5倍は区別できない。抑揚・アクセントはどの水準も有意差なし | 完了 |
| ~~T2~~ | batch size / 学習対象 | **完了。梃子ではなかった**（R-036 / D-044）。batch16 は同step なら -1.91pt だが同sample なら +1.67pt で、**効いているのは計算量**。head凍結は +0.78pt で悪化 | 完了 |
| ~~F1~~ | 評価を実運用と揃える | **完了**。実運用の主値は **読みCER 12.36%**（比較用 13.38%） | 完了 |
| ~~F3~~ | 新最良を frontend 込みで測る | M4a-split が上書きした（7.58%） | 完了 |
| ~~M4c~~ | 韻律を構造で受け取る | **完了**（R-054 / D-052）。**輪郭は +0.118 制御できる**（DiT head の adaLN + 先読み4 patch）。**差し込む場所が効き、量は効かない**。**アクセント核は真の F0 を与えても動かない**。**既定にしない**（実運用で与えるものが無く、条件が外れると素より悪い）。**韻律転写の機能としては成立** | 完了 |
| ~~M4b~~ | テキストから韻律を予測 | **完了**（R-053）。**辞書は輪郭の形を一切持っていない**（相関 -0.001。床は +0.006）。アクセント核は45%当てるのに連続的な動きは持たない。学習した予測器も **+0.113** で、M4c の利得と掛けると検出限界に埋もれる | 完了 |
| ~~M4d~~ | 位置を条件に足す | **完了**（R-056）。**仮説は支持されなかった** — 長さが 4.99 → 4.97秒で不変、アクセントも動かない。「届いてはいるが位置がずれている」は違った | 完了 |
| ~~M4e~~ | 条件を外しても劣化しないようにする | **完了**（R-057）。**依存と能力は同じもので、連続的に取引できる**。dropout 0.5 では条件の効果も消えたが、**0.1 なら効果を91%保ったまま劣化の大半が消える**（運用点） | 完了 |
| ~~M2~~ | 抑揚・アクセントの天井 | **完了**（R-037 / D-045）。天井は **輪郭 +0.38 / アクセント 64.5%**。**現行は隔たりの約32%しか埋めていない**ので、抑揚に投資する根拠が出た | 完了 |
| ~~D1~~ | データ量の再測定 | **完了**（R-038 / D-046）。**読みCER -3.82pt で梃子だった**（3,000 step では -1.90pt）。**抑揚は動かない**ので S2 は抑揚の答えにならない | 完了 |
| ~~G1~~ | 停止の健全性 | **完了**（R-040）。`training.stopping` で指標化。base 3.2% → 現行 1.2%（-2.00pt、有意）。**試聴の44%は評価setでは再現しない** | 完了 |
| ~~M3~~ | 抑揚の手段の設計 | **設計済み**。仮説（情報のボトルネック）は **M3b で否定された**（R-041）。**参照から韻律を読む経路が無い**ので、学習の変更が要る（D-047） | 設計済み |
| ~~F2~~ | frontend の音への効果 | **完了**。数詞 **-11.44pt**（有意）。J3 がアクセントを壊すかは**判定不能**（-3.00pt、CI上限 +0.14）。喋り続けは参照音声に依存しない（1.2% で同じ） | 完了 |
| ~~M3b~~ | 韻律転写の診断 | **完了**（R-041 / D-047）。同じ台詞の参照でも +0.160（天井 +0.382）。**声質と声域は取るが輪郭は取らない** | 完了 |
| ~~D2~~ | データ量の3点目（80h） | **完了**（R-042）。15.22%。傾きは **-3.05 / -3.02pt/10倍 で一致**（対数直線）。S2 の外挿はそのまま | 完了 |
| ~~C1~~ | 計算量4倍 | **完了**（R-043 / D-048）。**読みCER 11.98%（-1.40pt、有意）で新記録**。喋り続け 1.2%→0.5%。抑揚は動かない | 完了 |
| ~~T3~~ | flow_copies / condition_dropout | **完了**（R-044 / D-049）。**既定値のままが最良**（`flow_copies=2` は +0.79pt で有意に悪い）。抑揚も4条件すべて有意差なし | 完了 |
| ~~M4a~~ | アクセント核をテキストに明示 | **完了**（R-045 / D-050）。**読みCER 7.58%（-5.80pt）で新記録**。`phonetic` は 2.43%。**抑揚は +0.93pt で動かず**（基準 +5pt 未達） | 完了 |
| ~~C2~~ | 計算量4倍 | **完了**（R-055 / D-053）。**-0.33pt で有意差なし＝頭打ち**。読みへの計算量の投資はここで止める | 完了 |
| **M4f** | **アクセント核が条件に入っているか** | **次**。12.5 Hz の条件から読んだ核と元の音声から読んだ核を比べる。**CPU のみ・数分**。一致しなければモデルを疑う必要が無くなる | 提案 |
| M4g | 条件を隣接モーラの**差分**にする | アクセント核は相対関係で決まる。**M4f で「条件には入っている」と分かってから** | 提案 |
| M4h | 韻律転写を**本番データ**で回す | 325.9h / `--f0-dropout 0.1`。約13h / 約$2.6。**17.9h の検証しかしていない** | 提案 |
| S2 | 1,000時間 | 外挿では **-1.5pt**。**抑揚には効かない**。前処理が重い（P1e Pass B が要る） | 後回し |


**棄却した仮説**: R-026（referenceの長さ）。対象文の長さと r=0.947 で交絡し、
直接検証も一貫しなかった。`ensure_minimum_duration` は実装済みだが**効果の裏づけは無い**。

### 測定と学習で二度と繰り返さないこと（2026-09-02 確定）

- **bf16パラメータを optimizer で直接更新しない（R-020）。** lr が小さいと
  更新が丸め幅を下回り、round-to-nearest-even が毎step捨てる。
  `promote_to_float32` で fp32 に上げ、`export_for_inference(dtypes=)` で戻す。
  **`ParameterDrift` の「重みが動いた割合」を毎runのmetricsで必ず確認する。**
  100%から大きく外れていたら、そのrunの比較は無意味。
- **n=30 の評価setで 2〜3pt の差を語らない。** 検出できる最小差は **6.9pt**。
  `scripts/summarize_eval_runs.py --compare A B` で信頼区間を必ず出す。
  **評価set v3（`data/eval/eval_set_v3.json`、600文）を使うこと**（MDE 1.5〜2.0pt）。
  少数標本では点推定が大きく振れる（同じcheckpointで v2 -8.07pt / v3 -5.23pt。
  v2のCI [-13.22,-3.02] は v3 の値を含むので矛盾ではない）。
  **点推定の順位ではなく信頼区間で判断する。** 補正係数をかけて使うことはできない。
- **打ち切り生成をCERに混ぜない（R-021）。** `max_decode_length`（64.0秒）
  張り付きは停止の失敗であって発音誤りではない。`summarize_eval_runs.py` が
  打ち切り率を別勘定で出す。停止健全性はCERとは独立のゲートとして扱う。
- **評価setは結果を見てから変えない。** v1→v2の差し替えだけで基準線が
  5.2pt動いた（改善幅の主張自体は共通27文で再現するが、絶対ゲート値は事後編集の下流）。
- **実行の対応付けはindexではなくテキストで行う。** 評価setが差し替わると
  indexがずれ、別の文どうしを比較する（v1→v2で実際に起きた）。

### S1で判明した落とし穴（再発させない）

- **golのtarは `_partN` に分割されている**。ファイル名をそのまま game_id に
  使うと、分割されたgameが **エラーも出さずに丸ごと落ちる**。
  S1では5 game中2 game（170時間・52%）が消えていた。
- **クラスタの粒度は用途ごとに逆向きの要求を持つ**（R-014 / D-027）。
  `voice_cluster_id`（完全連結・細かい）は PairSampler の単位で、粗いと
  **別の声をreferenceにして学習する**（S1実測で26.9%）。
  `split_group_id`（単連結・粗い）は split の単位で、細かいと
  **同じ声がtrainとzero-shotに現れる**（実測15話者）。片方の粒度では両立しない。
- **out_of_domain はデータ量では直らない**（D-026）。golのcorpusで
  数字を含む文は1.3%。S1のゴールから外した。
- ~~データはクラスタ密度で選ぶ（D-029 / R-018）~~ → **差し戻した**（2026-09-02）。
  根拠の -6.0pt は有意でなく（95%CI [-14.25, +0.05]）、差の約60%が1文の
  挿入発散によるもので、step数とmoe比率も交絡していた。しかも**すべて
  backbone凍結下の観測**（R-020）。S2の選定基準は fp32 で測り直してから決める。
- **CERは必ず測る**（R-015）。flow loss は CER と逆相関することがある。
  20,000 step実行では flow 最良の点で CER が最悪（54.2%）だった。

**packingを触るときの注意**: 行index（`target_batch_index`）と
sample index（`target_sample_index`）は別物。unpackedでは一致するので
混同しても露見しない。packingすると1行に複数sampleが入り、speaker の
対応付けが壊れる。
