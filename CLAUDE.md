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
pip install -e ".[ja]"   # J3（読み付与）を使うとき。pyopenjtalk-plus（D-035）
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

```bash
.venv/Scripts/python.exe -m pytest tests/training -v   # テスト
.venv/Scripts/python.exe scripts/<name>.py --config configs/japanese/<name>.yaml
```

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
07章の意思決定表（D-001〜D-035）は項目を削除せず、状態と理由を追記して更新する。

### 進捗（2026-09-02）

**P0 / P1 / P2 / S0 完了。S1の失敗原因が確定し、修正して最良値を更新した。**

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
| S1 | **原因確定・修正済** | bf16でbackboneが凍結していた（R-020）。修正後 **20.10%**（v3・600文。base比 -15.77pt） |

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
tools/                  mutation_check（テストが実際に効くかの検証）
tests/training/         全件PASS（slowマーカーは実checkpointを要する）
```

S1のデータは [tts-dataset/cutetts-ja-latents](https://huggingface.co/datasets/tts-dataset/cutetts-ja-latents)
（public / **gated: manual**）にある。約2 GBの取得だけで学習を開始できる。
**音声そのものは置いていない**（latentは復元可能なので同じ扱い）。

実行手順は `.claude/skills/cutetts-ja-pipeline/SKILL.md` にまとめてある。

### 絶対に守ること

- **ローカルGPUを使う処理は実行前にユーザーへ確認する**（D-023）。
  バックグラウンド実行でも同じ。所要時間やVRAMが小さいことは省略の理由にならない。
  GPUジョブを並列起動しない（1本ずつ直列）。
- **`artifacts/` 配下の音声をコミット・公開しない**。MoeSpeech LICENSEは
  「音声ファイルを1つであっても公開することは再配布とみなす」と規定している。
- **話者IDを匿名化として扱わない**。golのIDは `SHA-256(表示名)[:32]` で辞書攻撃可能。
- 重いGPU処理は vast.ai へ回す（D-024）。起動前に費用見積もりを提示する。

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
- ~~zero-shot split の話者不足（R-013）~~ → S1前処理で解消（119 cluster）。

### 未探索の手段（2026-09-10 時点）

**探索したのは dtype / step数 / データ量 の3軸だけ。**
学習17回すべてが **lr=2e-5 固定**で、`batch_size=4` / `flow_copies=4` /
`condition_dropout=0.1` も全runで同じ（変えたのは warmup だけ）。
しかも 2e-5 は backbone が凍結していた時期に選んだ値（R-020）。
**「データ量は弱い、step数は頭打ち」はこの3軸の中での結論にすぎない。**

効いた順（実測）:

| 手段 | 効果 | 学習の要否 |
|---|---:|---|
| dtype修正（bf16→fp32） | **-15.77pt** | 要（済） |
| **J2 読み展開（数詞）** | **-11.80pt** | **不要** |
| step数 3,000 → 30,000 | -4.20pt | 要（済） |
| **データ量 17h → 325.9h（19倍）** | **-1.90pt** | 要 |

**データ量は測った中で最も弱い。** 聴取で残った指摘は
読み間違い45% / 抑揚40% / アクセント30%。

| # | フェーズ | 内容 | 状態 |
|---|---|---|---|
| **J3** | 読み付与 frontend | 誤読を frontend で消す。**再学習不要**。仮名置換で `中華`→`ちゅうか`、`湊`→`みなと` が直ることを実証（R-027） | **次に着手** |
| J4 | Tokenizer 互換拡張 | byte-fallback を減らす。J3の結果を見て価値を測る（R-024） | 待機 |
| **T1** | 学習率の探索 | 17回すべて lr=2e-5 固定。しかも凍結時代に選んだ値（R-025） | 未実施 |
| T2 | batch size / 学習対象 | 未検証 | 待機 |
| M1 | 抑揚・アクセントの測定 | CERでは測れない。測る手段が無いと判定できない | 並行 |
| S2 | 1,000時間 | **保留**。規模を上げる根拠が実測で得られていない | 保留 |

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
