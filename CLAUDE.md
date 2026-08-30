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

セットアップ（Python 3.10+、実際の想定は3.12）:

```bash
pip install torch==2.5.1 torchaudio==2.5.1  # CUDA 12.1なら --index-url https://download.pytorch.org/whl/cu121
pip install -e .
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

test / lintは **未整備**。pyprojectにdev依存もlint設定もなく、さらに `.gitignore` が `tests/` を
除外しているため、`docs/japanese-training/04-training-implementation.md` が計画する `tests/training/`
を追加する際は **先に .gitignore から `tests/` を外す** 必要がある。

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

README → 01〜07（背景・方針・設計）→ 08（実行計画）→ references の順で読む。
**実作業に入る前に読むべきは `08-execution-plan.md`**。各フェーズの目的・ゴール（完了条件）・
成果物・判断ゲートが定義されており、05のロードマップを実行単位へ落としたもの。

文書は情報を **確認済み / 決定済み / 提案 / 未確定** の4状態で区別して書く規約がある。
新しい記述を追加するときもこの区別を維持し、「実装した」と「日本語学習が成功した」を混同しない。
07の意思決定表（D-001〜D-012）は項目を削除せず、状態と理由を追記して更新する。

進め方（05-experiment-roadmap.md）:
`P0 公開baseの推論再現` → `P1 日本語preflight（Tokenizer coverage / VAE reconstruction / manifest）`
→ `P2 training forward復元` → `Stage 0 (10〜30h overfit)` → `Stage 1 (100〜500h)` →
`Stage 2 (1,000h)` → `Stage 3 (3,000〜10,000h)` → `Stage 4 JA VAE（条件付き）` → `Stage 5 distill`。
各段階にexit gateがあり、checkpoint・config・seed・manifest checksum・評価artifactが揃って完了。

初期方針: Audio VAEとSpeaker Encoderはfreeze、text embedding / Patch Encoder / backbone /
Diffusion Head / Stop Predictorをtrain、full fine-tuningが主案（すべて「提案」であり実験で再判定）。

未実装で、実装する場合に仕様を自分で確定する必要がある箇所（04章）:
flow-matching loss（`x_t = (1-t)ξ + tP`、target velocity `P - ξ`、`t = sigmoid(u), u~N(0,1)`）、
stop targetの位置とmask、sequence packingのattention遮断、condition dropout、latent cache、
checkpoint resume。stop loss周りは論文にも規定がなく、推論側の停止挙動と一致するテストが必須。
