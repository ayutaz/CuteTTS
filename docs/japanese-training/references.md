# 一次資料

最終確認: 2026-08-30

## 公式資料

### CuteTTS repository

- [OPPO-Mente-Lab/CuteTTS](https://github.com/OPPO-Mente-Lab/CuteTTS)
- [Official README](https://github.com/OPPO-Mente-Lab/CuteTTS/blob/main/README.md)

主に確認した内容:

- 約230Mのcontinuous autoregressive TTS
- 英語、中国語、フランス語、ドイツ語、スペイン語対応
- Python API、CLI、Web demo、streaming API
- `OPPOer/CuteTTS` と `OPPOer/CuteTTS-distill`
- Apache-2.0
- 公開treeにtraining entrypointがないこと

### Paper

- [CuteTTS: Efficient and High-Quality Speech Synthesis via Autoregressive Modeling of Continuous Latents](https://arxiv.org/abs/2608.08638)
- [Paper HTML v2](https://arxiv.org/html/2608.08638v2)

主に確認した内容:

- v1: 2026-08-09
- v2: 2026-08-26
- architectureとflow-matching objective
- semantically aligned causal sigma-VAE
- explicit speaker conditioning
- patch size 2、12.5 Hz latent、6.25 LM token/s
- model/component parameters
- base、VAE、distillationのtraining configuration
- 550,000時間の内部多言語corpus
- objective/subjective/efficiency evaluation

### Public checkpoints

- [OPPOer/CuteTTS](https://huggingface.co/OPPOer/CuteTTS)
- [OPPOer/CuteTTS-distill](https://huggingface.co/OPPOer/CuteTTS-distill)

主に確認した内容:

- `config.json`
- `tokenizer/`
- `weights/audio_vae/`
- `weights/speaker_encoder/`
- `weights/tts/`
- Audio VAE: 24 kHz、12.5 Hz、64-dim
- Speaker Encoder: 16 kHz、256-dim
- extended vocabulary size 16,385
- Qwen3系configと7-layer keep設定

### 学習コードの公開状況（2026-09-10確認）

**確認済み: 公式にもコミュニティにも学習コードは存在しない。**

- [upstream issue #1](https://github.com/OPPO-Mente-Lab/CuteTTS/issues/1)（2026-09-06）で
  学習コードとconfigの公開が要望され、論文第一著者のYuqian Zhang（`rulerman`）が
  2026-09-07に回答した: 「training configurationは論文のappendixを参照。
  training codeは当面open-source化の予定はない。簡単なfine-tuningならAIで実装できるだろう」
- GitHubのfork 8件のうち、upstream以降にpushがあるのはこのforkだけ
- GitHubのcode検索で `CuteTTSModel` / `from cutetts` / `OPPOer/CuteTTS` を探しても、学習系の実装は見つからない。
  ヒットしたのはupstream本体と、推論の組み込み（`xRetr00/Marvi-OS`）、awesome-list、論文要約だけ
- Hugging Face: `OPPOer/CuteTTS` を base_model に指定した派生modelは0件。
  `niobures/CuteTTS` はupstreamの推論コードzip + weight + 論文PDFのミラーで、学習コードは入っていない。
  `hugging-apps/cutetts-demo` はHF公式チームが作った推論demo
- 論文appendixが明記している精度は **VAEのFP32だけ**。base / distillの学習精度は書かれていない。
  appendixのhyperparameterは[04章](04-training-implementation.md)第3節の値と一致する

## ローカル実装

このforkで確認した主要file:

- [`../../README.md`](../../README.md)
- [`../../pyproject.toml`](../../pyproject.toml)
- [`../../src/cutetts/modeling/configuration.py`](../../src/cutetts/modeling/configuration.py)
- [`../../src/cutetts/modeling/model.py`](../../src/cutetts/modeling/model.py)
- [`../../src/cutetts/modeling/tokenizer.py`](../../src/cutetts/modeling/tokenizer.py)
- [`../../src/cutetts/modeling/processor.py`](../../src/cutetts/modeling/processor.py)
- [`../../src/cutetts/modeling/diffusion_head.py`](../../src/cutetts/modeling/diffusion_head.py)
- [`../../src/cutetts/modeling/audio_adapter.py`](../../src/cutetts/modeling/audio_adapter.py)
- [`../../src/cutetts/inference/generation.py`](../../src/cutetts/inference/generation.py)
- [`../../src/cutetts/inference/conditioning.py`](../../src/cutetts/inference/conditioning.py)
- [`../../src/cutetts/runtime.py`](../../src/cutetts/runtime.py)

## 根拠の強さ

### 公式に確認済み

- architecture、公開config、公開weight構成
- 公式pretraining hyperparameters
- 公式benchmark条件と結果
- 公式対応言語
- ライセンス

### ローカル実装で確認済み

- 現在のforkがinference-orientedであること
- Tokenizer、model component、streaming generationのinterface
- training entrypoint、dataset、loss、trainerが未実装であること

### プロジェクト提案

- 日本語10〜30時間から段階的に拡大すること
- VAE/Speaker Encoderをfreezeして始めること
- full fine-tuningを主案にすること
- 日本語90〜95% + replay 5〜10%の候補
- RTX 4090 1台でのPoC可能性、H100 8台の容量計画候補
- Japanese VAEやdistillationを後段へ回すこと

プロジェクト提案は公式が日本語学習について推奨した値ではありません。実験結果によって更新します。
