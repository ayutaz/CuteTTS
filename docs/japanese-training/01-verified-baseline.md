# 確認済みベースラインとアーキテクチャ

最終更新: 2026-09-01

## 1. CuteTTSの位置づけ

CuteTTSは、離散codec tokenではなく連続Audio VAE latentをpatch単位で自己回帰生成するTTSです。全体は次の役割分担になっています。

```text
target text
    +
reference audio
    │
    ├─ Speaker Encoder ────────────┐
    ├─ Causal Audio VAE Encoder ─┐ │
    │                            │ │
    ▼                            ▼ ▼
text tokens + reference patches + speaker embedding
    │
    ▼
Patch Encoder + Causal LM backbone
    │
    ▼
Flow-Matching Diffusion Head
    │
    ▼
next continuous latent patch
    │
    ▼
Causal Audio VAE Decoder
    │
    ▼
streaming waveform
```

確認済みの特徴:

- 24 kHz waveformを12.5 Hz、64次元の連続latentへ変換する。
- 2 latent frameを1 patchとして扱う。
- LMの自己回帰token rateは `12.5 / 2 = 6.25 token/s`。
- Patch Encoderがpatch内部を集約し、Causal LMがpatch間の依存を扱う。
- Diffusion Headが、LM hidden state、直前patch、speaker embeddingを条件に次patchを生成する。
- Stop Predictorが生成終了を判定する。
- Causal VAE Decoderが生成済みpatchを逐次waveformへ変換できる。

先行会話には一般例として4 frame/patchの説明がありましたが、公開CuteTTSの実設定は2 frame/patchです。

## 2. 公開モデルの構成

論文v2と公開configで確認できる値です。

| Component | 主な設定 | Parameters |
|---|---|---:|
| Audio VAE | 24 kHz、12.5 Hz、64-dim、causal sigma-VAE | 127.9M |
| Patch Encoder | 2 layers、hidden 1024、16/2 attention/KV heads、FFN 4096 | 31.0M |
| Causal Backbone | 7 layers、hidden 1024、16/8 attention/KV heads、FFN 3072 | 126.9M |
| Diffusion Head | 4 layers、hidden 1024、16/2 attention/KV heads、FFN 4096 | 70.5M base |
| TTS全体 | Patch Encoder + Backbone + Diffusion Head等 | 228.6M base |
| Distilled TTS全体 | guidance/step embeddingを追加 | 231.8M |

公開configのQwen3内部設定には28 layersが含まれますが、`lm_keep_num_hidden_layers=7` により実際のbackboneは7 layersに切り詰められます。

## 3. Audio VAE

確認済み:

- 24 kHz入力
- frame rate 12.5 Hz
- latent dimension 64
- fixed posterior standard deviation 0.15
- encoder stride `3/5/8/16`
- decoder stride `16/8/5/3`
- causal convolutional architecture
- DAC由来の実装要素を含む
- 学習時にはfrozen WavLM teacherとのsemantic alignmentを使用

VAEの公開weightはTTS本体とは別componentです。ローカル実装の `CuteTTSProcessor.acoustic_feature_forward()` はVAE posteriorのmeanを音響特徴として返します。

## 4. Speaker Encoder

公開checkpointのSpeaker EncoderはTTS本体と別componentです。

| 項目 | 値 |
|---|---:|
| Architecture | ECAPA-style student |
| Input sample rate | 16 kHz |
| Mel bins | 80 |
| Embedding dimension | 256 |

論文では、frozen WavLM Large speaker-verification teacherからstudentをdistillしたと説明されています。ローカル推論ではreference audioからこの256次元embeddingを計算し、LM側のprojectionとDiffusion Head側のadaptive normalizationの両方へ渡します。

## 5. Baseとdistill

### CuteTTS base

- voice cloning時はLM-level CFGを使用
- 公開論文の既定条件はguidance weight 2
- 各patchについてconditional/unconditionalの2 branchを使う
- 各branchで10 diffusion-head NFE

### CuteTTS-distill

- guidance strengthとstep sizeをDiffusion Headの条件に追加
- unconditional branchを使わず、1/2/4 step budgetを同一checkpointで扱う
- 論文の主評価は4 NFE

日本語継続学習はbaseから始め、品質が成立した後で日本語baseをteacherとしてdistillする順番を採用します。

## 6. 公式評価の読み方

公式READMEと論文は、約230Mのbaseモデルについて、RTX 4090で約49 msの平均first-audio latency、RTF 0.184を報告しています。distill版は同じ評価手順で約37.6 ms、RTF 0.109です。READMEの「約40 ms、約9倍real time」はdistill側の丸めた説明に相当します。

これらは、固定された英語voice-cloningベンチマーク、単一RTX 4090、warm-service、batch size 1などの条件に基づく公式報告です。このforkで再測定した結果でも、日本語で達成済みの結果でもありません。

## 7. 現在のローカルfork

確認時点:

- branch: `feat/japanese-training`
- 起点: `main`
- snapshot commit: `ca9dbd3b82c05f5b067466088449b93ee2aa5a0c`
- `origin`: `https://github.com/ayutaz/CuteTTS.git`

現在の実装には次が含まれます。

- Python API、CLI、Web demo
- base/distill checkpoint loader
- Tokenizer、Processor、Segment Manager
- Audio VAE adapterとstreaming decoder
- Speaker Encoder loader
- Patch Encoder、Qwen3系backbone、Diffusion Head、Stop Predictor
- streaming/offline推論

現在の実装に含まれないもの:

- training entrypoint
- 学習Dataset/DataLoader
- reference/target pair sampler
- latent cache生成
- sequence packer
- training用の統合forward
- flow-matching training loss
- stop target/loss
- optimizer/scheduler/checkpoint resume
- distributed training設定
- 日本語frontend
- 日本語評価pipeline

`src/cutetts/modeling/model.py` と `src/cutetts/modeling/processor.py` は明示的にinference-onlyと記述されています。公開moduleは学習再現の基礎になりますが、そのまま `train.py` を実行できる状態ではありません。

**2026-08-31追記:** 学習pathは `src/cutetts/training/` として**新規に追加**され、
S0（7.15時間の日本語継続学習）を通過しました。既存の推論pathには手を入れていません。
`training_forward` は公開moduleの `prepare_input_embeds` / `forward_lm` / `head._predict` を
そのまま呼び、学習時だけ `config.scale_acoustic_latent` を一時的に無効化します
（正規化はdataset側で済ませるため）。詳細は [04章](04-training-implementation.md)。

## 8. ライセンス

リポジトリ本体はApache License 2.0です。DAC由来部分とF5-TTS由来部分にはそれぞれMITライセンスのnoticeがあります。

これはforkで学習機能を追加できることを示しますが、次は別途確認が必要です。

- 学習データごとの利用・再配布条件
- 音声提供者の同意とvoice cloning用途
- 公開checkpointへデータ由来の権利が及ぼす制約
- モデルカード、NOTICE、attribution

## 関連資料

- [日本語継続学習の方針](02-continual-training-strategy.md)
- [学習コード復元・実装計画](04-training-implementation.md)
- [一次資料](references.md)
