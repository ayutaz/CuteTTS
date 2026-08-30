# 学習コード復元・実装計画

最終更新: 2026-08-30

## 1. 現在の境界

公開リポジトリは推論に必要な主要moduleとweight loaderを含みますが、完全なtraining pipelineは含みません。

再利用できる実装:

- `CuteTTSModel`
  - Patch Encoder (`locenc`)
  - acoustic-to-LM projection
  - speaker-to-LM projection
  - Qwen3系Causal Backbone
  - `AudioDiTHead`
  - Stop Predictor
- `CuteTTSProcessor`
  - Tokenizer
  - Audio VAE adapter
  - reference prompt segment構築
- `SegmentManager`
  - text/speech/speaker slotの結合とmask
- inference
  - autoregressive patch生成
  - CFG branch
  - streaming/offline VAE decode

不足している実装:

- 学習sample schemaとloader
- reference/target pair sampler
- raw audioからlatentを作るcache pipeline
- text/reference/targetを学習sequenceへ組み立てる処理
- sequence packingとattention/padding mask
- teacher-forcing用のshift
- flow-matching objective
- stop targetとloss
- conditional dropout
- optimizer、schedule、gradient scaling/clipping
- checkpoint save/resume
- single/multi-GPU launch
- validation生成とmetric集計

## 2. 論文から確認できる学習式

### Audio VAE

公式VAEは次のlossを組み合わせます。

```text
15.0 * multi-resolution mel reconstruction
+ 1.0 * adversarial
+ 2.0 * feature matching
+ 0.1 * KL
+ 1.0 * semantic alignment
```

VAEは初期日本語継続学習ではfreezeするため、最初のtraining pipelineにこのlossを実装する必要はありません。将来Japanese VAEへ進む場合の別milestoneとします。

### TTS flow matching

clean target patch `P` とGaussian noise `xi` に対し、論文は次のlinear pathを使います。

```text
x_t = (1 - t) * xi + t * P
target velocity = P - xi
```

`t = sigmoid(u), u ~ N(0, 1)` とし、Diffusion Headの予測velocityとtarget velocityのMSEを最小化します。

公開pretrainingでは各target patchを4つの独立noise/timeで複製します。日本語継続学習でもまず同じ意味論を再現し、その後メモリ・速度とのtrade-offを測ります。

### Stop loss

公開modelは2-class Stop Predictorを持ちますが、論文v2はstop targetの細部、padding patchとの関係、loss weightを十分に規定していません。実装時に次を明示します。

- continuation/stop labelを置く位置
- padding patchをlossから除外するmask
- target末尾とpacked sample境界
- class imbalance対策
- flow lossとのweight

推論で期待される停止位置と一致するunit/integration testが必要です。

## 3. 公式pretraining設定

次は論文v2の確認済み設定です。継続学習へそのまま採用する値ではなく、再現の基準です。

### CuteTTS base

| 項目 | 公式設定 |
|---|---:|
| Steps | 1,000,000 |
| Global batch | 最大81,920 packed tokens |
| Target patch copies | 4 |
| Condition dropout | 0.1 |
| Optimizer | AdamW |
| Peak learning rate | 5e-4 |
| Betas | 0.9 / 0.95 |
| Weight decay | 0.01 |
| Warmup | 5,000 steps |
| Schedule | cosine |

### Audio VAE

| 項目 | 公式設定 |
|---|---:|
| Steps | 1,000,000 |
| Precision | FP32 |
| Effective batch | 256 |
| Crop | 2.5 seconds |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Betas | 0.8 / 0.99 |
| Weight decay | 0.01 |
| Decay | exponential, gamma 0.999996 |

### CuteTTS-distill

| 項目 | 公式設定 |
|---|---:|
| Updated module | Diffusion Head only |
| Steps | 100,000 |
| Global batch | 最大65,536 packed tokens |
| Target patch copies | 2 |
| Peak learning rate | 1e-5 |
| Warmup | 1,000 steps |
| Schedule | cosine |

日本語baseが成立するまでdistillationは実装優先度を下げます。

## 4. 継続学習で決め直す値

公式の1M-step pretraining設定は、550,000時間の内部多言語corpus向けです。継続学習では次を実測で決めます。

- learning rate
- warmup steps/ratio
- packed token budget
- microbatch
- gradient accumulation
- reference duration
- target duration
- mixed precision
- activation checkpointing
- flow target copies
- gradient clipping
- evaluation/checkpoint間隔
- 日本語/replay比率

最初は短いLR range testまたは複数の小規模runを行い、本番GPUを確保する前にVRAMとthroughputを測ります。

## 5. Training sequenceの復元

推論sequenceと論文の条件から、training sampleは概念的に次を含みます。

```text
instruction text
+ speaker slot
+ reference latent patches
+ target text
+ teacher-forced target patch history
```

各target patch位置で必要なもの:

- causal LM hidden state
- speaker embedding
- previous target patch
- current clean target patch
- flow time/noise
- stop label

実装時に必ず検証する点:

- reference latentとtarget latentのnormalization
- Patch Encoderへ入れるtensor shapeとpatch padding
- speech/text/speaker mask
- LM position ID
- packed sample間のattention遮断
- first target patchのprevious condition
- odd latent frame数のpadding
- target末尾patchとstop位置
- condition dropout時に消す条件
- loss maskと分母

## 6. Latent cache

VAEをfreezeする初期段階では、毎step waveformをencodeするより、manifestに対応したlatent cacheを作る方が効率的です。

cacheに記録するもの:

- source `utterance_id`
- source audio checksum
- VAE checkpoint revision/checksum
- preprocessing version
- latent dtype/shape
- original sample数とduration
- normalization情報

cache manifestやstatus file自身は更新されるため、immutable input checksumと分けます。VAE checkpointやpreprocessingが変わったcacheを混在させません。

## 7. 提案するコード構成

実装時の候補です。現時点では未作成です。

```text
configs/
└─ japanese/
   ├─ tokenizer-coverage.yaml
   ├─ vae-reconstruction.yaml
   ├─ overfit.yaml
   └─ continual.yaml

scripts/
├─ analyze_japanese_tokenizer.py
├─ evaluate_japanese_vae.py
├─ prepare_japanese_manifest.py
├─ cache_audio_latents.py
└─ train_continual.py

src/cutetts/training/
├─ dataset.py
├─ pairing.py
├─ packing.py
├─ collator.py
├─ objectives.py
├─ trainer.py
└─ checkpointing.py

tests/training/
├─ test_pairing.py
├─ test_packing.py
├─ test_flow_objective.py
├─ test_stop_targets.py
├─ test_condition_dropout.py
└─ test_checkpoint_resume.py
```

現状の `.gitignore` は `tests/` と `.pytest_cache/` を除外しています。上記の `tests/training/` を
追加する場合は、先に `tests/` の除外を解除する必要があります。また、`pyproject.toml` には
テスト・lintの依存も設定も含まれていないため、テスト実行方法（pytestの導入とconfig）は
P2着手時に決めます。

## 8. 実装の検証境界

最低限必要な証拠:

- tiny deterministic batchでshape/mask/lossを固定
- packed/unpackedで同じsample lossになる
- referenceとtargetの取り違えを検出
- stop labelの1位置ずれを検出
- condition dropoutが指定条件だけを落とす
- frozen VAE/Speaker Encoderへgradientが流れない
- train対象moduleへgradientが流れる
- checkpoint resume前後で次stepが一致する
- 1 sampleを意図的にoverfitできる
- inference pathで保存checkpointをloadできる

学習lossが下がることだけでは十分ではありません。配線を外した場合にテストが失敗すること、生成音声と固定evaluationで効果を確認することが必要です。

## 9. 公式学習コードが公開された場合

upstreamのtraining/fine-tuningコードが公開された場合は、自前実装を即座に置換せず、次を比較します。

- sample schema
- reference sampling
- packing/mask
- stop target
- flow objective
- condition dropout
- optimizer/schedule
- checkpoint互換性

差分を記録した上で、公式実装へ寄せるか、fork側の実装を維持するか決めます。upstream公開を監視する価値はありますが、それを前提に日本語データ準備を止めません。

## 関連資料

- [確認済みベースライン](01-verified-baseline.md)
- [段階的な実験ロードマップ](05-experiment-roadmap.md)
- [一次資料](references.md)
