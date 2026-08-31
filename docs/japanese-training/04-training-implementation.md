# 学習コード復元・実装計画

最終更新: 2026-08-31

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

### S0で実際に使った値（2026-08-31・通過実績あり）

| 値 | S0での設定 | 根拠 |
|---|---|---|
| learning rate | **2e-5** | 論文pretrainingの 5e-4 の1/25。継続学習なので大幅に下げた |
| warmup | 100 step | |
| scheduler | cosine（floor 10%） | |
| microbatch | 4 | peak VRAM 4.15 GB |
| gradient accumulation | なし | 16 GBに収まるため不要 |
| target duration上限 | 188 patch（約30秒） | |
| reference duration | 10秒目標（`target_reference_seconds`） | 推論側の30秒想定と実発話長4.55秒の乖離を埋める |
| mixed precision | bf16（headのみfp32） | checkpointのdtype構成に従う |
| activation checkpointing | 未使用 | 不要 |
| flow target copies | 4 | 論文どおり |
| gradient clipping | 1.0 | |
| condition dropout | 0.1（speaker + reference、joint） | |
| 日本語/replay比率 | 100%日本語（replayなし） | D-009はS1で判断 |

**bf16の注意:** 値が1.0付近のパラメータ（LayerNorm weight）は、
lr×weight_decay 程度の更新がbf16の分解能（相対4e-3）に埋もれて消える。
S0の実測でも locenc のLayerNorm weight は3000step後も変化がゼロだった。
S1で学習が停滞する場合、fp32 master weight の導入を検討する。

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

## 7. 実際のコード構成（2026-08-31時点）

```text
configs/japanese/     tokenizer-coverage.yaml, vae-reconstruction.yaml

scripts/
  P0/P1  reproduce_baseline, analyze_japanese_tokenizer, evaluate_japanese_vae,
         prepare_japanese_manifest, cache_audio_latents, build_voice_clusters
  S0     build_eval_set, evaluate_japanese_cer, train_continual,
         diagnose_flow_loss, check_reference_following,
         benchmark_training_memory, vastai_bootstrap.sh

src/cutetts/training/
  P1     artifacts, manifest, text_rules, pairing, latents,
         speaker_cache, voice_clusters
  P2     objectives, collator, dataset, forward, packing, checkpointing, prompt

tests/training/        全件PASS（slowマーカーは実checkpointを要する）
tools/                 mutation_check（テストが実際に効くかの検証）
```

`trainer.py` は作らず、`scripts/train_continual.py` が P2 の部品を繋ぐだけの
薄いdriverになっています。学習semanticsは `src/cutetts/training/` 側にあります。

`pyproject.toml` に `[project.optional-dependencies] dev`（pytest / pyyaml）と
`[tool.pytest.ini_options]` を追加済み。`.gitignore` の `tests/` 除外も解除済みです。
`gpu` マーカーは既定で除外されます（`addopts = "-q -m 'not gpu'"`）。lint設定は未整備。

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

### S0で追加した検証（実際に事故を踏んだため）

| 検証 | テスト | 何を防ぐか |
|---|---|---|
| 条件付けの因果性 | `test_leakage.py` | target patch i を予測する hidden が patch i を見ていないこと。摂動の影響行列が厳密に上三角になることを確認する |
| sampler が進むこと | `test_pair_stream.py` | `PairSampler.sample()` を毎step呼ぶと同じペアが返る。200step引けば全発話に触れることを固定 |
| 損失の絶対値 | `scripts/diagnose_flow_loss.py` | 「条件付けを使わない予測器（常に0）」の loss（約2.0）を併記し、何に対して小さいのかを示す |
| train と dev の乖離 | 同上 | 記憶と汎化を切り分ける。未学習baseも同じ経路で測る |

**S0の1回目はこれらが無かったために、丸暗記を成功と読み違えた。**
flow matching は原理的に velocity を完全には当てられないので、
loss が 0 に近づくこと自体が異常のサインになる（[R-012](07-risks-and-decisions.md)）。

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
