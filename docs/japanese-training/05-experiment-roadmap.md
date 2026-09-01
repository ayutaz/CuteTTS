# 段階的な実験ロードマップ

最終更新: 2026-09-01

## 原則

- 10,000時間を最初から投入しない。
- 1段階につき主な不確定要素を1つずつ減らす。
- 次段階へ進む前に、固定evaluationと音声sampleを保存する。
- 実装済み、実行済み、品質合格を別statusとして扱う。
- seed、checkpoint revision、manifest checksum、config、hardware、software versionを記録する。

## Phase P0: 公開baseの再現

### 目的

forkと公開checkpointが正しく動く基準を作ります。

### 作業

- baseとdistillのweightをrevision固定で取得
- 英語TTSとvoice cloningのoffline推論
- streaming推論
- 同じ入力・seedで再実行
- first-audio latency、RTF、VRAMのローカル基準値
- 出力音声、config、実行logを保存

### Exit gate

- base checkpointを再現可能にloadできる
- offline/streamingで破損のないwaveformが出る
- 固定prompt/reference/seedのartifactが保存される

## Phase P1: 日本語preflight

### 目的

training codeがなくても確認できる最大リスクを先に潰します。

### 作業

1. 既存Tokenizerの日本語coverage
2. 公式Audio VAEの日本語encode/decode
3. 日本語metadata validator
4. speaker-disjoint evaluation split
5. fixed Japanese challenge text

### VAE評価

- waveform reconstruction
- mel distance
- PESQ/STOI（適用条件を固定）
- ASR CERのoriginal/reconstruction差
- UTMOS等の自動品質指標
- 日本語話者によるblind listening

### Exit gate

- Tokenizer方針を「既存維持 / 互換拡張 / reading追加」のいずれかへ決められる
- 公式VAEをfreezeして進める根拠、または再学習が必要な根拠がある
- 学習・評価manifestが分離されている

## Phase P2: Training forward復元

### 目的

公開moduleから、最小のteacher-forced training stepを構成します。

### 作業

- latent cache
- reference/target pair sampler
- unpacked single-sample batch
- flow-matching objective
- stop target/loss
- condition dropout
- optimizer step
- checkpoint save/load
- inference checkpoint互換性
- packingはunpacked correctnessの後に追加

### Exit gate

- deterministic tiny batchでlossが再現する
- 期待するmoduleだけにgradientが流れる
- 1 utteranceをoverfitできる
- save/resume後の次stepが一致する
- packingがunpackedの結果を変えない

## Stage 0: 10〜30時間 overfit — **完了（2026-08-31）**

先行案の10〜50時間を、最初の既定範囲10〜30時間に絞りました。
**実際には 7.15時間で主ゲートを通過**したため、拡大は不要でした。
結果の全文は [S0-GATE.md](S0-GATE.md)。

| 項目 | 実績 |
|---|---|
| データ | 5,431発話 / 7.15時間 / 63 voice cluster |
| 設定 | 3000 step、lr 2e-5、batch 4、warmup 100、condition dropout 0.1 |
| 所要 | 9分（vast.ai RTX 3090）、peak VRAM 4.15 GB |
| in_domain CER | 35.8% → **28.4%** |
| reference追随 | 4話者4択で 12/12 |

### 目的

品質競争ではなく、既存VAE + 日本語textから日本語speechを学習できるか確認します。

### 作業

- 少数の高品質話者
- data/minibatchを目視可能な規模に固定
- raw textまたはP1で選んだ最小frontend
- base checkpointからTTS本体を継続学習
- train/dev sampleを短い間隔で生成
- learning rateとfreeze variantを小規模比較

### Success evidence（実績）

| 項目 | 結果 |
|---|---|
| target textに対応した日本語音声が出る | **ゲートにならなかった**。未学習baseで既に成立していたため、CERの改善幅に置き換えた |
| loss低下だけでなく発音改善がある | CER -7.4pt。実聴取は未実施 |
| textを入れ替えると出力内容も追随する | 達成（未学習52文でCER 28.4%。追随しなければ約100%） |
| reference speakerを入れ替えるとspeaker identityが追随する | 達成（12/12、自己0.833 vs 他者0.600） |
| 未学習文でも完全なmemorization以上の挙動が見える | 達成（評価文は学習manifest外、dev-zero-shotも悪化せず） |

### Stop/rollback条件（確認結果）

| 条件 | 該当 |
|---|---|
| VAE reconstruction自体に重大な欠陥 | なし（P1c: CER差 +0.58pt） |
| Tokenizerが情報を失っている | なし（P1b: `<unk>` 0%） |
| target/reference leakageで見かけ上成功 | なし（`tests/training/test_leakage.py` で因果性を確認） |
| stop headが学習できず無限生成または早期停止 | なし（stop loss 全splitで -72〜80%） |
| NaN/overflowが再現 | なし |

**ただし1回目の学習は別の理由で無効だった。** `PairSampler.sample()` の誤用で
3000step全部が同じ4発話になり、loss 0.003 は丸暗記だった（[R-012](07-risks-and-decisions.md)）。
上のどの停止条件にも該当せず、損失曲線だけ見れば成功に見えた。
**Stage 1以降も、学習ループの損失だけで判断しない（D-025）。**

## Stage 1: 100〜500時間 PoC — **次はここ**

100〜300時間で最初の判断を行い、結果が良い場合に500時間まで拡大します。

### 前処理は完了（2026-09-01）

| 項目 | S0 | S1 |
|---|---:|---:|
| 学習データ | 7.15時間 / 63 cluster | **265.7時間 / 894 cluster** |
| zero-shot の話者 | 3クラスタ | **119クラスタ** |

gol 5ゲーム（326時間・215 GB）を vast.ai 上で前処理し、latent cache だけを
[tts-dataset/cutetts-ja-latents](https://huggingface.co/datasets/tts-dataset/cutetts-ja-latents)
へ置いた。以降は約2 GBの取得だけで学習できる。所要 約9時間 / $2.8。

[R-013](07-risks-and-decisions.md)（zero-shot話者不足）は解消した。

**out_of_domain はS1のゴールに含めない（D-026）。** S0で 74.7% → 76.4% と
改善しなかったが、golのcorpusで数字を含む文は **1.3%**（10h以上の434ゲームでも
中央値0.9%）しかなく、データ量では解決しない。S1では読み誤りの内訳を
集計するだけにして（D-008）、対処はS2で決める。

### 目的

本格的な日本語化とzero-shot voice cloningの成立を確認します。

### 評価

- Japanese CER
- speaker similarity
- 自然性
- アクセント・prosody
- long-form stability
- streaming latency/RTF
- seen/zero-shot speaker差
- 既存言語forgetting

### 比較実験

- Patch Encoder train vs freeze（[D-005](07-risks-and-decisions.md)。S0はtrainのみ実施）
- 100%日本語 vs replay混合（D-009）
- raw/normalized text
- 必要ならtext + reading
- full fine-tuning vs部分freeze（S0はfull。16 GBに収まることを確認済み・D-006確定）

### Exit gate

- 日本語の内容一致と自然性が継続して改善
- zero-shot speakerで成立
- streaming生成が壊れていない
- Stage 2へ拡大するconfigを1つに絞れる

## Stage 2: 1,000時間

### 目的

データ分布と学習安定性を拡張し、`CuteTTS-JA v0.1` 候補を作ります。

### 作業

- speaker/domain/style samplingを本番相当にする
- checkpoint resumeと障害復旧を検証
- 固定evaluationを継続実行
- quality bucketごとの寄与を分析
- 生成artifactとtraining manifestを公開候補形式にまとめる

### Exit gate

- 100〜500時間PoCより複数軸で改善
- 重要なsubsetで回帰がない
- 3,000〜10,000時間へ拡大する費用対効果が説明できる

## Stage 3: 3,000〜10,000時間

### 目的

全データを使う最終日本語baseモデルを作ります。

### 作業

- accepted dataだけを段階的に追加
- speaker/domain/style exposureを監視
- checkpointを一定intervalで保持
- data mixtureとLR scheduleを変更した時点を記録
- 途中checkpointを同一evaluationで比較

「10,000時間を1 epoch回す」ではなく、packed token、audio seconds、optimizer steps、speaker exposureで予算を管理します。

### Exit gate

- 固定テストで最良checkpointを選定
- 主観評価を完了
- model card、データ説明、制限事項、ライセンスを準備
- baseモデルの再現可能な推論手順を用意

## Stage 4: Japanese Audio VAE（条件付き）

P1およびStage 1〜3の失敗分析でVAEがボトルネックと示された場合だけ実施します。

### 目的

24 kHz、12.5 Hz、64-dimの互換構造を保ちながら、日本語音声分布へVAEを適応します。

### 注意

- GAN discriminator、multi-resolution mel、WavLM teacherによりTTS本体より重い可能性がある
- VAE変更後はlatent分布が変わるためTTS本体の再学習が必要
- 既存TTS checkpointとの直接互換性を期待しない

## Stage 5: Guidance-step distillation

日本語baseの品質が確定した後にのみ実施します。

### 目的

- first-audio latency低減
- RTF低減
- 1/2/4 step budget
- baseに近い日本語品質とspeaker similarityの維持

### Exit gate

- 同じ日本語evaluation setでbaseとの品質差を測定
- 同一hardware/protocolでlatencyとRTFを比較
- baseとdistillの両方を保持

## GPU計画

先行案:

- 開発PoC: RTX 4090 24 GB ×1でも可能性あり
- 大規模run: H100 80 GB ×8を初期容量計画の候補

### 実測値（S0前段・S0、vast.ai RTX 3090）

| 項目 | 実測 |
|---|---|
| microbatch 1 の peak VRAM（full fine-tuning、bf16） | **4.15 GB** |
| throughput | **150〜177 ms/step**（batch 4、target 188 patch上限） |
| 16 GBでfull fine-tuningが載るか | **載る**（D-006確定） |
| S0（7.15時間・3000step）の実所要 | **9分** |

**開発PoCにH100は不要。** 24 GB級どころか16 GBで足りる。
S1の規模（100〜500時間）でも、同じmicrobatchなら1 GPUで回せる見込み。

まだ測っていないもの:

- gradient accumulation時のthroughput
- activation checkpointingの効果
- latent cache I/O（S1のデータ量で律速するか）
- single GPUとdistributedのscaling

最終GPU数は、目標終了時間、利用可能budget、failure recovery、checkpoint I/Oを含めて決定します。
