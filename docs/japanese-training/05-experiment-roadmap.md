# 段階的な実験ロードマップ

最終更新: 2026-08-30

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

## Stage 0: 10〜30時間 overfit

先行案の10〜50時間を、最初の既定範囲10〜30時間に絞ります。30時間で判断できない場合のみ50時間へ拡大します。

### 目的

品質競争ではなく、既存VAE + 日本語textから日本語speechを学習できるか確認します。

### 作業

- 少数の高品質話者
- data/minibatchを目視可能な規模に固定
- raw textまたはP1で選んだ最小frontend
- base checkpointからTTS本体を継続学習
- train/dev sampleを短い間隔で生成
- learning rateとfreeze variantを小規模比較

### Success evidence

- target textに対応した日本語音声が出る
- loss低下だけでなく、聞き取れる発音改善がある
- textを入れ替えると出力内容も追随する
- reference speakerを入れ替えるとspeaker identityが追随する
- 未学習文でも完全なmemorization以上の挙動が見える

### Stop/rollback条件

- VAE reconstruction自体に重大な欠陥が見つかる
- Tokenizerが情報を失っている
- target/reference leakageで見かけ上成功している
- stop headが学習できず無限生成または早期停止する
- NaN/overflowが再現する

## Stage 1: 100〜500時間 PoC

100〜300時間で最初の判断を行い、結果が良い場合に500時間まで拡大します。

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

- Patch Encoder train vs freeze
- 100%日本語 vs replay混合
- raw/normalized text
- 必要ならtext + reading
- full fine-tuning vs部分freeze

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

これは実測結果ではありません。GPU契約前に次を測ります。

- parameter/optimizer/gradient memory
- sequence length別activation memory
- microbatch 1のpeak VRAM
- gradient accumulation時のthroughput
- activation checkpointingの効果
- latent cache I/O
- single GPUとdistributedのscaling

最終GPU数は、目標終了時間、利用可能budget、failure recovery、checkpoint I/Oを含めて決定します。
