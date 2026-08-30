# リスク、意思決定、未解決事項

最終更新: 2026-08-30

## 1. 意思決定記録

| ID | 内容 | 状態 | 根拠・次のgate |
|---|---|---|---|
| D-001 | 既存base checkpointから継続学習する | 決定済み | ユーザーの明示方針 |
| D-002 | distill checkpointを最初の起点にしない | 提案採用 | base適応後にdistillする方が分析しやすい |
| D-003 | 初期はAudio VAEをfreeze | 提案採用 | P1の日本語再構成で再判定 |
| D-004 | 初期はSpeaker Encoderをfreeze | 提案採用 | zero-shot SIMで再判定 |
| D-005 | Patch Encoderをtrainする案を主案にする | 提案 | freeze variantとStage 0/1で比較 |
| D-006 | 最初はfull fine-tuning | 提案 | VRAM・安定性・forgettingで再判定 |
| D-007 | 既存Tokenizerを先に測る | 提案採用 | coverage reportで維持/拡張を決定 |
| D-008 | Raw textから開始し、reading/accentを段階追加 | 提案採用 | 読み誤り分析で追加 |
| D-009 | 日本語90〜95% + replay 5〜10% | 未確定 | 100%日本語との比較が必要 |
| D-010 | Japanese VAEは条件付き | 提案採用 | 公式VAEがボトルネックの場合のみ |
| D-011 | 10〜30h → 100〜500h → 1,000h → 3,000〜10,000h | 提案採用 | 各stageのexit gateを満たして進む |
| D-012 | Guidance-step distillationは最後 | 提案採用 | 日本語base品質確定後 |

## 2. 主要リスク

### R-001: 公式training codeがない

影響:

- training semanticsの細部を誤る
- checkpointが学習できても公式設計を再現していない可能性

対策:

- 論文式と公開推論pathの両方から復元
- mask、shift、stop target、CFG dropoutを小さなtestで固定
- upstream公開時に構造diffを取る

### R-002: 日本語Tokenizer coverage不足

影響:

- `<unk>`による情報欠落
- 文字あたりtoken数増加
- 読み・固有名詞・数字の不安定化

対策:

- 数千文以上の固定coverage調査
- 既存ID互換の拡張
- raw text、normalized text、text + readingを段階比較

### R-003: 公式VAEが日本語音韻を十分保持しない

影響:

- TTS本体を改善してもCER・音質が頭打ち

対策:

- TTS学習前に日本語reconstructionを評価
- original/reconstructionのASR CER差とblind listening
- 問題が確認できた場合だけJapanese VAE

### R-004: Reference/target leakage

影響:

- 見かけ上のvoice cloning成功
- 未知話者で性能崩壊

対策:

- 同一utterance・近重複を禁止
- speaker-disjoint split
- pair provenanceをartifactへ保存

### R-005: Catastrophic forgetting

影響:

- 既存5言語の能力低下
- speaker/stop/streaming挙動の回帰

対策:

- 英語・中国語固定evaluation
- replay混合ablation
- freeze/full fine-tuning比較

### R-006: データ品質より時間数を優先する

影響:

- transcript error、noise、speaker label errorを大量学習
- zero-shot性能の低下

対策:

- accepted hoursとraw hoursを分離
- quality/speaker/domain分布を可視化
- 高品質subsetから段階拡張

### R-007: GPU規模の見積もり誤り

影響:

- 大規模GPU契約後にthroughput不足
- optimizer/activation/I/Oがボトルネック

対策:

- training forward完成後にmicrobatch benchmark
- 1 GPUでmemory breakdown
- distributed scalingを短時間runで測る

### R-008: Streaming品質の回帰

影響:

- offlineでは良いがchunk境界にartifact
- first-audio latencyやRTFが悪化

対策:

- 各stageでoffline/streaming両方を評価
- chunk timingと境界artifactを保存
- stop behaviorとlong-formを固定test

### R-009: データ・voice cloningの権利

影響:

- checkpointやdatasetを公開できない
- 意図しない声の再現、同意範囲逸脱

対策:

- dataset単位のlicense/consent追跡
- public/private modelの境界を先に決める
- model cardに用途・制限・known riskを記載

## 3. 未解決事項

### 目的と公開範囲

- 日本語専用性能を最優先するか、多言語能力を残すか。
- checkpointを公開するか、内部利用に限定するか。
- zero-shot voice cloningをどの利用条件で提供するか。

### データ

- 実際のaccepted hours、speaker数、speaker分布。
- transcriptの精度と取得方法。
- speaker IDの信頼性。
- style/domain/収録環境。
- ライセンスとmodel training/redistribution可否。

### Frontend

- 既存Tokenizerの実coverage。
- normalization仕様。
- G2P toolと辞書。
- reading/accentの入力表現。
- tokenizer拡張時の既存embedding互換。

### Training

- stop targetとloss weight。
- sequence packingの正確なmask。
- condition dropoutの対象。
- continual learningのLR。
- reference/target duration。
- replay比率。
- precision、activation checkpointing、distributed方式。

### Evaluation

- 固定する日本語ASR。
- speaker similarity model。
- 日本語母語話者による主観評価体制。
- 公開前の合格閾値。
- safety/abuse evaluation。

## 4. 次に確定すべき順序

1. 公開checkpointのローカル推論再現
2. データの権利・speaker分布・品質の概要
3. Tokenizer coverage
4. VAE reconstruction
5. metadata/split/reference pairing
6. training forwardとobjective
7. Stage 0のLR/freeze config
8. 日本語/replay比率
9. G2P/accentの追加可否
10. Stage 2以降のGPU予算

## 5. 変更ルール

この文書の決定を変更するときは、元の項目を消さずに状態と理由を更新します。実験による変更では、該当run ID、config、評価artifactへの参照を追加します。
