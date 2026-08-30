# CuteTTS 日本語継続学習

最終更新: 2026-08-30

## 目的

公式のbase checkpoint `OPPOer/CuteTTS` を起点に、次の性質を持つ日本語モデルを段階的に作ることを目標とします。

- 約230Mパラメータ級
- 24 kHz出力
- ストリーミング生成
- 高品質な日本語
- multi-speaker
- zero-shot voice cloning

これは目標仕様であり、現時点で日本語学習や日本語品質の検証が完了したことを意味しません。

## 現在地

### 完了

- forkをローカルへclone済み
- `main` から `feat/japanese-training` ブランチを作成済み
- 先行会話、ローカル実装、公式README、論文v2、公開checkpoint構成を調査済み
- 日本語データ候補の特定（gol-dataset 10,654 h、moe-speech-plus 152 GB）。
  実測値は [データ棚卸し](data-inventory.md) を参照

### 未実施

- 公式checkpointのローカル推論ベースライン
- 公式Tokenizerの日本語coverage測定
- 公式Audio VAEによる日本語再構成評価
- 日本語metadataの実データ検証
- gol-datasetの利用条件の確定（license記載がなく、S1以降の規模を左右する）
- 学習用forward、loss、packing、trainerの実装
- 日本語のoverfit test、PoC、継続学習
- 日本語モデルの主観・客観評価

## 読み方

各文書では、情報を次の状態に分けます。

| 状態 | 意味 |
|---|---|
| 確認済み | ローカル実装、公開checkpoint、公式README、論文v2のいずれかで確認できる |
| 決定済み | ユーザーが明示したプロジェクト方針 |
| 提案 | 現時点の推奨案。実験結果により変更し得る |
| 未確定 | 実測またはユーザー判断が必要 |

## 文書一覧

1. [確認済みベースラインとアーキテクチャ](01-verified-baseline.md)
2. [日本語継続学習の方針](02-continual-training-strategy.md)
3. [データセットと日本語frontend](03-data-and-frontend.md)
4. [学習コード復元・実装計画](04-training-implementation.md)
5. [段階的な実験ロードマップ](05-experiment-roadmap.md)
6. [評価計画](06-evaluation-plan.md)
7. [リスク、意思決定、未解決事項](07-risks-and-decisions.md)
8. [対応計画（実行フェーズ定義）](08-execution-plan.md)
9. [データ棚卸し](data-inventory.md)
10. [一次資料](references.md)

## 現時点の要約

### 決定済み

- ゼロからの学習ではなく、既存の `OPPOer/CuteTTS` base checkpointから日本語継続学習を開始する。
- `CuteTTS-distill` は最初の学習起点にしない。

### 最初に検証すること

1. 公式Tokenizerが日本語をどの程度表現できるか。
2. 公式Audio VAEが日本語の音質と発音情報を保ったまま再構成できるか。
3. 公開推論実装からtraining forward、flow-matching loss、stop loss、packingを再現できるか。
4. 10〜30時間規模で日本語へのoverfitが成立するか。

### 初期の推奨構成

| Component | 初期方針 | 状態 |
|---|---|---|
| Audio VAE | freeze | 提案 |
| Speaker Encoder | freeze | 提案 |
| Text embedding | train | 提案 |
| Patch Encoder | trainを既定とする | 提案 |
| Causal LM backbone | train | 提案 |
| Flow/Diffusion Head | train | 提案 |
| Stop Predictor | train | 提案 |

先行会話の途中には「最初だけPatch Encoderもfreezeする」案もありました。最新の整理ではPatch Encoderをtrainする案を主案とし、freeze案は比較実験またはメモリ不足時の代替案として残します。

## 重要な境界

- 公式モデルの英語・中国語等での公開評価値は、日本語品質を保証しません。
- 約10,000時間という量だけでは成功を保証できません。話者多様性、音質、文字起こし精度、権利、重複、収録分布が重要です。
- 論文のpretraining設定は再現の参考値であり、継続学習の最適設定ではありません。
- RTX 4090 1台やH100 8台という規模感は初期の容量計画上の仮説です。実際の必要量はtraining forward完成後のmicrobatch実測で決めます。
- 「パイプラインを準備した」と「日本語学習が成功した」は別の状態として記録します。
