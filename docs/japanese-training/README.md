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

- forkをローカルへclone、`feat/japanese-training` ブランチで作業
- 計画文書の整備（01〜08 + データ棚卸し）
- **P0: 推論ベースライン再現**（`gate_passed: true`。base 7/7、distill 7/7）
- **P1a: データ実態調査**（accepted **10,466.4 h** / 18,279話者ID を確定）
- **P1b: Tokenizer coverage**（`<unk>` 0%、byte-fallback 9.66% → 既存Tokenizerで開始可）
- **P1c: Audio VAE 日本語再構成**（CER差 +0.58pt → **freeze確定**、Stage 4見送り）
- **P1d: manifest / split / voiceクラスタ**（t=0.92 較正、leakage 0件）
- **P1e: 前処理パス Pass A**（6,112発話、外挿 65.3 GB / 239 GPU時間）
- 実装: `src/cutetts/training/` 7モジュール + `scripts/` 5本、**テスト354件PASS**

数値の一覧は [P0/P1 実測結果まとめ](RESULTS.md) を参照。

### 未実施

- **P2: 学習forward復元**（flow-matching loss、stop target、packing、trainer）← 次はここ
- P1e Pass B（gol全体7 TB。S2直前まで実施しない）
- Stage 0以降の日本語学習と評価
- 固定評価set（text-challenge、英語・中国語のforgetting用）
- 日本語母語話者による主観評価
- モデル公開範囲の決定（R-009の残件）

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
10. [P0/P1 実測結果まとめ](RESULTS.md)
11. [一次資料](references.md)

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
