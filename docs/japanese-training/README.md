# CuteTTS 日本語継続学習

最終更新: 2026-09-01

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
- **P2: 学習forward復元**（flow-matching loss、stop target、packing、trainer。変異テスト9/9検出）
- **S0: 10〜30時間 overfit** → **主ゲート通過**。in_domain CER **35.8% → 28.4%**
- **ASRの誤り床を測定**（人間の実音声で CER 10.4%）。CERを0%基準で読まない根拠
- **S1の前処理**（gol 5ゲーム・**265.7時間**・1,197話者ID）。成果物は
  [tts-dataset/cutetts-ja-latents](https://huggingface.co/datasets/tts-dataset/cutetts-ja-latents)（gated: manual）
- 実装: `src/cutetts/training/` 15モジュール + `scripts/` 12本、**テスト全件PASS**

数値の一覧は [実測結果まとめ](RESULTS.md)、S0のゲートと結果は [S0-GATE.md](S0-GATE.md) を参照。

### 未実施

- **S1: 100〜500時間 PoC の学習** ← 次はここ。**前処理は完了済み**（265.7時間）
- P1e Pass B（gol全体7 TB。S2直前まで実施しない）
- 英語・中国語のforgetting用の固定評価set
- 日本語母語話者による主観評価（S0の生成音声は `artifacts/` にある）
- モデル公開範囲の決定（R-009の残件）

### S0で分かったこと

- **日本語継続学習は成立する。** 7.15時間・3000step・9分でCERが7.4pt改善した。
- **未学習baseが既に部分的な日本語を出す。** 「音声が出る」はゲートにならない。
- **out-of-domain（数字・固有名詞）は改善しない。** 74.7% → 76.4%（[R-010](07-risks-and-decisions.md)）。
  golのcorpusで数字を含む文は1.3%しかなく、**データ量では解決しない**。S1のゴールから外した（D-026）。
- **CERには約10%の床がある。** 人間の実音声を同じ経路で測ると 10.4%。
  S0の28.4%のうちTTS由来は約18pt。
- **クラスタの粒度は用途ごとに逆向きの要求を持つ**（[R-014](07-risks-and-decisions.md)）。
  PairSamplerの単位が粗いと別の声をreferenceにして学習し、splitの単位が細かいと
  同じ声がtrainとzero-shotに現れる。単位を2つに分けた（D-027）。
- **学習ループの損失だけでは成否を判断できない。** 1回目の学習は同じ4発話を
  3000step繰り返しており、loss 0.003 は丸暗記だった（[R-012](07-risks-and-decisions.md)）。

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

1〜4はすべて検証済み（4は7.15時間で成立）。次の問いは
**100〜500時間で品質とzero-shot voice cloningが実用水準に達するか**。

### 初期の推奨構成

| Component | 方針 | 状態 |
|---|---|---|
| Audio VAE | freeze | **確定**（P1c: CER差 +0.58pt。D-003） |
| Speaker Encoder | freeze | 提案（zero-shot SIMで再判定。D-004） |
| Text embedding | train | S0で実施 |
| Patch Encoder | train | S0で実施（freeze版との比較はD-005として残る） |
| Causal LM backbone | train | S0で実施 |
| Flow/Diffusion Head | train | S0で実施 |
| Stop Predictor | train | S0で実施（loss -72〜80%） |

S0では上記すべてを同時にtrainするfull fine-tuningで通過しました。
VRAMは4.15 GB（microbatch 1）で、16 GBに収まることを確認済み（D-006確定）。
Patch Encoderのfreeze版との比較（D-005）はS1で行います。

## 重要な境界

- 公式モデルの英語・中国語等での公開評価値は、日本語品質を保証しません。
- 約10,000時間という量だけでは成功を保証できません。話者多様性、音質、文字起こし精度、権利、重複、収録分布が重要です。
- 論文のpretraining設定は再現の参考値であり、継続学習の最適設定ではありません。
- RTX 4090 1台やH100 8台という規模感は初期の容量計画上の仮説です。実際の必要量はtraining forward完成後のmicrobatch実測で決めます。
- 「パイプラインを準備した」と「日本語学習が成功した」は別の状態として記録します。
