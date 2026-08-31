# データセットと日本語frontend

最終更新: 2026-08-31

## 0. 現状（2026-08-31）

manifest と前処理は実装済みで、S0 まで動いています。

| 項目 | 実績 |
|---|---|
| manifest | `data/manifests/all_clustered.jsonl`（`scripts/prepare_japanese_manifest.py`） |
| 実際に学習へ投入できた量 | **5,431発話 / 7.15時間 / 63 voice cluster**（cache済みのぶん） |
| accepted（cache未実施を含む） | 10,466.4時間 / 18,279話者ID |
| frontend | **raw text（J0）のみ**。S0はこれで CER 35.8% → 28.4% |
| split | voice cluster単位。zero-shot は3クラスタしかない（[R-013](07-risks-and-decisions.md)） |

**J1（normalized）以降は未実施。** S0で out_of_domain CER が
74.7% → 76.4% と改善しなかったため、数字・固有名詞の読み対応は
S1で normalized text と併せて判断します。

## 0.1 S1のデータ選定（2026-08-31決定）

**gol-dataset を主体にする。** textがゲームスクリプト（正解）で音声と1:1対応し、
tarがゲーム単位（596本）なので `metadata.tsv` から事前に選定できます。

| 選定案（約333h / 978話者ID / 219 GB） | 時間 | 話者 | 数字文% |
|---|---:|---:|---:|
| 9381931FAB68… | 93.0 | 249 | 3.7 |
| AA538DEF78C6… | 76.9 | 215 | 0.1 |
| F1C923982610… | 61.7 | 140 | 2.3 |
| 2A8DB5A70357… | 52.2 | 166 | 1.2 |
| 1FBDD444767B… | 49.4 | 208 | 4.4 |

moe-speech-plus（621.4h / 473話者 / 152 GB）は補助として使えます。
textはASR出力ですが、**実測でラベル雑音は約10%、数字の読みも保たれる**ため
当初の懸念より小さいことを確認しました（[RESULTS.md](RESULTS.md) の「ASRの誤り床」）。
話者あたり下限（最小14.3分・121ファイル）が保証されている点は gol に無い利点です。

**ただし out_of_domain はどちらでも解決しません。** golは数字を含む文が
corpus全体の1.3%しかなく、moeも同様の分布です。別施策が要ります。

## 1. データ設計の目的

約10,000時間のデータを「音声ファイルの集合」として扱うのではなく、次の用途を再現可能にするmanifestへ変換します。

- target textからtarget audio latentを学習する
- 同一話者の別発話をreferenceとしてsampleする
- zero-shot speakerを学習話者から分離して評価する
- 収録品質、style、domain、話者分布を制御してsampleする
- 入力テキストのraw/normalized/readingを比較する
- 元データの権利と変換履歴を追跡する

## 2. 最小metadata

最低限必要な情報:

```json
{
  "audio": "/data/ja/example.wav",
  "text": "今日はいい天気ですね。",
  "speaker_id": "speaker_0001"
}
```

実運用では、次のようなJSONL schemaを推奨します。

```json
{
  "utterance_id": "dataset_a:000001",
  "audio": "/data/ja/example.wav",
  "text_raw": "今日は8月30日です。",
  "text_normalized": "今日は八月三十日です。",
  "reading": null,
  "accent": null,
  "speaker_id": "speaker_0001",
  "language": "ja",
  "duration": 5.28,
  "sample_rate": 48000,
  "channels": 1,
  "quality_score": 0.96,
  "style": "neutral",
  "domain": "audiobook",
  "dataset_id": "dataset_a",
  "license_id": "internal-cleared-v1",
  "split": "train"
}
```

`reading` と `accent` は最初から必須にしません。後から追加してもraw inputと対応を検証できるよう、`text_raw` を不変の監査フィールドとして残します。

## 3. 音声の前処理

学習用の標準化は、元音声を破壊せず派生artifactとして作成します。

確認項目:

- decode可能か
- sample rateとchannel数
- duration
- clipping、DC offset、極端な無音
- signal-to-noiseの代理指標
- transcriptと音声の整合
- 重複・近重複
- BGM、効果音、複数話者、クロストーク
- 過度なdenoise、codec artifact、帯域制限

初期のTTS学習入力は24 kHz monoを想定します。ただし、raw sourceと変換後fileのchecksum、変換command/version、duration差を記録します。

## 4. テキスト正規化

正規化は同じ入力から常に同じ出力を作る決定論的pipelineにします。

対象例:

- Unicode normalization
- 全角・半角
- 数字、桁区切り、小数、負数
- 日付、時刻、曜日
- 通貨、単位、百分率
- 英字、略語、型番
- URL、メールアドレス
- 括弧、引用符、句読点、ダッシュ
- 絵文字と読み上げない記号
- 伏字、個人情報、禁止語

元表記が必要な固有名詞や文脈依存の読みを単純規則で壊さないことが重要です。正規化の各変換にはrule IDを付け、どのruleが適用されたかを追跡できるようにします。

## 5. Reading/G2P

Raw textだけのPoC後、読み誤りの内訳を集計してから追加します。

評価対象:

- 今日、一日、生、日本橋などの多義読み
- 人名、地名、作品名、企業名
- カウンターと助数詞
- 長音、促音、撥音
- 無声化
- 英数字混在
- アクセント句境界
- 疑問、列挙、括弧挿入

G2P toolを採用する場合は、tool名だけでなく辞書version、ユーザー辞書、前処理、出力記法をconfigへ固定します。

## 6. Reference/target sampling

Voice cloning学習では、同一 `speaker_id` から異なる発話を選びます。

```text
speaker A
├─ reference utterance ─→ reference VAE latent + speaker embedding
└─ target utterance
   ├─ target text
   └─ target VAE latent
```

制約:

- referenceとtargetに同じutteranceを使わない
- 同一録音から切り出した近重複を避ける
- reference durationの分布を固定・記録する
- 1発話しかないspeakerをvoice-cloning pairに使用しない
- speaker labelの誤りを検出する
- target transcriptをreference側から漏らさない

話者ごとの発話数が大きく偏る場合、単純なutterance-uniform samplingでは一部話者へ集中します。speaker-uniform、temperature sampling、duration cap等を比較します。

## 7. Split設計

最低限、次を分けます。

| Split | 目的 |
|---|---|
| train | 継続学習 |
| dev-seen | 学習話者の品質・収束監視 |
| test-seen | 学習話者での最終評価 |
| dev-zero-shot | 未学習話者での開発評価 |
| test-zero-shot | 未学習話者での最終評価 |
| text-challenge | 読み、数字、固有名詞、長文等の固定文 |
| vae-reconstruction | VAE単体の固定評価 |

zero-shot splitはspeaker-disjointである必要があります。同一人物の別名義、同一録音session、重複音声がsplitを跨がないようにします。

## 8. 10,000時間の使い方

10,000時間を毎epochすべて回す前提にはしません。CuteTTSは低rate latentとpackingを使うため、管理単位を次にします。

- total source audio seconds
- accepted audio seconds
- target latent frames
- packed LM tokens
- optimizer steps
- unique speakers
- speaker/domain/styleごとのexposure

量より先に次を可視化します。

- speaker数と上位speakerの占有率
- duration分布
- domain/style/収録環境
- 性別・年代等、利用可能で適法な属性分布
- transcript confidence
- quality score
- 重複率
- rights status

## 9. 既存言語replay

多言語能力を残す場合は、日本語90〜95%・既存言語5〜10%を初期候補とします。ただし、元の550,000時間内部corpusは利用できません。

replay dataには次が必要です。

- 適法に利用できる音声・text・speaker情報
- 公式対応言語の固定evaluation
- 日本語とreplayのsampling比率記録
- language別loss/exposure

100%日本語とreplay混合を同じstep予算で比較し、catastrophic forgettingと日本語性能のtrade-offを測ります。

## 10. Data gate

大規模学習を開始する前に、少なくとも次を満たします。

- schema validatorが全recordを検証できる
- audio decodeと24 kHz変換が再現可能
- transcriptと音声のspot checkを完了
- speaker-disjoint splitを検証
- reference/target leakageがない
- dataset/license/consentが追跡可能
- manifest versionとchecksumを保存
- 小規模subsetを同じpipelineで再生成できる
- 固定evaluation setを学習から除外

## 関連資料

- [日本語継続学習の方針](02-continual-training-strategy.md)
- [段階的な実験ロードマップ](05-experiment-roadmap.md)
- [評価計画](06-evaluation-plan.md)
