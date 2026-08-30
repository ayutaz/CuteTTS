# 日本語継続学習の方針

最終更新: 2026-08-30

## 1. 採用する起点

### 決定済み

最初からAudio VAE、Speaker Encoder、TTS本体をすべて再学習せず、公開base checkpoint `OPPOer/CuteTTS` から日本語継続学習を開始します。

```text
OPPOer/CuteTTS base
    ↓ Japanese continual training
CuteTTS-JA base
    ↓ optional guidance-step distillation
CuteTTS-JA-distill
```

`OPPOer/CuteTTS-distill` は日本語学習の起点にしません。distill版は推論高速化を目的にDiffusion Headをstudent化したモデルであり、まずbase側を日本語へ適応させる方が比較と失敗分析を行いやすいためです。

## 2. 初期freeze方針

初期PoCの推奨構成:

| Component | 方針 | 理由 |
|---|---|---|
| Audio VAE | freeze | 日本語再構成で問題がない限り、最も重い再学習を避ける |
| Speaker Encoder | freeze | まず公開256-dim speaker表現の言語横断性を測る |
| Text embedding | train | 日本語tokenへの適応が必要 |
| Patch Encoder | train | 日本語音響patchへの適応を許す |
| Causal LM backbone | train | textからpatch系列への言語依存対応を学ぶ |
| Diffusion Head | train | 日本語の局所音響分布へ適応する |
| Stop Predictor | train | 日本語文長・句読点・終端へ適応する |

これは提案であり、実験で比較します。特にPatch Encoderについては次を残します。

- A: Patch Encoderを含むTTS本体をfull fine-tune
- B: Patch Encoderをfreezeし、LM + Diffusion Head + Stop Predictorをtrain
- C: LM中心の最小更新

主案はAです。B/Cは、VRAM、初期不安定性、catastrophic forgettingを調べるablationです。

## 3. Full fine-tuningとLoRA

約230MのTTS本体であるため、最初はfull fine-tuningを主案とします。音響分布への適応では、LMだけへLoRAを入れるとPatch EncoderやDiffusion Headの適応を制限する可能性があります。

ただし、これは未検証の設計判断です。次の場合はLoRAまたは部分freezeを再検討します。

- 1 GPUのメモリへ収まらない
- full fine-tuningで既存言語やspeaker identityが急激に崩れる
- 小規模PoCで更新対象を絞った方が明確に安定する
- 複数日本語variantを低コストに管理する必要が生じる

## 4. 日本語Tokenizerのdecision gate

公開Tokenizerは `CuteTTSSentencePieceTokenizer` で、model vocabularyは16,384 piece、追加special tokenを含むextended vocabは16,385です。日本語は公式対応言語に含まれないため、実測せず「そのままで十分」または「必ず差し替える」と決めません。

### T0: 既存Tokenizerのcoverage測定

数千文以上の固定corpusを用意し、少なくとも次を測ります。

- `<unk>` の文単位・文字単位発生率
- 文字数あたりtoken数
- token長のP50/P95/P99
- 漢字、ひらがな、カタカナ、ASCII、数字、記号、emoji別のcoverage
- Unicode normalization前後の差
- 固有名詞、英数字混在、日付、単位、URLの分割
- 既存special tokenとの衝突

### 分岐

1. coverageと系列長が許容できる場合、既存Tokenizerとembedding行を保持したまま継続学習する。
2. `<unk>` または過剰分割が重大な場合、既存token IDとの互換性を壊さないvocabulary拡張を設計する。
3. 文字入力だけでは読み精度が不足する場合、Tokenizer全交換の前にreading/G2Pを追加入力する。

### T0 実測結果（2026-08-30）

gol-datasetの実テキスト 200,000文 / 5,594,489 tokenで測定（公式tokenizer、`model/CuteTTS/tokenizer`）。

| 指標 | 実測値 |
|---|---:|
| **文単位 `<unk>` 率** | **0.0000%**（0文） |
| **token単位 `<unk>` 率** | **0.0000%**（0/5,594,489） |
| tokens per char | 1.1381（= 0.879 文字/token） |
| token長 P50 / P95 / P99 / max | 25 / 58 / 75 / 211 |
| NFKC正規化でtoken数が変わる文 | 2.357% |
| **byte-fallback token率** | **9.658%**（540,330 token） |
| **byte-fallbackを含む文** | **45.64%** |

**`<unk>` が0なのは、tokenizerが256個のbyte-fallbackピースを持つため**であり、
日本語をよく表現できているからではない。実際には次が単一ピースを持たない。

- ひらがな15字種（小書き仮名 `ぅ ぉ ぃ ゅ` 等。会話文で頻出）
- 漢字643字種
- カタカナ18字種
- `～ ― ♪` 等の記号

例: `龗` → 4 token、`😀` → 5 token、`麻` → 2 token。

special tokenのID: `<|im_start|>`=4、`<|im_end|>`=5、`<|endofprompt|>`=16384（拡張分）。

#### 実効text予算

`config.json` の `processor.segment.max_length` は既定の4096ではなく **10240**。

| mode | prompt overhead | reference patch | 日本語text予算 |
|---|---:|---:|---:|
| tts | 15 token | 0 | 10,225 token ≒ 8,984 文字 |
| voice_clone（ref 10秒） | 40 token | 63 | 10,137 token ≒ 8,907 文字 |
| voice_clone（ref 30秒） | 40 token | 188 | 10,012 token ≒ 8,797 文字 |

**系列長は制約にならない**（gol発話の平均は23文字）。

#### 判断（提案）

情報欠落が無いため **分岐1（既存Tokenizer維持）で学習を開始できる**。
一方、byte-fallbackが45.6%の文に及ぶため **分岐2（互換拡張）の価値は高い**。
頻出する約700字種を単一ピースとして追加すれば系列長を約10%削減でき、
モデルがbyte列から文字を再構成する負担も消える。
拡張は既存token ID・embedding行を保持したまま追加分のみ行う設計が前提（D-007）。

SentencePiece modelの拡張は、単に別modelへ置き換えればよいわけではありません。既存token ID、embedding行、special token、checkpoint loadを保持する設計と変換テストが必要です。

## 5. 日本語frontendの実験順

アーキテクチャ変更と日本語front-end変更を同時に行うと原因が分からなくなるため、段階を分けます。

### J0: Raw text

```text
日本語文字列
    ↓ existing SentencePiece
CuteTTS
```

目的は最小変更で日本語が学習可能かを見ることです。

### J1: Normalized text

数字、日付、時刻、単位、通貨、記号、全半角、Unicodeを決定論的に正規化します。raw textは監査用に保持します。

### J2: Text + reading

読み分け誤りが支配的な場合に、surfaceとreadingを併用します。

```text
<text>今日は東京に行きます。</text>
<reading>キョーワ トーキョーニ イキマス</reading>
```

### J3: Accent/prosody

アクセント句、アクセント核、ポーズ、疑問・強調などを追加する価値を比較します。最初から必須にはしません。

候補toolにはOpenJTalk/pyopenjtalk、SudachiPy、MeCab等がありますが、採用tool、辞書version、ライセンス、再現性は未決定です。

## 6. 日本語と既存言語のmix

先行案ではcatastrophic forgetting対策として次を提案しました。

```text
Japanese: 90–95%
original languages: 5–10%
```

これは未決定です。判断は最終用途によります。

- 日本語専用性能を最大化するなら100%日本語も候補。
- 多言語能力を保持するならreplay dataを混ぜる。
- 公式の元学習データは非公開のため、同一分布を再現できるとは限らない。

最低限、英語と中国語の固定evaluation subsetを保持し、日本語比率ごとのforgettingを測ってから決めます。

## 7. Audio VAE再学習の条件

日本語専用VAEは将来候補ですが、最初から実施しません。

再学習を検討する条件:

- 日本語再構成でASR CERが明確に悪化する
- 子音、母音長、促音、撥音、無声化等の情報が系統的に失われる
- 公式VAE再構成音の主観品質が目標に届かない
- VAEの問題とTTS本体の問題を分離した評価で、VAEがボトルネックと確認できる

再学習しない条件:

- 日本語でも音質・明瞭度が十分
- TTS学習側の誤りが支配的
- VAE変更によるTTS全再学習コストが便益を上回る

## 8. 成果物の順序

1. 再現可能なTokenizer coverage report
2. 日本語VAE reconstruction reportと音声sample
3. metadata validatorと固定evaluation set
4. training forward/lossの最小実装
5. 10〜30時間overfit checkpoint
6. 100〜500時間PoC checkpoint
7. 1,000時間checkpoint (`CuteTTS-JA v0.1`候補)
8. 3,000〜10,000時間full checkpoint
9. 必要な場合だけJapanese VAE
10. 最後にCuteTTS-JA-distill

各段階は、コードがあるだけで完了とはしません。checkpoint、設定、seed、入力manifest、評価結果、音声sampleが揃った状態を完了条件とします。
