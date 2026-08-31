# 評価計画

最終更新: 2026-08-31

## 1. 評価の原則

- 自動metricだけで合否を決めない。
- 同じtext、reference、seed、推論設定でcheckpointを比較する。
- seen speakerとzero-shot speakerを分ける。
- base checkpoint、直前stage、候補checkpointを同じprotocolで比較する。
- 平均値だけでなく分布、P50/P95、失敗例を保存する。
- 日本語内容、speaker、音質、prosody、streamingを別軸で評価する。

## 2. Audio VAE単体

### 入力

- 性別、年代、pitch、style、収録環境の異なる日本語話者
- 母音長、促音、撥音、無声化を含む発話
- clean/帯域制限/軽いnoise等の固定subset

### 自動metric

- mel distance
- PESQ
- STOI
- UTMOS等の品質予測
- speaker embedding similarity
- ASR CER

PESQ/STOIはsample rateや実装条件を固定し、適用範囲を明記します。UTMOS等は主観評価の代替ではなく、変化検出の補助とします。

### 比較

```text
original audio
vs
official VAE reconstruction
```

将来Japanese VAEを作った場合:

```text
official VAE
vs
Japanese VAE
```

### 主観評価

- 音質劣化
- 音韻情報の欠落
- metallic/phase/codec artifact
- breath、無音、語尾
- speaker identity

## 3. 日本語内容一致

### 確定済みの測定条件（S0で固定・2026-08-31）

S0以降のCER測定はこの条件で行う。値の一覧は [S0-GATE.md](S0-GATE.md)。

| 項目 | 値 |
|---|---|
| ASR | `kotoba-tech/kotoba-whisper-v2.0`（D-019） |
| 評価set | `data/eval/s0_eval_set.json`（version 2、seed 20260831、52文） |
| 構成 | in_domain 30 / out_of_domain 12 / phonetic 10 |
| mode / reference | voice_clone / `assets/default_reference.wav` |
| seed / max_decode_length | 42 / 400 |

**評価setに語彙的内容を持たない発話を入れない。** S0の初版では
`ふあぁぁぁぁっ、あぁぁ、ああぁぁぁ！` のような感情表現が3文混入し、
CERが構造的に84%以上に張り付いていた。除外基準はテキストの性質のみで
定義し（同一文字4回以上の連続、異なり文字比率0.45未満、漢字・カタカナ皆無）、
**CERを参照して選別しない**。旧版は `s0_eval_set_v1_superseded.json` に保存。

### 指標

- Character Error Rate
- 誤り分類
  - substitution
  - deletion
  - insertion
  - repetition
  - early stop
  - non-stop

### 固定challenge

- ひらがな、カタカナ、漢字
- 多義語
- 人名、地名、固有名詞
- 数字、日付、時刻、単位、通貨
- 英数字混在
- 短文、長文
- 句読点、括弧、引用
- 疑問、列挙、強調

ASR modelとnormalization ruleをversion固定します。ASR誤りとTTS誤りを区別するため、代表失敗は人手で確認します。

## 4. Speaker similarityとvoice cloning

### Split

- seen speaker
- zero-shot speaker
- near-domain zero-shot
- out-of-domain zero-shot

**必要な話者数（R-013・2026-08-31追加）**

| split | 下限 | 理由 |
|---|---:|---|
| zero-shot | **20 voice cluster以上** | S0時点で3人しかなく、reference追随テストが2択になった |
| seen | 20 voice cluster以上 | 同上 |

この下限を満たさないうちは、**speaker similarity の数値をゲートに使わない**（参考値扱い）。
voice cluster単位でsplitを切っているため、クラスタ総数が少ないと
zero-shotに回せる話者が構造的に不足する。

### 指標

- speaker embedding cosine similarity
- reference duration別の性能
- reference/target style mismatch
- reference/target language mismatch（必要な場合）
- 人手によるspeaker identity比較

同じspeakerの別発話をreference/targetに使い、同一音声や近重複を禁止します。

### reference 追随性テスト（S0で確立）

同じtextを複数のreferenceで生成し、生成音声のspeaker embeddingが
**自分のreferenceに最も似ているか**を見る（`scripts/check_reference_following.py`）。

自己類似度と他者類似度に差が無ければ、referenceが無視され固定の声が出ている。
**2択では偶然でも当たるため、4話者以上の多択で測る。**
S0の実測は dev-seen 4話者×3文で 12/12、自己0.833 vs 他者0.600。

## 5. 自然性とprosody

自動metric:

- UTMOS等のquality estimator
- duration
- silence ratio
- pitch/energy/duration統計

人手評価:

- naturalness
- sound quality
- word accent
- accent phrase
- phrase-final intonation
- pause placement
- speaking rate
- emotion/style consistency

日本語母語話者によるblind A/BまたはMOSを行い、model名とcheckpointを隠します。

## 6. Long-form stability

評価項目:

- 1文、段落、複数段落
- repetition
- omission
- text/audio drift
- speaker drift
- tempo drift
- early stop/non-stop
- chunk boundary artifact
- peak memoryの増加

長文を単に短文へ分割した場合と、モデルへ長文を直接渡した場合を区別します。

## 7. Streaming efficiency

最低限記録する値:

- first PCM latency
- P50/P95 first-audio latency
- RTF
- total synthesis time
- generated audio duration
- peak VRAM/RAM
- chunk durationとchunk間隔

protocol:

- hardware、driver、PyTorch、deviceを固定
- batch size 1
- modelを常駐
- warm-upを別扱い
- cold/warm file cacheを明記
- text処理、reference load/resample、speaker encode、LM prefill、latent生成、VAE decode、CPU転送の包含範囲を記録
- seedとrequest順序を固定

公式論文の値と比較する場合も、ローカルprotocol差を明記します。

## 8. Catastrophic forgetting

日本語100%とreplay混合を比較する場合、少なくとも英語と中国語の固定subsetを用意します。

- intelligibility
- speaker similarity
- naturalness
- stop behavior
- streaming latency

日本語性能の改善と既存言語の劣化を同じreportに載せます。

## 9. Stage別の主評価

| Stage | 主な評価 |
|---|---|
| P0 | 公開baseの推論再現、streaming、速度 |
| P1 | Tokenizer coverage、VAE reconstruction |
| P2 | loss/gradient/mask/checkpoint correctness |
| Stage 0 (S0) | **完了**。日本語学習可能性、memorization/leakage排除 |
| Stage 1 | CER、SIM、自然性、accent、zero-shot、streaming |
| Stage 2 | 分布拡張、安定性、回帰 |
| Stage 3 | 最終blind評価、model card |
| Stage 4 | official/JA VAE比較 |
| Stage 5 | base/distill品質と速度 |

## 10. Artifact

各評価runで保存するもの:

- evaluation manifest checksum
- model checkpoint ID/checksum
- config
- seed
- source revision
- environment versions
- raw metric per item
- aggregate metric
- generated waveform
- failure tags
- manual review result

評価pipelineを用意しただけでは結果を完了扱いにしません。実際の音声生成、自動評価、人手評価のどこまで完了したかを別々に記録します。

## 11. 合格基準

S0で分布を取得済み。以降も次の形でgateを定義します。

- 必須の重大failureがない
- 直前stageより主要metricが改善
- 改善が特定speaker/domainだけに限定されない
- 人手評価が自動metricと大きく矛盾しない
- streaming/stop behaviorに重大な回帰がない

閾値は結果を見た後に都合よく変更せず、次stage開始前に固定します。
S0では [S0-GATE.md](S0-GATE.md) に基準線と合格条件を先に書き、
結果が出た後もその値を変更していません。

### 学習ループの損失をゲートに使わない（D-025）

S0の1回目は flow loss が 1.02 → 0.003 まで落ちたが、これは同じ4発話を
3000step繰り返した丸暗記であり、モデルは未学習baseより悪化していた
（[R-012](07-risks-and-decisions.md)）。**必ず次の3点を同じ経路で測る:**

1. train split の損失
2. dev split（未学習発話・未学習話者）の損失
3. **未学習baseを同じ経路で測った損失**（下回っているかの確認）

flow loss は「条件付けを使わない予測器（常に0を出す）」が約2.0になる。
損失の絶対値はこの値との比で判断する（`scripts/diagnose_flow_loss.py` が併記する）。
