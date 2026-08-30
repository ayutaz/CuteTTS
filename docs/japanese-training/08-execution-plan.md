# 対応計画（実行フェーズ定義）

最終更新: 2026-08-30

## この文書の位置づけ

[05-experiment-roadmap.md](05-experiment-roadmap.md) が「どの順序で不確定要素を減らすか」という
実験設計であるのに対し、この文書は **各フェーズで何を作り、何をもって完了とするか** を定義します。

- 目的: そのフェーズが答えを出す問い
- ゴール: 測定可能な完了条件。これを満たすまで次フェーズへ進まない
- 成果物: 実際に残るファイル・checkpoint・評価artifact
- 判断ゲート: フェーズ終了時に確定させる意思決定（[07-risks-and-decisions.md](07-risks-and-decisions.md) のIDと対応）

状態表記は他の文書と同じく **確認済み / 決定済み / 提案 / 未確定** を使います。
この文書に書かれたスクリプト名・ディレクトリ構成・判定手順は、特記のない限り **提案** です。

## 全体像

| ID | フェーズ | 目的（答える問い） | 前提 | 並行 |
|---|---|---|---|---|
| P0 | 推論ベースライン再現 | 公開checkpointがこの環境で正しく動くか | weight取得 | — |
| P1a | データ実態調査 | 実際に学習へ投入できるデータは何時間・何話者か | データ所在の把握 | P0と並行可 |
| P1b | Tokenizer coverage | 公式Tokenizerは日本語を表現できるか | P0 | P1cと並行可 |
| P1c | Audio VAE 日本語再構成 | 公式VAEをfreezeしたまま進めてよいか | P0 + 日本語音声 | P1bと並行可 |
| P1d | Manifest / split / pairing | 再現可能なデータ入口があるか | P1a | — |
| P2 | 学習forward復元 | 公開moduleから正しい学習stepを構成できるか | P0, P1d | — |
| S0 | 10〜30h overfit | 日本語がそもそも学習できるか | P1b, P1c, P2 | — |
| S1 | 100〜500h PoC | 日本語品質とzero-shot cloningが成立するか | S0 | — |
| S2 | 1,000h | 分布を広げても安定するか（v0.1候補） | S1 | — |
| S3 | 3,000〜10,000h | 最終baseモデルを作れるか | S2 | — |
| S4 | Japanese Audio VAE | VAEがボトルネックの場合のみ実施 | 条件付き | — |
| S5 | Guidance-step distillation | 日本語baseを高速化できるか | S3 | — |

**最初に着手すべきはP0とP1a。** P1aはコードを1行も書かずに進められ、かつ以降すべてのフェーズの
規模・スケジュール・公開可否がここに依存するため、遅らせるほど手戻りが大きくなります。

## 共通ルール（提案）

### artifactの保存

評価・計測結果は `artifacts/<phase>/` に保存し、gitには入れません（`.gitignore` に追加する）。
各runで最低限次を同じディレクトリに残します。

```text
artifacts/p0/2026-08-30T12-00-00/
├─ run.json          # phase, seed, 実行コマンド, 開始/終了時刻
├─ env.json          # OS, GPU, driver, torch, transformers, cutetts commit
├─ inputs.json       # checkpoint repo/revision/checksum, 入力text/audioのchecksum
├─ metrics.json      # そのフェーズの数値
└─ samples/          # 生成・再構成音声
```

`run.json` に **cutetts側のcommit hash** を必ず含めます。コードが変わった後のartifactを
同じ表で比較しないためです。

### artifactの公開制限（決定済み）

学習データのライセンス（[data-inventory.md](data-inventory.md)）により、次を規約とします。

- `artifacts/` 配下の音声を **リポジトリに入れない・公開しない**。
  MoeSpeech LICENSEは「音声ファイルを1つであっても公開することは再配布とみなす」と規定しており、
  P0/P1cの `samples/` は元音声そのものを含む
- 評価reportに識別名（speaker hash）を載せる場合も、少数話者の特徴を再現した音声と
  対応づけて公開しない
- モデルcheckpointの公開はMoeSpeech LICENSE上は禁止されていない（「そのモデルを公開することは
  再配布とみなしません」と明記）。ただしgol-dataset側の条件が未確定のため、
  公開可否はP1aの結論を待つ

### フェーズの完了宣言

「スクリプトを実装した」ではなく「実行して成果物が揃った」で完了とします。
[06-evaluation-plan.md](06-evaluation-plan.md) の原則どおり、実装済み・実行済み・品質合格を別に記録します。

---

## P0: 推論ベースライン再現

### 目的

日本語以前に、このforkと公開checkpointがこの環境で正しく動くことを確認し、
以降のすべての比較の基準線（音声・速度・メモリ）を作る。

### ゴール

- [ ] base / distill の両weightをrevision固定で取得し、`inputs.json` にrepo id・revision・
      `model.safetensors` のchecksumを記録している
- [ ] 次の8通りで、無音・NaN・途中切れのないwaveformが生成される
      `{base, distill} × {tts, voice_clone} × {offline, streaming}`
- [ ] 同一text・同一reference・同一seedで2回実行し、結果が一致する（または差分の大きさを記録している）
- [ ] first-audio latency (TTFA)、RTF、peak VRAM のローカル基準値が `metrics.json` にある
- [ ] streaming出力とoffline出力の波形差が許容範囲であることを確認している

### 作業

```bash
mkdir -p ./model
hf download OPPOer/CuteTTS --revision <commit-sha> --local-dir ./model/CuteTTS
hf download OPPOer/CuteTTS-distill --revision <commit-sha> --local-dir ./model/CuteTTS-distill

cutetts --model-dir ./model/CuteTTS --mode tts --text "..." --seed 42 --output artifacts/p0/base_tts.wav
cutetts --model-dir ./model/CuteTTS-distill --mode voice_clone \
  --reference-audio assets/default_reference.wav --text "..." --seed 42 --output artifacts/p0/distill_clone.wav
```

計測は既存実装を流用します。`src/cutetts/demo/metrics.py` の `MetricsRecorder` が
TTFA・RTF・chunk間隔を計算済みで、`cutetts-demo` のWebSocket経路（`/api/generate`）から取得できます。
CLIにはこの計測がないため、streaming計測は `generate_stream()` を直接呼ぶ
`scripts/reproduce_baseline.py`（新規・提案）で行います。

### 成果物

- `scripts/reproduce_baseline.py`
- `artifacts/p0/<timestamp>/`（上記の共通構成）

### 判断ゲート

- ここで破損・例外が出る場合、原因が解消するまで日本語作業へ進まない
- distill側の制約（`diffusion_steps` は 1/2/4 のみ、sway sampling不可）を実機で確認し、
  以降の速度比較プロトコルを固定する

### 注意

公式READMEの「約40 ms / 約9倍real time」はdistill側・RTX 4090・warm serviceの公式報告値です。
ローカル値がこれと異なっても異常とは限りません。**比較対象は公式値ではなく、このP0の自己計測値**です。

---

## P1a: データ実態調査

### 目的

「約10,000時間」を、権利・品質・話者分布が判明した **投入可能な時間数** に置き換える。

### 進捗

候補datasetは特定済み。実測値は [data-inventory.md](data-inventory.md) にある。

- [midralab/gol-dataset](https://huggingface.co/datasets/midralab/gol-dataset):
  10,654 h / 19,349話者 / 7,405,094発話 / 7,019 GB / 48 kHz mono 32bit /
  visual novel domain。**license記載なし（未確定）**
- [ayousanz/moe-speech-plus](https://huggingface.co/datasets/ayousanz/moe-speech-plus):
  152 GB / anime domain / speechMOS・2系統transcription・感情ラベル付き。
  MoeSpeech LICENSE（著作権法30条の4、モデル公開は再配布に当たらないと明記）

**残る最大の未確定はgol-datasetの利用条件。** 時間数の面ではS3の目標を単独で満たすため、
ここが確定しない限りS1以降の規模計画が立ちません。

### ゴール

- [ ] dataset単位の棚卸し表が存在し、各行に次が埋まっている
      `dataset_id / 取得元 / 総時間 / 話者数 / transcript有無と取得方法 / sample rate /
      license / 学習利用可否 / 生成物の再配布可否 / voice cloning利用の同意状況`
- [ ] 「raw hours」と「学習に使ってよいaccepted hours（見込み）」が分離して集計されている
- [ ] S0（10〜30h）、S1（100〜500h）、S2（1,000h）に投入する候補datasetが指名されている
- [ ] 権利が不明なdatasetが、学習対象から明示的に除外されている
- [ ] checkpointを公開するか内部利用に限定するかの方針が仮決定されている

### 作業

コード不要。[03-data-and-frontend.md](03-data-and-frontend.md) 第8節（10,000時間の使い方）と
[07-risks-and-decisions.md](07-risks-and-decisions.md) の未解決事項「データ」に回答を埋める。

### 成果物

- `docs/japanese-training/data-inventory.md`（新規・提案）

### 判断ゲート

- R-006（時間数優先の回避）: 高品質subsetの実在を確認する
- R-009（権利）: public/privateの境界をここで先に決める。後から決めると学習をやり直す可能性がある

### 注意

このフェーズの結果次第で、以降のStageの時間数（10〜30h / 100〜500h / 1,000h / 10,000h）自体が
変わります。**S3が成立しないと分かった場合、S0〜S2の設計も見直します。**

---

## P1b: Tokenizer coverage

### 目的

公式SentencePiece Tokenizer（16,384 piece、extended vocab 16,385、日本語は公式対応言語外）が
日本語をどこまで表現できるかを実測し、frontend方針を確定する。

選択肢は[02-continual-training-strategy.md](02-continual-training-strategy.md) 第4節の3分岐:
既存Tokenizerを維持 / 既存token ID互換のvocabulary拡張 / reading・G2Pを入力へ追加。

### ゴール

- [ ] 数千文以上の固定corpusに対し、02章T0の全項目を測定したreportがある
      （`<unk>` 率、文字あたりtoken数、token長P50/P95/P99、文字種別coverage、
      正規化前後差、固有名詞・数字・URL等の分割、special token衝突）
- [ ] 上記3分岐のいずれかが、**実測値と理由つきで**選択されている
- [ ] 選択した方針で、S0の入力テキスト形式（raw / normalized / +reading）が確定している

### 作業

TTS本体のloadは不要で、CPUのみで実行できます。

```python
from cutetts.modeling.tokenizer import CuteTTSSentencePieceTokenizer
tokenizer = CuteTTSSentencePieceTokenizer.from_pretrained("model/CuteTTS/tokenizer")
```

`processor.py` のpromptテンプレート（英語のinstruction文と `<|im_start|>` / `<|im_end|>` /
`<|endofprompt|>`）も入力sequenceの一部なので、**テンプレート込みのtoken長**も測ります。
`SegmentManagerConfig.max_length`（checkpointの `config.json` 由来）に対して、
reference latent patch分を含めた実効的なtext予算を出します。

### 成果物

- `scripts/analyze_japanese_tokenizer.py`
- `configs/japanese/tokenizer-coverage.yaml`（corpusパス、正規化設定、出力先）
- `artifacts/p1b/<timestamp>/` の `metrics.json` と `report.md`

### 判断ゲート

- D-007（既存Tokenizerを先に測る）: 完了
- D-008（raw textから開始）: 実測に基づいて再確認

### 注意

vocabulary拡張を選ぶ場合、SentencePiece modelの差し替えだけでは済みません。
既存token ID・embedding行・special token・checkpoint loadの互換を保つ変換設計と、
変換前後でtoken IDが一致することを示すテストが必要です（この設計もP1bの成果物に含める）。

---

## P1c: Audio VAE 日本語再構成

### 目的

公式Audio VAE（24 kHz、12.5 Hz、64-dim）をfreezeしたまま日本語へ進んでよいかを判断する。
TTS本体の問題とVAEの問題を、学習を始める前に分離する。

### ゴール

- [ ] 話者・収録環境・音韻条件を分散させた日本語評価subset（[06章](06-evaluation-plan.md) 第2節の入力条件）が固定されている
- [ ] original と reconstruction について、mel距離・ASR CER差・話者embedding類似度・
      自動品質指標が測定されている
- [ ] 促音・撥音・長音・無声化を含む発話で、音韻情報の系統的欠落がないことを
      日本語話者の聴取で確認している
- [ ] 「VAEをfreezeして進める」か「S4（Japanese VAE）を検討する」かが決定されている

### 作業

TTS本体のloadは不要です。VAE adapterだけでencode → decodeが完結します。

```python
from cutetts.modeling.audio_adapter import AudioAcousticVAEAdapter
vae = AudioAcousticVAEAdapter("model/CuteTTS/weights/audio_vae").eval()
latent = vae.encode(wav_24k_mono).mean   # posterior mean
recon = vae.decode(latent)
```

streaming decode経路（`vae.streaming_decode()`）でも同じ入力を通し、offline decodeとの差分を
記録します。ここが壊れているとS1以降のstreaming評価が無効になります。

### 成果物

- `scripts/evaluate_japanese_vae.py`
- `configs/japanese/vae-reconstruction.yaml`
- `artifacts/p1c/<timestamp>/`（metrics.json、original/reconstructionのペア音声、聴取結果）

### 判断ゲート

- D-003（初期はVAE freeze）: 実測で確認または再判定
- R-003（VAEが日本語音韻を保持しない）: S4の要否をここで仮決定

---

## P1d: Manifest / split / pairing

### 目的

学習と評価が同じ入口を共有し、reference/target leakageを構造的に防げる状態を作る。

### ゴール

- [ ] JSONL schema（[03章](03-data-and-frontend.md) 第2節）が確定し、validatorが全recordを検証できる
- [ ] `train / dev-seen / test-seen / dev-zero-shot / test-zero-shot / text-challenge /
      vae-reconstruction` のsplitが生成され、zero-shot splitがspeaker-disjointであることを
      テストで確認している
- [ ] reference/target pairが同一utterance・同一録音の近重複を選ばないことをテストで確認している
- [ ] 固定評価set（日本語challenge text、英語・中国語のforgetting用subset）が学習から除外されている
- [ ] manifest versionとchecksumが記録され、小規模subsetを同じpipelineで再生成できる

### 成果物

- `src/cutetts/training/manifest.py`（schema + validator。[04章](04-training-implementation.md) 第7節の構成に追加する）
- `src/cutetts/training/pairing.py`
- `src/cutetts/training/text_normalize.py`（P1bでJ1（正規化テキスト）を選択した場合のみ。
  [03章](03-data-and-frontend.md) 第4節の決定論的変換とrule ID記録を実装する）
- `scripts/prepare_japanese_manifest.py`
- `tests/training/test_manifest.py`, `tests/training/test_pairing.py`

### 実データ由来の設計課題（[data-inventory.md](data-inventory.md)）

- **speaker IDの粒度**: gol-datasetの `speaker` がgame横断で一意かが未確定。
  game内でのみ一意なら、同一声優が別IDとしてtrain/zero-shot splitの両側に現れる。
  `metadata.tsv` 取得後、最初にこれを判定する
- **reference長**: 発話の平均は5.18秒・中央値4.55秒だが、推論側の `prepare_reference_audio` は
  VAE用に先頭30秒を想定している。1発話をそのままreferenceにすると学習と推論が乖離するため、
  (A) 同一speakerの複数発話を連結 / (B) reference長を実分布に合わせ推論側の既定値も見直す /
  (C) reference長をランダム化、のいずれかを決める
- **非音声・極端に短い発話**: 0-1秒が4.39%あり、テキストが `…………` の発話も実在する。
  validatorで除外条件を定義する

### 判断ゲート

- R-004（leakage）: pair provenanceがartifactに残ることを確認する
- D-009（日本語/replay比率）: replay用の既存言語データが実在するかをここで確定する
- reference長の方針（上記A/B/Cのいずれか）を確定する

### 前提作業

現在の `.gitignore` は `tests/` を除外しています。**このフェーズの最初に除外を解除**し、
`pyproject.toml` にテスト依存（pytest）と設定を追加します。

---

## P2: 学習forward復元

### 目的

推論専用の公開moduleから、最小のteacher-forced training stepを構成する。
このフェーズの誤りは以降すべてのStageを無効にするため、**品質ではなく正しさ**だけを扱う。

### ゴール

- [ ] deterministic tiny batchでloss値が再現する
- [ ] 学習対象moduleにだけgradientが流れ、frozen VAE / Speaker Encoderには流れない
- [ ] 1 utteranceを意図的にoverfitできる
- [ ] checkpoint save/resume前後で次stepの結果が一致する
- [ ] packing有無で同一sampleのlossが一致する
- [ ] 保存したcheckpointを既存の推論path（`CuteTTS.from_pretrained`）でloadできる
- [ ] 配線を1箇所外すと、対応するテストが必ず失敗する

### 推論コードから確認済みの学習仕様

実装前に、以下は推論実装から確定できています。**推測で決めないこと。**

| 項目 | 確認済みの内容 | 根拠 |
|---|---|---|
| 動作する潜在空間 | Diffusion Headの入出力・previous cond・LM入力はすべて **正規化後** の空間。`(latent + speech_bias_factor) * speech_scaling_factor` | `model.py: forward_speech_features` |
| 非正規化 | waveform decodeの直前だけ `pred / speech_scaling_factor - speech_bias_factor` | `generation.py:986` |
| Head呼び出し | `head._predict(x=[N,P,64], t=[N], z=[N,1024], cond=[N,P,64], speaker_embedding=[N,256])` → velocity `[N,P,64]` | `diffusion_head.py:544` |
| previous cond | 直前patchの正規化latent。系列先頭はprefix末尾のspeech patch、無ければzeros | `conditioning.initial_previous_from_prefix`, `generation.py:242` |
| stop label | 位置iのLM hiddenが「patch iが最終patchか」を2値で予測する。生成側はpatch生成**前**に判定し、生成後にbreakする | `generation.py:337, 935, 1003` |
| patch size | `locenc_patch_size` と `diff_dit_patch_size` は一致が必須（不一致は例外）。公開値は2 | `generation.py:242` |
| speaker条件 | 同じ256-dim embeddingをLM側（`lm_speaker_linear`）とDiT側（adaLN-Zero）の両方へ渡す | `model.py`, `api.py` |
| dtype | backbone/locencはcheckpoint dtype、`head` はfp32固定 | `model.py` |

論文から取る式（[04章](04-training-implementation.md) 第2節）:
`x_t = (1-t)ξ + tP`、target velocity `P - ξ`、`t = sigmoid(u), u ~ N(0,1)`、
target patchを4つの独立noise/timeで複製。

**論文にもコードにも規定がなく、自分で決めて記録する必要がある項目**:
padding patchのloss除外方法、packed sample境界でのstop label、stop lossのclass imbalance対策、
flow lossとstop lossの重み、condition dropoutが落とす条件の範囲。

### タスク分解

各タスクは「失敗するテストを書く → 失敗を確認 → 最小実装 → 成功を確認 → commit」で進めます。
テストは `tests/training/` に置きます（P1dで `.gitignore` の `tests/` 除外を解除済みであること）。

**Task 1: latent cache**
- Create: `src/cutetts/training/latents.py`, `scripts/cache_audio_latents.py`
- Test: `tests/training/test_latents.py`
- 検証: 同じwaveformから2回生成したlatentが一致する / cacheにVAE checkpoint revisionと
  preprocessing versionが記録され、不一致のcacheをloadすると例外になる
- 規模の根拠（[data-inventory.md](data-inventory.md)）: gol-dataset 10,654時間ぶんのlatentは
  fp16で約61 GB（元音声は7,019 GB）。**cache生成後は学習時に元音声が不要**になる。
  ローカルに7 TBを常駐させないよう、tar単位で「取得 → 24 kHz変換 → encode → cache書き出し →
  音声破棄」のストリーミング処理にする（提案）

**Task 2: reference/target pairing（学習用samplerへの拡張）**
- Modify: `src/cutetts/training/pairing.py`（P1dで作成済み）
- Test: `tests/training/test_pairing.py`
- 検証: referenceとtargetのutterance_idが必ず異なる / 発話が1つしかないspeakerがpairに選ばれない /
  speaker-uniformとutterance-uniformで話者分布が期待どおり変わる

**Task 3: training sequence組み立て**
- Create: `src/cutetts/training/collator.py`
- Test: `tests/training/test_collator.py`
- 検証: 組み立てたsequenceのtext/speech/speaker maskが `SegmentManager` の推論経路と一致する /
  奇数latent frameのpaddingが期待位置に入る / target textがreference側へ漏れない

**Task 4: flow-matching objective**
- Create: `src/cutetts/training/objectives.py`
- Test: `tests/training/test_flow_objective.py`
- 検証: 固定seedでlossが再現する / `t=1` で `x_t` が `P` に一致する / target velocityが `P - ξ` と一致する /
  4複製がbatch次元に正しく展開される / velocity予測を正解に置き換えるとlossが0になる

**Task 5: stop target / loss**
- Modify: `src/cutetts/training/objectives.py`
- Test: `tests/training/test_stop_targets.py`
- 検証: 長さNのtargetでstop labelが位置N-1にだけ1が立つ / **1位置ずらすとテストが失敗する** /
  padding位置がlossの分母に入らない

**Task 6: condition dropout**
- Modify: `src/cutetts/training/objectives.py`
- Test: `tests/training/test_condition_dropout.py`
- 検証: dropout率0で条件が一切変化しない / 率1で指定した条件だけが落ち、他が残る

**Task 7: trainer / checkpoint**
- Create: `src/cutetts/training/trainer.py`, `src/cutetts/training/checkpointing.py`,
  `scripts/train_continual.py`, `configs/japanese/overfit.yaml`
- Test: `tests/training/test_checkpoint_resume.py`, `tests/training/test_gradient_flow.py`
- 検証: frozen moduleのgradientがNone / 学習対象moduleのgradientが非None /
  save→resume後の次stepのlossとparameterが中断なし実行と一致する /
  保存checkpointを `CuteTTS.from_pretrained` がloadできる

**Task 8: sequence packing**（正しさが確認できた後に追加）
- Create: `src/cutetts/training/packing.py`
- Test: `tests/training/test_packing.py`
- 検証: packed/unpackedで同一sampleのlossが一致する / packed sample間にattentionが漏れない /
  position IDがsampleごとにリセットされる

### 判断ゲート

- R-001（公式training codeがない）: 上記の「自分で決めた項目」を文書に記録する
- upstreamが学習コードを公開した場合、即置換せず[04章](04-training-implementation.md) 第9節の比較を行う

---

## S0: 10〜30時間 overfit

### 目的

品質ではなく **可能性** の確認。既存VAEと日本語textから日本語音声が学習できるか。

### ゴール

- [ ] target textに対応した日本語音声が生成される
- [ ] lossの低下だけでなく、聴取可能な発音改善がある
- [ ] textを入れ替えると出力内容が追随する
- [ ] referenceを入れ替えるとspeaker identityが追随する
- [ ] 未学習文でも、完全なmemorizationではない挙動が確認できる
- [ ] microbatch 1のpeak VRAMとthroughputが実測され、S1以降のGPU計画の根拠になっている

### 中止・巻き戻し条件

VAE再構成に重大な欠陥、Tokenizerの情報欠落、target/reference leakageによる見かけの成功、
stop headが学習できず無限生成または早期停止、NaN/overflowの再現。

### 判断ゲート

- D-005（Patch Encoder train）: freeze版と小規模比較して主案を確定
- D-006（full fine-tuning）: VRAM実測と安定性で再判定
- R-007（GPU見積もり）: ここで初めて実測値が出る。**これ以前に大規模GPUを契約しない**

---

## S1: 100〜500時間 PoC

### 目的

日本語品質とzero-shot voice cloningの成立を確認し、S2へ拡大する構成を1つに絞る。

### ゴール

- [ ] Japanese CER、speaker similarity、自然性、アクセント、long-form安定性、
      streaming latency/RTFが[06章](06-evaluation-plan.md)のprotocolで測定されている
- [ ] seen speakerとzero-shot speakerの差が定量化されている
- [ ] 英語・中国語の固定subsetでforgettingが測定されている
- [ ] 比較実験（Patch Encoder train/freeze、100%日本語 vs replay混合、raw/normalized text）の
      結果から、S2で使うconfigが1つに決まっている
- [ ] streaming生成がoffline同等の品質を保っている

### 判断ゲート

- D-009（日本語/replay比率）: 100%日本語との比較結果で確定
- D-008: 読み誤りの内訳を集計し、reading/G2P追加（J2）の要否を確定

---

## S2: 1,000時間

### 目的

データ分布と学習安定性を拡張し、`CuteTTS-JA v0.1` 候補を作る。

### ゴール

- [ ] speaker/domain/style samplingが本番相当になっている
- [ ] checkpoint resumeと障害復旧が、実際の中断を伴って検証されている
- [ ] S1より複数の評価軸で改善し、重要subsetに回帰がない
- [ ] S3へ拡大する費用対効果が、実測throughputに基づいて説明できる

---

## S3: 3,000〜10,000時間

### 目的

全データを使う最終日本語baseモデルを作る。

### ゴール

- [ ] accepted dataだけが段階的に投入され、speaker/domain/style exposureが監視されている
- [ ] 固定テストで最良checkpointが選定されている
- [ ] 日本語母語話者によるblind主観評価が完了している
- [ ] model card、データ説明、制限事項、ライセンスが用意されている
- [ ] 再現可能な推論手順が用意されている

予算管理は「10,000時間を1 epoch」ではなく、packed token・audio seconds・optimizer steps・
speaker exposureで行います。

---

## S4: Japanese Audio VAE（条件付き）

### 目的

P1cまたはS1〜S3の失敗分析で **VAEがボトルネックと確認できた場合のみ**、
24 kHz / 12.5 Hz / 64-dim の互換構造を保ったまま日本語音声分布へ適応する。

### ゴール

- [ ] 公式VAEがボトルネックであることの証拠が揃っている（実施の前提条件）
- [ ] 日本語VAEが公式VAEを再構成品質で上回る
- [ ] latent分布の変化に伴うTTS本体の再学習コストが見積もられている

GAN discriminator・multi-resolution mel・WavLM teacherを含むため、TTS本体より重くなる可能性があります。
既存TTS checkpointとの直接互換は期待しません。

---

## S5: Guidance-step distillation

### 目的

日本語baseの品質確定後に、first-audio latencyとRTFを下げる。

### ゴール

- [ ] Diffusion Headのみを更新するdistillationが実装されている
- [ ] 1/2/4 stepを同一checkpointで扱える
- [ ] 同じ日本語評価setでbaseとの品質差が測定されている
- [ ] 同一hardware・同一protocolでlatencyとRTFが比較されている
- [ ] baseとdistillの両方が保持されている

---

## 依存関係

```text
P0 ──┬── P1b ──┐
     └── P1c ──┤
               ├── S0 ── S1 ── S2 ── S3 ── S5
P1a ── P1d ── P2┘                │
                                 └── S4（条件付き・S1〜S3の失敗分析から分岐）
```

- P0とP1aは同時着手できる（P1aはコード不要）
- P1bとP1cは互いに独立で、どちらもTTS本体のloadを必要としない
- P2はP1d（manifest）に依存する。データが無いと学習sampleを組み立てられない
- S0はP1b・P1c・P2の3つが揃って初めて意味を持つ

## 決定ゲート一覧

| 決定 | 確定するフェーズ | 決めるのに必要な材料 |
|---|---|---|
| Tokenizer方針（維持/拡張/reading追加） | P1b | coverage report |
| Audio VAEをfreezeで進めるか | P1c | reconstruction metric + 聴取 |
| 公開範囲（public checkpoint / 内部利用） | P1a | license・consentの棚卸し |
| stop target / loss weightの仕様 | P2 | 推論の停止挙動と一致するテスト |
| Patch Encoder train / freeze | S0 | 小規模ablation |
| full fine-tuning / 部分freeze / LoRA | S0 | VRAM実測と安定性 |
| 日本語/replay比率 | S1 | forgetting測定 |
| reading/G2P追加の要否 | S1 | 読み誤りの内訳 |
| GPU規模（4090 1台 / H100 8台 等） | S0の実測後、S2着手前 | microbatch benchmark、throughput |
| Japanese VAEの要否 | P1cで仮決定、S1〜S3で確定 | VAEがボトルネックである証拠 |

## 着手前に回答が必要な事項

以下はコードでは決められず、この計画の規模そのものを変えます。

1. ~~**日本語データは現時点で手元にあるか。**~~ 解決。gol-dataset（10,654 h）と
   moe-speech-plus（152 GB）を特定済み。[data-inventory.md](data-inventory.md) 参照。
   **ただしgol-datasetのlicenseが未記載** であり、これがS1以降の規模を決める最大の未確定事項。
2. **checkpointを公開するか、内部利用に限定するか。** 公開する場合、voice cloningの提供条件と
   データ側の再配布可否がP1aの必須項目になる。MoeSpeech LICENSEはモデル公開を許容するが、
   gol-dataset側は未確定。
3. **日本語専用性能を最優先するか、既存5言語の能力を残すか。** replay data確保の要否が変わる。
4. **利用可能なGPU。** 現時点で使える機材（開発機・クラウド）が、S0の実施可否を直接決める。
   あわせて、latent cache生成（7 TBのI/O）を実行できるストレージと帯域も必要。
5. **日本語母語話者による主観評価の実施体制。** S1以降のexit gateに聴取評価が含まれる。
6. **目標を「日本語TTS一般」から「日本語の表現的な多話者TTS」へ言い直すか。**
   利用可能な両datasetがanime / visual novel domainに偏っており、中立的な朗読音声を
   ほぼ含まない。別途データを足すか、目標を実データに合わせるかの判断が要る。

## 関連資料

- [段階的な実験ロードマップ](05-experiment-roadmap.md)
- [学習コード復元・実装計画](04-training-implementation.md)
- [評価計画](06-evaluation-plan.md)
- [リスク、意思決定、未解決事項](07-risks-and-decisions.md)
