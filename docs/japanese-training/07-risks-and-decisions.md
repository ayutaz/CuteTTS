# リスク、意思決定、未解決事項

最終更新: 2026-08-30

## 1. 意思決定記録

| ID | 内容 | 状態 | 根拠・次のgate |
|---|---|---|---|
| D-001 | 既存base checkpointから継続学習する | 決定済み | ユーザーの明示方針 |
| D-002 | distill checkpointを最初の起点にしない | 提案採用 | base適応後にdistillする方が分析しやすい |
| D-003 | 初期はAudio VAEをfreeze | **確定** | P1c実測で支持。CER差 +0.58pt、original/reconstruction間CERの中央値0.00%、speaker cos 0.939（[02章 §7](02-continual-training-strategy.md)） |
| D-004 | 初期はSpeaker Encoderをfreeze | 提案採用 | zero-shot SIMで再判定 |
| D-005 | Patch Encoderをtrainする案を主案にする | **確定** | VRAM実測でfreezeの節約は0.2GB（5%）・速度差ほぼゼロ。freezeする理由がない |
| D-006 | 最初はfull fine-tuning | **確定** | 30秒発話でも peak 4.15 GB。24GBに余裕、16GBでも載る（[RESULTS.md](RESULTS.md)） |
| D-007 | 既存Tokenizerを先に測る | **完了** | P1b実測済み。`<unk>` 0%だがbyte-fallbackがtokenの9.66%・文の45.6%（[02章 §4](02-continual-training-strategy.md)） |
| D-008 | Raw textから開始し、reading/accentを段階追加 | 提案採用 | 読み誤り分析で追加 |
| D-009 | 日本語90〜95% + replay 5〜10% | 未確定 | 100%日本語との比較が必要 |
| D-010 | Japanese VAEは条件付き | **当面見送り** | P1cで公式VAEがボトルネックである証拠は得られなかった。S4は着手しない |
| D-011 | 10〜30h → 100〜500h → 1,000h → 3,000〜10,000h | 提案採用 | 各stageのexit gateを満たして進む |
| D-012 | Guidance-step distillationは最後 | 提案採用 | 日本語base品質確定後 |
| D-013 | 学習データは `midralab/gol-dataset`（10,654 h）と `ayousanz/moe-speech-plus`（621 h）を使う | 決定済み | ユーザーが提示。実測値は[データ棚卸し](data-inventory.md) |
| D-014 | gol-datasetの利用条件は解決済みとして進める | 決定済み | ユーザー確認（2026-08-30）。R-009のうちデータ利用可否の部分はクローズ |
| D-015 | splitはspeaker IDではなく **voiceクラスタ単位** で行う | 提案 | 両datasetのspeaker IDが声の識別子でないことが実測で判明（R-004参照）。P1dのクラスタリング結果で確定 |
| D-016 | 総称ラベル話者・記号のみ発話・markup発話を学習から除外する | 提案 | 実測で対象を特定済み（[データ棚卸し](data-inventory.md) 第6節）。P1dのvalidatorで実装 |
| D-017 | S0は moe-speech-plus、S1以降は gol-dataset を主軸にする | 提案 | moe側は話者あたり最小14.3分を保証しspeechMOSを持つ。gol側は規模を持つ |
| D-018 | 既存Tokenizerを維持したままStage 0を開始する | 提案 | P1c実測で情報欠落なし。互換拡張はStage 0/1の結果を見て判断（D-007の分岐1） |
| D-019 | 日本語ASRは `kotoba-tech/kotoba-whisper-v2.0` に固定する | 提案採用 | P1cのCER測定で使用。06章が要求する「ASRのversion固定」に対応 |
| D-020 | voiceクラスタリングの閾値は **0.92** | 提案 | P1dで較正。既定0.70は77話者中62を1クラスタへ併合し破綻した。Pass B規模で再較正が必要 |
| D-021 | leakage防止では **過剰併合を安全側** とする | 提案採用 | 併合しすぎても学習話者が減るだけだが、併合し損ねるとzero-shot splitに同じ声が漏れる |
| D-022 | Windowsでは `triton-windows` を導入する | 決定済み | 未導入だとdistillが `torch.compile` 失敗で全滅する（P0で実証） |
| D-023 | ローカルGPUを使う処理は実行前にユーザーへ確認する | 決定済み | ユーザー指示（2026-08-30）。作業マシンを占有し他の作業を妨げるため |
| D-024 | 重いGPU処理は **vast.ai** へ回す | 決定済み | ユーザー指示。APIキーは未設定。起動前に費用見積もりを提示する |
| D-025 | 学習ループの損失だけで成否を判断しない | **確定** | S0の1回目で loss 0.003 が4発話の丸暗記だった（R-012）。train/dev/未学習baseを同じ経路で測り、損失の絶対値を理論値と突き合わせる |

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

**2026-08-30更新: 具体化した。当初想定より深刻。**

実測の結果、**両datasetのspeaker IDは声の識別子ではない**ことが判明した。

- gol-dataset: `speaker` = `SHA-256(キャラクター表示名)[:32]`。
  同名異キャラが同一IDへ統合され、総称ラベル（`？？？`『女の子』等）は
  複数の声が1 IDに混在する
- moe-speech-plus: `uuid4().hex[:8]` のランダム値。READMEに
  「同一声優が複数キャラを演じる場合・同一キャラが複数gameに登場する場合、
  別の識別子が割り当てられる」と明記

影響:

- 見かけ上のvoice cloning成功
- 未知話者で性能崩壊
- **speaker-disjoint splitを行ってもvoice-actor-disjointにならず、
  zero-shot評価が楽観側にバイアスする**

対策:

- 同一utterance・近重複を禁止
- **voiceクラスタ単位のsplit**（D-015）。frozenの公式Speaker Encoder
  （16 kHz → 256-dim）でspeaker IDごとの重心embeddingを作り、
  cosine類似度でクラスタリングしてからsplitする
- 総称ラベルはクラスタ内分散で検出して除外する（D-016）
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

**2026-08-30更新: 実測により解消。VRAMは想定より遥かに少なくて済む。**

学習forwardの実測（vast.ai RTX 3090、228.6M full fine-tuning、microbatch 1）:

| target長 | peak VRAM |
|---|---:|
| 5秒 | 3.01 GB |
| 30秒 | **4.15 GB** |

**16GBのローカルGPUでも十分載る。** VRAMはボトルネックではなかった。
残る制約はスループット（150 ms/step）とデータ量。

| 項目 | 05章の想定 | 実機 |
|---|---|---|
| 開発PoC GPU | RTX 4090 24 GB | **RTX 4070 Ti SUPER 16 GB** |
| ストレージ空き | 未記載 | C: 3.2 TB / D: 1.5 TB |

VRAMが想定の2/3しかないため、D-006（full fine-tuning を主案とする）は
S0のmicrobatch実測で早期に再判定する必要があります。
gol-dataset全体（7 TB）のダウンロードは容量的には可能です。

影響:

- 大規模GPU契約後にthroughput不足
- optimizer/activation/I/Oがボトルネック
- **16 GBでfull fine-tuningが載らない場合、部分freezeまたはLoRAへ切り替える**

対策:

- training forward完成後にmicrobatch benchmark
- 1 GPUでmemory breakdown
- distributed scalingを短時間runで測る
- P1eの前処理パスで、まずVAE encode + Speaker Encoderの実VRAM使用量を測る

### R-008: Streaming品質の回帰

影響:

- offlineでは良いがchunk境界にartifact
- first-audio latencyやRTFが悪化

対策:

- 各stageでoffline/streaming両方を評価
- chunk timingと境界artifactを保存
- stop behaviorとlong-formを固定test

### R-009: データ・voice cloningの権利

**2026-08-30更新: データ利用可否の部分はクローズ（D-014）。公開範囲の判断は残る。**

影響:

- ~~datasetを利用できない~~ 解決
- checkpointや生成物の公開範囲が未確定
- 意図しない声の再現、同意範囲逸脱

対策:

- dataset単位のlicense/consent追跡
- **artifactの音声を公開しない**（MoeSpeech LICENSEは音声ファイルを1つでも公開すれば
  再配布とみなすと規定。`artifacts/` と `data/` はgitignore済み）
- **話者IDを匿名化として扱わない**。gol-datasetのIDは `SHA-256(表示名)[:32]` であり、
  一般的な名前は辞書攻撃で復元できる（本調査でも総称ラベル91件を復元した）
- public/private modelの境界を決める（未確定）
- model cardに用途・制限・known riskを記載

### R-010: 学習データのdomainが偏っている

**2026-08-30追加。実測により判明。**

利用可能な両datasetはanime / visual novelの声優演技であり、中立的な朗読音声をほぼ含まない。

影響:

- 話者多様性と表現力では有利（実効話者数 約2,000〜3,500）
- **数字・日付・単位・英数字混在の音声がほぼ無い。**
  gol-datasetでASCII数字を含む発話は0.11%、ラテン文字は1.41%
- [06章](06-evaluation-plan.md) 第3節の `text-challenge` が、学習分布の外になる
- 生成音声が全体としてキャラクター演技寄りになる

対策:

- 目標を「日本語TTS一般」ではなく「日本語の表現的な多話者TTS」と言い直すか、
  中立朗読データを別途追加するかを決める（未確定）
- P1bのcoverage corpusを、学習分布（会話文）と評価分布（数字・固有名詞等）の
  両方で分けて測る
- `text-challenge` の結果を、他のmetricと同列に扱わず「分布外性能」として別記する

### R-011: 実行環境の再現性

**2026-08-30追加。**

システム既定のPythonは3.14で、**torch 2.5.1 が対応していない**（3.9〜3.12のみ）。
このリポジトリの作業は `.venv`（Python 3.12 + torch 2.5.1+cu121）で行う必要があります。

対策:

- 実行コマンドは `.venv/Scripts/python.exe` を使う
- `artifacts/*/env.json` にpython / torch / GPU / cutetts commitを必ず記録する
  （`cutetts.training.artifacts.env_snapshot`）
- Windowsでは `triton-windows` を導入する（D-022）。未導入だとdistillが全滅する

### R-012: 「損失が下がった」を学習成功と読み違える

**2026-08-31追加。S0の1回目の学習で実際に踏んだ。**

`PairSampler.sample(n)` は **呼ぶたびにRNGを作り直す**（docstringに明記）。
`train_continual.py` はこれをstepごとに呼んでいたため、3000 step すべてが
**同じ4発話**だった。flow loss は 1.02 -> 0.003 まで落ちたが、これは
4発話の丸暗記であり、学習ではない。

同じ重みを条件を変えて測り直すと矛盾が露呈した:

| 測り方 | flow loss | R2 |
|---|---:|---:|
| 学習ハーネス（seed 42 = 記憶した4発話） | 0.0021 | — |
| 診断（seed 1234 = 見ていない発話） | 4.15 | -1.01 |
| 未学習base（同じ診断） | 0.78 | +0.62 |

**学習後のモデルは未学習baseより悪くなっていた。** 学習ループの損失曲線だけ
見ていれば「大成功」に見える。

気づけた理由は **損失の絶対値を理論値と突き合わせたこと**。flow matching の
target velocity は `clean - noise` で、正規化済み clean の分散が約1、
noise が N(0,1) なので二乗平均は約2.0。0.003 は決定係数 0.998 に相当し、
条件付けが答えを決定していない限り原理的に到達できない。

対策:

- **損失の絶対値を必ず基準と比較する**。`diagnose_flow_loss.py` は
  「条件付けを使わない予測器（常に0）」の loss を併記する
- **train と dev を同じ手順で測る**。train だけを見ない
- **未学習baseを同じ経路で測り、下回っているか確認する**
- teacher forcing の因果性は `tests/training/test_leakage.py` で固定
  （target patch i は z[i] に影響しない／過去は効いている）
- sampler の使い方は `tests/training/test_pair_stream.py` で固定
  （200 step 引けば全発話に触れること）

D-025 として「S0以降、学習ループの損失だけで成否を判断しない」を追加。

### R-013: zero-shot split の話者数が足りない

**2026-08-31追加。S0のreference追随性テストで判明。**

現在の manifest（5,431発話 / 63 voice cluster）の split 内訳:

| split | 発話 | voice cluster | うち4秒以上を含む |
|---|---:|---:|---:|
| dev-seen | 146 | 32 | 29 |
| dev-zero-shot | 228 | **3** | **2** |
| test-zero-shot | 434 | **5** | 4 |

**zero-shot split に話者が3〜5人しかいない。** speaker similarity や
zero-shot voice cloning の評価は、この人数では統計的に意味を持たない
（S0のreference追随性テストは2択になった）。

原因は voice cluster 単位で split を切っているため。クラスタ総数が63と
少ないうちは、zero-shot に回せるクラスタが構造的に少なくなる。

対策（S1までに）:

- データ量を増やしてクラスタ総数を増やす（S1は100〜500時間想定）
- zero-shot split に回すクラスタ数の**下限を明示する**
  （06章の評価protocolに「zero-shot話者20人以上」等）
- それまで zero-shot の数値は **参考値** として扱い、ゲートに使わない

### GPU確認フックの実効性（2026-08-30 検証）

`.claude/settings.json` に PreToolUse フックを置き、GPUを使うコマンドで
`permissionDecision: "ask"` を返すようにした。検証の結果:

| 項目 | 結果 |
|---|---|
| フックが実行されるか | ○（センチネルで確認） |
| `ask` を返すか | ○（GPUコマンドのみ。nvidia-smi参照やpytestは素通り） |
| **確認プロンプトが出るか** | **×** |

**セッションの権限モードが `ask` を素通りさせるため、フックだけでは歯止めにならない。**
権限モードに依存しない `systemMessage` を併記したが、**一次的な歯止めは
CLAUDE.md の D-023 と記憶（`confirm-before-local-gpu`）に置く**こと。
フックは default モードでの補助と位置づける。

**2026-08-31追記（誤検知の修正）。** 判定がコマンド全文の部分一致だったため、
**git commit のメッセージ本文に script 名が入るだけで確認を求めていた**。
判定条件を次の3点に絞った:

1. `git commit` 以降は判定対象から外す
2. script 名は `python` 起動の直後にある場合のみ対象（単なる言及は無視）
3. `--device cuda|auto`、`cutetts` CLI、`pytest -m gpu` も対象に追加

`.claude/settings.json` は権限ホック本体のため、auto mode では既定で
アシスタントの書き込みが分類器にブロックされる。**ユーザーの明示的な許可**を
得たうえで適用した（2026-08-31）。

適用後、10ケースで判定を検証した:

| コマンド | 期待 | 結果 |
|---|---|---|
| commit message に script 名 | 素通り | ○ |
| heredoc 内に script 名 | 素通り | ○ |
| `grep -n "train_continual"` | 素通り | ○ |
| `pytest tests/training -q` | 素通り | ○ |
| ssh 経由のリモート実行 | 素通り | ○ |
| `python scripts/train_continual.py --device cuda` | ask | ○ |
| `python scripts/evaluate_japanese_cer.py` | ask | ○ |
| `cutetts --model-dir ...` | ask | ○ |
| `pytest -m gpu` | ask | ○ |
| `--device auto` | ask | ○ |

## 3. 未解決事項

### 目的と公開範囲

- 日本語専用性能を最優先するか、多言語能力を残すか。
- checkpointを公開するか、内部利用に限定するか。
- zero-shot voice cloningをどの利用条件で提供するか。

### データ

2026-08-30時点。詳細は[データ棚卸し](data-inventory.md)。

- ~~ライセンスとmodel training可否~~ 解決（D-014）。**redistribution / モデル公開範囲は未確定**。
- ~~raw hours、speaker数、speaker分布~~ 実測済み。gol 10,654 h / 話者ID 19,349（実効 約2,000〜3,500）、
  moe 621 h / 473話者。
- ~~transcriptの取得方法~~ 解決。gol=ゲームスクリプト（正解）、moe=ASR 2系統。
- ~~speaker IDの信頼性~~ 解決（**声の識別子ではない**。R-004参照）。voiceクラスタリングで対処する。
- **accepted hours**（除外条件適用後の実数）。P1dのvalidator実行で確定する。
- **同名異キャラの衝突率**（gol横断IDの61.3%が対象）。voiceクラスタリングで確定する。
- gol側のtranscript精度（正解テキストと音声の実際の対応。`%bd` 等の変数を含む発話がある）。
- 収録品質（BGM/SE混入、複数話者、クロストーク）。gol側は品質指標を持たない。
- moe側ASR 2系統の一致率。

### Frontend

- ~~既存Tokenizerの実coverage~~ 解決（P1b）。`<unk>` 0%、byte-fallback 9.66%。
- **vocabulary拡張を行うか**（byte分解される約700字種の追加）。Stage 0/1の結果で判断。
- normalization仕様（NFKCで2.36%の文のtoken数が変わる）。
- G2P toolと辞書。golのルビmarkup `<rかな>` を読み情報に転用できるかは未検証。
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

- ~~固定する日本語ASR~~ 暫定確定（D-019: `kotoba-tech/kotoba-whisper-v2.0`）。
- ~~speaker similarity model~~ 公式Speaker Encoder（ECAPA student 256次元）を使用。
- PESQ / STOI / UTMOS は未導入。P1cでは測っていない。
- 日本語母語話者による主観評価体制。
- 公開前の合格閾値。
- safety/abuse evaluation。

## 4. 次に確定すべき順序

2026-08-30更新。2はクローズ、1が全体のボトルネックになった。

1. **公開checkpointのローカル推論再現**（P0）。weight取得は3・4・5すべての前提でもある
2. ~~データの権利・speaker分布・品質の概要~~ 解決（D-013、D-014、[データ棚卸し](data-inventory.md)）
3. Tokenizer coverage（P1b）
4. VAE reconstruction（P1c）
5. voiceクラスタリング → metadata/split/reference pairing（P1d、D-015）
6. training forwardとobjective（P2）
7. Stage 0のLR/freeze config
8. 日本語/replay比率
9. G2P/accentの追加可否
10. Stage 2以降のGPU予算
11. **モデル公開範囲**（R-009の残件。S3のmodel card作成までに確定させる）
12. **目標の言い直しの要否**（R-010。中立朗読データを足すか、表現的多話者TTSに寄せるか）

## 5. 変更ルール

この文書の決定を変更するときは、元の項目を消さずに状態と理由を更新します。実験による変更では、該当run ID、config、評価artifactへの参照を追加します。
