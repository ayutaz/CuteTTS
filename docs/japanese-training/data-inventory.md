# データ棚卸し（P1a）

最終更新: 2026-08-30

[08-execution-plan.md](08-execution-plan.md) のP1aの成果物。
Hugging Face APIとファイル実体から確認できた事実と、まだ確定していない項目を分けて記載します。

調査時点でHFに `ayousanz` として認証済み、`midralab` org のメンバー。両datasetとも
`gated: manual` だがアクセス権あり。

## 概要

| | [midralab/gol-dataset](https://huggingface.co/datasets/midralab/gol-dataset) | [ayousanz/moe-speech-plus](https://huggingface.co/datasets/ayousanz/moe-speech-plus) |
|---|---:|---:|
| 時間 | 10,654.32 h | 621.4 h |
| 話者ID数 | 19,349（実効 約2,000〜3,500） | 473 |
| 発話数 | 7,405,094 | 395,170 |
| 平均発話長 | 5.18 s | 5.66 s |
| 容量 | 7,019 GB | 152 GB |
| 音声 | 48 kHz / 32 bit / mono | 44.1 kHz / 16 bit / mono |
| テキスト | **ゲームスクリプト（正解）** | **ASR（anime-whisper + parakeet）** |
| 品質フィルタ | なし | NISQA + Silero VAD 適用済み |
| 話者IDの実体 | `SHA-256(表示名)[:32]` | `uuid4().hex[:8]`（ランダム） |
| domain | visual novel | anime |
| license | 記載なしだが**利用条件は解決済み**（D-014） | MoeSpeech LICENSE（著作権法30条の4） |

**gol-dataset単独で、当初目標の「約10,000時間」を時間数の上では満たします。**
利用条件も解決済み（D-014）のため、**律速は前処理コストとGPU/ストレージへ移りました。**

---

## midralab/gol-dataset

### 確認済み（HF API + tar実体）

**規模**

- 総発話数: 7,405,094
- 総話者（キャラクター）数: 19,349
- 合計再生時間: 10,654時間19分19.27秒
- 平均 5.1796秒 / 中央値 4.552秒 / 最小 0.30秒 / 最大 58.6987秒
- 四分位: P25 2.686秒、P75 7.005秒
- 長さ分布: 0-1s 4.39%、1-2s 11.91%、2-3s 12.91%、3-4s 13.63%、4-5s 12.55%、
  5-6s 10.68%、6-7s 8.89%、7-8s 7.10%、8-9s 5.37%、9-10s 3.91%、10s+ 8.67%

**格納形式**

- WebDataset形式の `.tar` × 602、合計 7,019 GB
- 1 tarあたり 2.6〜20.8 GB（= 1 game に相当）
- tar内部構造:

```text
<game_id>.tar
├─ index.json                      # UTF-8。Speaker / Text / FilePath / Duration
└─ <speaker_hash>/<file>.wav
```

- トップレベルに `metadata.tsv`（1.68 GB）と `metadata.json`（2.43 GB）。同じ内容で
  全発話を横断できる。列は `game_id / speaker / text / file_path / duration`

```tsv
game_id	speaker	text	file_path	duration
00E16AD74D230360BD7CC5EEAD1945C5	151DBA2BBBC6785FD8C3F3B912A84110	そうだ！貴官のことも教えていただけませんか？	00E16AD74D230360BD7CC5EEAD1945C5/151DBA2BBBC6785FD8C3F3B912A84110/z1001#00622.wav	5.4092
```

- `game_id` と `speaker` はいずれも32桁hex。実名そのものは含まれない
  （`speaker` の生成方式は後述。**声の識別子ではない**）
- テキストはUTF-8で正しく格納されている（tar内 `index.json` も同様。文字化けなし）

**音声フォーマット**

- 48,000 Hz / mono / 32 bit / 無圧縮WAV（byte rate 192,000 B/s）
- 10,654 h × 3600 s × 192,000 B/s ≒ 7.36 TB で、公称容量と整合する

CuteTTSは24 kHz入力なので **整数比（1/2）でダウンサンプルできる**。
リサンプリング品質の観点では扱いやすい構成。

### 確認済み（metadata.tsv 全7,405,094行を解析）

`metadata.tsv`（1,676,537,850 bytes）を取得し全行を集計した結果。

**テキストの出自**: ゲームスクリプト由来の**正解テキスト**（ユーザー確認済み）。
ASR由来ではないため、transcript精度のリスクは基本的に無い。

**話者IDの正体**: `speaker` は **`SHA-256(キャラクター表示名)` の先頭32桁（大文字hex）**。
複数の総称ラベルでハッシュが一致することから確認した。
つまり **声の識別子ではなく「表示名」の識別子** である。

| 項目 | 実測値 |
|---|---:|
| 総行数 | 7,405,094 |
| game数（metadata上） | 596（tarは602。**6件が未対応・未確定**） |
| 話者ID数 | 19,349 |
| (game, speaker) の組 | 30,193 |
| 合計時間 | 10,654.32 h（READMEと一致） |
| テキストが空の行 | 1,055 |

**話者あたりの分布（重要）**

| 指標 | 値 |
|---|---:|
| 最大 | 53.87 h |
| P99 | 8.96 h |
| **中央値** | **0.01 h（約36秒）** |
| 最小 | 0.0001 h |

| 閾値 | 該当話者数 | 累積時間 |
|---|---:|---:|
| ≥ 0.25 h | 3,583（18.5%） | 10,276 h（96.5%） |
| ≥ 0.5 h | 2,799（14.5%） | 9,992 h |
| ≥ 1 h | 2,095（10.8%） | 9,493 h（89%） |
| ≥ 2 h | 1,583（8.2%） | 8,764 h |
| ≥ 5 h | 621（3.2%） | 5,510 h |

上位1,000話者で全体の66.9%、上位100で16.6%を占めます。
**「19,349話者」は名目値で、学習に足る量を持つ実効話者数は約2,000〜3,500です。**

**game横断の話者ID**

- 複数gameに出現する話者ID: 3,522件（18.2%）。合計 6,532 h（**全体の61.3%**）
- 出現game数が最多のIDは総称ラベル。総称ラベル91件を特定し、合計 47.4 h（0.44%）、37,923発話

| 出現game数 | 時間 | 発話 | ラベル |
|---:|---:|---:|---|
| 114 | 6.66 h | 5,787 | `？？？` |
| 115 | 3.40 h | 3,119 | `女の子` |
| 109 | 1.02 h | 976 | `店員` |
| 66 | 0.84 h | 690 | `女性` |
| 56 | 0.78 h | 560 | `教師` |

総称ラベルは**複数の異なる声が1つのIDに混在**するため、speaker条件付けを壊します。除外が必要です。
一方、総称ラベルを除いたgame横断IDの大半（約6,485 h）は、続編・ファンディスク等に登場する
同一キャラクター名と考えられます（**同名異キャラの衝突リスクは残る・未確定**）。

**テキストの性質**

| 指標 | 値 |
|---|---:|
| 文字数 平均 / 中央値 / P95 / 最大 | 23.0 / 20 / 50 / 371 |
| 句読点・記号のみの発話 | 152,605（**2.06%**） |
| 構造化markupを含む発話 | 3,766（0.05%） |
| ASCII数字を含む発話 | 8,099（0.11%） |
| ラテン文字を含む発話 | 104,190（1.41%） |

markupの内訳（上位）: `</r>` 2,547、`<ハ>` 2,211、`[n]` 332、`</d>` 263、`%bd` 252、
`<rたま>` 151、`@ruby` 123。

- `<rかな>` … `</r>` は**ルビ（振り仮名）markup**。難読漢字の読みが埋め込まれており、
  J2（text + reading）の材料として利用できる可能性がある（提案・未検証）
- `%bd` は主人公名の差し替え変数。テキストと音声が一致しない
- `[n]` は改行制御

**数字を含む発話が0.11%、ラテン文字が1.41%しかない**点は、P1bのcoverage corpusと
`text-challenge` splitの設計に直接影響します（後述）。

### 未確定（P1d以降で確定する）

| 項目 | 状況 | 影響 |
|---|---|---|
| 同名異キャラの衝突 | game横断IDの61.3%について、同一キャラか別キャラかを検証していない | speaker条件付けとzero-shot splitの正当性 |
| 品質指標 | MOS・SNR等のスコアが付随しない | S0で「少数の高品質話者」を選ぶ根拠が無い。moe-speech-plusのspeechMOSと対照的 |
| 収録・分離品質 | BGM/SE混入、複数話者、クロストークの有無が未評価 | [03章](03-data-and-frontend.md) 第3節の確認項目 |
| tar 6件の扱い | tarは602、metadata上のgameは596 | 対応関係の確認が必要 |

### 匿名化に関する注意

話者IDは `SHA-256(表示名)[:32]` であり、**辞書攻撃で一般的な名前は復元できます**
（本調査でも総称ラベル91件を復元した）。IDそのものが匿名化として機能しない前提で、
公開物における識別名の扱いを決める必要があります（[07章](07-risks-and-decisions.md) R-009）。

---

## ayousanz/moe-speech-plus

### 確認済み

**規模（`info.csv` 全473行を集計）**

| 項目 | 値 |
|---|---:|
| キャラクター数 | 473 |
| ファイル数 | 395,170 |
| 合計時間 | **621.4 h** |
| 平均発話長 | 5.66 秒 |
| 話者あたり時間 中央値 | 53.7 分 |
| 話者あたり時間 最小 / P25 / P75 / 最大 | 14.3分 / 30.8分 / 108.2分 / 428.9分 |
| 話者あたりファイル数 最小 / 中央値 / 最大 | 121 / 541 / 4,675 |
| f0平均 最小 / 中央値 / 最大 | 100.9 / 290.5 / 473.4 Hz |
| f0 < 180 Hz の話者 | 84 / 473（17.8%） |
| 上位10話者の占有 / 上位50話者 | 9.2% / 30.7% |

**話者あたりの下限が保証されている**点（最小14.3分・121ファイル）がgol-datasetとの大きな違いです。
f0分布から**女性キャラクターに偏っている**ことが確認できます（READMEにも明記あり）。

**音声フォーマット**

- 44.1 kHz / 16 bit / mono WAV、1発話 2〜15秒
- スタジオ収録、ノイズ・BGMなしと明記
- 品質フィルタ済み: NISQA MOSスコア（話者ごとに閾値決定）、
  Silero VADによる発話比率 ≥ 0.5、話者あたり100ファイル以上かつ15分以上

44.1 kHz → 24 kHz は非整数比（147:80）のリサンプルになります。
gol-datasetの 48 kHz → 24 kHz（2:1）とは扱いが異なる点に注意。

**構成**

- `.zip` × 473（52.7 MB〜1,652 MB）、合計 151.6 GB
- 付随ファイル: `LICENSE.md`, `README.md`, `info.csv`（473行、`name,num_files,total_duration_min,f0_mean`）,
  `finished_uuids.txt`, `stats.png`, `upload_json.py`
- 話者IDは `uuid.uuid4().hex[:8]` によるランダム8文字。**名前由来ではない**
- `viewer: false`、タグに `not-for-all-audiences`

**テキストの出自**: ASRによる文字起こし（ユーザー確認済み）。
anime-whisperとparakeetの2系統があるため、**両者の不一致をtranscript信頼度の代理指標に使えます**（提案）。

**話者IDに関するREADMEの明記（重要）**

> The same voice actor may play multiple characters, or the same character may appear across
> multiple games. In such cases, they are assigned different identifiers.

同一声優・同一キャラであっても別IDが割り当てられます。gol-datasetとは逆方向の問題ですが、
**結論は同じで、speaker IDは声の識別子ではありません。**

**MoeSpeechへの付加情報**（各音声と同名のJSONとして格納）

| フィールド | 生成元 |
|---|---|
| `parakeet_jp_transcription` | nvidia/parakeet-tdt_ctc-0.6b-ja |
| `anime_whisper_transcription` | litagin/anime-whisper |
| `speechMOS` | UTMOS v2 (T05) |
| `duration` | — |
| `DeBERTa_Sentiment` / `_Top_Label` / `_Top_Score` | microsoft/deberta-v3-large |
| `Audio_Emotion_Scores` / `_Top_Label` / `_Top_Score` | litagin/anime_speech_emotion_classification |
| `Qwen2_Emotion_Label` | Qwen/Qwen2-Audio-7B-Instruct |

**このdatasetの強み**: `speechMOS` があるため、S0で要求される「少数の高品質話者」を
客観的な基準で選抜できます。gol-datasetには無い性質です。
また transcription が2系統あるため、両者の一致・不一致をtranscript信頼度の代理指標に使えます。

### 確認済み（LICENSE.md 全文より、要点）

原典は litagin の MoeSpeech LICENSE。**日本語版が優先**と明記されています。

| 区分 | 内容 |
|---|---|
| 許諾範囲 | 日本の著作権法 第三十条の四（情報解析＝機械学習等）による利用のみ |
| **モデル公開** | 「このデータセットを用いて機械学習モデルを作成した場合、そのモデルを公開することは**再配布とみなしません**」 |
| 再配布 | 改変の有無に関わらず**禁止** |
| 音声の公開 | **音声ファイルを1つであっても公開することは、特定識別名のデータの再配布とみなされます** |
| 識別名の扱い | 少数の識別名のみを使用しその特徴が再現されているような共有物については、使用した識別名を公開しないこと |
| 同定の禁止 | 実際のキャラクター名・声優名・出典元を明らかにして公開することを禁止 |
| その他の禁止 | 特定識別名のデータを享受目的で利用すること。音声を元ゲームのシナリオ順に並べる試み。出典元が同じキャラクター識別名群を明らかにすること |
| 免責 | 具体的にどの利用が可能かは利用者の判断に委ねられ、提供者は責任を負わない |

### 未確定

- ASR転写の実精度（2系統の一致率を測っていない）
- 各zipの実際の格納構造が上流MoeSpeechと同一か（READMEは `data/{uuid}/wav/` を記載）
- 「性的な内容を含む可能性がある」とREADMEが明記する発話の扱い

---

## このプロジェクトへの影響

### 0. speaker IDは「声」の識別子ではない（両datasetに共通・最重要）

| dataset | speaker IDの実体 | 帰結 |
|---|---|---|
| gol-dataset | `SHA-256(キャラクター表示名)[:32]` | 同名異キャラが同一IDへ統合される。総称ラベルは多数の声が1 IDに混在する |
| moe-speech-plus | `uuid.uuid4().hex[:8]`（ランダム） | READMEに「同一声優・同一キャラでも別IDを割り当てる」と明記 |

**どちらも speaker-disjoint split が voice-actor-disjoint を保証しません。**
[03章](03-data-and-frontend.md) 第7節と[06章](06-evaluation-plan.md) 第4節が前提にする
zero-shot評価は、このままでは**楽観側にバイアスします**（学習済みの声が
別IDでzero-shot splitに現れる）。[07章](07-risks-and-decisions.md) R-004の具体化です。

#### 対策（提案・P1dで実施）

公式Speaker Encoderが**すでにfrozenで利用可能**なので、これを使って声ベースで検証できます。

```python
# runtime.load_runtime() が返す speaker_encoder（ECAPA student, 16 kHz → 256-dim）
# speaker IDごとに数発話をembedし、ID間のcosine類似度で「別IDの同一声」を検出する
```

1. 各speaker IDから数発話をサンプルし、256-dim embeddingの重心を計算する
2. ID間のcosine類似度行列を作り、閾値以上のIDを同一voiceクラスタへまとめる
3. **splitはIDではなくvoiceクラスタ単位で分割する**
4. 総称ラベルのIDは、クラスタ内分散が大きいことで自動的に検出できる（除外候補）

これにより、gol-datasetの「同名異キャラ」とmoe-speech-plusの「同一声優別ID」の
両方に同じ手段で対処できます。moe-speech-plusには
[Moe Speech Similarity Map](https://huggingface.co/spaces/litagin/moe-speech-similarity-map)
という既存の類似度可視化もあり、参考にできます。

### 1. 規模の制約が外れ、律速がlicenseと前処理へ移った

[05-experiment-roadmap.md](05-experiment-roadmap.md) のS3が想定する3,000〜10,000時間を、
gol-dataset単独で満たせます。当初「10,000時間をどう集めるか」だった課題は、
**「10,654時間をどう捌くか」と「使ってよいか」** に置き換わりました。

### 2. latent cacheが決定的に有効（確認済みの計算）

VAEをfreezeする前提なら、7 TBの音声を毎step読む必要はありません。

| 表現 | 1秒あたり | 10,654時間ぶん |
|---|---:|---:|
| 元音声（48 kHz / 32 bit） | 192,000 B | 7,019 GB |
| 24 kHz / 16 bit へ変換後 | 48,000 B | 1,755 GB |
| **VAE latent（12.5 Hz × 64 dim, fp16）** | **1,600 B** | **61 GB** |
| VAE latent（fp32） | 3,200 B | 123 GB |
| speaker embedding（256 dim fp32、発話あたり1 kB） | — | 7.4 GB |

**全データのlatent cacheが約61 GB**に収まります。一度cacheを作れば、学習時に7 TBは不要です。
[04章](04-training-implementation.md) 第6節のlatent cache方針を、この規模が強く裏づけます。

ただし cache生成は全音声を1回ずつVAEへ通す必要があり、7 TBのI/Oとencodeコストが発生します。
実測はP2 Task 1で行い、tar単位で「取得 → 24 kHz変換 → encode → cache書き出し → 音声破棄」の
ストリーミング処理にすれば、ローカルに7 TBを常駐させずに済みます（提案）。

### 3. reference長の制約（設計上の注意）

発話の平均は5.18秒、中央値4.55秒です。一方、推論側の `prepare_reference_audio` は
VAE用に**先頭30秒**を想定しています。1発話をそのままreferenceにすると、
学習時のreferenceは平均5秒程度（約32 patch）にしかならず、**推論時の想定と乖離します。**

対応案（いずれも提案・P1dで決定）:

- A: 同一speakerの複数発話を連結して長いreferenceを作る
- B: reference長を実データ分布（約5秒）に合わせ、推論側の既定値も見直す
- C: reference長をランダム化し、推論時の長さ変動に頑健にする

[06章](06-evaluation-plan.md) 第4節が「reference duration別の性能」を評価軸に挙げているため、
ここは評価と対で設計します。

### 4. domainが偏っている（ロードマップの前提修正）

両datasetとも **anime / visual novel の声優演技** です。次の帰結があります。

- 感情表現が豊かで話者数が非常に多い（19,349話者）。multi-speaker・表現力の面では有利
- 一方で**中立的な朗読・ニュース・実用文の音声がほぼ無い**。できあがるモデルは
  その方向に寄る
- テキストが会話文中心。[03章](03-data-and-frontend.md) 第4節が挙げる数字・日付・単位・
  URL・型番の出現が乏しい可能性が高い。P1bのcoverage corpusと、
  `text-challenge` splitは**学習データ分布の外**になる
- `…………` のようなテキストを持つ発話（sample中に実在）や、0-1秒が4.39%ある。
  非音声・極端に短い発話のフィルタが必要

実測でも裏づけられました。gol-datasetのテキストで **ASCII数字を含む発話は0.11%、
ラテン文字は1.41%** しかありません。[06章](06-evaluation-plan.md) 第3節が要求する
「数字、日付、時刻、単位、通貨、英数字混在」のchallenge textは、
**ほぼ完全に学習データ分布の外**になります。

「日本語TTS一般」ではなく **「日本語の表現的な多話者TTS」** を作っている、と目標を
言い直すか、中立朗読データを別途足すかの判断が要ります（未確定）。

### 6. 前処理で除外・変換が必要な発話（実測ベース）

P1dのvalidatorに入れる条件（提案）。

| 条件 | gol-datasetでの実測 | 扱い |
|---|---:|---|
| テキストが空 | 1,055件 | 除外 |
| テキストが句読点・記号のみ（`…………` 等） | 152,605件（2.06%） | 除外。音声はあるが言語内容が無い |
| 構造化markupを含む | 3,766件（0.05%） | markupを除去。`%bd`（主人公名の変数）はテキストと音声が一致しないため**除外** |
| 総称ラベル話者（`？？？`『女の子』等） | 91ラベル / 47.4 h | speaker条件付けから除外 |
| 0〜1秒の発話 | 4.39% | 下限を決めて除外（閾値は未確定） |
| 話者あたり総時間が極端に短い | 中央値36秒、81.5%が0.25 h未満 | reference/targetのpairを作れないため、voice cloning学習から除外 |

ルビmarkup `<rかな>...</r>`（2,547件）は、除去せず**読み情報として保持する**選択肢があります
（J2の材料。提案・未検証）。

### 5. artifactの取り扱い規約（licenseからの直接の帰結）

moe-speech-plusのライセンスにより、次を**プロジェクトの規約として固定**します（提案）。

- `artifacts/` 配下の音声は**リポジトリに入れない・公開しない**。
  P0/P1cの `samples/` は元音声そのものを含むため、1ファイルでも公開すれば再配布に当たる
- 評価reportに識別名（speaker hash）を載せる場合も、少数話者の特徴を再現した音声と
  対応づけて公開しない
- モデルcheckpoint自体の公開はライセンス上禁止されていない（「再配布とみなしません」と明記）

gol-datasetはlicense記載が無いため、**上記より緩い扱いはしません**。
確定するまでは同等以上に保守的に扱います。

---

## S0〜S3へのdataset割り当て（提案）

| Stage | 規模 | 割り当て案 | 理由 |
|---|---|---|---|
| P1c（VAE再構成） | 数十発話 | moe-speech-plus | speechMOS上位から話者・収録条件を分散させて選抜できる |
| S0 | 10〜30 h | moe-speech-plus（speechMOS上位の少数話者） | 客観スコアで高品質話者を選べる。目視可能な規模に絞りやすい |
| S1 | 100〜500 h | moe-speech-plus 全体 + gol-dataset subset | 話者数を増やしてzero-shotを評価する |
| S2 | 1,000 h | gol-dataset subset | speaker/domain samplingを本番相当にする |
| S3 | 3,000〜10,000 h | gol-dataset 全体 | 唯一この規模を持つ |

利用条件は解決済み（D-014）のため、この割り当てを妨げる要因はデータ側にはありません。
実行上の制約は **P1eの前処理コスト（gol全体で7 TBのI/O）とGPU/ストレージの確保** に移ります。

---

## 次のアクション

1. ~~gol-datasetの利用条件を確定する~~ **完了。** ユーザー確認により解決（D-014）。
   モデル公開範囲の決定はS3までの残件（R-009）
2. ~~`speaker` がgame横断で一意かを判定する~~ 完了。IDは `SHA-256(表示名)[:32]` であり、
   声の識別子ではないことが判明した
3. ~~`text` の出自を確認する~~ 完了。gol=スクリプト（正解）、moe=ASR
4. ~~moe-speech-plusの総時間数・話者数を実測する~~ 完了。473話者 / 621.4 h
5. **Speaker Encoderによるvoiceクラスタリング**を実施し、split単位をIDからvoiceクラスタへ
   移す（P1d。上記「影響 0」の対策）
6. checkpointの公開/内部利用の方針を仮決定する（[07章](07-risks-and-decisions.md) R-009）
7. gol-datasetの同名異キャラ衝突を、5のクラスタリング結果から検証する

## 関連資料

- [対応計画（実行フェーズ定義）](08-execution-plan.md)
- [データセットと日本語frontend](03-data-and-frontend.md)
- [リスク、意思決定、未解決事項](07-risks-and-decisions.md)
