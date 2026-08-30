# データ棚卸し（P1a）

最終更新: 2026-08-30

[08-execution-plan.md](08-execution-plan.md) のP1aの成果物。
Hugging Face APIとファイル実体から確認できた事実と、まだ確定していない項目を分けて記載します。

調査時点でHFに `ayousanz` として認証済み、`midralab` org のメンバー。両datasetとも
`gated: manual` だがアクセス権あり。

## 概要

| dataset | 時間 | 話者 | 発話数 | 容量 | domain | license |
|---|---:|---:|---:|---:|---|---|
| [midralab/gol-dataset](https://huggingface.co/datasets/midralab/gol-dataset) | 10,654 h | 19,349 | 7,405,094 | 7,019 GB | visual novel | **記載なし（未確定）** |
| [ayousanz/moe-speech-plus](https://huggingface.co/datasets/ayousanz/moe-speech-plus) | 未確定 | 未確定 | 100K〜1M | 152 GB | anime | MoeSpeech LICENSE（著作権法30条の4） |

**gol-dataset単独で、当初目標の「約10,000時間」を時間数の上では満たします。**
規模がボトルネックではなくなった一方、律速は license の確定と前処理コストへ移りました。

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

- `game_id` と `speaker` はいずれも32桁hex（MD5相当）。実名は含まれない
- テキストはUTF-8で正しく格納されている（tar内 `index.json` も同様。文字化けなし）

**音声フォーマット**

- 48,000 Hz / mono / 32 bit / 無圧縮WAV（byte rate 192,000 B/s）
- 10,654 h × 3600 s × 192,000 B/s ≒ 7.36 TB で、公称容量と整合する

CuteTTSは24 kHz入力なので **整数比（1/2）でダウンサンプルできる**。
リサンプリング品質の観点では扱いやすい構成。

### 未確定（P1aで確定が必要）

| 項目 | 状況 | 影響 |
|---|---|---|
| **license** | README・HF metadata・リポジトリのいずれにも記載がない。`license:` タグ自体が存在しない | **最重要。** 08計画のP1aゲート「権利が不明なdatasetを学習対象から明示的に除外」に現状は抵触する。midralabメンバーとして原典の取り決めを確認する必要がある |
| `text` の出自 | ゲームスクリプト由来（正解テキスト）か、ASR出力かがREADMEに書かれていない | 正解テキストならtranscript精度の懸念が消え、P1aのdata gateが大きく前進する。ASRなら信頼度フィルタが必要 |
| `speaker` の粒度 | game横断で一意か、game内でのみ一意かが不明。キャラクター単位か声優単位かも不明 | speaker-disjoint splitの正当性に直結。同一声優が別gameで別IDなら、zero-shot splitに同じ声が漏れる |
| 品質指標 | MOS・SNR等のスコアが付随しない | S0で「少数の高品質話者」を選ぶ根拠が無い。moe-speech-plusのspeechMOSと対照的 |
| 収録・分離品質 | BGM/SE混入、複数話者、クロストークの有無が未評価 | [03章](03-data-and-frontend.md) 第3節の確認項目 |

`speaker` の粒度は、`metadata.tsv` 取得後に次で判定できます（P1dのタスク）。

```bash
# 同一speakerが複数のgame_idに出現するか
tail -n +2 metadata.tsv | cut -f1,2 | sort -u | cut -f2 | sort | uniq -d | head
```

---

## ayousanz/moe-speech-plus

### 確認済み

**規模と構成**

- `.zip` × 473、合計 151.6 GB
- `size_categories: 100K<n<1M`
- 付随ファイル: `LICENSE.md`, `README.md`, `info.csv`, `finished_uuids.txt`,
  `stats.png`, `upload_json.py`
- `viewer: false`、タグに `not-for-all-audiences`

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

- 総時間数（上流MoeSpeechのchangelogに v0.4.1 で「621 hours」の記載があるが、
  この派生版の実測値は未確認）
- 話者（識別名）数
- 各話者の発話数分布

---

## このプロジェクトへの影響

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

「日本語TTS一般」ではなく **「日本語の表現的な多話者TTS」** を作っている、と目標を
言い直すか、中立朗読データを別途足すかの判断が要ります（未確定）。

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

**S1以降はgol-datasetのlicense確定が前提です。** 確定しない場合、
到達可能な上限はmoe-speech-plusの規模（未確定・上流changelog基準で数百時間）に制限され、
S2以降の設計を見直す必要があります。

---

## 次のアクション

1. **gol-datasetの利用条件を確定する**（最優先）。midralab内の取り決め、原典データの
   扱い、モデル公開の可否。ここが決まらないとS1以降の計画が立たない
2. `metadata.tsv` を取得し、`speaker` がgame横断で一意かを判定する（上掲コマンド）
3. gol-datasetの `text` がスクリプト由来か ASR 由来かを確認する
4. moe-speech-plusの総時間数・話者数を実測する
5. checkpointの公開/内部利用の方針を仮決定する（[07章](07-risks-and-decisions.md) R-009）

## 関連資料

- [対応計画（実行フェーズ定義）](08-execution-plan.md)
- [データセットと日本語frontend](03-data-and-frontend.md)
- [リスク、意思決定、未解決事項](07-risks-and-decisions.md)
