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

| ID | フェーズ | 目的（答える問い） | 状態 |
|---|---|---|---|
| P0 | 推論ベースライン再現 | 公開checkpointがこの環境で正しく動くか | **完了**（gate_passed: true） |
| P1a | データ実態調査 | 実際に学習へ投入できるデータは何時間・何話者か | **完了**（accepted 10,466.4 h） |
| P1b | Tokenizer coverage | 公式Tokenizerは日本語を表現できるか | **完了**（`<unk>` 0%、byte-fallback 9.66%） |
| P1c | Audio VAE 日本語再構成 | 公式VAEをfreezeしたまま進めてよいか | **完了**（CER差 +0.58pt → freeze確定） |
| P1d | Manifest / split / pairing | 再現可能なデータ入口があるか | **完了**（voiceクラスタ t=0.92、leakage 0） |
| P1e | 前処理パス | 全音声1パスでlatentとspeaker embeddingを作れるか | **Pass A完了**（Pass BはS2直前） |
| P2 | 学習forward復元 | 公開moduleから正しい学習stepを構成できるか | **完了**（ゴール7件すべて達成） |
| S0 | 10〜30h overfit | 日本語がそもそも学習できるか | **次はここ**（GPU必要） |
| S1 | 100〜500h PoC | 日本語品質とzero-shot cloningが成立するか | 未着手 |
| S2 | 1,000h | 分布を広げても安定するか（v0.1候補） | 未着手 |
| S3 | 3,000〜10,000h | 最終baseモデルを作れるか | 未着手 |
| S4 | Japanese Audio VAE | VAEがボトルネックの場合のみ実施 | **見送り**（P1cで根拠なし・D-010） |
| S5 | Guidance-step distillation | 日本語baseを高速化できるか | 未着手 |

**2026-08-30時点で P0 / P1 / P2 は完了。次に着手すべきは S0（10〜30時間 overfit）です。**
P2はすべてCPUで検証した（縮小した実CuteTTSModelを使用）。
**S0からはGPUが必要**で、ローカルGPUは使わず vast.ai を使う（D-023 / D-024）。

実行に使うコマンドは `docs/japanese-training/RESULTS.md` と
`.claude/skills/cutetts-ja-pipeline/SKILL.md` にまとまっています。

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

### 状態: 完了（2026-08-30、`gate_passed: true`）

- [x] base / distill の両weightをrevision固定で取得し、HF revisionと全weightのsha256を記録
- [x] 8通りすべてで無音・NaN・途中切れのないwaveformを生成（base 7/7、distill 7/7）
- [x] 同一seedで2回実行し差分を記録（下表）
- [x] TTFA / RTF / peak VRAM のローカル基準値を `metrics.json` に記録
- [x] streaming と offline の波形差を記録

#### 実測基準値（この環境の基準。公式値との比較対象ではない）

| checkpoint | streaming TTFA | RTF (streaming) | peak VRAM | model load |
|---|---:|---:|---:|---:|
| base | 251 ms | 1.89 | 2.04 GB | 44.4 s |
| distill | 210 ms | 0.52 | 2.05 GB | 56.3 s |

distill step sweep: steps=1 → RTF 0.262 / steps=2 → 0.328 / steps=4 → 0.470

| 比較 | base tts | base voice_clone |
|---|---:|---:|
| 再現性（run1 vs run2） | max_abs 2.38e-04 | **完全一致** |
| streaming vs offline | max_abs 8.37e-04 | **完全一致** |

**tts経路は再現性がない**（2.4e-04）。今後checkpointの差を波形で判定する際の**ノイズ下限**になる。

#### 環境固有の落とし穴（重要）

WindowsのPyTorchには **triton が同梱されない**ため、`torch.compile` を使う sampler
（`set_sampler_compile_mode("full-sampler")`）が失敗し、**distillは当初7ケース全滅した**。
`triton-windows` の導入と `--sampler-compile-mode auto` で解消。
現在も compile mode は `eager` で動いており、RTFが公式報告値より高い一因になっている。

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

### 状態: 実質完了（2026-08-30）

候補datasetを特定し、`metadata.tsv` 全7,405,094行と `info.csv` 全473行を実測済み。
利用条件はユーザー確認により解決（D-014）。詳細は [data-inventory.md](data-inventory.md)。

- [midralab/gol-dataset](https://huggingface.co/datasets/midralab/gol-dataset):
  10,654.32 h / 話者ID 19,349（実効 約2,000〜3,500）/ 7,405,094発話 / 7,019 GB /
  **44.1/48 kHz 混在** mono 32bit / **テキストはゲームスクリプト由来の正解** / visual novel domain
- [ayousanz/moe-speech-plus](https://huggingface.co/datasets/ayousanz/moe-speech-plus):
  621.4 h / 473話者 / 395,170発話 / 152 GB / 44.1 kHz mono 16bit /
  **テキストはASR（2系統）** / NISQA + VAD 品質フィルタ済み / anime domain。
  MoeSpeech LICENSE（著作権法30条の4、モデル公開は再配布に当たらないと明記）

**S3の目標時間はgol-dataset単独で満たせます。** 規模はもうボトルネックではありません。

### ゴール

- [x] dataset単位の棚卸し表が存在する
- [x] S0（10〜30h）、S1（100〜500h）、S2（1,000h）に投入する候補datasetが指名されている（D-017）
- [x] 権利の確認が完了している（D-014）
- [x] 「raw hours」と「accepted hours」が分離して集計されている（下表）
- [ ] checkpointを公開するか内部利用に限定するかの方針が仮決定されている（R-009の残件）

#### accepted hours の実測（全7,405,094行に D-016 を適用）

| | 発話数 | 時間 | 話者ID |
|---|---:|---:|---:|
| raw | 7,405,094 | 10,654.3 h | 19,349 |
| **accepted** | **6,916,974** | **10,466.4 h** | **18,279** |
| excluded | 488,120 | 187.9 h（**1.8%**） | 1,070 |

除外内訳: too_short 324,890 / punctuation_only 181,486 / generic_speaker 37,314 /
markup 4,543 / empty_text 1,055 / too_long 340 / name_placeholder 252。
**S3の目標（3,000〜10,000時間）はacceptedだけで満たせる。**

### 確定した除外条件（実測値つき・D-016）

| 条件 | gol-datasetでの実測 |
|---|---:|
| テキストが空 | 1,055件 |
| テキストが句読点・記号のみ（`…………` 等） | 152,605件（2.06%） |
| markupを含む（`%bd` は主人公名の変数で音声と不一致） | 3,766件（0.05%） |
| 総称ラベル話者（`？？？`『女の子』『店員』等 91件） | 47.4 h |
| 0〜1秒の発話 | 4.39% |
| 話者あたり総時間が短くpairを作れない | 中央値36秒、81.5%が0.25 h未満 |

ルビmarkup `<rかな>…</r>`（2,547件）は除去せず**読み情報として保持する**選択肢がある（提案・未検証）。

### 残作業

- ~~除外条件をvalidatorへ実装し、accepted hoursを確定する~~ 完了（上表）
- モデル公開範囲の決定（R-009。S3のmodel card作成までに必要）

### 判断ゲート

- D-013 / D-014 / D-017: 確定
- R-006（時間数優先の回避）: moe-speech-plusが話者あたり最小14.3分とspeechMOSを持ち、
  高品質subsetの実在を確認済み
- R-009（権利）: データ利用可否はクローズ。public/privateの境界は未確定
- R-010（domain偏り）: 新規に認識。目標の言い直しの要否が未確定

---

## P1b: Tokenizer coverage

### 目的

公式SentencePiece Tokenizer（16,384 piece、extended vocab 16,385、日本語は公式対応言語外）が
日本語をどこまで表現できるかを実測し、frontend方針を確定する。

選択肢は[02-continual-training-strategy.md](02-continual-training-strategy.md) 第4節の3分岐:
既存Tokenizerを維持 / 既存token ID互換のvocabulary拡張 / reading・G2Pを入力へ追加。

### 状態: 完了（2026-08-30）

- [x] gol実テキスト 200,000文 / 5,594,489 token で T0 の全項目を測定
- [x] 分岐1（既存Tokenizer維持）を実測値と理由つきで選択（D-018）
- [x] S0の入力テキスト形式を raw text に確定（D-008）

| 指標 | 実測 |
|---|---:|
| 文単位 / token単位 `<unk>` 率 | **0.0000% / 0.0000%** |
| **byte-fallback token率** | **9.658%** |
| **byte-fallbackを含む文** | **45.64%** |
| tokens per char | 1.1381 |
| token長 P50 / P95 / P99 | 25 / 58 / 75 |
| NFKCでtoken数が変わる文 | 2.357% |
| 実効text予算（ref 30秒） | 10,012 token ≒ 8,797 文字 |

**`<unk>` が0なのは256個のbyte-fallbackピースが受けているため**で、日本語をよく
表現できているからではない。小書き仮名15字種・漢字643字種・カタカナ18字種が
単一ピースを持たない。→ 既存Tokenizerで開始可、**互換拡張の価値は高い**。
詳細は [02章 §4](02-continual-training-strategy.md)。

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

### 状態: 完了（2026-08-30）

- [x] f0 101〜473 Hz に分散した10話者の日本語評価subsetを固定
- [x] mel距離・ASR CER差・話者embedding類似度を測定
- [ ] **日本語話者による聴取は未実施**（自動指標のみ）
- [x] 「VAEをfreezeして進める」を決定（D-003確定、S4は見送り）

| 指標 | 実測 |
|---|---:|
| speaker cos類似度（80発話 / 10話者） | mean **0.9392** / min 0.8643 |
| SNR | mean 9.27 dB |
| log-mel L1 | mean 0.651 |
| latent frame rate | 12.589 Hz（仕様12.5） |
| **CER差（reconstruction − original）** | **+0.58 pt** |
| **original ASR vs reconstruction ASR** | **mean 2.21% / P50 0.00%** |

**中央値0.00%は「半数の発話がVAE往復後も完全に同一の転写になる」ことを意味する。**
ASRは `kotoba-tech/kotoba-whisper-v2.0` に固定（D-019）。
未実施: PESQ / STOI / UTMOS、聴取評価、促音・撥音に特化した固定subset。

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

### 状態: 完了（2026-08-30）

- [x] JSONL schemaを確定し、validatorが全recordを検証（`cutetts.training.manifest`）
- [x] split生成、zero-shot話者のtrain重複 **0件** をテストで確認
- [x] PairSampler 100ペアで **leakage 0件**、平均reference 9.61秒
- [x] manifest checksumを記録（`artifacts/p1d/*/metrics.json`）
- [ ] 固定評価set（text-challenge、英語・中国語のforgetting用）は未作成

#### voiceクラスタリングの較正（D-015）— 既定値は使えない

| 分布 | mean | P5 | P50 | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|
| 話者内 cos | 0.825 | **0.608** | 0.861 | — | — | — |
| 話者間 cos | 0.540 | — | 0.586 | **0.837** | 0.883 | 0.959 |

**話者内P5が話者間P95を下回り、両分布が大きく重なる。** 単連結は連鎖しやすく、
既定の t=0.70 では77話者中62が1クラスタへ併合された。

| 閾値 | クラスタ数 | 複数話者クラスタ | 最大 |
|---:|---:|---:|---:|
| 0.70 | 15 | 2 | 62 |
| **0.92** | **71** | **5** | **3** |
| 0.95 | 76 | 1 | 2 |

**採用: t = 0.92。** leakage防止では過剰併合が安全側（併合しすぎても学習話者が減るだけだが、
併合し損ねるとzero-shot splitに同じ声が漏れる）。

#### R-004 が実データで確認された

| cos | 話者A | 話者B |
|---:|---|---|
| 0.9594 | gol | gol（別ID） |
| **0.9299** | **gol** | **moe** ← dataset跨ぎの同一声 |

**dataset単位でsplitを分けても声は漏れる。** 現在のsplitはvoiceクラスタ確定前の暫定
（speaker_id単位）なので、`data/manifests/all_clustered.jsonl` で作り直すこと。

### 成果物

- `src/cutetts/training/manifest.py`（schema + validator。[04章](04-training-implementation.md) 第7節の構成に追加する）
- `src/cutetts/training/pairing.py`
- `src/cutetts/training/text_normalize.py`（P1bでJ1（正規化テキスト）を選択した場合のみ。
  [03章](03-data-and-frontend.md) 第4節の決定論的変換とrule ID記録を実装する）
- `scripts/prepare_japanese_manifest.py`
- `tests/training/test_manifest.py`, `tests/training/test_pairing.py`

### 実データ由来の設計課題（[data-inventory.md](data-inventory.md)）

- **speaker IDが声の識別子でない（確認済み・最重要）**: gol-datasetの `speaker` は
  `SHA-256(キャラクター表示名)[:32]`、moe-speech-plusは `uuid4().hex[:8]` のランダム値で
  「同一声優でも別ID」とREADMEに明記。**どちらもspeaker-disjoint splitが
  voice-actor-disjointを保証しない。** 対策として、frozenの公式Speaker Encoder
  （`runtime.load_runtime()` が返す `speaker_encoder`、16 kHz → 256-dim）で
  speaker IDごとの重心embeddingを作り、cosine類似度でvoiceクラスタへまとめ、
  **splitをIDではなくvoiceクラスタ単位で行う**。総称ラベル（複数の声が1 IDに混在）も
  クラスタ内分散で検出できる
- **除外対象（実測値）**: テキストが空 1,055件 / 句読点・記号のみ 152,605件（2.06%）/
  markup含み 3,766件（`%bd` は主人公名の変数でテキストと音声が不一致）/
  総称ラベル話者 91件・47.4 h / 0-1秒の発話 4.39%
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

## P1e: 前処理パス（speaker embedding + latent cache）

### 目的

P1dのvoiceクラスタリングとP2のlatent cacheは、**どちらも全音声を1回ずつ読む**必要があります。
別々に実施すると7 TBのI/Oを2回払うため、**1回のストリーミングパスで両方を生成**します。

### 設計（提案）

tar 1本ごとに完結させ、元音声をローカルに常駐させません。

```text
for each tar (602本):
    download tar
    for each wav:
        24 kHz mono へ変換（gol 48k は 2:1、moe 44.1k は 147:80）
        ├─ Audio VAE encoder  → latent [T, 64] を fp16 で cache へ
        └─ Speaker Encoder    → 256-dim embedding を cache へ
    tar を破棄
```

Speaker Encoderは16 kHz入力、Audio VAEは24 kHz入力なので、**リサンプルは2系統必要**です
（`prepare_reference_audio` が推論で行っているのと同じ構成）。

### 容量（確認済みの計算）

| 表現 | 1秒あたり | gol全体（10,654 h） |
|---|---:|---:|
| 元音声（48 kHz / 32 bit） | 192,000 B | 7,019 GB |
| 24 kHz / 16 bit へ変換後 | 48,000 B | 1,755 GB |
| **VAE latent（12.5 Hz × 64 dim, fp16）** | **1,600 B** | **61 GB** |
| speaker embedding（256 dim fp32、発話あたり1 kB） | — | 7.4 GB |

**cache生成後は学習時に元音声が不要**になります。

### 段階実行（提案）

全602 tarを一度に処理せず、2段階に分けます。

| パス | 対象 | 用途 | 規模 |
|---|---|---|---|
| **Pass A** | moe-speech-plus 全体 + gol-dataset の数十tar | P1c、P1d設計、S0、S1 | 152 GB + 数百 GB |
| **Pass B** | gol-dataset 全体 | S2、S3 | 7,019 GB |

Pass Aで手順とcache形式を固めてからPass Bを流します。
Pass Bを走らせる前に、Pass Aの実測からI/O時間とGPU時間を見積もります。

### 状態: Pass A完了（2026-08-30）。Pass BはS2直前まで実施しない

- [x] Pass A完了。latent cache と speaker embedding cache を生成
- [x] cacheに checksum / preprocessing version を記録し、不一致で `CacheMetaMismatch`
- [x] 再開機能を確認（2回目の実行で既存300発話をスキップ）
- [x] Pass Bの総時間を見積もり
- [x] 元音声を常駐させず、書庫1本ずつで完走

| 指標 | 実測（6,112発話 / 8.13 h） |
|---|---:|
| スループット | **44.5× realtime** |
| latent frame rate | 12.599 Hz（仕様12.5） |
| peak VRAM | **2.48 GB** |
| 失敗 | **0件** |

#### gol全体（10,654 h）への外挿

| | 実測外挿 | 計画見積 |
|---|---:|---:|
| latent cache | **65.3 GB** | 61 GB |
| 所要 | **239 GPU時間** | — |

計画の `12.5 Hz × 64次元 × 2 byte = 1600 B/s → 61 GB` が実データで裏づけられた。
**239 GPU時間はPass Bをクラウド（vast.ai）で回す際の費用見積の基礎。**

#### Pass B を今やらない理由

accepted データは確定したが、**voiceクラスタの閾値0.92は77話者での較正値**であり
Pass B規模（数千話者）では再較正が必要。先にP2を作り、S0で必要なデータ規模を
見極めてからの方が無駄がない。

### 成果物

- `src/cutetts/training/latents.py`, `src/cutetts/training/speaker_cache.py`
- `scripts/cache_audio_latents.py`
- `tests/training/test_latents.py`

### 注意

このフェーズはP2 Task 1と成果物が重なります。**P2 Task 1はここへ統合**し、
P2側ではcacheのload側だけを扱います。

---

## P2: 学習forward復元

### 目的

推論専用の公開moduleから、最小のteacher-forced training stepを構成する。
このフェーズの誤りは以降すべてのStageを無効にするため、**品質ではなく正しさ**だけを扱う。

### 状態: 完了（2026-08-30）。ゴール7件すべて達成

実装（`src/cutetts/training/`）:

| module | 役割 |
|---|---|
| `objectives.py` | flow matching / stop / condition dropout |
| `collator.py` | teacher forcing の sequence 組み立て |
| `dataset.py` | latent cache の読み出しと正規化 |
| `forward.py` | 学習forward本体、`freeze_all_but` |
| `packing.py` | segment境界を遮断する packing |
| `checkpointing.py` | save/resume（RNG含む）と推論用export |

検証結果（すべてCPU、`tests/training` 全件PASS）:

| ゴール | 結果 |
|---|---|
| deterministic tiny batchでlossが再現 | ✓ |
| 学習対象moduleにだけgradientが流れる | ✓ 6module全てに到達、freezeで停止 |
| 1 utteranceをoverfitできる | ✓ flow lossが30%以上低下 |
| save/resume後の次stepが一致 | ✓ 完全一致。RNG未復元だと不一致になることも確認 |
| packingがunpackedの結果を変えない | ✓ loss一致、hidden stateも個別実行と一致 |
| 推論pathでcheckpointをloadできる | ✓ `runtime.load_runtime` で確認 |
| 配線を外すとテストが落ちる | ✓ 変異テスト9/9検出、packing境界で3件失敗 |

#### packing で見つかった実バグ

行index と sample index の混同。packingすると1行に複数sampleが入るため
speaker slot と target の対応付けが壊れる。unpackedでは両者が一致するので
packingを書くまで露見しなかった。`target_batch_index`（行）と
`target_sample_index`（sample）を分離して解決。

#### 自分で決めた事項（04章「規定が無い箇所」への回答）

| 項目 | 決定 |
|---|---|
| padding patchのloss除外 | 分子からも分母からも除く |
| stopラベルの位置 | 位置iのhiddenが「patch iが最終patchか」。STOP_STOP=1固定 |
| stopのclass imbalance | `positive_weight`（重み付き平均） |
| flow/stopの重み | `stop_weight=1.0` 既定。Stage 0で調整 |
| condition dropoutの対象 | speaker + reference、既定はjoint |

#### 旧: 着手時に揃っていた前提

**すぐ使える資産**

| 用途 | 実体 |
|---|---|
| 学習sampleの入口 | `data/manifests/all_clustered.jsonl`（voice_cluster_id付き） |
| latent | `data/cache/latents/`（`LatentCacheReader`、fp16、`[T,64]`） |
| speaker embedding | `data/cache/speaker/`（`SpeakerEmbeddingCacheReader`、`[256]`） |
| reference/target pair | `cutetts.training.pairing.PairSampler`（leakage検査つき） |
| 除外ルール | `cutetts.training.text_rules`（D-016） |
| run記録 | `cutetts.training.artifacts`（run/env/inputs/metrics） |

**splitはvoiceクラスタ単位で作り直すこと。** `all.jsonl` のsplitはspeaker_id単位の暫定で、
`all_clustered.jsonl` のクラスタを使って再生成する必要がある（D-015）。

VRAM実測の基準: P1eのVAE encode + Speaker Encoderで peak 2.48 GB。
学習forwardのVRAMは別物で、**P2の最後にmicrobatch 1で必ず実測する**（R-007）。
16 GBに載らなければ D-006（full fine-tuning主案）を見直す。

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

**Task 1: latent cacheのload**（生成側はP1eへ統合済み）
- Create: `src/cutetts/training/dataset.py`
- Test: `tests/training/test_dataset.py`
- 検証: cacheから読んだlatentがVAE encode結果と一致する / VAE revisionや
  preprocessing versionが不一致のcacheをloadすると例外になる /
  manifestのutterance_idとcacheの対応が壊れていると検出される

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

### 状態: 完了（2026-08-31、ゲート通過）

**結果の全文は [S0-GATE.md](S0-GATE.md)。** in_domain CER 35.8% -> **28.4%**（-7.4pt）で
主ゲートを満たした。7.15時間・3000step・9分（RTX 3090）。

1回目の学習は `PairSampler.sample()` の誤用で無効（同じ4発話を3000step、[R-012](07-risks-and-decisions.md)）。
2回目が有効な結果。


基準線の測定と固定は完了。**ゲート値は [S0-GATE.md](S0-GATE.md) に確定済みで、結果を見て変更しない。**

| subset | n | CER mean | median |
|---|---:|---:|---:|
| in_domain | 30 | **35.8%** | 30.2% |
| out_of_domain | 12 | 74.7% | 71.4% |
| phonetic | 10 | 46.9% | 42.3% |

**主ゲート: in_domain mean が 35.8% から有意に低下**（目安30%未満）。
「日本語音声が出る」はゲートにならない — 未学習baseで既に部分的に成立している。

### 実データの制約（S0開始時点）

手元でlatent cache済みの学習データは **5,431発話 / 7.15時間 / 63 voice cluster**
（gol 4.79h + moe 2.37h）。この章が想定した10〜30時間には届かない。

段階を分けて判断する:

1. 7.15時間で先に走らせる（学習5分・評価15分・$0.06）
2. 改善が出れば S0 は成立。出なければデータを10時間以上に増やして再試行

安い方から試す順序であり、**7.15時間で失敗しても「日本語学習は不可能」と結論しない**。
データ量と手法のどちらが原因か切り分かないため。

### ゴール

- [x] 固定評価setでの基準線CERを測定し、ゲート値として固定している
- [x] **in_domain CERが基準線（35.8%）から明確に改善している** → 28.4%
- [x] lossの低下だけでなく、発音改善がある（CERで確認。実聴取は未実施）
- [x] textを入れ替えると出力内容が追随する（未学習52文でCER 28.4%）
- [x] referenceを入れ替えるとspeaker identityが追随する（4話者4択で12/12）
- [x] 未学習文でも、完全なmemorizationではない挙動が確認できる（評価文は学習manifest外）
- [x] microbatch 1のpeak VRAMとthroughputが実測されている（4.15 GB / 150 ms/step）
- [x] **16 GBでfull fine-tuningが載るかを確認している**（載る。D-006確定）

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

2026-08-30時点。P1aは完了（取り消し線）。

```text
             ┌── P1b ──────────────┐
P0（weight）─┼── P1c ──────────────┤
             └── P1e（Pass A）─ P1d ─ P2 ─┬─ S0 ─ S1 ─ S2 ─ S3 ─ S5
                                          │       │
        ~~P1a~~（完了）───────────────────┘       └── S4（条件付き）

P1e（Pass B, gol全体）────────────────────────────── S2 以降で必要
```

- **P0のweight取得が全体のボトルネック。** P1b（Tokenizer）はtokenizerディレクトリを、
  P1c（VAE）はAudio VAE weightを、P1e（前処理）はVAEとSpeaker Encoderのweightを必要とする
- P1bとP1cは互いに独立
- P1dのvoiceクラスタリングはP1eのspeaker embeddingを消費する。順序はP1e → P1d
- P1e Pass BはS2の直前までに完了していればよく、S0/S1と並行して流せる
- S0はP1b・P1c・P2の3つが揃って初めて意味を持つ

## 決定ゲート一覧

| 決定 | 状態 | 確定するフェーズ | 決めるのに必要な材料 |
|---|---|---|---|
| 使用するdataset（D-013） | **確定** | P1a | 実測済み |
| データ利用条件（D-014） | **確定** | P1a | ユーザー確認 |
| S0〜S3のdataset割り当て（D-017） | 提案 | P1a | 実測済み。S0開始時に再確認 |
| 除外条件（D-016） | 提案 | P1d | 実測済み。validator実行で確定 |
| split単位＝voiceクラスタ（D-015） | 提案 | P1d | クラスタリング結果 |
| Tokenizer方針（維持/拡張/reading追加） | 未確定 | P1b | coverage report |
| Audio VAEをfreezeで進めるか | 未確定 | P1c | reconstruction metric + 聴取 |
| reference長の扱い（A/B/C） | 未確定 | P1d | 発話長分布（実測済み）と評価設計 |
| stop target / loss weightの仕様 | 未確定 | P2 | 推論の停止挙動と一致するテスト |
| Patch Encoder train / freeze | 未確定 | S0 | 小規模ablation |
| full fine-tuning / 部分freeze / LoRA | 未確定 | S0 | VRAM実測と安定性 |
| 日本語/replay比率 | 未確定 | S1 | forgetting測定 |
| reading/G2P追加の要否 | 未確定 | S1 | 読み誤りの内訳 |
| GPU規模（4090 1台 / H100 8台 等） | 未確定 | S0の実測後、S2着手前 | microbatch benchmark、throughput |
| Japanese VAEの要否 | 未確定 | P1cで仮決定、S1〜S3で確定 | VAEがボトルネックである証拠 |
| **モデル公開範囲**（R-009残件） | 未確定 | S3まで | 公開/内部利用の方針 |
| **目標の言い直しの要否**（R-010） | 未確定 | S1まで | domain偏りの影響度 |

## 着手前に回答が必要な事項

以下はコードでは決められず、この計画の規模そのものを変えます。2026-08-30時点。

1. ~~**日本語データは現時点で手元にあるか。**~~ **解決。** gol-dataset（10,654 h）と
   moe-speech-plus（621 h）を実測済み（D-013）。利用条件も解決済み（D-014）。
2. **利用可能なGPUとストレージ。**（最優先の未回答）
   S0の実施可否を直接決めるほか、P1eの前処理パスに次が必要:
   - Pass A: 数百 GBの一時領域 + latent cache 数 GB
   - Pass B: gol全体 7 TBのdownload帯域（音声は都度破棄するため常駐は不要）+ cache 61 GB
   - GPU: VAE encoderとSpeaker Encoderを7.4M発話へ適用するGPU時間（Pass Aで実測する）
3. **checkpointを公開するか、内部利用に限定するか。** MoeSpeech LICENSEはモデル公開を
   明示的に許容している。S3のmodel card作成までに確定させる（R-009残件）。
4. **日本語専用性能を最優先するか、既存5言語の能力を残すか。** replay data確保の要否が変わる。
5. **日本語母語話者による主観評価の実施体制。** S1以降のexit gateに聴取評価が含まれる。
6. **目標を「日本語TTS一般」から「日本語の表現的な多話者TTS」へ言い直すか。**
   両datasetがanime / visual novel domainに偏っており、gol-datasetで数字を含む発話は
   0.11%しかない（R-010）。中立朗読データを足すか、目標を実データに合わせるかの判断が要る。

## 関連資料

- [段階的な実験ロードマップ](05-experiment-roadmap.md)
- [学習コード復元・実装計画](04-training-implementation.md)
- [評価計画](06-evaluation-plan.md)
- [リスク、意思決定、未解決事項](07-risks-and-decisions.md)
