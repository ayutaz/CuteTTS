# 対応計画（実行フェーズ定義）

最終更新: 2026-09-13

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

### 準備フェーズ（すべて完了）

| ID | フェーズ | 目的（答える問い） | 状態 |
|---|---|---|---|
| P0 | 推論ベースライン再現 | 公開checkpointがこの環境で正しく動くか | **完了**（gate_passed: true） |
| P1a | データ実態調査 | 実際に学習へ投入できるデータは何時間・何話者か | **完了**（accepted 10,466.4 h） |
| P1b | Tokenizer coverage | 公式Tokenizerは日本語を表現できるか | **完了**（`<unk>` 0%、byte-fallback 9.66%） |
| P1c | Audio VAE 日本語再構成 | 公式VAEをfreezeしたまま進めてよいか | **完了**（CER差 +0.58pt → freeze確定） |
| P1d | Manifest / split / pairing | 再現可能なデータ入口があるか | **完了**（voiceクラスタ t=0.92、leakage 0） |
| P1e | 前処理パス | 全音声1パスでlatentとspeaker embeddingを作れるか | **Pass A完了** |
| P2 | 学習forward復元 | 公開moduleから正しい学習stepを構成できるか | **完了**（ゴール7件すべて達成） |
| S0 | 10〜30h overfit | 日本語がそもそも学習できるか | **完了**（7.15hで通過） |
| S1 | 100〜500h PoC | 日本語品質とzero-shot cloningが成立するか | **完了**（原因確定・修正済み） |

### 現在の到達点（2026-09-15）

評価set v3（in_domain 600文）での実測:

| checkpoint | 素CER mean / median | **読みCER mean** |
|---|---:|---:|
| base（未学習） | 35.86 / 31.91 | 30.94 |
| **fp32 30,000 step** | **20.10 / 16.67** | **13.38** |
| 同（**J2+J3込み＝実運用**） | 19.71 / — | **12.36** |
| ASR床（人間の実音声） | 10.40 | 5.59 |

base比 **-15.77pt**（95%CI [-15.49, -12.75]、600文中457文で改善）。
TTS由来の誤りは **25.5pt → 9.7pt（62%削減）**。

**素のCERはASRの表記選択を誤りと数える**（[R-029](07-risks-and-decisions.md)）。
読みへ直して測ると base比 **-17.56pt**（95%CI [-18.89, -16.31]）で、
床との差も **7.79pt**。**学習の効果は素CERで見るより大きい。**

抑揚とアクセントも測れるようになった（M1）。**学習が両方を有意に改善している。**

| | base | **30,000 step** | 人間 |
|---|---:|---:|---:|
| 輪郭の相関 | +0.024（床と区別できない） | **+0.122**（有意） | 床 -0.009 |
| アクセント核（対人間） | 35.2% | **43.6%** | 辞書が 44.8% |

**天井は分かっていない**（同一文・同一話者の人間の別テイクが無い）。
**盲検A/Bで 15/18（83%、p=0.0038）** と知覚できる差があり、
会話文では 13/14（93%、p=0.0009）で CER と聴取が一致する。

### 何が律速しているか（実測で並べた）

**S1の失敗はデータ不足ではなく学習実装のバグだった**（[R-020](07-risks-and-decisions.md)）。
以降の測定で、効いた順は次のとおり。

| 手段 | 効果 | 学習の要否 |
|---|---:|---|
| **dtype修正**（bf16→fp32 master weights） | **-15.77pt** | 要（済） |
| **J2 読み展開（数詞）** | **-11.80pt**（数詞set） | **不要** |
| step数 3,000 → 30,000 | -4.20pt | 要（済） |
| **データ量 17h → 325.9h（19倍）** | **-1.90pt** | 要 |

**データ量は測った中で最も弱い。** 一方、聴取で残った指摘は

| 指摘 | 割合 | 状態 |
|---|---:|---|
| **読み間違い** | **45%** | **J3 で対処済み**（読みCER -4.23pt。再学習不要） |
| **抑揚が不自然** | **40%** | **測れるようになった**（M1）。学習で輪郭の相関 +0.024→+0.122（有意）。**まだ人間に届いていない** |
| **アクセントが違う** | **30%** | **測れるようになった**（M1）。学習で対人間 35.2%→43.6%（有意）。**まだ人間に届いていない** |

誤読の主因は **byte-fallback による文字の分解**（[R-027](07-risks-and-decisions.md)）。
`華` は単独pieceを持たず3つのバイト断片になり、モデルは文字として見ていない。
仮名に置き換えると `中華`→`ちゅうか`、`湊`→`みなと` が直る。**再学習を要しない。**

### これからの段階（データ規模ではなく律速要因で切る）

**従来の S2（1,000h）→ S3（3,000〜10,000h）という規模の段階は保留する。**
19倍のデータで -1.90pt という実測に対し、frontend は学習なしで -11.80pt を出した。
**規模を先に上げる根拠が無い。**

| ID | フェーズ | 答える問い | 学習 | 状態 |
|---|---|---|---|---|
| ~~J3~~ | 読み付与 frontend | 誤読（指摘45%）を frontend で消せるか | **不要** | **完了**。専用set 読みCER -4.23pt、会話文でも -1.26pt（どちらも有意） |
| ~~J4~~ | Tokenizer 互換拡張 | byte-fallback を減らすと何が改善するか | 要 | **見送り**（D-039）。J3 が内容語の fallback をほぼ吸収した（R-030） |
| ~~T1~~ | 学習率の探索 | lr=2e-5 は凍結時代の値。最適値は別か | 要 | **完了**（R-035 / D-043）。**梃子ではなかった。** 半分と5倍は有意に悪く、2.5倍は区別できない。抑揚・アクセントはどの水準も有意差なし |
| ~~T2~~ | batch size / 学習対象 | 他のハイパーパラメータに余地があるか | 要 | **完了**（R-036 / D-044）。**梃子ではなかった。** 同じ計算量なら batch 4 と 16 は区別できない。head凍結は有意に悪い |
| ~~M1~~ | 抑揚・アクセントの測定 | CERの外にある指摘（40%/30%）をどう測るか | 不要 | **完了**（240文 / 53話者）。抑揚・アクセントとも学習で有意に改善。T1の判定に使える（輪郭の検出限界 0.045） |
| **F1** | 評価を実運用と揃える | 文書の主値は frontend 込みか | 不要 | **次に着手**。評価は既定で J2/J3 を適用していない（`--assign-yomi` は任意フラグ） |
| ~~M2~~ | 抑揚・アクセントの天井 | +0.122 / 45.3% は天井のどこか | 要（軽） | **完了**（R-037 / D-045）。天井は **輪郭 +0.38 / アクセント 64.5%**。現行は隔たりの約32% |
| **D1** | データ量の再測定 | 30,000 step でもデータ量は弱いか | 要 | **提案**。S2 の保留を解く条件に直接答える |
| **C1** | 計算量を4倍にする | 収穫逓減はどこで止まるか | 要 | **提案**。T2 で「効くのは計算量」と分かった |
| T3 | `flow_copies` / `condition_dropout` | 残る一次パラメータに余地はあるか | 要 | 提案。優先度は低い |
| S2 | 1,000時間 | 分布を広げても安定するか | 要 | **保留**。**D1 の結果で進退を決める** |
| S3 | 3,000〜10,000時間 | 最終baseモデルを作れるか | 要 | 保留 |
| S4 | Japanese Audio VAE | VAEがボトルネックの場合のみ | 要 | **見送り**（D-010） |
| S5 | Guidance-step distillation | 日本語baseを高速化できるか | 要 | 未着手 |

**順序の根拠（2026-09-15 更新）:**

1. ~~J3 が最優先~~ → **完了**。指摘の最多（45%）に対応し、再学習なしで
   読みCER -4.23pt（専用set）/ -1.26pt（会話文）
2. ~~J4~~ → **見送り**（D-039 / R-030）。J3 が内容語の fallback をほぼ吸収した
3. ~~M1 は並行~~ → **完了**。**測る手段が無いまま学習に金を払うのを避けられた。**
   実際、測定器を3回直すまで正しい値が出なかった（R-032 / R-033 / R-034）
4. ~~T1（lr）が次~~ → **完了。梃子ではなかった**（R-035 / D-043）。
   4水準を3指標で比較し、**現行の 2e-5 がほぼ底**だと分かった。
   「凍結時代の値だから最適でないはず」は外れで、偶然良い値を引いていた。
   **これで「データ量は弱い、step数は頭打ち」という結論の射程が一段広がった**
5. ~~T2~~ → **完了。梃子ではなかった**（R-036 / D-044）。
   batch16 は同じ step 数なら有意に良いが、同じサンプル数なら有意に悪い。
   **効いているのは計算量で、batch の大きさではない**
   （同じ計算量の batch4・30,000 step と区別できない）。head凍結は有意に悪い
6. **S2 は保留のまま** — 学習設定（lr / batch / 学習対象）には余地が無かった。
   残るのは「計算量」「データ量」「データの質」と、未探索の `flow_copies` / `condition_dropout`
7. **次は F1 → M2 → D1 → C1**（下表）。**GPUを使う前に、CPUで答えられる問いを
   先に片づける。** M1 で「測れないまま学習に金を払わない」を学んだのと同じ理由

**これからの順序（2026-09-16）:**

| 順 | ID | 内容 | GPU | 見積もり | 判定（これが出たら次へ） |
|---|---|---|---|---|---|
| 1 | **F1** | 評価を実運用（J2+J3）と揃え、文書の主値を差し替える | 不要 | 約1h / $0 | 主値が **12.36%**（frontend込み）になる |
| ~~2~~ | ~~**M2**~~ | 抑揚・アクセントの**天井**を別テイクで測る | 要（軽） | 実測 約40分 / 約$0.15 | **完了。天井 +0.38 で判定を満たした** → 抑揚に投資（D-045） |
| 3 | **D1** | データ量（17h 対 325.9h）を **30,000 step** で測り直す | 要 | 約3h / 約$0.5 | **有意かつ2pt以上 → S2へ。有意でない → S2 見送り** |
| 4 | **C1** | 計算量を4倍にする（batch16 × 30,000 step） | 要 | 約6.5h / 約$1.1 | **1pt以上の有意改善 → 最終モデルは長時間学習で作る** |
| 5 | T3 | `flow_copies` / `condition_dropout` | 要 | 約4〜8h / 約$1.3 | 有意差が出れば採用 |

**S2 に進まない場合の合計は約$2。** 着手前に費用を提示する（D-024）。

**判定に使う指標（T1以降はこの3つで見る）:**

| 指標 | 評価set | 検出限界 | 基準 |
|---|---|---|---|
| **読みCER** | `eval_set_v3.json` 600文 | 約1.5pt | 人間の床 5.59% |
| **輪郭の相関** | `prosody_eval_set_v2.json` 240文 | 0.045 | 床 -0.009 |
| **アクセント核** | 同上（1,491句） | — | 人間対辞書 44.8% / 固定回答 42.3% |

**CERだけで判定しない。** 読みが直っても抑揚が壊れる可能性があり、
その逆もある。3つを併記する。

**S4（Japanese VAE）だけは扱いが変わらない。** P1cで根拠が得られず見送り（D-010）。
ただし聴取で「高音域の発音がつぶれる」という指摘が1件あり、
**高音域はP1cで評価していない**。M1 で測る対象に含める。

### 中国語は諦めた（D-032）

日本語学習で中国語CERが 11.5% → 77.2% に劣化する
（[R-022](07-risks-and-decisions.md)）。原因は漢字の読みが日本語に上書きされること。
ユーザー判断（2026-09-12）で**日本語特化modelとする**。
**英語は replay なしで保たれる**（WER 1.7%、baseと同値）ので影響しない。
これにより D-009（replay 5〜10%）は不要になり、S2以降も100%日本語で進める。

### 実行環境

GPUはローカルを使わず vast.ai を使う（D-023 / D-024）。
実行コマンドは [RESULTS.md](RESULTS.md) と
`.claude/skills/cutetts-ja-pipeline/SKILL.md` にまとまっています。
最良checkpointは `checkpoints/s1v2-fp32-30000/`（ローカル退避済み、
`strict=True` でロード確認済み）。

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

### 状態: 原因確定・修正済み（2026-09-02）

**S1が失敗していた原因はデータではなく学習実装のバグだった（[R-020](07-risks-and-decisions.md)）。**
公開checkpointの `qwen_backbone` / `locenc` は bf16 で、`AdamW` がそれを直接更新すると
lr=2e-5 の更新量が bf16 の丸め幅を下回り、**3,000 step 回しても backbone は 3.68% しか
動いていなかった**（`ParameterDrift` 実測。fp32 なら 100%）。
学習されていたのは fp32 の DiT head だけで、**19回の試行はすべてこの条件下の観測**。

修正後の到達点（評価set v3・600文・`--param-dtype float32`）:

| 実行 | in_domain mean / median |
|---|---:|
| base | 35.86 / 31.91 |
| bf16 対照 3,000 step | 31.21 / 28.00 |
| fp32 3,000 step | 25.98 / 22.86 |
| **fp32 12,000 step** | **21.78 / 18.90** |

base → 12,000 step は **-14.09pt**（95%CI [-15.49, -12.75]、600文中440文で改善）。
ASR床 10.4% に対し、TTS由来の誤りは **25.5pt → 11.4pt**（55%削減）。

**以下は修正前の記録。結論はすべて再検証を要する。**

305時間で学習しても、S0（7.15時間）の 28.4% に届かない。
S1系の最良は **30.8%**（密なクラスタ17.5時間）。

原因は2つに絞れた（[R-018](07-risks-and-decisions.md) / [R-019](07-risks-and-decisions.md)）:

1. **クラスタ密度**。S1は1クラスタ median 5発話で、`PairSampler` が
   組み合わせを作れない。密なクラスタに絞ると 36.8% → **30.8%**
2. **S0データ固有の性質**（未解明）。混ぜると 29.6%（median 25.8%）まで届く

**データ量は効かない。** 305h でも 31.8% が上限で、7.15h の 28.6% に劣る。
step数・moe比率・話者あたりの学習量も棄却済み。詳細は [RESULTS.md](RESULTS.md)。

データは [tts-dataset/cutetts-ja-latents](https://huggingface.co/datasets/tts-dataset/cutetts-ja-latents)
（gated: manual）に置いた。以降のインスタンスは約2 GBの取得だけで学習できる。

| split | 発話 | 時間 | voice cluster |
|---|---:|---:|---:|
| train | 159,964 | **265.7h** | 894 |
| dev-zero-shot | 5,410 | 8.1h | 52 |
| test-zero-shot | 22,847 | 31.0h | 67 |
| dev-seen | 4,182 | 6.9h | 254 |
| test-seen | 4,203 | 6.9h | 277 |

gol 5ゲーム（326時間・1,197話者ID・215 GB）を vast.ai 上で前処理した。
215 GBはローカルへ落としていない。所要 約9時間・**$2.8**。

前処理の過程で2つの静かな欠陥を見つけて修正した（[R-014](07-risks-and-decisions.md)、
分割tarの取りこぼし）。詳細は[RESULTS.md](RESULTS.md)。

### ゴール

- [x] **Japanese CER**（v3・600文、base 35.86% → **21.78%**）、
      **streaming latency/RTF**（TTFA 84ms、RTF 0.435、peak VRAM 1.89 GiB）
      — [ ] 自然性・アクセントは未測定（日本語向けの信頼できる自動指標が無い。
      聴取での定性確認に置き換えるかの判断が要る）
- [~] seen speakerとzero-shot speakerの差 — 測定はしたが
      **指標が鈍い**（zero-shot 12/12・差 +0.247 だが base 自身も 12/12 で通る）。
      本当に定量化するには話者数を増やした SIM-o / SIM-r が要る
- [x] **英語・中国語のforgetting**（[R-022](07-risks-and-decisions.md)）—
      英語 WER 1.7% で無傷、**中国語 CER 11.5% → 77.2% で壊滅**。
      原因は忘却ではなく**漢字の読みが日本語に上書きされたこと**
- [~] S2で使うconfig — **fp32 master weights は必須**（R-020）、
      **step数は12,000でも飽和しない**（5,000→12,000 で -2.78pt）、
      **データ量よりstep数が効く**（19倍のデータで -1.90pt vs 4倍のstepで -4.20pt）。
      Patch Encoder train/freeze と replay混合比は未実施
- [x] **streaming生成がoffline同等** — `voice_clone` は**ビット一致**、
      7/7 ok・corrupt 0。`tts` の乖離 2.45e-03 は −52 dB で再現性ノイズと同オーダー

**out_of_domain（数字・固有名詞）はS1のゴールに含めない（D-026）。**
golのcorpusで数字を含む文は1.3%しかなく、データ量では解決しないため。
S1ではD-008の「読み誤りの内訳を集計する」だけを行い、対処はS2で決める。

### 判断ゲート

- D-009（日本語/replay比率）: 100%日本語との比較結果で確定
- D-008: 読み誤りの内訳を集計し、reading/G2P追加（J2）の要否を確定

---

## J3: 読み付与 frontend — **完了**

### 結論

**誤読を frontend で消せる。再学習は要らない。**
専用set（300文）で読みCER **-4.23pt** [-5.44, -3.05]、
会話文（600文）でも **-1.26pt** [-1.73, -0.80]。**どちらも有意。**
**悪化しないどころか改善する**（J2 は +0.02pt で差なしだった）。

### 目的

誤読を frontend で消す。聴取4回すべてで最多だった指摘（読み間違い 45%）に直接対応する。

### 根拠（実測済み）

誤読の主因は byte-fallback による文字の分解（[R-027](07-risks-and-decisions.md)）。

| 語 | 分割 | byte-fallback |
|---|---|---:|
| **中華** | `['▁','中','<0xE8>','<0x8F>','<0xAF>']` | **3** |
| すなぎも | `['▁','す','な','ぎ','も']` | **0** |

同一モデル・同一reference・同一seedで表記だけを替えると直る:

| 入力 | ASR転写 |
|---|---|
| ……中華ですね | ……シュカですね ✗ |
| ……**ちゅうか**ですね | ……**中華**ですね ○ |
| それじゃ**湊**さんを | それじゃ保さんを ✗ |
| それじゃ**みなと**さんを | それじゃ**みなと**さんを ○ |

聴取でも 2/2 で仮名が良いと判定された。
先行する J2（数詞の読み展開）は同じ機構で **-11.80pt**（95%CI [-17.45, -6.00]、有意）。

### ゴール

- [x] 任意の日本語テキストから読みを取れる（`pyopenjtalk-plus`。D-035で選定済み）
- [x] **どの語を置換するかの規則が決まっている**（D-036）。
      byte-fallback を含む語のみ。1文字語も対象、記号は除外。
      `src/cutetts/training/yomi.py`
- [x] **専用の評価setで効果が測られている**
      （`data/eval/yomi_eval_set.json`、300文/300話者/114game、学習との重複0）。
      素CER **-2.64pt** [-4.07, -1.24]、読みCER **-4.23pt** [-5.44, -3.05]、
      どちらも有意（[R-028 / R-029](07-risks-and-decisions.md)）
- [x] **通常の会話文で悪化しない**（`eval_set_v3` in_domain 600文、同一GPUで対照）。
      素CER -0.60pt [-1.12, -0.10]、読みCER **-1.26pt** [-1.73, -0.80]、
      どちらも**有意に改善**。J2（+0.02pt、差なし）より良い
- [x] `scripts/synthesize_japanese.py` に組み込まれ、既定で有効（`--no-yomi` で無効）

### 射程（原理的な限界。実測済み）

| 誤読の種類 | J3で直るか | 例 |
|---|---|---|
| **一般語の難読** | **直る** | 砂肝、会釈、中華、十分 |
| **一般的な人名・地名** | **直る** | 湊、若葉、青葉、千弘 |
| 作品固有のキャラクター名 | **直らない** | 藤宮高邦、聖、珠音、剣丞 |
| 文脈依存の同形異音 | 直らない | 一日 |

**作品固有名にはユーザー辞書が必要で、別作業として切り分ける。**
`聖` を `キヨシ` と読むのは一般語として正しく、人名で `せい` と読むかは
文脈では決まらない。**J3は「すべての誤読を直す」ものではない。**

### 設計課題

1. ~~依存関係の追加~~ → **確定（D-035）: `pyopenjtalk-plus`**。
   4候補を実測比較して選んだ（読み12/14、記号を壊さない、発音形を返す、
   アクセント情報も取れる、MIT、Windows wheel でビルド不要）。
   `pyproject.toml` の `[ja]` extra に置いた（upstream推論には不要なので core に入れない）。
   **`[onnxruntime]` extra は効果ゼロだったので入れない。**
   詳細は [07章の選定節](07-risks-and-decisions.md#形態素解析器の選定d-0352026-09-13)
2. **置換の対象をどう選ぶか**。候補:
   - byte-fallback を含む語だけ（R-027 と直結。機械的に決まる）
   - 辞書の読みが一意でない語だけ
   - 出現頻度が低い語だけ
   **最初の案が最も根拠がある**（fallbackが原因だと実測できている）
3. **数詞との順序**。J2（漢数字→仮名）と J3（語→仮名）を同じ経路に入れる

### 判断ゲート

- 専用評価setで **有意な改善**が出るか
- 会話文で悪化しないか（J2 と同じ基準）
- 悪化する場合は置換規則を絞る

---

## J4: Tokenizer 互換拡張

### 目的

byte-fallback を減らす。gol 30万文で単独pieceを持たない文字は **1,115種**あり、
`ゆ` `ぬ` のような常用ひらがなすら3 tokenに分解される。

### 根拠と限界（[R-024](07-risks-and-decisions.md)）

**実装は安価**。`lm_head` が無く入力embeddingだけを拡張すればよく、
`PreTrainedTokenizer` の added-token 機構がそのまま効くので**推論コードを変えずに済む**。
上位20文字で未収録文字の出現の **73.8%** をカバーする。

| 文 | 拡張前 | 拡張後 |
|---|---:|---:|
| `ゆっくり` | 7 token | **5 token** |
| `価格は千二百八十円` | 12 token | **8 token** |

**ただし J2 の読みはほとんど改善しない**（14 → 13 token）。
`▁`（語頭マーカー）の位置がずれる副作用もある。

**J3 の後に評価が変わる。** J3 は仮名を大量に入力するので、
`ちゅうか` に残る `ゅ` の fallback（3 token）が効いてくる。
**J3 を先にやり、その結果を見てから J4 の価値を測る。**

### ゴール

- [ ] 追加する文字集合が決まっている（頻度順。上位N文字）
- [ ] 新しい embedding の初期化方法が決まっている
      （その文字が現在分解される byte-fallback token の平均が素直）
- [ ] `▁` のずれが学習・推論で一致している
- [ ] 拡張前後で CER を比較し、**有意差の有無**が出ている

---

## T1: 学習率の探索 — **完了。梃子ではなかった**

### 結論（[R-035 / D-043](07-risks-and-decisions.md)）

**`lr=2e-5` は変える必要がない。** 4水準を同一条件で比較した
（10,000 step / batch 4 / seed 42 / fp32。**全水準 `ParameterDrift` 1.0000**）。

| lr | 読みCER | 輪郭の相関 | アクセント対人間 |
|---|---:|---:|---:|
| 1e-5 | 17.87% | +0.065 | 40.7% |
| **2e-5（現行）** | **15.80%** | +0.083 | 41.6% |
| 5e-5 | 16.11% | **+0.096** | **43.0%** |
| 1e-4 | 18.81% | **+0.106** | 38.8% |

現行との対応のある検定（読みCER 600文、検出限界 約1.0pt）:

| | 差 | 95%CI | 判定 |
|---|---:|---|---|
| 2e-5 → 1e-5 | **+2.06pt** | [+1.37, +2.74] | **有意に悪い** |
| 2e-5 → 5e-5 | +0.31pt | [-0.42, +1.03] | **有意差なし** |
| 2e-5 → 1e-4 | **+3.00pt** | [+2.14, +3.92] | **有意に悪い** |

輪郭の相関とアクセントは**3水準すべて有意差なし**。

**「2e-5 は凍結時代に選んだ値だから最適でないはず」という仮説は
支持されなかった。** 偶然良い値を引いていた。

### ゴールの達成状況

- [x] 3〜4水準を同一step数で比較している
- [x] **CER・抑揚・アクセントの3つ**で判定している
- [x] 差に信頼区間が付いている
- [x] `ParameterDrift` が全runで記録されている（全水準 1.0000）
- [x] 最良値が 2e-5 以外なら step数の収穫逓減も測り直す
      → **2e-5 が最良だったので不要**

### 3指標を分けて測った意味が出た

**`lr=1e-4` は輪郭の相関では最良（+0.106）なのに、CERは最悪で
アクセントも最下位（38.8%）。** 固定回答（42.3%）すら下回る。
**CERだけ、あるいは抑揚だけを見ると逆の結論になる。**

### 実行の記録

vast.ai RTX 3090（$0.123/h）、**約6.5時間 / 約$0.80**。
評価は `--shard 3` で並列（単独だとGPU使用率15〜71%、3並列で98%）。
実装の誤りを6件見つけて直した（[RESULTS.md](RESULTS.md) の T1 節）。

---

## T2: batch size / 学習対象 — **完了。梃子ではなかった**

### 目的

**lr 以外の一次パラメータを初めて確かめる。**

### 根拠

T1 で lr は梃子でないと分かった（R-035）。学習19回すべてで次が固定のまま:

| 項目 | 値 | 未検証の理由 |
|---|---|---|
| `batch_size` | **4** | VRAM 16GB のローカル制約で決めた値。vast.ai なら上げられる |
| `flow_copies` | **4** | 1発話あたりのflow sampling数。増やすと勾配のノイズが減る |
| `condition_dropout` | **0.1** | CFGのためのdropout率。日本語で最適とは限らない |
| 学習対象 | **6 module全部** | **head は凍結時代に唯一学習されていた**ので過適合の可能性 |

**学習対象がとくに怪しい。** R-020 以前は実質 head（70.5M）だけが
学習されていた。いま全部を動かしているが、**head を凍結した方が良い**
可能性は一度も試していない。

### ゴール

- [x] `batch_size` を 2水準以上（4 / 16）で比較している。**16 は同step と同sample の2本で挟んだ**
- [x] **学習対象**を 2水準以上（全部 / head凍結）で比較している
- [x] **CER・抑揚・アクセントの3つ**で判定している
- [x] 差に信頼区間が付いている
- [x] `ParameterDrift` が全runで記録されている（head凍結の run は head=0.0000、他は 1.0000）
- [ ] `flow_copies` / `condition_dropout` は**扱っていない**（T3 候補）

### 注意

* **batch を変えると実効的な学習率が変わる。** 比較するなら
  lr を線形に合わせるか、合わせないことを明記する
* **多重比較**になる。主指標を読みCERに決め打ちし、
  抑揚とアクセントは「壊れていないかの確認」として読む（T1と同じ扱い）

### 見積もり

T1 と同じ規模なら **約6時間 / $0.80**（vast.ai RTX 3090、評価3分割並列）。
水準を増やすなら比例して伸びる。

### 結果（2026-09-16、[R-036](07-risks-and-decisions.md) / D-044）

**batch size も学習対象も梃子ではなかった。**

| run | batch | step | 見たサンプル | 学習時間 | 読みCER | 基準線との差 |
|---|---:|---:|---:|---:|---:|---|
| 基準線（T1 2e-5） | 4 | 10,000 | 4万 | 0.51h | 15.80% | — |
| head凍結 | 4 | 10,000 | 4万 | 0.44h | 16.58% | **+0.78pt 有意に悪い** |
| batch16・同sample | 16 | 2,500 | 4万 | 0.40h | 17.47% | **+1.67pt 有意に悪い** |
| batch16・同step | 16 | 10,000 | 16万 | 1.61h | **13.90%** | **-1.91pt 有意に良い** |

* batch16・同step は **30,000 step（batch 4、13.38%）と区別できない**
  （+0.51pt [-0.19, +1.25]）。学習時間もほぼ同じ。**効いているのは計算量**
* 輪郭の相関とアクセントは、基準線に対してどの run も有意差なし
* lr は線形に合わせていない（全runで 2e-5）

実行: vast.ai RTX 3090（T1 と同じインスタンス）、学習 2.45h + 評価 約3.5h、
**約6時間 / 約$1.0（推定）**。

---


## F1: 評価を実運用と揃える（**次に着手**、GPU不要）

### 目的

**文書の主値を、実際に使う経路の値にする。**

### 根拠

`evaluate_japanese_cer.py` は **既定では frontend を適用しない**
（`--expand-numerals`（J2）と `--assign-yomi`（J3）は任意フラグ）。
T1 / T2 / step数sweep の値はすべて **frontend 無し**である。
一方、実運用の entrypoint `synthesize_japanese.py` は **J2 / J3 が既定で有効**。

同じ 30,000 step checkpoint・同じ600文での実測:

| 経路 | 素CER | 読みCER |
|---|---:|---:|
| frontend 無し（checkpoint比較用） | 20.10% | 13.38% |
| **J2 + J3（実運用）** | **19.71%** | **12.36%** |

**-1.02pt の差が、文書のどこにも主値として出ていない。**

### ゴール

- [ ] 現行最良の主値を **12.36%**（frontend込み）に統一する
- [ ] checkpoint どうしの比較は **frontend 無しで揃える**（過去の値と比較可能にするため）
- [ ] 以後の評価では両方を記録する規約を書く
- [ ] **抑揚・アクセントは frontend 込みで一度も測っていない**ことを明記し、
      M2 か D1 のついでに1回測る

### 見積もり

約1時間。**GPU不要**（すでにある metrics を読み替えるだけ）。

---

## M2: 抑揚・アクセントの天井を測る — **完了。天井は +0.38**

### 目的

**+0.122 や 45.3% が「どこまで行けるのか」を出す。**
天井が無いと、抑揚に投資すべきかを決められない。

### 根拠

M1 では「同一文・同一話者の別テイクが無い」と書いたが、gol の metadata を
全走査すると、**同じ game・同じ話者・同じ台詞で長さの違う録音が
27,367組（557 game / 4,138話者）ある**（2〜15秒・10文字以上に限った数）。
人間 対 人間で測れば、そのまま指標の天井になる。

ただし **既存の抑揚set（6 game）の中には2組しかない。** 別の game の音声が要る
（tar は game 単位。組数の多い上位3 game で 66.4 GB、190組 / 60話者）。

### ゴール

- [x] 別テイク対を **150組以上**（話者30人以上）集める → **200組 / 70話者**
- [x] 人間 対 人間で **輪郭の相関**と**アクセント核の一致**を測る
- [x] 現行モデル（+0.122 / 45.3%）を床と天井の間に位置づける → **隔たりの約32%**

### 手順

1. metadata から別テイク対を選ぶ（`build_prosody_set.py` と同じ規約:
   2〜15秒 / 内容語つき / 1話者4組まで）
2. tar を **1本ずつ落として必要な wav だけ取り出し、tar を消す**
   （`fetch_prosody_audio.py` と同じ方式。ディスクは35 GB以上空ける）
3. `evaluate_prosody.py` の「モデル音声」の位置に、もう一方のテイクを入れて測る

### 見積もり

vast.ai RTX 3090 で **約1.5時間 / 約$0.3**（66.4 GB のダウンロードが主。
アラインメント自体は MMS_FA で数分）。

### 判定

| 天井 | 読み方 | 次の行動 |
|---|---|---|
| **+0.30 以上** | 現行 +0.122 は道半ば | 抑揚を主目標にする根拠になる |
| **+0.15 前後** | ほぼ届いている | 抑揚から手を引き、読みと声質へ |

**別テイクは同じ台詞でも感情や文脈が違いうる**ので、出る値は
**天井の下限**として読む。

### 結果（2026-09-17、[R-037](07-risks-and-decisions.md) / D-045）

**天井は +0.38。現行モデルは隔たりの約32%しか埋めていない。**

別テイク **200組 / 70話者**（6 game、tar 22.4 GB）。

| | 床 | **現行モデル** | **天井（別テイク）** |
|---|---:|---:|---:|
| 輪郭の相関 | -0.009（v2） / **+0.015**（天井set） | **+0.122** | **+0.38**（中央値 +0.42） |
| アクセント核（句単位） | 固定回答 40.5% | **45.3%** | **64.5%**（1,092句） |

* 天井setでの床との差 **+0.366**（95%CI [+0.320, +0.412]、有意）
* **アクセントは辞書と同水準で止まっている**（人間 対 辞書 43.2%、モデル 45.3%、天井 64.5%）
* **抑揚の幅は天井側とほぼ同じ**（テイクA 14.33 / テイクB 14.49）。足りないのは中身
* **判定「+0.30 以上なら抑揚に投資」を満たした** → D-045

**測定器の一貫性も確認できた。** 別の game・別の話者で作ったsetなのに、
床（+0.015 対 -0.009）と人間 対 辞書（43.2% 対 44.8%）が近い値になった。

実行: vast.ai RTX 3090、**約40分 / 約$0.15**（22.4 GB の取得が主）。

---

## D1: データ量の再測定（**提案**）

### 目的

**S2（1,000時間）へ進むかを決める。**

### 根拠

データ量の比較は **3,000 step でしか行っていない**（17h → 325.9h で -1.90pt）。
その条件では 17h がほぼ1 epoch、325.9h は **1 epochの5%** しか見ておらず、
**大きいデータ側に不利**だった。T2 で「効いているのは計算量」と分かったので、
**計算量を現行最良と同じ（30,000 step）にして測り直す。**

### ゴール

- [ ] 17h と 325.9h を **30,000 step / batch 4 / lr 2e-5 / seed 42** で比較する
- [ ] 3指標すべてで差に信頼区間を付ける
- [ ] `ParameterDrift` を記録する

### 手順

latent は全量キャッシュ済みなので **前処理は要らない**（manifest を絞るだけ）。
部分集合は **クラスタ単位で無作為・seed固定**で選ぶ（大きいクラスタから採ると
密度が交絡する。2026-09-02 と同じ規約）。325.9h 側は既存の
30,000 step checkpoint を再利用する。`scripts/build_data_subset.py` が要る（未実装）。

### 見積もり

学習 約1.5h + 評価 約1.5h = **約3時間 / 約$0.5**。

### 判定

| 結果 | 意味 | 次 |
|---|---|---|
| **有意かつ 2pt 以上** | データ量は梃子だった（3,000 step の比較が不利だっただけ） | **S2 に進む** |
| 有意でない | 規模は梃子でない | **S2 を見送り**、データの質と計算量へ |

---

## C1: 計算量を4倍にする（**提案**）

### 目的

**収穫逓減がどこで止まるかを1点で測る。**

### 根拠

T2 で分かったのは「効いているのは batch でも lr でもなく **計算量**」だった。
step数の効きは 10,000→20,000 で -1.88pt、20,000→30,000 で -0.96pt と
**半減している**。この先に 1pt 以上が残っているかは測っていない。

### ゴール

- [ ] 現行最良（30,000 step / 12万サンプル）の **4倍の計算量**を1本走らせる
      （batch 16 × 30,000 step = 48万サンプル、約4.8h）
- [ ] 3指標で現行最良と比較する

### 見積もり

学習 約4.8h + 評価 約1.5h = **約6.5時間 / 約$1.1**。

### 判定

| 結果 | 次 |
|---|---|
| **1pt 以上の有意改善** | 最終モデルは長時間学習で作る。規模より先に計算量 |
| 有意でない | 計算量も頭打ち。**残るのはデータの質**（S2 / S3 の意味が変わる） |

---

## T3: flow_copies / condition_dropout（**提案・優先度低**）

### 目的

学習設定で唯一残った2つを確かめる。

### 根拠

`--flow-copies` / `--condition-dropout` は `train_continual.py` に
**既に引数がある**（追加実装は要らない）。ただし lr / batch / 学習対象が
すべて梃子でなかったので、**期待値は低い**。

### 見積もり

2〜4本で **約4〜8時間 / 約$0.7〜1.3**。

---

## M1: 抑揚・アクセントの測定 — **完了**

### 何ができるようになったか

**CERの外にある指摘を測れるようにした。** 評価setは240文 / 53話者
（`data/eval/prosody_eval_set_v2.json`）。同一文・同一話者の**人間の実音声**と比べる。

| 指標 | 実装 | 床 | 検出限界 |
|---|---|---:|---:|
| **抑揚の幅** | `ProsodyStats.semitone_range` | 人間 16.79半音 | — |
| **輪郭の相関** | `contour_similarity` | **-0.009** | 0.045 |
| **アクセント核** | `observed_nucleus` | 固定回答 42.3% / 当てずっぽう 29.7% | — |

### 実測（[RESULTS.md](RESULTS.md) / [R-032](07-risks-and-decisions.md)）

| | base | **30,000 step** | 人間 |
|---|---:|---:|---:|
| 抑揚の幅（半音） | 16.21 | 17.08 | 16.79 |
| 輪郭の相関 | +0.024 | **+0.122** | — |
| アクセント核（対人間） | 35.2% | **43.6%** | 辞書が 44.8% |

* **未学習モデルは人間の抑揚を再現していない**（床と区別できない）。学習が与えている
* **アクセントも学習で有意に良くなる**（+8.32pt [+4.46, +12.25]）
* **抑揚の幅は元から人間並み。** 足りないのは「量」ではなく「中身」

### 作るのに要した修正

**測定器を3回直してからでないと正しい値が出なかった。**

| | 何が間違っていたか |
|---|---|
| n=67 | 検出限界 0.084 に対して観測値が小さすぎ、**何も判定できなかった** |
| F0（R-033） | **広い範囲だと倍音に乗る。** 男声110Hzを392Hzと報告した |
| 核の規則（R-034） | **「落差最大」は固定回答に負ける。** 核は「最後に高いモーラ」 |

**n=67 で出した結論は2つとも覆った。**

### 手段の選定

| | 採用 | 理由 |
|---|---|---|
| F0推定（D-040） | `pyworld`（harvest + stonemask） | 調波信号で誤差0.01%、**無声判定がある**。`torchaudio` のものは有声/無声を判定しない |
| 強制アラインメント（D-041） | `torchaudio` の **MMS_FA** | **追加依存なし**。モーラを「単語」として渡すとモーラ単位の区間が返る |
| 構造（D-042） | **full-context label** | `run_frontend` の `acc`/`chain_flag` だと促音でモーラ数がずれ、平板の符号化も取り違える |

### まだ測れないもの

* **平板と尾高の区別**（句の内部ではどちらも「下がらない」。次の句まで要る）
* **話速・間の取り方**（`seconds` は記録しているが指標にしていない）
* **声質**。聴取であった「高音域がつぶれる」は未測定。
  **P1c も高音域を評価していない**（log-mel L1 / SNR / speaker cos のみ）。
  VAE往復で帯域別に測れば S4（Japanese VAE）の再判断材料になる

### ~~天井が分かっていない~~ → **測った（M2、2026-09-17）**

M1 の時点では「別テイクが無いので天井が測れない」と書いたが、gol の metadata に
**27,367組**（557 game / 4,138話者）あった。別テイク 200組で測った天井は
**輪郭 +0.38 / アクセント 64.5%（句単位）**。

**現行モデルは輪郭で隔たりの約32%しか埋めていない**（床 +0.015 → +0.122 → 天井 +0.38）。
アクセントは辞書（43.2%）と同水準で、天井まで約19pt ある。→ [R-037](07-risks-and-decisions.md) / D-045

---


## S2: 1,000時間（**保留**）

### 状態: 保留（2026-09-16 再確認）

**データ規模を上げる根拠が実測で得られていない。**

| 手段 | 効果 | 学習 |
|---|---:|---|
| dtype修正（bf16→fp32） | -15.77pt | 要（済） |
| **J2 読み展開（数詞）** | **-11.80pt** | **不要** |
| step数 3,000 → 30,000 | -4.20pt | 要（済） |
| **J3 読み付与（語）** | **-4.23pt**（読みCER） | **不要** |
| **データ量 17h → 325.9h（19倍）** | **-1.90pt** | 要 |
| **学習率（T1で探索）** | **効果なし**（2e-5が最良） | 要（済） |
| **batch size / 学習対象（T2）** | **同じ計算量なら効果なし**。head凍結は悪化 | 要（済） |

19倍のデータで -1.90pt に対し、frontend は学習なしで -11.80pt / -4.23pt を出した。
**T1 で lr、T2 で batch size と学習対象も梃子でないと分かった**（R-035 / R-036）。
残る未探索は `flow_copies` / `condition_dropout` だけ。

#### 保留を解く条件

次のいずれかが満たされたとき、規模を上げる判断材料になる:

* ~~T2 で有意な改善が出る~~ → **出なかった**（同じ計算量では差なし。R-036）
* **学習設定はほぼ出尽くした** → 残るのは「計算量」「データ量」「データの質」
* **D1（データ量を 30,000 step で測り直す）が判定になる。**
  有意かつ 2pt 以上なら **S2 に進む**。有意でなければ **S2 は見送り**
* **C1（計算量4倍）で改善が続くなら、規模より先に計算量へ投資する**
* ~~抑揚・アクセントが規模で伸びる見込みが立つ~~ → **天井は出た**（M2 / R-037）。
  ただし「規模を上げれば抑揚が伸びる」という根拠は**まだ無い**。伸びしろの存在と、
  それを埋める手段がデータ規模かどうかは別の問い

いずれにせよ **CER・抑揚・アクセントの3つで測る**（M1完了で可能になった）。

ただしデータ量の比較には**自分で認めた限界がある**。比較は 3,000 step で行い、
325.9h はそこで1 epochの5%しか見ていない（17h はほぼ1 epoch）。
**大きいデータセットに不利な条件だった。** **これを 30,000 step で測り直すのが D1。**
S2 の進退はその結果で決める。

### 着手前に決めること

- D-009（replay混合）は**不要**になった（D-032 で中国語を諦めたため）。100%日本語で進める
- P1e Pass B（全音声の前処理）が必要。実測外挿で **65.3 GB / 239 GPU時間**
- voiceクラスタ閾値 t=0.92 の再較正（D-020。Pass B規模で未検証）
- [R-026](07-risks-and-decisions.md) で実装した `ensure_minimum_duration` を
  残すか消すか（効果の裏づけが無い）

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

**2026-08-30 時点の表に、T2 後の決定（M2 / D1 / C1）を追記した。**
確定済みの決定は [07章の D 表](07-risks-and-decisions.md)（D-001〜D-044）を見る。

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
| **抑揚に投資するか** | **確定: 投資する**（D-045） | ~~M2~~ 完了 | 天井 **+0.38**（現行 +0.122 は隔たりの約32%）。アクセントも天井 64.5% に対し現行 45.3% |
| **S2（1,000時間）へ進むか** | 未確定 | **D1** | 30,000 step でのデータ量比較。**有意かつ2pt以上で進む** |
| **最終モデルを長時間学習で作るか** | 未確定 | **C1** | 計算量4倍での3指標比較。1pt以上の有意改善で採用 |

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
