# S0 ゲート定義（確定）

確定日: 2026-08-31
**この文書の数値は結果を見て変更しない。** 変更する場合は理由と差分を必ず残す。

## 評価set

| 項目 | 値 |
|---|---|
| ファイル | `data/eval/s0_eval_set.json` |
| version | 2 |
| checksum | `9dd568218a49ad76...` |
| seed | 20260831 |
| 構成 | in_domain 30 / out_of_domain 12 / phonetic 10 = 52文 |

`in_domain` は gol の実テキストから **1話者1文**、学習manifestに含まれない発話から
seed固定で選出。v2 で語彙的内容を持たない感情表現を除外した
（除外基準はテキストの性質のみで、CERを参照していない）。
v1 は `data/eval/s0_eval_set_v1_superseded.json` に保存。

## 測定条件（固定）

| 項目 | 値 |
|---|---|
| ASR | `kotoba-tech/kotoba-whisper-v2.0`（D-019） |
| mode | voice_clone |
| reference | `assets/default_reference.wav` |
| seed | 42 |
| max_decode_length | 400 |

## 基準線（未学習 base checkpoint）

| subset | n | mean | median | p90 |
|---|---:|---:|---:|---:|
| **in_domain** | 30 | **35.8%** | **30.2%** | — |
| out_of_domain | 12 | 74.7% | 71.4% | 100.0% |
| phonetic | 10 | 46.9% | 42.3% | 84.6% |

artifact: `artifacts/s0-cer/2026-08-30T15-59-54/`

参考: v1（感情表現を含む）では in_domain mean 41.0% / median 33.1% だった。

## 合格条件

S0 は品質競争ではなく **可能性の確認** なので、閾値は控えめに置く。

| ゴール | 判定条件 |
|---|---|
| **主ゲート** | in_domain CER **mean が 35.8% から有意に低下**。目安は 30% 未満 |
| 副次 | phonetic CER が 46.9% から悪化しない |
| 副次 | out_of_domain が 74.7% から大きく悪化しない（R-010より改善は期待しない） |
| 定性 | textを入れ替えると出力内容が追随する |
| 定性 | referenceを入れ替えるとspeaker identityが追随する |
| 定性 | 未学習文でも完全なmemorizationではない |

**「日本語音声が出る」はゲートにしない。** 未学習baseで既に成立しているため。

## 中止・巻き戻し条件

- NaN / overflow の再現
- stop head が学習できず無限生成または早期停止
- target/reference leakage による見かけの成功
- in_domain CER が基準線より **悪化**

## 既に達成済みのゴール

| ゴール | 結果 |
|---|---|
| microbatch 1 の peak VRAM と throughput を実測 | 4.15 GB / 150 ms/step |
| 16 GB で full fine-tuning が載るか | 載る（D-006確定） |

---

## 結果（2026-08-31）

### 1回目: 無効（R-012）

`PairSampler.sample()` の誤用で3000stepすべてが同じ4発話だった。
flow loss 1.02 -> 0.003 は丸暗記で、学習後のモデルは未学習baseより悪化していた。
詳細は [07章 R-012](07-risks-and-decisions.md)。**この結果は破棄。**

### 2回目: 主ゲート通過

修正版（`iter_pairs()` の stream から引く）で 3000 step、lr 2e-5、
batch 4、warmup 100、condition dropout 0.1、group_key=voice_cluster_id。
学習データ 5,431発話 / 7.15時間 / 63 voice cluster。所要 9分（RTX 3090）。

**CER（S0の主ゲート）**

| subset | 基準線 | 学習後 | 変化 |
|---|---:|---:|---:|
| **in_domain** | 35.8% / 30.2% | **28.4% / 25.5%** | **-7.4pt / -4.7pt** |
| phonetic | 46.9% / 42.3% | **35.9% / 29.4%** | **-11.0pt / -12.9pt** |
| out_of_domain | 74.7% / 71.4% | 76.4% / 71.1% | +1.7pt / -0.3pt |

主ゲート「in_domain mean が 35.8% から有意に低下（目安30%未満）」を満たす。
out_of_domain の +1.7pt は n=12 の誤差範囲（median は -0.3pt）で、
R-010（domain偏り）の予想どおり改善しない。

**flow / stop loss（同じ診断経路で base と比較）**

| split | base flow | 学習後 | base stop | 学習後 |
|---|---:|---:|---:|---:|
| train | 0.7775 | **0.5930** | 0.1375 | **0.0284** |
| dev-seen | 0.7612 | 0.7206 | 0.1483 | 0.0408 |
| dev-zero-shot | 0.7680 | 0.7498 | 0.0917 | 0.0258 |

flow は train で -23.7%、未知話者では -2.4% にとどまる。
**stop はすべての split で -72%〜-80%** と大きく改善しており、
「stop head が学習できず無限生成または早期停止」の中止条件には該当しない。

CER が 7.4pt 改善した一方で dev-zero-shot の flow がほぼ動かないのは、
CER評価が固定reference（`assets/default_reference.wav`）の1話者条件で、
改善の主体が **text -> 音韻の対応** にあるため。任意話者の音響予測とは別物。

### ゴールの達成状況

| ゴール | 結果 |
|---|---|
| 基準線CERをゲート値として固定 | 達成（本文書） |
| **in_domain CER が基準線から明確に改善** | **達成**（35.8% -> 28.4%） |
| textを入れ替えると出力内容が追随 | 達成（未学習52文でCER 28.4%。追随しなければ約100%） |
| 未学習文でも完全なmemorizationではない | 達成（評価文は学習manifest外。dev-zero-shot も悪化せず） |
| referenceを入れ替えるとspeaker identityが追随 | `check_reference_following.py` で測定 |
| microbatch 1 の peak VRAM と throughput | 達成（4.15 GB / 150 ms/step） |
| 16 GB で full fine-tuning が載るか | 達成（載る。D-006確定） |
| 聴取可能な発音改善 | CERが客観指標。実聴取は未実施 |

### 中止・巻き戻し条件の確認

| 条件 | 該当 |
|---|---|
| NaN / overflow の再現 | なし |
| stop head が学習できず無限生成または早期停止 | なし（stop loss -72%〜-80%） |
| target/reference leakage による見かけの成功 | なし（`test_leakage.py` で因果性を確認） |
| in_domain CER が基準線より悪化 | なし（-7.4pt 改善） |
