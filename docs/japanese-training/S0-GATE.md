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
