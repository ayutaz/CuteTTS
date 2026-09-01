# P0 / P1 実測結果まとめ

最終更新: 2026-09-01

各フェーズの数値を1枚にまとめたもの。詳細と根拠は各章へのリンク先にある。
**ここに載っているのはすべて実行済みの実測値**であり、提案や見積もりは明記している。

## 実行環境

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER **16 GB**（05章の想定は4090 24GB。R-007） |
| Python | 3.12.13（`.venv`。システム既定の3.14ではtorch 2.5.1が動かない。R-011） |
| torch / transformers | 2.5.1+cu121 / 4.51.0 |
| 追加導入 | `triton-windows` 3.8.0（distillに必須）、`pytest`、`kotoba-whisper-v2.0` |
| ストレージ | C: 3.2 TB 空き / D: 1.5 TB 空き |

**実行コマンドは必ず `.venv/Scripts/python.exe` を使う。**

## P0: 推論ベースライン再現 — `gate_passed: true`

| checkpoint | ケース | streaming TTFA | RTF (streaming) | peak VRAM | model load |
|---|---|---:|---:|---:|---:|
| base | 7/7 ok | 251 ms | 1.89 | 2.04 GB | 44.4 s |
| distill | 7/7 ok | 210 ms | 0.52 | 2.05 GB | 56.3 s |

distill step sweep: steps=1 → RTF **0.262** / steps=2 → 0.328 / steps=4 → 0.470

### 再現性（今後の比較のノイズ下限）

| 比較 | base tts | base voice_clone |
|---|---:|---:|
| run1 vs run2 | max_abs **2.38e-04** | **完全一致** |
| streaming vs offline | max_abs 8.37e-04 | **完全一致** |

**tts経路はビット再現しない。** checkpointの差を波形で判定するなら 2.4e-04 が下限。

### 環境固有の落とし穴

WindowsのPyTorchには triton が同梱されず、`torch.compile` を使う sampler が失敗して
**distillは当初7ケース全滅した**。`triton-windows` 導入と `--sampler-compile-mode auto` で解消。
現在も compile mode は `eager`。RTFが公式報告値（0.109）より高い一因。

## P1a: データ実態 — accepted 10,466.4 h

| | 発話数 | 時間 | 話者ID |
|---|---:|---:|---:|
| raw | 7,405,094 | 10,654.3 h | 19,349 |
| **accepted** | **6,916,974** | **10,466.4 h** | **18,279** |
| excluded | 488,120 | 187.9 h（1.8%） | 1,070 |

| 除外理由 | 件数 | 時間 |
|---|---:|---:|
| too_short（<1秒） | 324,890 | 55.5 h |
| punctuation_only | 181,486 | 72.1 h |
| generic_speaker | 37,314 | 45.8 h |
| markup | 4,543 | 9.7 h |
| empty_text | 1,055 | 1.3 h |
| too_long（>30秒） | 340 | 3.1 h |
| name_placeholder（`%bd`） | 252 | 0.4 h |

moe-speech-plus: 473話者 / 395,170発話 / **621.4 h**（話者あたり最小14.3分を保証）。

**「19,349話者」は名目値。** 1時間以上を持つのは2,095話者で、それが89%を占める。
実効話者数は約2,000〜3,500。

## P1b: Tokenizer coverage — 既存維持で開始可

200,000文 / 5,594,489 token（gol実テキスト）で測定。

| 指標 | 実測 |
|---|---:|
| 文単位 / token単位 `<unk>` 率 | **0.0000% / 0.0000%** |
| **byte-fallback token率** | **9.658%** |
| **byte-fallbackを含む文** | **45.64%** |
| tokens per char | 1.1381 |
| token長 P50 / P95 / P99 / max | 25 / 58 / 75 / 211 |
| NFKCでtoken数が変わる文 | 2.357% |
| 実効text予算（tts / clone ref30秒） | 10,225 / 10,012 token（≒ 8,984 / 8,797 文字） |

**`<unk>` が0なのは256個のbyte-fallbackピースが受けているため。**
小書き仮名15字種（`ぅ ぉ ぃ ゅ` — 会話文で頻出）、漢字643字種、カタカナ18字種、
`～ ― ♪` が単一ピースを持たない。例: `龗`→4 token、`😀`→5 token。

special token: `<|im_start|>`=4、`<|im_end|>`=5、`<|endofprompt|>`=16384。
`max_length` は既定4096ではなく **10240** で、系列長は制約にならない（gol発話は平均23文字）。

→ **分岐1（既存維持）で開始（D-018）。分岐2（互換拡張）の価値は高い。**

## P1c: Audio VAE 日本語再構成 — freeze確定

### 音響（80発話 / 10話者、f0 101〜473 Hz）

| 指標 | mean | P50 | min | max |
|---|---:|---:|---:|---:|
| **speaker cos類似度** | **0.9392** | 0.9429 | 0.8643 | 0.9759 |
| SNR (dB) | 9.27 | 9.30 | 2.63 | 14.08 |
| log-mel L1 | 0.651 | 0.651 | 0.461 | 0.985 |
| latent frame rate | 12.589 Hz（仕様12.5） | | | |

### ASR CER（60発話 / 10話者、`kotoba-whisper-v2.0`）

| 比較 | mean | P50 | P90 |
|---|---:|---:|---:|
| dataset転写 vs original音声ASR | 7.83% | 4.35% | 23.53% |
| dataset転写 vs reconstruction音声ASR | 8.41% | 4.55% | 25.00% |
| **original ASR vs reconstruction ASR** | **2.21%** | **0.00%** | 8.33% |

**CER差 +0.58ポイント。中央値0.00%は半数の発話がVAE往復後も完全に同一の転写になる**ことを意味する。
3行目はASR自身の誤りが相殺されるためVAE劣化の直接指標。

→ **D-003確定（VAE freeze）。D-010（Japanese VAE / Stage 4）は見送り。**

未実施: PESQ / STOI / UTMOS、日本語母語話者のblind listening、促音・撥音・長音・無声化の固定subset。

## P1d: manifest / split / voiceクラスタ

- validator適用後 accepted 6,412発話、zero-shot話者のtrain重複 **0件**
- PairSampler 100ペアで **leakage 0件**、平均reference 9.61秒
- split: train 5,431 / test-seen 173 / dev-seen 146 / test-zero-shot 434 / dev-zero-shot 228

### voiceクラスタリングの較正 — 既定値0.70は使えない

| 分布 | mean | P5 | P50 | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|
| 話者内 cos | 0.825 | **0.608** | 0.861 | — | — | — |
| 話者間 cos | 0.540 | — | 0.586 | **0.837** | 0.883 | 0.959 |

話者内P5が話者間P95を下回り、両分布が大きく重なる。単連結は連鎖しやすく、
t=0.70では77話者中62が1クラスタへ併合された。

| 閾値 | クラスタ数 | 複数話者クラスタ | 最大 |
|---:|---:|---:|---:|
| 0.70 | 15 | 2 | 62 |
| **0.92** | **71** | **5** | **3** |
| 0.95 | 76 | 1 | 2 |

**採用 t=0.92。** leakage防止では過剰併合が安全側。

### R-004 が実データで確認された

| cos | 話者A | 話者B |
|---:|---|---|
| 0.9594 | gol | gol（別ID） |
| 0.9465 | gol | gol（別ID） |
| **0.9299** | **gol** | **moe** ← dataset跨ぎの同一声 |

**dataset単位でsplitを分けても声は漏れる。**

## P1e: 前処理パス Pass A

| 指標 | 実測（6,112発話 / 8.13 h） |
|---|---:|
| スループット | **44.5× realtime** |
| latent frame rate | 12.599 Hz |
| peak VRAM | **2.48 GB** |
| latent cache | 49.8 MB |
| speaker cache | 3.6 MB |
| 失敗 | **0件** |

### gol全体（10,654 h）への外挿

| | 実測外挿 | 計画見積 |
|---|---:|---:|
| latent cache | **65.3 GB** | 61 GB |
| 所要 | **239 GPU時間** | — |

**Pass BはS2直前まで実施しない**（voiceクラスタ閾値の再較正が先）。

## P2: 学習forward復元 — ゴール7件すべて達成

実装は `src/cutetts/training/`（objectives / collator / dataset / forward /
packing / checkpointing）。**すべてCPUで検証**（縮小した実CuteTTSModelを使用）。

| ゴール | 結果 |
|---|---|
| deterministic tiny batchでlossが再現 | 完全一致 |
| 学習対象moduleにだけgradient | 6module全てに到達。freezeで停止 |
| 1 utteranceのoverfit | flow lossが30%以上低下 |
| save/resume後の次stepが一致 | 完全一致（RNG未復元だと不一致になることも確認） |
| packingがunpackedの結果を変えない | loss一致、hidden stateも個別実行と一致 |
| 推論pathでcheckpointをload | `runtime.load_runtime` で確認 |
| 配線を外すとテストが落ちる | 変異テスト **9/9 検出**、packing境界で3件失敗 |

### 変異テストの結果（`tools/mutation_check.py`）

| 変異 | 検出 |
|---|---|
| stopラベルを1つ手前へ | 4件失敗 |
| velocity符号反転 | 2件失敗 |
| 補間のt反転 | 2件失敗 |
| joint dropout無効化 | 1件失敗 |
| flow lossのmask無視 | 2件失敗 |
| stopのpadding除外を無効化 | 2件失敗 |
| STOP_STOPを0にする | 2件失敗 |
| copiesを無視して1固定 | 2件失敗 |
| speaker dropout無効化 | 1件失敗 |

### 自分で決めた事項

論文にもコードにも規定が無い項目（04章）への回答。

| 項目 | 決定 |
|---|---|
| padding patchのloss除外 | 分子からも分母からも除く |
| stopラベルの位置 | 位置iのhiddenが「patch iが最終patchか」。`STOP_STOP=1` 固定 |
| stopのclass imbalance | `positive_weight`（重み付き平均） |
| flow/stopの重み | `stop_weight=1.0` 既定。Stage 0で調整 |
| condition dropoutの対象 | speaker + reference、既定はjoint |

### packing で見つかった実バグ

行index と sample index の混同。packingすると1行に複数sampleが入るため
speaker slot と target の対応付けが壊れる。unpacked では両者が一致するので
**packingを書くまで露見しなかった**。

## S0前段: 学習forwardのVRAM/throughput実測（2026-08-30、vast.ai RTX 3090）

**R-007（GPU規模の見積もり誤り）とD-006（full fine-tuning主案）の再判定材料。**
公開checkpointの実サイズ（228.6M）で測定。microbatch 1、flow copies 4、bfloat16。

| 構成 | 学習パラメータ | t=32（5秒） | t=64（10秒） | t=128（20秒） | t=188（30秒） |
|---|---:|---:|---:|---:|---:|
| **full**（D-005主案） | 228.6M | 3.01 GB | 3.01 GB | 3.43 GB | **4.15 GB** |
| freeze_patch_encoder | 198.7M | 2.77 GB | 2.77 GB | 3.25 GB | 3.94 GB |
| lm_only | 197.4M | 2.76 GB | 2.76 GB | 3.24 GB | 3.94 GB |

速度（full）: 100.4 / 108.4 / 142.0 / 189.6 ms/step。
モデル常駐は 0.62 GB で、残りはactivationとoptimizer state。

### 判定

- **D-006（full fine-tuning）は成立する。** 30秒の発話でも 4.15 GB で、
  24GBに対して大きな余裕がある
- **D-005 は full を選んでよい。** Patch Encoder を freeze しても
  節約は 0.2 GB（5%）、速度差はほぼゼロで、得られるものがない
- **R-007 は解消。** 16GBのローカルGPUでも十分載る（4.15 GB）。
  「VRAM不足で構成を見直す」事態は起きない

### S0本番の費用見積もり（実測ベース）

| 前提 | 値 |
|---|---:|
| GPU | RTX 3090 24GB @ $0.179/h（60GBディスク込みの実単価） |
| step時間 | 約150 ms（平均発話長） |
| 50,000 step | 約2.1時間 → **$0.38** |
| latent cache生成込み | **$1〜2** |

### この実行で見つかった実バグ

`torch.randn(..., generator=cpu_generator, device='cuda')` が
`Expected a 'cuda' device type for generator but found 'cpu'` を投げる。
再現性のためCPU generatorを使いつつGPUで学習する構成では必ず踏む。
**CPUだけのテストでは絶対に露見しない**種類の不具合で、
vast.aiで実際に回したことで発見できた。

## S0 基準線: 未学習baseの日本語CER（確定）

**確定値は [S0-GATE.md](S0-GATE.md) を参照。** 固定評価set（52文, version 2, seed 20260831）で測定。

| subset | n | CER mean | median |
|---|---:|---:|---:|
| in_domain（gol会話文） | 30 | **35.8%** | 30.2% |
| out_of_domain（数字・固有名詞） | 12 | 74.7% | 71.4% |
| phonetic（音韻的難所） | 10 | 46.9% | 42.3% |

主ゲート: **in_domain mean が 35.8% から有意に低下**（目安30%未満）。
artifact: `artifacts/s0-cer/2026-08-30T15-59-54/`

### 予備測定（12文、参考のみ）

S0の固定評価set作成前に12文で測った値。標本が小さく、**ゲートには使わない**。

| 区分 | n | CER mean | median |
|---|---:|---:|---:|
| in-domain | 8 | 29.6% | 20.9% |
| out-of-domain | 4 | 77.2% | 84.7% |

例: `そうだ！貴官のことも教えていただけませんか？` → `そうだ本番のことも教えていただけませんか`（CER 10.0%）

**out-of-domainがin-domainの2倍以上悪い**のは R-010（domain偏り）の定量的裏づけ。
日本語未学習のbase checkpointが既に理解可能な日本語を生成するため、
**S0のゲートは「日本語音声が出る」ではなく「このCERから測定可能に改善する」**とした。

## S0: 継続学習の成立確認（2026-08-31、vast.ai RTX 3090）

学習データ 5,431発話 / **7.15時間** / 63 voice cluster（S0想定10〜30hには未達）。
3000 step、lr 2e-5、batch 4、condition dropout 0.1。所要 **9分**。

### CER（主ゲート・通過）

| subset | 基準線 | 学習後 | 変化 |
|---|---:|---:|---:|
| **in_domain** | 35.8% | **28.4%** | **-7.4pt** |
| phonetic | 46.9% | **35.9%** | **-11.0pt** |
| out_of_domain | 74.7% | 76.4% | +1.7pt（n=12、medianは-0.3pt） |

### flow / stop loss（base と同一経路で比較）

| split | base flow | 学習後 | base stop | 学習後 |
|---|---:|---:|---:|---:|
| train | 0.7775 | 0.5930 | 0.1375 | 0.0284 |
| dev-seen | 0.7612 | 0.7206 | 0.1483 | 0.0408 |
| dev-zero-shot | 0.7680 | 0.7498 | 0.0917 | 0.0258 |

stop が全splitで -72%〜-80%。flow は train -23.7%、未知話者では -2.4% に留まる。
CER が大きく改善した主体は **text -> 音韻の対応**であり、任意話者の音響予測ではない。

### reference 追随性

| 条件 | 自己類似度 | 他者類似度 | 差 | argmax正解 |
|---|---:|---:|---:|---:|
| dev-seen 4話者 × 3文 | 0.833 | 0.600 | +0.232 | **12/12** |
| dev-zero-shot 2話者 × 3文 | 0.821 | 0.548 | +0.273 | 6/6 |

zero-shot split に voice cluster が3つしか無い（[R-013](07-risks-and-decisions.md)）。

### 1回目の学習は無効（R-012）

`PairSampler.sample()` を step ごとに呼んでいたため、3000 step すべてが
**同じ4発話**だった。flow loss 1.02 -> 0.003 は丸暗記で、
学習後のモデルは未学習baseより悪化していた（診断 4.15 vs base 0.78）。

| 測り方 | flow loss | R2 |
|---|---:|---:|
| 学習ハーネス（記憶した4発話） | 0.0021 | — |
| 診断（見ていない発話） | 4.15 | -1.01 |
| 未学習base | 0.78 | +0.62 |

**学習ループの損失曲線だけでは「大成功」に見えた。** 詳細は
[07章 R-012](07-risks-and-decisions.md)、ゲートと結果は [S0-GATE.md](S0-GATE.md)。

## ASR の誤り床（2026-08-31、vast.ai RTX 3090）

**人間の実音声**（gol原音声）を kotoba-whisper で文字起こしし、ゲームスクリプトと比較した。
`evaluate_japanese_cer.py` と同じASR・同じ正規化・同じCER実装を使っている。
標本は通常文40 + 数字入り40（計80件・459秒）。

| 区分 | n | CER mean | median | p90 |
|---|---:|---:|---:|---:|
| 通常文 | 40 | **10.4%** | 6.7% | 23.5% |
| 数字入り | 40 | 13.1% | 10.3% | 27.3% |
| 全体 | 80 | 11.7% | 8.3% | — |

CER 0%（完全一致）は 22/80。

### 1. CER指標には約10%の床がある

**S0の学習後CER 28.4% は「0%が理想」として読めない。** 人間の実音声でも
同じ経路で10%前後出る。TTS由来の誤りは概ね **28.4% − 10.4% ≒ 18pt** と読むべき。
S1のゲートを引くときはこの床を明示する。

### 2. ASRは数字を正しく読む

数字列が完全一致した文は **30/40（75%）**。誤りの実体は数字ではなく
固有名詞と表記ゆれだった。

```
正解: じゃあダメージ７５％で発動にしとくぞい
ASR : じゃあダメージ75%で発動にしとくぞい          CER 0.0%

正解: みこの活躍、これからも隣で見守っててね。３５Ｐ！
ASR : ミコの活躍これからも隣で見守っててねミコピン   CER 28.6%
```

### 3. ASR出力は漢字を減らさない

漢字比率は 正解text 17.2% → ASR出力 21.7%（**+26%**）。
ゲームスクリプトが可読性のため仮名で書く箇所を、ASRは標準的な漢字表記にする。
**「ASR textだと漢字の読みを学習できない」という懸念は成り立たない。**

### 判断への影響

- **moe-speech-plus のASR textは学習ラベルとして使える。** ラベル雑音は約10%で、
  数字の読みも保たれる。当初の懸念より小さい
- ただし gol のスクリプトは正解textなので、入手できる範囲では gol を優先する
- **out_of_domain（数字・固有名詞）はどちらのデータでも解決しない。** これは
  ASRの問題ではなく corpus の問題（下記）

## gol-dataset のtext分布（metadata.tsv 全7.4M行）

| 項目 | 値 |
|---|---:|
| ゲーム数 | 596（tarはゲーム単位。**事前に選定できる**） |
| 最大ゲーム | 93.0 h / 249話者 |
| **数字を含む文の割合** | **1.3%**（10h以上の434ゲームでも中央値0.9%、最大5.9%） |

**S0で out_of_domain CER が改善しなかったのはデータ量の問題ではない。**
corpus に数字がほとんど無い。golを10倍落としても解決しない。
text正規化で読みを与えるか、数字を含む文を別途補強する必要がある。

集計結果は `data/gol_game_stats.json`（596ゲーム分の時間・話者数・text性質）。

## S1 前処理（2026-09-01、vast.ai RTX 3090）

gol 5ゲームを **vast.ai上で完結** させ、latent cache だけを永続化した。
215 GBはローカルへ落としていない。所要 約9時間 / **$2.8**。

成果物: [tts-dataset/cutetts-ja-latents](https://huggingface.co/datasets/tts-dataset/cutetts-ja-latents)（public / gated: manual）

| 中身 | サイズ |
|---|---:|
| latents | 1.8 GB |
| speaker embeddings | 109 MB |
| manifests | 357 MB |

### データ

| split | 発話 | 時間 | voice cluster |
|---|---:|---:|---:|
| train | 159,964 | **265.7h** | 894 |
| dev-zero-shot | 5,410 | 8.1h | 52 |
| test-zero-shot | 22,847 | 31.0h | 67 |
| dev-seen | 4,182 | 6.9h | 254 |
| test-seen | 4,203 | 6.9h | 277 |

S0比で train **37倍**、zero-shot話者 **3 → 119クラスタ**。R-013は解消した。

### 見つけた2つの静かな欠陥

**1. 分割tarでゲームが丸ごと落ちていた（実行前に発見）**

tarのファイル名をそのまま game_id として使っていた。golの大きいゲームは
`_part1.tar` / `_part2.tar` に分割されており、選定5ゲーム中2ゲーム
（**170時間・全体の52%**）がエラーも出さずにmanifestから消えていた。
「tar 602本 vs metadata上のgame 596」という既知の差の正体でもあった。

**2. クラスタの粒度が用途に対して逆向きだった（[R-014](07-risks-and-decisions.md)）**

単連結クラスタリングの連鎖で 45話者・86.1時間（trainの30.5%）が
1クラスタになり、**学習ペアの26.9%でreferenceと違う声をtargetにしていた**。

| linkage @0.92 | クラスタ数 | 最大話者 | ペア不一致 | 境界をまたぐ同一声 |
|---|---:|---:|---:|---:|
| single | 974 | **42** | **26.9%** | 0話者 |
| average | 1,006 | 4 | 13.1% | — |
| complete | 1,021 | 2 | 8.3% | **15話者** |

完全連結だけにすると逆に、同じ声がtrainとzero-shotへ分かれた。
**用途ごとに粒度が逆向きに必要**なので、単位を2つに分けた（D-027）。

| 単位 | linkage | 用途 |
|---|---|---|
| `voice_cluster_id` | 完全連結（細かい） | PairSamplerのreference/target選択 |
| `split_group_id` | 単連結（粗い） | splitの割り当て |

修正後の検証: voice_clusterのsplitまたぎ **0件** /
境界をまたぐ最大cos **0.9198**（< 0.92） / クラスタ内の最小cos **0.9204**（≥ 0.92）。

**閾値0.92そのものは妥当だった。** 壊れていたのは併合の仕方であって閾値ではない。

## S1 学習の試行（2026-09-01、vast.ai RTX 3090）

**S1はまだ成功していない。** 265.7時間で学習しても、S0（7.15時間）の
CER 28.4% に届いていない。

### 測定した全実行

| 実行 | データ | step | in_domain CER mean/median |
|---|---|---:|---|
| base（未学習） | — | — | 35.8 / 30.2 |
| **S0** | 7.15h（gol 70% + **moe 30%**） | 3,000 | **28.4 / 25.5** |
| S1 lr5e-5 | 265.7h（golのみ） | 8,000で中断 | — |
| S1 lr2e-5 | 265.7h | 500 | 31.8 / 27.4 |
| S1 lr2e-5 | 265.7h | 1,000 | 32.4 / 27.7 |
| S1 lr2e-5 | 265.7h | 2,000 | 31.2 / 30.9 |
| S1 lr2e-5 | 265.7h | 8,000 | **51.6 / 42.9** |
| S1 lr2e-5 | **228.6h（浄化後）** | 2,000 | 34.1 / 31.0 |

### 分かったこと

**1. lr 5e-5 は高すぎた。** 同じ step 2,000 で dev-zero-shot flow が
0.8150（5e-5）vs 0.8026（2e-5）。2e-5 が全区間で上回る。

**2. 8,000 step で生成が崩壊する。** 64秒（`max_decode_length 400`）に
張り付く生成が 3件 → 8件へ増え、CER が 31.2% → 51.6% になる。
暴走区間は 0.05〜0.5 文字/秒（正常5.74）で、text に対応しない音。

**3. flow loss では崩壊を検知できない。** 同じ区間で dev-zero-shot flow は
0.8026 → 0.7909 と **改善し続けた**。D-025（学習ループの損失で判断しない）
だけでは足りず、**dev損失ですら不十分**だった。CERを測る以外に方法がない。

**4. データ汚染は主因ではなかった（仮説の棄却）。** golの1 game
（9381931FAB68、93時間）は text と音声が対応しない record を41.6%含み、
S1全体の12.5%（37時間）が `duration/文字数 > 0.40` だった（S0は1.4%）。
これを除いて再学習したが、CER は 31.2% → **34.1%** で改善しなかった。
生成長の分布も変わらない（64秒張り付き 3件 → 3件）。

  ただし **8,000 step での崩壊への寄与は未検証**（浄化データで長い学習を
  していない）。フィルタ自体は入口の品質ゲートとして残す価値がある
  （`text_audio_mismatch`、閾値0.40）。

### 残る最有力候補: moe の有無

S0（成功）と S1（未達）の最大の構成差は **moe-speech-plus 30% の有無**。

| | gol | moe |
|---|---|---|
| 音源 | ゲーム音声 | **スタジオ収録** |
| 品質フィルタ | なし | **NISQA MOS + Silero VAD 済み** |
| BGM/SE | **混入の可能性（未評価）** | 明示的に「なし」 |

7.15時間で28.4%、265.7時間で34.1%という結果は、**量ではなく質の問題**を示す。
golには秒/文字比では捉えられない品質問題がある可能性が高い
（data-inventory.md の「収録・分離品質は未評価」に該当）。

## artifact の所在

```text
artifacts/p0/2026-08-30T15-26-33/          base（distill失敗時）
artifacts/p0/2026-08-30T16-05-00-distill/  distill（triton導入後、gate_passed: true）
artifacts/p1d/2026-08-30T16-35-00/         manifest + accepted hours
artifacts/p1d-clusters/2026-08-30T17-05-00/ voiceクラスタ（t=0.92）
artifacts/s0-cer/2026-08-30T15-59-54/     S0基準線CER（評価set v2、確定値）
artifacts/s0-train/2026-08-30T23-02-51/   S0学習（修正版・3000step）
artifacts/s0-diagnose/2026-08-30T23-40-15/ 学習後のflow/stop診断
artifacts/s0-diagnose/2026-08-30T23-44-47/ 未学習baseの同一診断
artifacts/s0-cer/2026-08-30T23-40-47/     S0学習後のCER（主ゲート）
artifacts/asr-floor/2026-08-31T15-49-53/  ASRの誤り床（人間の実音声）
artifacts/p1e/2026-08-30T16-45-00-passA/   前処理パス Pass A
artifacts/s0-memory/2026-08-30T14-56-42/   VRAM/throughput実測（vast.ai RTX 3090）
```

`artifacts/` と `data/` は gitignore 済み。
**`artifacts/*/samples/` の音声はライセンス上コミットも公開もしてはならない。**

## 関連資料

- [対応計画（フェーズ定義）](08-execution-plan.md)
- [データ棚卸し](data-inventory.md)
- [リスク・意思決定](07-risks-and-decisions.md)
- [日本語継続学習の方針](02-continual-training-strategy.md)
