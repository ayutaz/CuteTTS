# P0 / P1 実測結果まとめ

最終更新: 2026-08-30

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
