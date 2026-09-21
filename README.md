[日本語](README.md) | [English (upstream)](README_en.md) | [中文 (upstream)](README_zh.md)

## <sup><sup><sup><img src="assets/logo.png" alt="CuteTTS-jp logo" height="72" align="middle"></sup></sup></sup> CuteTTS-jp

<a href="https://huggingface.co/OPPOer/CuteTTS"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Model-CuteTTS-yellow" alt="CuteTTS Hugging Face model"></a>
<a href="https://arxiv.org/abs/2608.08638"><img src="https://img.shields.io/badge/Paper-CuteTTS-red" alt="paper"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="Apache 2.0"></a>

**CuteTTS の日本語継続学習。**
[OPPO-Mente-Lab/CuteTTS](https://github.com/OPPO-Mente-Lab/CuteTTS) の fork で、
公開base checkpoint `OPPOer/CuteTTS`（約230M・24 kHz・streaming）を起点に
**日本語へ継続学習**し、その過程と実測値をすべて記録しています。

upstream は **推論専用**で、学習コード（trainer / dataset / loss / packing）を含みません。
CuteTTS-jp が追加したのは `src/cutetts/training/`、`scripts/`、`tests/` で、
**既存の推論pathは変更していません**（F0条件づけの受け口を追加しただけです）。

---

## 目次

| | |
|---|---|
| [何ができるか](#何ができるか) | 到達点と、できていないこと |
| [結果](#結果) | 3指標の実測値 |
| [セットアップ](#セットアップ) | Python 3.12 固定 |
| [使う](#使う) | 合成 / 韻律転写 |
| [学習を再現する](#学習を再現する) | 約2 GBの取得から始められます |
| [評価する](#評価する) | 600文の評価setと信頼区間 |
| [CuteTTS-jp で分かったこと](#cutetts-jp-で分かったこと) | 学習と測定の落とし穴 |
| [ドキュメント](#ドキュメント) | 詳細の行き先 |
| [データとライセンス](#データとライセンス上の注意) | **公開してはいけないもの** |

---

## 何ができるか

**日本語の読みは人間の床の手前 2pt まで来ています。抑揚とアクセントは届いていません。**

| | 状態 |
|---|---|
| 日本語の読み | **読みCER 7.58%**（人間の実音声が 5.59%）。base は 30.94% |
| zero-shot voice cloning | 12/12 で追随（upstream の能力を保持） |
| streaming | `voice_clone` は offline と**ビット一致** |
| **韻律転写** | **CuteTTS-jp の追加**。同じ台詞の人間の読みから F0 の輪郭を写せる |
| 抑揚・アクセント | **人間に届いていない**。輪郭は隔たりの約1/4、アクセント核は辞書にも未達 |
| 英語 | **無傷**（WER 1.7%、base と同値） |
| 中国語 | **壊滅**（CER 11.5% → 77.2%）。日本語特化と決めた結果です |

> **学習済み checkpoint は公開していません。** 公開範囲が未確定のためです。
> 前処理済みデータは公開してあるので、**約2 GBの取得だけで学習を再現できます**
> （[学習を再現する](#学習を再現する)）。

## 結果

評価set v3（in_domain 600文）。**すべて対応のある検定つきです。**

| 指標 | 公開base | **現行** | 人間 / 上限 |
|---|---:|---:|---|
| 素CER | 35.86% | **16.39%** | 10.42% |
| **読みCER**（表記差を落とす） | 30.94% | **7.58%** | **床 5.59%** |
| phonetic（促音・撥音・長音・無声化） | 46.9% | **2.43%** | — |
| 輪郭の相関（240文） | +0.024 | **+0.091** | 上限 +0.367（codec） |
| アクセント核（対人間） | 35.2% | **43.6%** | 天井 64.5% / 辞書 44.8% |
| 喋り続け | 3.2% | **0.5%** | — |

現行 = 325.9時間 / 30,000 step / `--frontend accent`。
読みCER は **-23.4pt** で、ASR の床 5.59% を引くと
**TTS由来の誤りは 25.4pt → 2.0pt（92%削減）** です。

**残りの 2.0pt は規模では動きません。** データ量を18倍にしても -0.59pt
（95%CI [-1.41, +0.27]、有意差なし）、計算量4倍でも -0.33pt（有意差なし）でした。

数値の一覧は [RESULTS.md](docs/japanese-training/RESULTS.md) にあります。

## セットアップ

**Python 3.12 固定**です（torch 2.5.1 の対応は 3.9〜3.12。既定が 3.13 以降の環境では動きません）。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121      # CUDA 12.1
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python -e ".[ja]"       # 日本語の読み付与
uv pip install --python .venv/bin/python -e ".[eval]"     # CER評価（accelerate が要る）
uv pip install --python .venv/bin/python -e ".[dev]"      # テスト
```

Windows では `.venv/Scripts/python.exe` を使い、`triton-windows` も入れてください
（未導入だと distill 版が動きません）。`uv` が無ければ `py -3.12 -m venv .venv` でも作れます。

```bash
mkdir -p ./model
hf download OPPOer/CuteTTS --local-dir ./model/CuteTTS
```

## 使う

### 日本語で合成する

```bash
python scripts/synthesize_japanese.py \
  --model-dir <日本語checkpoint>/inference \
  --text "価格は千二百八十円、消費税込みです。" \
  --reference-audio assets/default_reference.wav \
  --output out.wav
```

**日本語のtext前処理が既定で有効**です。

- **漢数字の読み展開** — `千二百八十円` → `せんにひゃくはちじゅう円`。
  学習コーパスに複合漢数字は 0.32% しかなく桁の合成規則を学べませんが、
  **仮名ならモデルが既に読めます**。数詞の評価set 200文で **-11.80pt**。
  **再学習を要しません**（`--raw-text` で無効）
- **語の読み付与** — tokenizer の byte-fallback を含む語を読みへ置き換えます。
  `中華` → `チュウカ`。専用setで **-4.23pt**（`--no-yomi` で無効）

学習時に `--frontend accent`（全文片仮名 + アクセント核の記号）を使った checkpoint は、
**推論でも同じ frontend が要ります**。素の漢字テキストを渡すと分布外になります。

upstream の `cutetts` CLI と Python API はそのまま使えます（[README_en.md](README_en.md)）。

### 韻律を写す

**同じ台詞を読んだ人間の音声から F0 の輪郭を写せます。**

```bash
python scripts/synthesize_japanese.py \
  --model-dir <韻律転写対応checkpoint>/inference \
  --text "..." \
  --reference-audio <声質の参照> \
  --prosody-reference <同じ台詞を読んだ音声> \
  --output out.wav
```

`--reference-audio`（声質）と `--prosody-reference`（韻律）は別物で、両方渡せます。
輪郭の追随は **+0.104**（有意）。条件を渡さなくても読みは壊れません。

> **参照は必ずその文の読みにしてください。** 別の文の F0 を渡すと
> アクセントが **-3.78pt 壊れます**（有意）。
> なお**アクセント核は写りません** — 正しい F0 を渡しても条件なしと差がありません。

## 学習を再現する

前処理済みの latent は
[`tts-dataset/cutetts-ja-latents`](https://huggingface.co/datasets/tts-dataset/cutetts-ja-latents)
（public / gated: manual）にあります。**約2 GBの取得だけで始められます。**

```bash
hf download tts-dataset/cutetts-ja-latents --repo-type dataset --local-dir data/s1v2

python scripts/train_continual.py \
  --manifest data/s1v2/manifests-v2/all_clustered.jsonl \
  --latent-cache data/s1v2/latents-v2 --speaker-cache data/s1v2/speaker-v2 \
  --model-dir model/CuteTTS \
  --param-dtype float32 --frontend accent \
  --steps 30000 --batch-size 4 --lr 2e-5 --seed 42 \
  --save-every 30000 --export-every-save --out checkpoints/run --device cuda
```

RTX 3090 で 30,000 step が約 1.4 時間です。

> **`--param-dtype float32` を外さないでください**（既定）。
> 公開checkpointは bf16 で、`AdamW` が直接更新すると **backbone の 91% が1stepも動きません**。
> 詳細は [bf16 パラメータを optimizer で直接更新しない](#1-bf16-パラメータを-optimizer-で直接更新しない)。

**毎runの metrics にある `parameter_moved_ratio` を必ず確認してください。**
100% 付近でなければ、そのrunの比較は無意味です。

## 評価する

```bash
python scripts/evaluate_japanese_cer.py \
  --model-dir checkpoints/run/inference --frontend accent \
  --eval-set data/eval/eval_set_v3.json --label run --device cuda

# **点推定の順位ではなく信頼区間で判断する**
python scripts/summarize_eval_runs.py --metric cer_reading --compare v3-base run
```

**評価set v3（600文）を使ってください。** 旧v2（30文）は検出できる最小差が **6.9pt** で、
step数の順位すら取り違えました。v3 の検出限界は約 1.2pt です。

抑揚とアクセントは `scripts/evaluate_prosody.py`、停止の健全性は
`scripts/summarize_stop_health.py` で測ります。

## CuteTTS-jp で分かったこと

### 1. bf16 パラメータを optimizer で直接更新しない

**19回の学習をすべて外した原因**です。データではなく学習実装のバグでした。

公開checkpointは `qwen_backbone` / `locenc` が bf16 です。`AdamW` がそれを直接更新すると、
lr=2e-5 の更新量が bf16 の丸め幅（相対 2^-8）を下回り、round-to-nearest-even が
**毎step更新を捨てます**。同じ向きに積み上がらないので、何step回しても動きません。

| module | params | dtype | 更新が丸めで消える割合 |
|---|---:|---|---:|
| `qwen_backbone` | 126.9M | bf16 | **91.37%** |
| `locenc` | 31.0M | bf16 | **84.15%** |
| `head`（DiT） | 70.5M | fp32 | 0.00% |

**実際に学習されていたのは fp32 の DiT head だけでした。** 修正後、同一データ・
同一step・同一seed の対照実験で **-8.07pt**（有意）。詳細は
[R-020](docs/japanese-training/risks-and-decisions.md)。

### 2. text frontend が最大の梃子だった

**学習を要しない手段が、学習を要する手段より効きました。**

| 手段 | 効果 | 再学習 |
|---|---:|---|
| dtype 修正（bf16→fp32） | **-15.77pt** | 要 |
| **漢数字の読み展開** | **-11.80pt** | **不要** |
| **全文片仮名化** | **-4.40pt** | 要 |
| **語の読み付与** | **-4.23pt** | **不要** |
| step数 3,000 → 30,000 | -4.20pt | 要 |
| データ量 19倍 | -3.82pt | 要 |

**学習率・batch size・学習対象・`flow_copies`・`condition_dropout` はいずれも
梃子ではありませんでした**（すべて検定済み）。

### 3. 床に近づくと規模は効かなくなる

読みCER 13.38% の水準では計算量4倍が -1.40pt（有意）、データ量19倍が -3.82pt（有意）でした。
**ところが 7.58% の水準では、どちらも有意差が出ません**（-0.33pt / -0.59pt）。

データ量が治していた誤りは**漢字の読みの引き当て**で、
**全文片仮名化が消す誤りと同じもの**だったためです。

### 4. 測定でやってはいけないこと

CuteTTS-jp で実際に誤った結論を出したものです。

| 欠陥 | 何が起きたか | 対処 |
|---|---|---|
| **n=30 の検出力** | 検出限界 6.9pt の評価setで 2〜3pt の差を比較し、**step数の順位を逆に読んだ** | v3（600文）を使い、信頼区間を出す |
| **打ち切り生成の混入** | `max_decode_length` 張り付き（停止の失敗）を発音誤りとして数えていた | `mean_excluding_truncated` を見る |
| **表記の違い** | ASRは `なにも` と話しても `何も` と書く。**床の半分近くは発音でなく表記だった** | `cer_reading` を主に読む |
| **指標の上限** | 輪郭の指標には codec 由来の上限（+0.367）がある。「人間の天井」と呼んでいたものは**上限そのもの**だった | 同じ音声を VAE で往復させて上限を測る |

**flow loss は品質の指標になりません**（3回実証しました）。CER を測ってください。

## ドキュメント

| 文書 | 内容 |
|---|---|
| [RESULTS.md](docs/japanese-training/RESULTS.md) | **実測値の一覧。まずここ** |
| [risks-and-decisions.md](docs/japanese-training/risks-and-decisions.md) | **リスク（R-001〜R-061）と意思決定（D-001〜D-055）。この fork の中核** |
| [execution-log.md](docs/japanese-training/execution-log.md) | 各フェーズの目的・ゴール・結果 |
| [architecture.md](docs/japanese-training/architecture.md) | CuteTTS の構成と、変更で壊しやすい箇所 |
| [training-implementation.md](docs/japanese-training/training-implementation.md) | 学習式の復元、fp32 master weights、latent cache |
| [data-and-frontend.md](docs/japanese-training/data-and-frontend.md) | データ設計と日本語 frontend |
| [data-inventory.md](docs/japanese-training/data-inventory.md) | データの実態調査 |
| [references.md](docs/japanese-training/references.md) | 一次資料 |

文書は情報を **確認済み / 決定済み / 提案 / 未確定** の4状態で区別します。
「実装した」と「日本語学習が成功した」を混同しないための規約です。

## データとライセンス上の注意

学習データは [`midralab/gol-dataset`](https://huggingface.co/datasets/midralab/gol-dataset) と
[`ayousanz/moe-speech-plus`](https://huggingface.co/datasets/ayousanz/moe-speech-plus)（どちらも gated）です。

- **`artifacts/` 配下の音声をコミット・公開してはいけません。**
  MoeSpeech LICENSE は「音声ファイルを1つであっても公開することは再配布とみなす」と
  規定しています。**生成物も同じ扱いにしてください**
- **話者IDを匿名化として扱わないでください。** gol のIDは `SHA-256(表示名)[:32]` で
  辞書攻撃が可能です
- 公開している latent は VAE で音声へ復元できるため、**音声と同じ扱い**にしています
- 学習済みモデルの公開範囲は未確定です

## Acknowledgements

- [OPPO-Mente-Lab/CuteTTS](https://github.com/OPPO-Mente-Lab/CuteTTS) — 本体
- [Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec) — Audio VAE の一部
- [F5-TTS](https://github.com/SWivid/F5-TTS) — Sway Sampling
- [Qwen3](https://github.com/QwenLM/Qwen3) — backbone（[Transformers v4.51.0](https://github.com/huggingface/transformers/tree/v4.51.0/src/transformers/models/qwen3) 経由）
- [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) — 日本語の読みとアクセント

## License

Apache License 2.0。upstream の著作権表示（Copyright 2026 OPPO and Fudan University）は保持しています。
CuteTTS-jp が追加した日本語継続学習のコードは Copyright 2026 ayutaz です。
third-party の部分はそれぞれの著作権表示とライセンスに従います（[NOTICE](NOTICE)）。
