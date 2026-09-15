[日本語](README.md) | [EN](README_en.md) | [中文](README_zh.md)

## <sup><sup><sup><img src="assets/logo.png" alt="CuteTTS logo" height="72" align="middle"></sup></sup></sup> CuteTTS 日本語継続学習 fork

<a href="https://huggingface.co/OPPOer/CuteTTS"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Model-CuteTTS-yellow" alt="CuteTTS Hugging Face model"></a>
<a href="https://arxiv.org/abs/2608.08638"><img src="https://img.shields.io/badge/Paper-CuteTTS-red" alt="paper"></a>

**[OPPO-Mente-Lab/CuteTTS](https://github.com/OPPO-Mente-Lab/CuteTTS) の fork です。**
公開base checkpoint `OPPOer/CuteTTS`（約230M・24 kHz・streaming）を起点に、
**日本語の継続学習**を進めています。作業ブランチは `feat/japanese-training`。

upstream は **推論専用**で、学習コード（trainer / dataset / loss / packing）を含みません。
このforkが追加したのは `src/cutetts/training/`、`scripts/`、`tests/` です。
**既存の推論pathは変更していません。**

> upstream本体の説明（アーキテクチャ、多言語の性能表、Web demo）は
> [README_en.md](README_en.md) にそのまま残してあります。

<br>

## 現在の到達点

評価set v3（in_domain 600文）での実測です。

| checkpoint | mean | median |
|---|---:|---:|
| base（未学習） | 35.86% | 31.91% |
| **fp32 30,000 step** | **20.10%** | **16.67%** |
**抑揚とアクセントも測れるようになった**（M1）。学習は3指標すべてを
有意に改善しているが、**どれも人間には届いていない**。

| 指標 | base | **現行** | 人間 |
|---|---:|---:|---:|
| 読みCER（600文） | 30.94% | **13.38%** | 5.59% |
| 輪郭の相関（240文） | +0.024 | **+0.122** | 床 -0.009 |
| アクセント核（対人間） | 35.2% | **43.6%** | 辞書が44.8% |

| ASR床（人間の実音声） | 10.40% | — |

base比 **-14.09 〜 -15.77pt**（95%CI [-17.26, -14.35]、600文中457文で改善）。
TTS由来の誤りは **25.5pt → 9.7pt（62%削減）** です。

**盲検A/Bで 15/18（83%、p=0.0038）** と、CERの改善は聴いて分かる差になっています。
会話文では 13/14（93%、p=0.0009）で CER と聴取が一致します。

| 項目 | 状態 |
|---|---|
| 日本語のCER | **base比 -15.77pt**（有意） |
| 音韻（促音・撥音・長音・無声化） | 46.9% → **22.2%**（有意） |
| 数詞の読み | **J2で -11.80pt**（有意。**再学習不要**） |
| 話者追随（zero-shot） | 12/12、margin +0.247 |
| streaming | `voice_clone` は offline と**ビット一致** |
| 英語 | **無傷**（WER 1.7%、baseと同値） |
| 中国語 | **壊滅**（CER 11.5% → 77.2%）。日本語特化と決定（D-032） |
| 自然性・アクセント | **未測定**。日本語向けの信頼できる自動指標が無い |

数値の一覧は [`docs/japanese-training/RESULTS.md`](docs/japanese-training/RESULTS.md)。

<br>

## 何が分かったか（S1の失敗の原因）

**S1（100〜500時間）は19回の学習をすべて外しました。原因はデータではなく
学習実装のバグでした。**

公開checkpointは `qwen_backbone` / `locenc` が bf16 です。`AdamW` がそれを直接更新すると、
lr=2e-5 の更新量が bf16 の丸め幅（相対 2^-8）を下回り、round-to-nearest-even が
**毎step更新を捨てます**。同じ向きに積み上がらないので、何step回しても動きません。

| module | params | dtype | 更新が丸めで消える割合 |
|---|---:|---|---:|
| `qwen_backbone` | 126.9M | bf16 | **91.37%** |
| `locenc` | 31.0M | bf16 | **84.15%** |
| `head`（DiT） | 70.5M | fp32 | 0.00% |

**実際に学習されていたのは fp32 の DiT head だけでした。** データを 7時間 →
305時間に増やしても反応しないのは当然で、19回の試行はすべてこの条件下の観測です。

修正（`--param-dtype float32` が既定）後、同一データ・同一step・同一seedの
対照実験で **-8.07pt**（95%CI [-13.22, -3.02]、有意）。
詳細は [R-020](docs/japanese-training/07-risks-and-decisions.md)。

**学習するときは毎runの metrics にある `parameter_moved_ratio` を必ず確認してください。**
100%付近でなければ、そのrunの比較は無意味です。

<br>

## 使う

### セットアップ

**Python 3.12 固定**です（torch 2.5.1 の対応は 3.9〜3.12。既定が3.13以降の環境では動きません）。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121     # CUDA 12.1
uv pip install --python .venv/Scripts/python.exe -e .
uv pip install --python .venv/Scripts/python.exe -e ".[ja]"    # 読み付与（J2/J3）
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"   # テスト
```

`uv` が無ければ `py -3.12 -m venv .venv` でも作れます。
Windows では `triton-windows` も入れてください（未導入だと distill が全滅します）。

```bash
mkdir -p ./model
hf download OPPOer/CuteTTS --local-dir ./model/CuteTTS
```

### 日本語で合成する

```bash
.venv/Scripts/python.exe scripts/synthesize_japanese.py \
  --model-dir checkpoints/s1v2-fp32-30000 \
  --text "価格は千二百八十円、消費税込みです。" \
  --reference-audio assets/default_reference.wav \
  --output out.wav
```

**漢数字の読み展開（J2）が既定で有効**です。`千二百八十円` を
`せんにひゃくはちじゅう円` に展開してから渡します。学習コーパスに複合漢数字は
0.32% しかなく桁の合成規則を学べませんが、**仮名ならモデルが既に読めます**。
数詞専用の評価set 200文で **-11.80pt**、通常の会話文600文では +0.02pt で副作用なしです。
**再学習を要しません。** 無効にするには `--raw-text`。

upstream の `cutetts` CLI と Python API はそのまま使えます（[README_en.md](README_en.md)）。

### 学習する

```bash
# **--param-dtype float32 が必須**（既定）
.venv/Scripts/python.exe scripts/train_continual.py \
  --steps 30000 --batch-size 4 --lr 2e-5 --warmup 100 \
  --param-dtype float32 --group-key voice_cluster_id \
  --save-every 10000 --export-every-save --out checkpoints/run --device cuda
```

**評価は v3（600文）を使ってください。** 旧v2（30文）は検出できる最小差が **6.9pt** で、
step数の順位すら取り違えました。

```bash
.venv/Scripts/python.exe scripts/evaluate_japanese_cer.py \
  --model-dir checkpoints/run/inference \
  --eval-set data/eval/eval_set_v3.json --label v3-trained --device cuda

# **点推定の順位ではなく信頼区間で判断する**
.venv/Scripts/python.exe scripts/summarize_eval_runs.py --compare v3-base v3-trained
```

手順の全体と落とし穴は `.claude/skills/cutetts-ja-pipeline/SKILL.md` にまとめてあります。

<br>

## 測定でやってはいけないこと

このforkで実際に誤った結論を出した3件です。

| 欠陥 | 何が起きたか | 対処 |
|---|---|---|
| **n=30 の検出力** | 検出限界6.9ptの評価setで2〜3ptの差を比較し、**step数の順位を逆に読んだ** | v3（600文）を使う。`summarize_eval_runs --compare` で信頼区間を出す |
| **打ち切り生成の混入** | `max_decode_length` 張り付き（停止の失敗）を発音誤りとして数えていた。S0系は0件、S1系は1〜7件で、除外すると差がほぼ消えた | `mean_excluding_truncated` を見る |
| **数字表記の不一致** | ASRは `1280円` と書くが参照は `千二百八十円`。**正しく読めるほど素のCERは悪化する** | `cer_numeric`（数字正規化CER）を見る |

**flow loss は品質の指標になりません**（3回実証しました）。CERを測ってください。

<br>

## これから

**データ規模ではなく律速要因で段階を切り直しました。**
19倍のデータで -1.90pt に対し、frontend は学習なしで -11.80pt を出したためです。

| ID | 内容 | 状態 |
|---|---|---|
| ~~J3~~ | 読み付与 frontend（`pyopenjtalk-plus`） | **完了**。読みCER -4.23pt（再学習不要） |
| ~~J4~~ | Tokenizer 互換拡張 | **見送り**。J3が内容語のfallbackをほぼ吸収した |
| ~~M1~~ | 抑揚・アクセントの測定 | **完了**。学習が両方を有意に改善していると分かった |
| ~~T1~~ | 学習率の探索 | **完了。梃子ではなかった**（2e-5がほぼ底） |
| **T2** | batch size / 学習対象 | **次に着手**。head凍結が未検証 |
| S2 | 1,000時間 | **保留**。T2の後に再判断 |

フェーズ定義は [`docs/japanese-training/08-execution-plan.md`](docs/japanese-training/08-execution-plan.md)。

<br>

## ドキュメント

| 文書 | 内容 |
|---|---|
| [RESULTS.md](docs/japanese-training/RESULTS.md) | **実測値の一覧**。まずここ |
| [08-execution-plan.md](docs/japanese-training/08-execution-plan.md) | フェーズ定義とゴール |
| [07-risks-and-decisions.md](docs/japanese-training/07-risks-and-decisions.md) | リスク（R-001〜R-027）と意思決定（D-001〜D-035） |
| [README.md](docs/japanese-training/README.md) | プロジェクトの概要 |
| [01〜06章](docs/japanese-training/) | アーキテクチャ、戦略、データ、学習実装、実験計画、評価計画 |
| [S0-GATE.md](docs/japanese-training/S0-GATE.md) | S0時点の記録（数値は凍結） |

文書は情報を **確認済み / 決定済み / 提案 / 未確定** の4状態で区別します。
「実装した」と「日本語学習が成功した」を混同しないための規約です。

<br>

## データとライセンス上の注意

学習データは [`midralab/gol-dataset`](https://huggingface.co/datasets/midralab/gol-dataset) と
[`ayousanz/moe-speech-plus`](https://huggingface.co/datasets/ayousanz/moe-speech-plus)（どちらも gated）。
前処理済みの latent は
[`tts-dataset/cutetts-ja-latents`](https://huggingface.co/datasets/tts-dataset/cutetts-ja-latents)（gated: manual）にあり、
約2 GBの取得だけで学習を再開できます。**音声そのものは置いていません。**

- **`artifacts/` 配下の音声をコミット・公開してはいけません。**
  MoeSpeech LICENSE は「音声ファイルを1つであっても公開することは再配布とみなす」と規定しています。
  生成物も同じ扱いにしてください
- **話者IDを匿名化として扱わないでください。** gol のIDは `SHA-256(表示名)[:32]` で辞書攻撃が可能です
- モデルの公開範囲は未確定です（R-009）

<br>

## Acknowledgements

- [OPPO-Mente-Lab/CuteTTS](https://github.com/OPPO-Mente-Lab/CuteTTS) — 本体
- [Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec) — Audio VAE の一部
- [F5-TTS](https://github.com/SWivid/F5-TTS) — Sway Sampling
- [Qwen3](https://github.com/QwenLM/Qwen3) — backbone（[Transformers v4.51.0](https://github.com/huggingface/transformers/tree/v4.51.0/src/transformers/models/qwen3) 経由）
- [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) — 日本語の読み付与

## License

Apache License 2.0。upstream の著作権表示（Copyright 2026 OPPO and Fudan University）は保持しています。
このforkが追加した日本語継続学習のコードは Copyright 2026 ayutaz です。
third-party の部分はそれぞれの著作権表示とライセンスに従います（[NOTICE](NOTICE)）。
