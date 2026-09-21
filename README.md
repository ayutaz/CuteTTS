[日本語](README.md) | [English (upstream)](README_en.md) | [中文 (upstream)](README_zh.md)

## <sup><sup><sup><img src="assets/logo.png" alt="CuteTTS-jp logo" height="72" align="middle"></sup></sup></sup> CuteTTS-jp

<a href="https://huggingface.co/OPPOer/CuteTTS"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Model-CuteTTS-yellow" alt="CuteTTS Hugging Face model"></a>
<a href="https://arxiv.org/abs/2608.08638"><img src="https://img.shields.io/badge/Paper-CuteTTS-red" alt="paper"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="Apache 2.0"></a>
<a href="https://huggingface.co/ayousanz/CuteTTS-jp"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-CuteTTS--jp-brightgreen" alt="CuteTTS-jp model"></a>

**CuteTTS の日本語継続学習。**
[OPPO-Mente-Lab/CuteTTS](https://github.com/OPPO-Mente-Lab/CuteTTS) の fork で、
公開されている学習済みモデル `OPPOer/CuteTTS`（約2億3千万パラメータ・24 kHz 出力・
逐次生成に対応）を出発点に、
**日本語へ継続学習**し、その過程と実測値をすべて記録しています。

元の CuteTTS には**音声を作る部分しか入っておらず、学習するためのコードがありません**。
CuteTTS-jp が足したのは `src/cutetts/training/`、`scripts/`、`tests/` で、
**音声を作る部分は変えていません**（抑揚を外から与えるための受け口を1つ足しただけです）。

---

## 目次

| | |
|---|---|
| [何ができるか](#何ができるか) | どこまで良くなり、何ができていないか |
| [結果](#結果) | 測った値と、その読み方 |
| [セットアップ](#セットアップ) | 1行で終わります |
| [使う](#使う) | 音声を作る / 抑揚を写す |
| [学習を再現する](#学習を再現する) | 2 GB のダウンロードから始められます |
| [評価する](#評価する) | 600 文で測り、差のばらつきも出す |
| [CuteTTS-jp で分かったこと](#cutetts-jp-で分かったこと) | 学習と測定でつまずいた点 |
| [ドキュメント](#ドキュメント) | 詳しい記録の置き場所 |
| [データとライセンス](#データとライセンス上の注意) | **公開してはいけないもの** |

---

## 何ができるか

**日本語を読み上げる精度は、人間の録音を同じ方法で測ったときの値まで
あと 2 ポイントのところまで来ました。一方、抑揚とアクセントは人間に届いていません。**

| | 状態 |
|---|---|
| 日本語の読み上げ | 読み間違いの割合 **7.12%**。学習前は 30.94%、人間の録音は 5.59% |
| 声のコピー | 初めて聞く声でも12例すべてで話者を寄せられる（元の CuteTTS の能力をそのまま保持） |
| 逐次生成 | 文全体を待たずに音声を流せる。まとめて生成した場合と**完全に同じ波形**になる |
| **抑揚のコピー** | **CuteTTS-jp が追加した機能。** 同じ台詞を読んだ人間の録音があれば、その声の高さの動きを写せる |
| 抑揚とアクセント | **人間に届いていません。** 声の高さの動きは差の約4分の1しか埋まらず、アクセントは辞書の正解率にも達していません |
| 英語 | **壊れていません**（単語誤り率 1.7%、学習前と同じ） |
| 中国語 | **壊れました**（文字誤り率 11.5% → 77.2%）。日本語に特化すると決めた結果です |

> **学習済みモデルを公開しました**: [`ayousanz/CuteTTS-jp`](https://huggingface.co/ayousanz/CuteTTS-jp)（Apache 2.0）。
> 学習に使う前処理済みのデータも公開してあるので、
> **2 GB ほどダウンロードすれば同じ学習をやり直せます**（[学習を再現する](#学習を再現する)）。

## 結果

日常会話 600 文で測った値です。

**同じ文どうしを比べて差を出し、そのばらつきも必ず併記しています**
（たまたま良かっただけの差を「改善した」と言わないため）。

| 測っているもの | 学習前 | **いまの値** | 人間 / 測定の限界 |
|---|---:|---:|---|
| 文字の間違いの割合 | 35.86% | **16.39%** | 人間の録音で 10.42% |
| **読み間違いの割合** | 30.94% | **7.58%** | **人間の録音で 5.59%** |
| 促音・撥音・長音・無声化の間違い | 46.9% | **2.43%** | — |
| 声の高さの動きが人間とどれだけ似ているか | +0.024 | **+0.091** | この方式では **+0.367 が限界** |
| アクセントの山の位置が人間と一致する割合 | 35.2% | **43.6%** | 人間どうしでも 64.5%。辞書は 44.8% |
| 話し終わらずに喋り続けてしまう割合 | 3.2% | **0.5%** | — |

**「人間の録音で 5.59%」は測定側の誤差です。** 人間が読んだ本物の音声を、
モデルの出力とまったく同じ手順（自動文字起こし → 比較）に通しても、
文字起こしが完璧ではないのでこれだけ誤りが出ます。
**この値より良くなることは原理的にありません**ので、そこからの差だけを見ます。

同じように、**声の高さの動きの +0.367 も測定側の限界**です。音声を一度圧縮して
戻すだけでこの値まで落ちるため、それ以上は測れません。

いまの値は、325.9 時間のデータで 30,000 回更新し、読み方を片仮名で
与えて生成したものです。読み間違いは **23.4 ポイント減り**、
測定側の誤差 5.59% を差し引くと、**モデル自身の誤りは 25.4 → 2.0 ポイントまで
92% 減った**ことになります。

**残った 2.0 ポイントは、データや計算を増やしても減りません。**
データを 18 倍にしても 0.59 ポイントしか減らず、この差はばらつきの範囲内でした
（95% の確率で -1.41 〜 +0.27 ポイントの間）。計算量を 4 倍にしても同じです。

数値の一覧は [RESULTS.md](docs/japanese-training/RESULTS.md) にあります。


## セットアップ

[uv](https://docs.astral.sh/uv/) を使います。**これ1行で終わります。**

```bash
uv sync --all-extras
```

仮想環境（Python 3.12）の作成、**NVIDIA GPU 向けの torch 2.5.1**、日本語の読みを求める
ライブラリ、抑揚を測るライブラリ、読み間違いを測るための自動文字起こし、
テスト用の道具まで一度に入ります。以降はすべて `uv run` を付けて実行します。

```bash
uv run python -m pytest tests/training -q     # テスト
uv run cutetts --help                         # 元の CuteTTS のコマンド
```

| | |
|---|---|
| **Python は 3.12** | torch 2.5.1 が動くのは 3.9〜3.12 です。`uv` が 3.12 を自動で用意します |
| **GPU 版の torch が入る** | 配布元を `pyproject.toml` に書いてあるので、**CPU 版で上書きされません** |
| **macOS** | NVIDIA 向けの配布が無いので、Apple GPU が使える通常版が入ります |
| **Windows** | 高速化ライブラリ（`triton-windows`）も自動で入ります。軽量版モデルに必要です |

ライブラリを追加するときも **`uv add <ライブラリ名>`** を使い、`pip` は使いません。

```bash
mkdir -p ./model
uv run hf download OPPOer/CuteTTS --local-dir ./model/CuteTTS
```


## 使う

### 日本語の音声を作る

```bash
uv run python scripts/synthesize_japanese.py \
  --model-dir <日本語モデル>/inference \
  --text "価格は千二百八十円、消費税込みです。" \
  --reference-audio assets/default_reference.wav \
  --output out.wav
```

**日本語向けのテキスト変換が最初から有効**です。モデルに渡す前に、
読み方があいまいな部分を仮名へ直します。

- **漢数字を読みに開く** — `千二百八十円` を `せんにひゃくはちじゅう円` にします。
  学習データに桁の組み合わさった漢数字は 0.32% しか含まれておらず、
  モデルは桁を合成する規則を覚えられません。**仮名なら元から読めます。**
  数字を含む 200 文で読み間違いが **11.80 ポイント減りました**。
  **学習し直す必要はありません**（`--raw-text` で無効にできます）
- **珍しい語に読みを振る** — モデルの語彙に無い漢字は、
  内部で細切れのバイト列になり読みを引けません。`中華` を `チュウカ` に
  置き換えます。専用の文集合で **4.23 ポイント減**（`--no-yomi` で無効）

学習するときに**全文を片仮名にしてアクセントの記号を付けた**モデル
（`--frontend accent` を付けて学習したもの）は、**音声を作るときも同じ変換が必要**です。
漢字のままのテキストを渡すと、学習で見たことのない形になり精度が落ちます。

元の CuteTTS の `cutetts` コマンドと Python API はそのまま使えます（[README_en.md](README_en.md)）。

### 抑揚を写す

**同じ台詞を読んだ人間の録音があれば、その声の高さの動きを写せます。**

```bash
uv run python scripts/synthesize_japanese.py \
  --model-dir <抑揚コピー対応のモデル>/inference \
  --text "..." \
  --reference-audio <声質をまねる録音> \
  --prosody-reference <同じ台詞を読んだ録音> \
  --output out.wav
```

`--reference-audio`（**声質**をまねる）と `--prosody-reference`（**抑揚**を写す）は
別のもので、両方同時に渡せます。抑揚の似かたは **+0.104 改善**します
（ばらつきを考えても確かな差です）。**抑揚の録音を渡さなくても読みは壊れません。**

> **渡す録音は必ずその文を読んだものにしてください。** 別の文の録音を渡すと、
> アクセントが **3.78 ポイント悪化します**。
> また**アクセントの山の位置は写りません** — 正しい録音を渡しても、
> 何も渡さない場合と差がありませんでした。


## 学習を再現する

音声をそのまま配るわけにいかないので、**学習に使う中間表現**（音声を圧縮した数値で、
元の音声へ戻せます）だけを
[`tts-dataset/cutetts-ja-latents`](https://huggingface.co/datasets/tts-dataset/cutetts-ja-latents)
に置いてあります。**2 GB ほどダウンロードすれば学習を始められます**
（利用には申請が必要です）。

```bash
hf download tts-dataset/cutetts-ja-latents --repo-type dataset --local-dir data/s1v2

uv run python scripts/train_continual.py \
  --manifest data/s1v2/manifests-v2/all_clustered.jsonl \
  --latent-cache data/s1v2/latents-v2 --speaker-cache data/s1v2/speaker-v2 \
  --model-dir model/CuteTTS \
  --param-dtype float32 --frontend accent \
  --steps 30000 --batch-size 4 --lr 2e-5 --seed 42 \
  --save-every 30000 --export-every-save --out checkpoints/run --device cuda
```

RTX 3090 で 30,000 回の更新に約 1.4 時間かかります。

> **`--param-dtype float32` を外さないでください**（初期値のままにしてください）。
> 公開モデルの重みは精度の低い形式（bf16）で保存されており、そのまま更新すると
> **更新量が小さすぎて丸められ、重みの 91% がまったく動きません**。
> 理由は [重みの形式を上げずに学習しない](#1-重みの形式を上げずに学習しない) に書きました。

**学習のたびに、出力される記録の `parameter_moved_ratio` を必ず見てください。**
これは「重みのうち実際に動いた割合」です。**100% 付近でなければ学習できていない**ので、
そのときの結果を他と比べても意味がありません。

## 評価する

```bash
uv run python scripts/evaluate_japanese_cer.py \
  --model-dir checkpoints/run/inference --frontend accent \
  --eval-set data/eval/eval_set_v3.json --label run --device cuda

# 2つの結果を同じ文どうしで比べ、差とそのばらつきを出す
uv run python scripts/summarize_eval_runs.py --metric cer_reading --compare v3-base run
```

**600 文の評価用データ（v3）を使ってください。** 以前使っていた 30 文の版では、
**6.9 ポイント以上の差がないと本物かどうか判断できず**、
学習回数の優劣すら逆に読み違えました。600 文なら約 1.2 ポイントの差から判断できます。

抑揚とアクセントは `scripts/evaluate_prosody.py`、
話し終わらずに喋り続けていないかは `scripts/summarize_stop_health.py` で測ります。


## CuteTTS-jp で分かったこと

### 1. 重みの形式を上げずに学習しない

**19 回の学習をすべて外した原因**です。データではなく、学習の実装の誤りでした。

公開モデルの重みの大半は **bf16** という、値を粗くしか表せない形式で保存されています。
この形式は「隣り合う表せる値の間隔」が広く、**そこへ間隔より小さい変化を足しても
四捨五入で元に戻ってしまいます**。学習率 2e-5 での 1 回ぶんの変化はまさにこの大きさで、
何回繰り返しても重みが動きませんでした。

| 部分 | パラメータ数 | 形式 | 更新が消えた割合 |
|---|---:|---|---:|
| 言語モデル本体 | 126.9M | bf16 | **91.37%** |
| 音声を読み取る部分 | 31.0M | bf16 | **84.15%** |
| 音声を作る部分 | 70.5M | fp32 | 0.00% |

**動いていたのは、たまたま精度の高い形式（fp32）だった一部分だけ**でした。
重みを fp32 に上げてから学習するよう直すと、**同じデータ・同じ回数・同じ乱数**でも
読み間違いが **8.07 ポイント**減りました。

### 2. 学習より、テキストの前処理のほうが効いた

**学習し直さずにできる工夫のほうが、学習の設定をいじるより効きました。**

| やったこと | 読み間違いの減少 | 学習し直す必要 |
|---|---:|---|
| 重みを fp32 に上げて学習する | **15.77 ポイント** | あり |
| **漢数字を読みに開く** | **11.80 ポイント** | **なし** |
| **本文を全部片仮名にして学習する** | **4.40 ポイント** | あり |
| **珍しい語に読みを振る** | **4.23 ポイント** | **なし** |
| 更新回数を 3,000 → 30,000 に増やす | 4.20 ポイント | あり |
| データを 19 倍にする | 3.82 ポイント | あり |

**学習率・1回に使う文の数・どの部分を学習するか、といった設定はどれも
効きませんでした**（すべて統計的に確かめました）。

### 3. 限界に近づくと、規模を増やしても効かなくなる

読み間違いが 13.38% だった頃は、計算量を 4 倍にすると 1.40 ポイント、
データを 19 倍にすると 3.82 ポイント減り、どちらも確かな差でした。
**ところが 7.58% まで来ると、同じことをしてもばらつきの範囲内**になります
（それぞれ 0.33 / 0.59 ポイント）。

データを増やして直っていた誤りは**漢字の読み方の取り違え**で、
**本文を片仮名にすれば消える誤りと同じもの**だったからです。
先に片仮名にしてしまうと、データを増やしても直すものが残りません。

### 4. 測り方を間違えると、間違った結論が出る

CuteTTS-jp で実際に誤った結論を出したものです。

| 失敗 | 何が起きたか | どうしたか |
|---|---|---|
| **評価する文が少なすぎた** | 30 文では 6.9 ポイント以上の差がないと判断できないのに、2〜3 ポイントの差を比べて**学習回数の優劣を逆に読んだ** | 600 文に増やし、差だけでなくばらつきも必ず出す |
| **途中で切れた音声を混ぜた** | 長さの上限に達して途中で終わった音声を、「発音を間違えた」として数えていた | 途中で切れたものを除いた値も併記する |
| **書き方の違いを誤りに数えた** | 自動文字起こしは `なにも` と話しても `何も` と書く。**誤りの半分近くは発音ではなく書き方の違い**だった | 読み方だけを比べる指標を主に見る |
| **測定の限界を見落とした** | 声の高さの動きは、音声を一度圧縮して戻すだけで +0.367 まで落ちる。これを「人間の上手さ」と取り違えていた | 同じ音声を圧縮・復元して限界値を先に測る |

**学習中に表示される損失の値は、音声の良し悪しを表しません**（3 回確かめました）。
必ず実際に音声を作り、読み間違いを測ってください。


## ドキュメント

| 文書 | 内容 |
|---|---|
| [RESULTS.md](docs/japanese-training/RESULTS.md) | **実測値の一覧。まずここ** |
| [risks-and-decisions.md](docs/japanese-training/risks-and-decisions.md) | **リスク（R-001〜R-061）と意思決定（D-001〜D-055）。この fork の中核** |
| [execution-log.md](docs/japanese-training/execution-log.md) | 各フェーズの目的・ゴール・結果 |
| [architecture.md](docs/japanese-training/architecture.md) | CuteTTS の構成と、変更で壊しやすい箇所 |
| [training-implementation.md](docs/japanese-training/training-implementation.md) | 学習の式をどう復元したか。重みの形式、音声の中間表現の作り置き |
| [data-and-frontend.md](docs/japanese-training/data-and-frontend.md) | データの設計と、日本語テキストの前処理 |
| [data-inventory.md](docs/japanese-training/data-inventory.md) | データの実態調査 |
| [references.md](docs/japanese-training/references.md) | 一次資料 |

文書では情報を **確認済み / 決定済み / 提案 / 未確定** の4つに分けて書いています。
「作った」と「良くなった」を混同しないための決まりです。

## データとライセンス上の注意

学習データは [`midralab/gol-dataset`](https://huggingface.co/datasets/midralab/gol-dataset) と
[`ayousanz/moe-speech-plus`](https://huggingface.co/datasets/ayousanz/moe-speech-plus) です。
どちらも利用には申請が必要です。

- **`artifacts/` の下にある音声をコミットしたり公開したりしてはいけません。**
  MoeSpeech の利用条件は「音声ファイルを1つでも公開すれば再配布とみなす」と
  定めています。**モデルが生成した音声も同じ扱いにしてください**
- **話者のIDを匿名化されたものとして扱わないでください。**
  このIDは表示名をハッシュしただけなので、**名前の一覧があれば元に戻せます**。
  公開したモデルには**話者IDも作品名も含めていません**
- **公開したモデルは参照音声から声をまねられます。**
  本人の同意なく実在の人の声をまねること、なりすまし、誤情報の拡散には使わないでください。
  生成した音声を公開するときは**合成音声であることを明示**してください
- 公開している中間表現は音声へ復元できるので、**音声と同じ扱い**にしています
- 学習済みモデルを公開してよいかは、まだ決めていません

## Acknowledgements

- [OPPO-Mente-Lab/CuteTTS](https://github.com/OPPO-Mente-Lab/CuteTTS) — 本体
- [Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec) — 音声を圧縮・復元する部分
- [F5-TTS](https://github.com/SWivid/F5-TTS) — 音声生成の途中経過の刻み方（Sway Sampling）
- [Qwen3](https://github.com/QwenLM/Qwen3) — 言語モデル部分（[Transformers v4.51.0](https://github.com/huggingface/transformers/tree/v4.51.0/src/transformers/models/qwen3) 経由）
- [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) — 日本語の読みとアクセント

## License

Apache License 2.0 です。元になった CuteTTS の著作権表示
（Copyright 2026 OPPO and Fudan University）はそのまま残しています。
CuteTTS-jp が追加した日本語学習のコードは Copyright 2026 ayutaz です。
他のプロジェクトから取り入れた部分は、それぞれの著作権表示とライセンスに従います
（[NOTICE](NOTICE)）。
