---
name: cutetts-ja-pipeline
description: Use when running, resuming, or debugging any CuteTTS Japanese continual-training phase (P0 baseline, P1b tokenizer, P1c VAE, P1d manifest, P1e latent cache) in this repository — covers the venv, GPU rules, exact commands, artifact layout, and the environment traps that make these scripts fail.
---

# CuteTTS 日本語学習パイプラインの実行

このリポジトリの P0/P1 スクリプトを正しく走らせるためのリファレンス。
**結果の数値は [`docs/japanese-training/RESULTS.md`](../../../docs/japanese-training/RESULTS.md)**、
フェーズ定義は `docs/japanese-training/08-execution-plan.md`。

## 実行前に必ず守る2点

1. **Python は必ず `.venv/Scripts/python.exe`。**
   システム既定は3.14で torch 2.5.1 が動かない（対応は3.9〜3.12）。
2. **GPUを使う処理は実行前にユーザーへ確認する。**
   バックグラウンド実行でも同じ。並列起動しない（1本ずつ直列）。

| スクリプト | GPU | 目安 |
|---|---|---|
| `analyze_japanese_tokenizer.py` | 不要 | 数分（CPU） |
| `prepare_japanese_manifest.py` | 不要 | 7.4M行で数分（CPU） |
| `reproduce_baseline.py` | **要** | 数分 / peak 2.1 GB |
| `evaluate_japanese_vae.py` | **要** | 話者数次第 / peak 1.5 GB |
| `cache_audio_latents.py` | **要** | 44.5× realtime / peak 2.5 GB |
| `build_voice_clusters.py` | 不要 | cache読むだけ（CPU） |

## コマンド

```bash
# P0: 推論ベースライン（gate_passed を確認する）
.venv/Scripts/python.exe scripts/reproduce_baseline.py --sampler-compile-mode auto
.venv/Scripts/python.exe scripts/reproduce_baseline.py --checkpoint CuteTTS-distill   # 個別

# P1b: Tokenizer coverage（CPU）
.venv/Scripts/python.exe scripts/analyze_japanese_tokenizer.py \
  --config configs/japanese/tokenizer-coverage.yaml

# P1c: VAE 日本語再構成
.venv/Scripts/python.exe scripts/evaluate_japanese_vae.py \
  --config configs/japanese/vae-reconstruction.yaml

# P1d: manifest + split + pairing（--skip-full-accounting で7.4M行集計を省略）
.venv/Scripts/python.exe scripts/prepare_japanese_manifest.py

# P1e: latent + speaker embedding を1パス生成（--limit でパイロット）
.venv/Scripts/python.exe scripts/cache_audio_latents.py --limit 300

# P1d: voiceクラスタ（既定0.70ではなく 0.92 を使う）
.venv/Scripts/python.exe scripts/build_voice_clusters.py --threshold 0.92

# テスト
.venv/Scripts/python.exe -m pytest tests/training -q -p no:warnings
```

`--checkpoint` は**ディレクトリ名**（`CuteTTS-distill`）であってパスではない。

## 環境の罠

| 症状 | 原因と対処 |
|---|---|
| `BackendCompilerFailed: Cannot find a working triton installation` | WindowsのPyTorchにtritonが同梱されない。`uv pip install --python .venv/Scripts/python.exe triton-windows`。未対処だと**distillが全ケース失敗する** |
| `UnicodeEncodeError: 'cp932'` | stdoutがcp932。`PYTHONIOENCODING=utf-8` を付けるか、ファイルへUTF-8明示で書く |
| `No module named pytest` | `-e .` はdev extrasを入れない。`uv pip install --python .venv/Scripts/python.exe pytest pyyaml` |
| `No usable checkpoint under ...` | `--checkpoint` にパスを渡した。ディレクトリ名を渡す |
| manifestの件数が倍 | moe zipの `.bak.json` を除外していない（本体と同数ある） |
| クラスタが1つに潰れる | 閾値0.70はこの埋め込み空間で破綻する。0.92を使う |

## データの前提（推測で埋めない）

- **gol の sample_rate は 44.1/48 kHz 混在。** dataset単位で仮定せず実ファイルから読む
- **speaker ID は声の識別子ではない。** gol は `SHA-256(表示名)[:32]`、moe は `uuid4()`。
  split は voice クラスタ単位で行う（実データで gol×moe 跨ぎの同一声 cos 0.93 を検出済み）
- 総称ラベル話者（`？？？`『女の子』等91件）は複数の声が1 IDに混在する。除外する

## artifact

```text
artifacts/<phase>/<timestamp>/
├─ run.json / env.json / inputs.json / metrics.json
└─ samples/     # 音声
```

`artifacts/` と `data/` は gitignore 済み。
**`samples/` の音声はライセンス上コミットも公開もしてはならない**
（MoeSpeech LICENSE: 音声ファイルを1つでも公開すれば再配布とみなす）。
