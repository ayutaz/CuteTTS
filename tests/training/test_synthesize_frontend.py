# Copyright 2026 ayutaz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`synthesize_japanese.py` が学習時と同じ表記を渡すかのテスト。

**2026-09-26 まで、このスクリプトには `--frontend` が無かった。** 公開した
`m4h-prosody` は `accent`（全文片仮名 + アクセント核の記号）で学習したのに、
スクリプトは J3/J2 しか掛けずに素の漢字を渡していた。

    学習時   コンニチワ、キョ'ーワイーテ'ンキデスネ。
    推論時   こんにちは。今日はいい天気ですね。

model card は `--frontend accent` と書いていたが受け取る側が持っておらず、
**手順どおりに使うと公開した読みCER 7.12% が出なかった**。M4a の実測では
表記を揃えるだけで -1.74pt、全文片仮名化で -4.40pt なので、黙って食い違うと
公開した値との差は説明のつかないものになる。

ここでは frontend の**決め方**だけを見る（合成そのものは実weightが要る）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import synthesize_japanese as sj  # noqa: E402

from cutetts.training.yomi import DEFAULT_FRONTEND  # noqa: E402


def _model_dir(tmp_path: Path, frontend: str | None) -> Path:
    payload: dict = {"model_type": "cutetts", "variant": "base"}
    if frontend is not None:
        payload["japanese_frontend"] = frontend
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def _resolve(model_dir: Path, *extra: str) -> str:
    args = sj.build_parser().parse_args(
        ["--model-dir", str(model_dir), "--text", "今日はいい天気ですね。", *extra])
    return sj.resolve_frontend(args)


def test_checkpointの記録に従う(tmp_path):
    """**既定でこれが効くのが肝。** 利用者がフラグを知らなくても揃う。"""
    assert _resolve(_model_dir(tmp_path, "accent")) == "accent"


def test_記録が別の表記でもその通りに従う(tmp_path):
    """`accent` を既定にしたからといって上書きしない。"""
    assert _resolve(_model_dir(tmp_path, "none")) == "none"
    assert _resolve(_model_dir(tmp_path, "kana_full")) == "kana_full"


def test_明示した指定が記録より優先する(tmp_path):
    assert _resolve(_model_dir(tmp_path, "accent"), "--frontend", "none") == "none"


def test_記録が無ければ既定を当てる(tmp_path, capsys):
    """**当てたことを必ず印字する。** 黙って当てると不具合が見えない。"""
    assert _resolve(_model_dir(tmp_path, None)) == DEFAULT_FRONTEND
    assert "japanese_frontend" in capsys.readouterr().out


def test_raw_textは前処理を全部止める(tmp_path):
    assert _resolve(_model_dir(tmp_path, "accent"), "--raw-text") == "none"


def test_raw_textとfrontendの同時指定を弾く(tmp_path):
    """どちらを尊重しても片方の意図を裏切るので、黙って選ばない。"""
    with pytest.raises(SystemExit):
        _resolve(_model_dir(tmp_path, "accent"), "--raw-text", "--frontend", "accent")


def test_未知のfrontendはargparseが弾く(tmp_path):
    with pytest.raises(SystemExit):
        _resolve(_model_dir(tmp_path, "accent"), "--frontend", "accent_v2")


# --- 書き込む側（export）------------------------------------------------------


def test_export_for_inferenceが学習時の表記をconfigに残す(tmp_path):
    """**記録する側が無いと読む側は永遠に None を見る。**

    `runtime.py` は未知の top-level キーを読まないので、付けても推論は壊れない。
    """
    import torch

    from cutetts.training.checkpointing import export_for_inference

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "cutetts", "variant": "base"}), encoding="utf-8")

    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

    out = export_for_inference(tmp_path / "out", model=_Stub(),
                               source_model_dir=source, frontend="accent")
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["japanese_frontend"] == "accent"
    # 元のキーを落としていない
    assert config["model_type"] == "cutetts"
    assert config["variant"] == "base"


def test_export_for_inferenceはfrontend未指定なら何も足さない(tmp_path):
    """**過去の checkpoint と同じ config になる。** 嘘の表記を書かない。"""
    import torch

    from cutetts.training.checkpointing import export_for_inference

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "cutetts", "variant": "base"}), encoding="utf-8")

    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

    out = export_for_inference(tmp_path / "out", model=_Stub(), source_model_dir=source)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert "japanese_frontend" not in config
