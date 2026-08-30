# Copyright 2026 OPPO and Fudan University
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

"""checkpoint の save / resume。

**2種類を明確に分ける。**

* `save_training_state` … 学習を再開するための一切（model / optimizer /
  scheduler / step / RNG）。resume 専用で、推論では読まない。
* `export_for_inference` … 推論の `CuteTTS.from_pretrained` がそのまま読める
  ディレクトリ（`config.json` + `weights/tts/model.safetensors` + tokenizer など）。

resume の再現性のために **RNG state も保存する**。これが無いと、
再開後に flow の noise と condition dropout の系列が変わってしまい、
「save/resume前後で次stepが一致する」というP2のゴールを満たせない。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from cutetts.modeling.model import CuteTTSModel

TRAINING_STATE_NAME = "training_state.pt"
METADATA_NAME = "checkpoint.json"


@dataclass
class TrainingState:
    """resume に必要なスカラー状態。"""

    step: int = 0
    epoch: int = 0
    samples_seen: int = 0
    best_metric: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _rng_state(generator: torch.Generator | None) -> dict[str, Any]:
    state = {
        "torch": torch.get_rng_state(),
        "generator": None if generator is None else generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any], generator: torch.Generator | None) -> None:
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if generator is not None and state.get("generator") is not None:
        generator.set_state(state["generator"].cpu().to(torch.uint8))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["cuda"]])


def save_training_state(
    directory: str | Path,
    *,
    model: CuteTTSModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    state: TrainingState | None = None,
    generator: torch.Generator | None = None,
) -> Path:
    """resume 用の状態を1ファイルへ保存する。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = state or TrainingState()
    payload = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "state": asdict(state),
        "rng": _rng_state(generator),
    }
    path = directory / TRAINING_STATE_NAME
    torch.save(payload, path)
    (directory / METADATA_NAME).write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_training_state(
    directory: str | Path,
    *,
    model: CuteTTSModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    generator: torch.Generator | None = None,
    map_location: str | torch.device = "cpu",
) -> TrainingState:
    """`save_training_state` が書いた状態を復元する。"""
    directory = Path(directory)
    path = directory / TRAINING_STATE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no training state at {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)

    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"model state did not load strictly: {missing} / {unexpected}")
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if payload.get("rng"):
        _restore_rng(payload["rng"], generator)
    return TrainingState(**payload["state"])


def export_for_inference(
    directory: str | Path,
    *,
    model: CuteTTSModel,
    source_model_dir: str | Path,
) -> Path:
    """推論の `CuteTTS.from_pretrained` が読めるディレクトリを書き出す。

    `config.json` / `tokenizer/` / `weights/audio_vae/` / `weights/speaker_encoder/`
    は ``source_model_dir`` からコピーし、`weights/tts/model.safetensors` だけを
    学習後の重みで置き換える。VAE と Speaker Encoder は freeze しているため
    そのままで整合する（D-003 / D-004）。
    """
    from safetensors.torch import save_file

    directory = Path(directory)
    source = Path(source_model_dir)
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"not a model directory: {source}")
    directory.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source / "config.json", directory / "config.json")
    for sub in ("tokenizer", "weights/audio_vae", "weights/speaker_encoder"):
        src = source / sub
        if src.is_dir():
            dst = directory / sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    tts_dir = directory / "weights" / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, str(tts_dir / "model.safetensors"))
    return directory
