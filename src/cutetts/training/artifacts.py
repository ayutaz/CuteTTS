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

"""Run artifact layout shared by every Japanese continual-training phase.

`docs/japanese-training/08-execution-plan.md` の「共通ルール / artifactの保存」を実装する。
1 runにつき ``artifacts/<phase>/<YYYY-MM-DDTHH-MM-SS>/`` を作り、その中へ

* ``run.json``     … phase / 実行コマンド / seed / 開始時刻 / 任意のextra
* ``env.json``     … OS・Python・torch・transformers・cutetts commit・GPU
* ``inputs.json``  … checkpoint revision や入力fileのchecksum など再現に要る入力
* ``metrics.json`` … そのフェーズの数値

を残す。artifact本体（音声など）は各フェーズが ``run_dir`` 配下へ自由に置いてよい。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "RUN_TIMESTAMP_FORMAT",
    "env_snapshot",
    "file_checksum",
    "new_run_dir",
    "write_metrics",
    "write_run_metadata",
]

#: run directory名のtimestamp書式。Windowsのpathで使えるよう ``:`` を ``-`` にしてある。
RUN_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"

#: このfileから見たrepository root（src/cutetts/training/artifacts.py → 3階層上）。
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: Any) -> None:
    """UTF-8 / ensure_ascii=False でJSONを書く（日本語をそのまま読めるようにする）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
    path.write_text(text + "\n", encoding="utf-8")


def new_run_dir(phase: str, root: str | Path = "artifacts", *, timestamp: str | None = None) -> Path:
    """``<root>/<phase>/<timestamp>/`` を作って返す。

    Args:
        phase: ``p0`` / ``p1b`` など、実行計画のフェーズID。
        root: artifact rootディレクトリ。既定は repository直下の ``artifacts/``。
        timestamp: ディレクトリ名。``None`` ならローカル時刻から生成する。
            テストや「同じrunへ追記する」用途では明示的に渡す。

    Notes:
        ``timestamp`` を明示した場合は **そのpathをそのまま** 使う（既存でもエラーにしない）。
        自動生成の場合のみ、同一秒での衝突を避けて ``-01`` 以降のsuffixを付ける。
    """
    if not phase:
        raise ValueError("phase must be a non-empty string.")
    base = Path(root) / phase
    if timestamp is not None:
        run_dir = base / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    stamp = datetime.now().strftime(RUN_TIMESTAMP_FORMAT)
    run_dir = base / stamp
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = base / f"{stamp}-{suffix:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _git_commit() -> str | None:
    """cutetts側のcommit hash。gitが無い/repoでない場合は ``None``。"""
    try:
        completed = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _module_version(name: str) -> str | None:
    """importできればversion文字列、できなければ ``None``。"""
    try:
        module = __import__(name)
    except Exception:  # noqa: BLE001 - 依存が無い環境でもsnapshotは失敗させない
        return None
    return getattr(module, "__version__", None)


def _gpu_info() -> tuple[str | None, float | None, str | None, int | None]:
    """(gpu名, VRAM GB, CUDA version, device数)。torchが無い/CUDAが無ければ ``None``。"""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None, None, None, None
    try:
        cuda_version = getattr(torch.version, "cuda", None)
        if not torch.cuda.is_available():
            return None, None, cuda_version, 0
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        return name, round(total_bytes / (1024**3), 2), cuda_version, count
    except Exception:  # noqa: BLE001 - driver不整合などでも例外を出さない
        return None, None, None, None


def env_snapshot() -> dict:
    """再現に必要な環境情報を集める。取得できない項目は ``None`` にする。

    torch / transformers が入っていない環境でも例外を投げない（CPUのみのpreflight用）。
    """
    gpu_name, vram_gb, cuda_version, gpu_count = _gpu_info()
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": _module_version("torch"),
        "transformers": _module_version("transformers"),
        "cuda": cuda_version,
        "cutetts_commit": _git_commit(),
        "gpu": gpu_name,
        "gpu_count": gpu_count,
        "vram_gb": vram_gb,
    }


def file_checksum(path: str | Path, *, algo: str = "sha256", chunk_size: int = 1 << 20) -> str:
    """fileのhex digest。巨大なmanifest/weightでも一定メモリで読む。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    digest = hashlib.new(algo)
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_run_metadata(
    run_dir: Path,
    *,
    phase: str,
    command: list[str],
    seed: int | None,
    inputs: dict,
    extra: dict | None = None,
) -> None:
    """``run.json`` / ``env.json`` / ``inputs.json`` を ``run_dir`` に書く。

    Args:
        run_dir: :func:`new_run_dir` が返したディレクトリ。
        phase: フェーズID。``run_dir`` の親名と一致させること。
        command: 実行コマンド（``sys.argv`` をそのまま渡してよい）。
        seed: 乱数seed。使っていないフェーズは ``None``。
        inputs: checkpoint revision・入力fileのchecksumなど、入力側の再現情報。
        extra: フェーズ固有のメモ（config path、subset名など）。
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "run.json",
        {
            "phase": phase,
            "command": list(command),
            "seed": seed,
            "started_at": datetime.now().astimezone().isoformat(),
            "extra": dict(extra) if extra else {},
        },
    )
    _write_json(run_dir / "env.json", env_snapshot())
    _write_json(run_dir / "inputs.json", dict(inputs))


def write_metrics(run_dir: Path, metrics: dict) -> None:
    """``run_dir/metrics.json`` を書く（UTF-8 / ensure_ascii=False）。"""
    _write_json(Path(run_dir) / "metrics.json", dict(metrics))
