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

"""run artifact層の決定的テスト（GPU・実データに依存しない）。"""

from __future__ import annotations

import json

import pytest

from cutetts.training.artifacts import (
    env_snapshot,
    file_checksum,
    new_run_dir,
    write_metrics,
    write_run_metadata,
)

# 既知の値: sha256("hello world") / md5("hello world")
HELLO_WORLD = b"hello world"
HELLO_WORLD_SHA256 = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
HELLO_WORLD_MD5 = "5eb63bbbe01eeed093cb22bb8f5acdc3"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# --- checksum ----------------------------------------------------------------


def test_file_checksum_matches_known_sha256(tmp_path) -> None:
    path = tmp_path / "hello.txt"
    path.write_bytes(HELLO_WORLD)

    assert file_checksum(path) == HELLO_WORLD_SHA256


def test_file_checksum_of_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert file_checksum(path) == EMPTY_SHA256


def test_file_checksum_is_chunk_size_independent(tmp_path) -> None:
    path = tmp_path / "hello.txt"
    path.write_bytes(HELLO_WORLD)

    assert file_checksum(path, chunk_size=1) == HELLO_WORLD_SHA256
    assert file_checksum(path, chunk_size=1 << 20) == HELLO_WORLD_SHA256


def test_file_checksum_supports_other_algorithms(tmp_path) -> None:
    path = tmp_path / "hello.txt"
    path.write_bytes(HELLO_WORLD)

    assert file_checksum(path, algo="md5") == HELLO_WORLD_MD5


def test_file_checksum_rejects_non_positive_chunk_size(tmp_path) -> None:
    path = tmp_path / "hello.txt"
    path.write_bytes(HELLO_WORLD)

    with pytest.raises(ValueError):
        file_checksum(path, chunk_size=0)


# --- run directory -----------------------------------------------------------


def test_new_run_dir_creates_explicit_timestamp_path(tmp_path) -> None:
    run_dir = new_run_dir("p1d", root=tmp_path, timestamp="2026-08-30T12-00-00")

    assert run_dir == tmp_path / "p1d" / "2026-08-30T12-00-00"
    assert run_dir.is_dir()


def test_new_run_dir_with_same_timestamp_returns_same_path(tmp_path) -> None:
    first = new_run_dir("p1d", root=tmp_path, timestamp="2026-08-30T12-00-00")
    second = new_run_dir("p1d", root=tmp_path, timestamp="2026-08-30T12-00-00")

    assert first == second
    assert first.is_dir()


def test_new_run_dir_generates_timestamp_when_omitted(tmp_path) -> None:
    run_dir = new_run_dir("p0", root=tmp_path)

    assert run_dir.parent == tmp_path / "p0"
    assert run_dir.is_dir()
    # YYYY-MM-DDTHH-MM-SS
    assert len(run_dir.name) == 19
    assert run_dir.name[4] == run_dir.name[7] == "-"
    assert run_dir.name[10] == "T"


def test_new_run_dir_does_not_reuse_an_existing_auto_directory(tmp_path) -> None:
    first = new_run_dir("p0", root=tmp_path)
    second = new_run_dir("p0", root=tmp_path)

    assert first != second
    assert first.is_dir() and second.is_dir()


def test_new_run_dir_rejects_empty_phase(tmp_path) -> None:
    with pytest.raises(ValueError):
        new_run_dir("", root=tmp_path)


# --- env snapshot ------------------------------------------------------------


def test_env_snapshot_has_required_keys() -> None:
    snapshot = env_snapshot()

    for key in ("os", "python", "torch", "transformers", "cutetts_commit", "gpu", "vram_gb"):
        assert key in snapshot

    assert isinstance(snapshot["os"], str) and snapshot["os"]
    assert isinstance(snapshot["python"], str) and snapshot["python"]


def test_env_snapshot_is_json_serialisable() -> None:
    json.dumps(env_snapshot(), ensure_ascii=False)


def test_env_snapshot_does_not_raise_when_torch_is_missing(monkeypatch) -> None:
    """torchの無い環境（CPUのみのpreflight）でも None を入れて完走する。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    snapshot = env_snapshot()
    monkeypatch.undo()

    assert snapshot["torch"] is None
    assert snapshot["gpu"] is None
    assert snapshot["vram_gb"] is None


# --- run metadata / metrics --------------------------------------------------


def test_write_run_metadata_writes_the_three_files(tmp_path) -> None:
    run_dir = new_run_dir("p1d", root=tmp_path, timestamp="2026-08-30T12-00-00")

    write_run_metadata(
        run_dir,
        phase="p1d",
        command=["python", "scripts/prepare_japanese_manifest.py", "--dataset", "gol"],
        seed=42,
        inputs={"manifest_checksum": "a" * 64, "metadata_tsv_rows": 7405094},
        extra={"config": "configs/japanese/manifest.yaml"},
    )

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    inputs = json.loads((run_dir / "inputs.json").read_text(encoding="utf-8"))

    assert run["phase"] == "p1d"
    assert run["command"][0] == "python"
    assert run["seed"] == 42
    assert run["started_at"]
    assert run["extra"] == {"config": "configs/japanese/manifest.yaml"}
    assert "cutetts_commit" in env
    assert inputs == {"manifest_checksum": "a" * 64, "metadata_tsv_rows": 7405094}


def test_write_run_metadata_accepts_none_seed_and_no_extra(tmp_path) -> None:
    run_dir = new_run_dir("p1b", root=tmp_path, timestamp="2026-08-30T12-00-00")

    write_run_metadata(run_dir, phase="p1b", command=["cutetts"], seed=None, inputs={})

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert run["seed"] is None
    assert run["extra"] == {}


def test_write_run_metadata_creates_missing_run_dir(tmp_path) -> None:
    run_dir = tmp_path / "p2" / "2026-08-30T12-00-00"

    write_run_metadata(run_dir, phase="p2", command=[], seed=0, inputs={})

    assert (run_dir / "run.json").is_file()


def test_write_metrics_keeps_japanese_readable(tmp_path) -> None:
    run_dir = new_run_dir("p1b", root=tmp_path, timestamp="2026-08-30T12-00-00")

    write_metrics(run_dir, {"unk_rate": 0.0123, "備考": "日本語コメント"})

    raw = (run_dir / "metrics.json").read_text(encoding="utf-8")
    metrics = json.loads(raw)

    assert "日本語コメント" in raw  # ensure_ascii=False
    assert metrics["unk_rate"] == pytest.approx(0.0123)
    assert metrics["備考"] == "日本語コメント"


def test_write_metrics_overwrites_previous_metrics(tmp_path) -> None:
    run_dir = new_run_dir("p1c", root=tmp_path, timestamp="2026-08-30T12-00-00")

    write_metrics(run_dir, {"a": 1})
    write_metrics(run_dir, {"b": 2})

    assert json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")) == {"b": 2}
