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

"""分割された gol の tar を落とさずに manifest 化できることを検証する。

gol-dataset の大きい game は `<game>_part1.tar` / `<game>_part2.tar` に
分割されている（tar 602本に対し metadata 上の game は 596）。

`gol_records` は当初 `p.stem` をそのまま game_id として使っていたため、
`..._part1` は metadata の game_id と一致せず、**分割された game が
エラーも出さずに丸ごと manifest から消えていた**。S1 で選定した5 game の
うち2 game（170時間）が該当し、実行前に気づけた。

エラーにならない欠落なので、テストで固定する。
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_japanese_manifest import gol_records, group_tars_by_game  # noqa: E402

RATE = 48000
SPLIT_GAME = "A" * 32
SINGLE_GAME = "B" * 32


def _wav_bytes(seconds: float = 1.0) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(RATE * seconds), dtype="float32"), RATE, format="WAV")
    return buffer.getvalue()


def _write_tar(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w") as archive:
        for name in names:
            payload = _wav_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


@pytest.fixture()
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    tar_dir = tmp_path / "tars"
    tar_dir.mkdir()
    # 分割game: part1 に 2件、part2 に 1件
    _write_tar(tar_dir / f"{SPLIT_GAME}_part1.tar", [f"{SPLIT_GAME}/a_0001.wav",
                                                     f"{SPLIT_GAME}/a_0002.wav"])
    _write_tar(tar_dir / f"{SPLIT_GAME}_part2.tar", [f"{SPLIT_GAME}/a_0003.wav"])
    # 非分割game
    _write_tar(tar_dir / f"{SINGLE_GAME}.tar", [f"{SINGLE_GAME}/b_0001.wav"])

    metadata = tmp_path / "metadata.tsv"
    rows = ["game_id\tspeaker\ttext\tfile_path\tduration"]
    for name in ("a_0001", "a_0002", "a_0003"):
        rows.append(f"{SPLIT_GAME}\tspk1\tこんにちは\t{SPLIT_GAME}/{name}.wav\t1.0")
    rows.append(f"{SINGLE_GAME}\tspk2\tおはよう\t{SINGLE_GAME}/b_0001.wav\t1.0")
    metadata.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return metadata, tar_dir


# ---------------------------------------------------------------- grouping

def test_part_suffix_is_stripped_to_recover_the_game_id(corpus):
    _, tar_dir = corpus
    groups = group_tars_by_game(tar_dir)
    assert set(groups) == {SPLIT_GAME, SINGLE_GAME}
    assert len(groups[SPLIT_GAME]) == 2
    assert len(groups[SINGLE_GAME]) == 1


def test_a_plain_tar_is_not_altered(tmp_path):
    tar_dir = tmp_path / "tars"
    tar_dir.mkdir()
    _write_tar(tar_dir / f"{SINGLE_GAME}.tar", [f"{SINGLE_GAME}/x.wav"])
    assert set(group_tars_by_game(tar_dir)) == {SINGLE_GAME}


# ----------------------------------------------------------------- records

def test_split_game_contributes_every_utterance(corpus):
    """part1 と part2 の両方から発話が出る。ここが元の不具合。"""
    metadata, tar_dir = corpus
    records = list(gol_records(metadata, tar_dir, frozenset()))
    ids = {r.utterance_id for r in records}
    assert ids == {
        f"gol:{SPLIT_GAME}:a_0001.wav",
        f"gol:{SPLIT_GAME}:a_0002.wav",
        f"gol:{SPLIT_GAME}:a_0003.wav",
        f"gol:{SINGLE_GAME}:b_0001.wav",
    }


def test_each_utterance_points_at_the_part_that_actually_contains_it(corpus):
    metadata, tar_dir = corpus
    by_id = {r.utterance_id: r for r in gol_records(metadata, tar_dir, frozenset())}
    assert by_id[f"gol:{SPLIT_GAME}:a_0001.wav"].audio_ref.split("::")[0].endswith("_part1.tar")
    assert by_id[f"gol:{SPLIT_GAME}:a_0002.wav"].audio_ref.split("::")[0].endswith("_part1.tar")
    assert by_id[f"gol:{SPLIT_GAME}:a_0003.wav"].audio_ref.split("::")[0].endswith("_part2.tar")
    assert by_id[f"gol:{SINGLE_GAME}:b_0001.wav"].audio_ref.split("::")[0].endswith(
        f"{SINGLE_GAME}.tar")


def test_metadata_rows_with_no_matching_wav_are_dropped(corpus):
    """どのpartにも無いwavは黙って採用しない。"""
    metadata, tar_dir = corpus
    text = metadata.read_text(encoding="utf-8")
    metadata.write_text(
        text + f"{SPLIT_GAME}\tspk1\t幽霊\t{SPLIT_GAME}/a_9999.wav\t1.0\n", encoding="utf-8")
    ids = {r.utterance_id for r in gol_records(metadata, tar_dir, frozenset())}
    assert f"gol:{SPLIT_GAME}:a_9999.wav" not in ids


def test_sample_rate_is_read_from_the_archive(corpus):
    metadata, tar_dir = corpus
    records = list(gol_records(metadata, tar_dir, frozenset()))
    assert {r.sample_rate for r in records} == {RATE}
