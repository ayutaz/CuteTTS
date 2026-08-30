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

"""latent cache のテスト。

大半は合成tensorだけを使う決定的なユニットテスト。
末尾の ``test_encode_waveform_*`` だけが ``model/CuteTTS/weights/audio_vae`` の
実weightを必要とし、無い環境では skip する（``@pytest.mark.slow``）。
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import pytest
import torch

from cutetts.training.latents import (
    CACHE_FORMAT_VERSION,
    LATENT_DIM,
    LATENT_FRAME_RATE,
    LATENT_HOP_LENGTH,
    LATENT_SAMPLE_RATE,
    PREPROCESSING_VERSION,
    CacheMeta,
    CacheMetaMismatch,
    LatentCacheReader,
    LatentCacheWriter,
    encode_waveform,
    expected_latent_frames,
    resample_to,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_VAE_DIR = REPO_ROOT / "model" / "CuteTTS" / "weights" / "audio_vae"

VAE_SHA = "0" * 64


def make_meta(**overrides) -> CacheMeta:
    payload = {
        "vae_checkpoint_sha256": VAE_SHA,
        "preprocessing_version": PREPROCESSING_VERSION,
        "sample_rate": LATENT_SAMPLE_RATE,
        "latent_dim": LATENT_DIM,
        "dtype": "float16",
    }
    payload.update(overrides)
    return CacheMeta(**payload)


def make_latent(frames: int, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(frames, LATENT_DIM, generator=generator)


# --- CacheMeta ---------------------------------------------------------------


def test_meta_roundtrip_through_json() -> None:
    meta = make_meta()

    assert CacheMeta.from_json(meta.to_json()) == meta


def test_meta_roundtrip_survives_a_real_json_file(tmp_path) -> None:
    meta = make_meta()
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta.to_json(), ensure_ascii=False), encoding="utf-8")

    assert CacheMeta.from_json(json.loads(path.read_text(encoding="utf-8"))) == meta


def test_meta_from_json_ignores_unknown_keys() -> None:
    payload = {**make_meta().to_json(), "cache_format": CACHE_FORMAT_VERSION, "row_dim": 64}

    assert CacheMeta.from_json(payload) == make_meta()


def test_meta_from_json_rejects_missing_keys() -> None:
    payload = make_meta().to_json()
    payload.pop("dtype")

    with pytest.raises(KeyError):
        CacheMeta.from_json(payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"vae_checkpoint_sha256": ""},
        {"preprocessing_version": ""},
        {"sample_rate": 0},
        {"latent_dim": 0},
        {"dtype": "int8"},
    ],
)
def test_meta_rejects_invalid_fields(overrides) -> None:
    with pytest.raises(ValueError):
        make_meta(**overrides)


# --- frame arithmetic --------------------------------------------------------


def test_expected_latent_frames_matches_hop_length_padding() -> None:
    assert expected_latent_frames(0) == 0
    assert expected_latent_frames(1) == 1
    assert expected_latent_frames(LATENT_HOP_LENGTH) == 1
    assert expected_latent_frames(LATENT_HOP_LENGTH + 1) == 2
    # 3秒 = 72000 サンプル -> ceil(72000 / 1920) = 38 = round(3 * 12.5)
    assert expected_latent_frames(3 * LATENT_SAMPLE_RATE) == 38
    assert LATENT_SAMPLE_RATE / LATENT_HOP_LENGTH == LATENT_FRAME_RATE


# --- writer / reader roundtrip ----------------------------------------------


def test_write_then_read_roundtrips_within_fp16_precision(tmp_path) -> None:
    meta = make_meta()
    latent = make_latent(17, seed=1)

    with LatentCacheWriter(tmp_path / "cache", meta) as writer:
        writer.write("utt-0001", latent)

    with LatentCacheReader(tmp_path / "cache", expect=meta) as reader:
        restored = reader.read("utt-0001")

    assert restored.shape == (17, LATENT_DIM)
    assert restored.dtype == torch.float32
    # 保存は fp16 なので、fp16 へ丸めた値とはビット単位で一致する。
    assert torch.equal(restored, latent.half().float())
    assert torch.allclose(restored, latent, atol=1e-2, rtol=1e-2)


def test_float32_cache_roundtrips_exactly(tmp_path) -> None:
    meta = make_meta(dtype="float32")
    latent = make_latent(5, seed=2)

    with LatentCacheWriter(tmp_path / "cache", meta) as writer:
        writer.write("utt", latent)
    with LatentCacheReader(tmp_path / "cache") as reader:
        assert torch.equal(reader.read("utt"), latent)


def test_reader_reports_meta_and_membership(tmp_path) -> None:
    meta = make_meta()
    with LatentCacheWriter(tmp_path / "cache", meta) as writer:
        for index in range(4):
            writer.write(f"utt-{index}", make_latent(3 + index, seed=index))

    with LatentCacheReader(tmp_path / "cache") as reader:
        assert reader.meta == meta
        assert len(reader) == 4
        assert sorted(reader.keys()) == ["utt-0", "utt-1", "utt-2", "utt-3"]
        assert "utt-2" in reader
        assert "utt-9" not in reader
        for index in range(4):
            assert reader.read(f"utt-{index}").shape == (3 + index, LATENT_DIM)


def test_variable_lengths_are_not_mixed_up(tmp_path) -> None:
    meta = make_meta()
    lengths = [1, 37, 4, 900, 12]
    latents = {f"utt-{i}": make_latent(n, seed=100 + i) for i, n in enumerate(lengths)}

    with LatentCacheWriter(tmp_path / "cache", meta) as writer:
        for key, latent in latents.items():
            writer.write(key, latent)

    with LatentCacheReader(tmp_path / "cache") as reader:
        for key, latent in latents.items():
            assert torch.equal(reader.read(key), latent.half().float())


def test_read_of_unknown_id_raises_key_error(tmp_path) -> None:
    with LatentCacheWriter(tmp_path / "cache", make_meta()) as writer:
        writer.write("known", make_latent(2))

    with LatentCacheReader(tmp_path / "cache") as reader:
        with pytest.raises(KeyError):
            reader.read("missing")


def test_read_on_empty_cache_raises_key_error(tmp_path) -> None:
    LatentCacheWriter(tmp_path / "cache", make_meta()).close()

    with LatentCacheReader(tmp_path / "cache") as reader:
        assert len(reader) == 0
        with pytest.raises(KeyError):
            reader.read("anything")


# --- determinism / duplicates ------------------------------------------------


def test_writing_the_same_input_twice_reads_back_identically(tmp_path) -> None:
    meta = make_meta()
    latent = make_latent(9, seed=7)

    with LatentCacheWriter(tmp_path / "cache", meta) as writer:
        writer.write("utt", latent)
        writer.write("utt", latent)

    with LatentCacheReader(tmp_path / "cache") as reader:
        first = reader.read("utt")
        second = reader.read("utt")

    assert len(list(LatentCacheReader(tmp_path / "cache").keys())) == 1
    assert torch.equal(first, second)
    assert torch.equal(first, latent.half().float())


def test_two_independent_caches_of_the_same_input_are_byte_identical(tmp_path) -> None:
    meta = make_meta()
    latent = make_latent(11, seed=8)
    for name in ("a", "b"):
        with LatentCacheWriter(tmp_path / name, meta) as writer:
            writer.write("utt", latent)

    shard_a = (tmp_path / "a" / "shard-00000.bin").read_bytes()
    shard_b = (tmp_path / "b" / "shard-00000.bin").read_bytes()
    assert shard_a == shard_b


def test_default_skip_duplicate_keeps_the_first_value(tmp_path) -> None:
    meta = make_meta()
    first = make_latent(4, seed=11)
    second = make_latent(6, seed=12)

    with LatentCacheWriter(tmp_path / "cache", meta) as writer:
        writer.write("utt", first)
        assert "utt" in writer
        writer.write("utt", second)

    with LatentCacheReader(tmp_path / "cache") as reader:
        assert torch.equal(reader.read("utt"), first.half().float())


def test_append_duplicate_mode_lets_the_last_write_win(tmp_path) -> None:
    meta = make_meta()
    first = make_latent(4, seed=11)
    second = make_latent(6, seed=12)

    with LatentCacheWriter(tmp_path / "cache", meta, on_duplicate="append") as writer:
        writer.write("utt", first)
        writer.write("utt", second)

    with LatentCacheReader(tmp_path / "cache") as reader:
        assert len(reader) == 1
        assert torch.equal(reader.read("utt"), second.half().float())


# --- resume / sharding -------------------------------------------------------


def test_reopening_a_matching_cache_appends_instead_of_truncating(tmp_path) -> None:
    meta = make_meta()
    with LatentCacheWriter(tmp_path / "cache", meta) as writer:
        writer.write("utt-a", make_latent(3, seed=21))

    with LatentCacheWriter(tmp_path / "cache", meta) as writer:
        assert "utt-a" in writer
        writer.write("utt-b", make_latent(5, seed=22))

    with LatentCacheReader(tmp_path / "cache") as reader:
        assert len(reader) == 2
        assert torch.equal(reader.read("utt-a"), make_latent(3, seed=21).half().float())
        assert torch.equal(reader.read("utt-b"), make_latent(5, seed=22).half().float())


def test_records_roll_over_into_multiple_shards(tmp_path) -> None:
    meta = make_meta()
    root = tmp_path / "cache"
    # 1 record = 8 frames * 64 dim * 2 byte = 1024 byte
    with LatentCacheWriter(root, meta, max_shard_bytes=2048) as writer:
        for index in range(7):
            writer.write(f"utt-{index}", make_latent(8, seed=index))

    shards = sorted(path.name for path in root.glob("shard-*.bin"))
    assert len(shards) == 4  # 2 records per shard, 7 records

    with LatentCacheReader(root) as reader:
        assert len(reader) == 7
        for index in range(7):
            assert torch.equal(
                reader.read(f"utt-{index}"), make_latent(8, seed=index).half().float()
            )


def test_a_torn_trailing_index_line_is_ignored(tmp_path) -> None:
    """.bin を書いた直後に落ちた場合を模す。壊れていないrecordは読めること。"""
    meta = make_meta()
    root = tmp_path / "cache"
    with LatentCacheWriter(root, meta) as writer:
        writer.write("utt-a", make_latent(3, seed=31))
        writer.write("utt-b", make_latent(4, seed=32))

    index_path = root / "shard-00000.idx"
    with index_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("utt-c\t99999\t7")  # 改行なし = 書きかけ

    with LatentCacheReader(root) as reader:
        assert len(reader) == 2
        assert "utt-c" not in reader
        assert torch.equal(reader.read("utt-b"), make_latent(4, seed=32).half().float())


def test_corrupted_index_line_raises(tmp_path) -> None:
    meta = make_meta()
    root = tmp_path / "cache"
    with LatentCacheWriter(root, meta) as writer:
        writer.write("utt-a", make_latent(3, seed=41))
    with (root / "shard-00000.idx").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("only-one-field\n")

    with pytest.raises(CacheMetaMismatch):
        LatentCacheReader(root)


def test_reader_is_picklable_for_dataloader_workers(tmp_path) -> None:
    meta = make_meta()
    with LatentCacheWriter(tmp_path / "cache", meta) as writer:
        writer.write("utt", make_latent(6, seed=51))

    reader = LatentCacheReader(tmp_path / "cache")
    reader.read("utt")  # file handle を開かせてから pickle する
    revived = pickle.loads(pickle.dumps(reader))

    assert torch.equal(revived.read("utt"), make_latent(6, seed=51).half().float())


# --- meta mismatch -----------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"vae_checkpoint_sha256": "f" * 64},
        {"preprocessing_version": "p1e-v2"},
        {"sample_rate": 16000},
        {"dtype": "float32"},
    ],
)
def test_writer_rejects_a_cache_built_with_other_settings(tmp_path, overrides) -> None:
    with LatentCacheWriter(tmp_path / "cache", make_meta()) as writer:
        writer.write("utt", make_latent(3))

    with pytest.raises(CacheMetaMismatch):
        LatentCacheWriter(tmp_path / "cache", make_meta(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"vae_checkpoint_sha256": "f" * 64},
        {"preprocessing_version": "p1e-v2"},
        {"sample_rate": 16000},
        {"dtype": "float32"},
    ],
)
def test_reader_rejects_a_cache_that_does_not_match_expect(tmp_path, overrides) -> None:
    with LatentCacheWriter(tmp_path / "cache", make_meta()) as writer:
        writer.write("utt", make_latent(3))

    with pytest.raises(CacheMetaMismatch):
        LatentCacheReader(tmp_path / "cache", expect=make_meta(**overrides))


def test_reader_rejects_a_missing_meta_file(tmp_path) -> None:
    root = tmp_path / "cache"
    root.mkdir()

    with pytest.raises(CacheMetaMismatch):
        LatentCacheReader(root)


def test_reader_rejects_an_unknown_cache_format(tmp_path) -> None:
    root = tmp_path / "cache"
    with LatentCacheWriter(root, make_meta()) as writer:
        writer.write("utt", make_latent(3))
    payload = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    payload["cache_format"] = "shard-v999"
    (root / "meta.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CacheMetaMismatch):
        LatentCacheReader(root)


def test_writer_rejects_a_shard_directory_without_meta(tmp_path) -> None:
    root = tmp_path / "cache"
    with LatentCacheWriter(root, make_meta()) as writer:
        writer.write("utt", make_latent(3))
    (root / "meta.json").unlink()

    with pytest.raises(CacheMetaMismatch):
        LatentCacheWriter(root, make_meta())


def test_writer_rejects_a_wrong_latent_dim(tmp_path) -> None:
    with pytest.raises(ValueError):
        LatentCacheWriter(tmp_path / "cache", make_meta(latent_dim=32))


# --- input validation --------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["", "with\ttab", "with\nnewline", "with\rcr"])
def test_writer_rejects_unusable_utterance_ids(tmp_path, bad_id) -> None:
    with LatentCacheWriter(tmp_path / "cache", make_meta()) as writer:
        with pytest.raises(ValueError):
            writer.write(bad_id, make_latent(3))


def test_writer_rejects_non_finite_latents(tmp_path) -> None:
    latent = make_latent(3)
    latent[1, 5] = float("nan")

    with LatentCacheWriter(tmp_path / "cache", make_meta()) as writer:
        with pytest.raises(ValueError):
            writer.write("utt", latent)


@pytest.mark.parametrize(
    "shape",
    [(64,), (3, 32), (1, 3, 64), (0, 64)],
)
def test_writer_rejects_wrong_shapes(tmp_path, shape) -> None:
    with LatentCacheWriter(tmp_path / "cache", make_meta()) as writer:
        with pytest.raises(ValueError):
            writer.write("utt", torch.zeros(*shape))


def test_writer_accepts_numpy_input(tmp_path) -> None:
    latent = make_latent(4, seed=61)
    with LatentCacheWriter(tmp_path / "cache", make_meta()) as writer:
        writer.write("utt", latent.numpy())

    with LatentCacheReader(tmp_path / "cache") as reader:
        assert torch.equal(reader.read("utt"), latent.half().float())


def test_unicode_utterance_ids_roundtrip(tmp_path) -> None:
    keys = ["gol/ある話者/0001", "moe/話者-01/utt_0002", "ascii-0003"]
    with LatentCacheWriter(tmp_path / "cache", make_meta()) as writer:
        for index, key in enumerate(keys):
            writer.write(key, make_latent(2, seed=index))

    with LatentCacheReader(tmp_path / "cache") as reader:
        assert sorted(reader.keys()) == sorted(keys)
        for index, key in enumerate(keys):
            assert torch.equal(reader.read(key), make_latent(2, seed=index).half().float())


# --- resample ----------------------------------------------------------------


def test_resample_to_is_a_passthrough_for_equal_rates() -> None:
    waveform = torch.randn(1, 100)

    assert resample_to(waveform, 24000, 24000) is waveform


def test_resample_to_changes_length_by_the_rate_ratio() -> None:
    waveform = torch.randn(1, 48000)

    assert resample_to(waveform, 48000, 24000).shape == (1, 24000)
    assert resample_to(waveform, 48000, 16000).shape == (1, 16000)


def test_resample_to_handles_the_moe_44100_ratio() -> None:
    waveform = torch.randn(1, 44100)

    assert resample_to(waveform, 44100, 24000).shape == (1, 24000)


def test_resample_to_rejects_invalid_rates() -> None:
    with pytest.raises(ValueError):
        resample_to(torch.randn(1, 10), 0, 24000)


# --- real Audio VAE weights --------------------------------------------------


def synthetic_waveform(seconds: float, sample_rate: int) -> torch.Tensor:
    """440 Hz サイン波 + 小さなノイズ。決定的に作る。"""
    generator = torch.Generator().manual_seed(1234)
    samples = int(round(seconds * sample_rate))
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    tone = 0.5 * torch.sin(2 * math.pi * 440.0 * time)
    noise = 0.01 * torch.randn(samples, generator=generator)
    return (tone + noise).unsqueeze(0)


@pytest.fixture(scope="module")
def audio_vae():
    if not AUDIO_VAE_DIR.is_dir():
        pytest.skip(f"Audio VAE weights are not available at {AUDIO_VAE_DIR}")
    from cutetts.modeling.audio_adapter import AudioAcousticVAEAdapter

    return AudioAcousticVAEAdapter(AUDIO_VAE_DIR).eval()


@pytest.mark.slow
def test_encode_waveform_returns_expected_shape_with_real_weights(audio_vae) -> None:
    waveform = synthetic_waveform(3.0, LATENT_SAMPLE_RATE)

    latent = encode_waveform(audio_vae, waveform)

    assert latent.shape == (round(3 * LATENT_FRAME_RATE), LATENT_DIM)
    assert latent.shape == (expected_latent_frames(waveform.shape[-1]), LATENT_DIM)
    assert latent.dtype == torch.float32
    assert torch.isfinite(latent).all()


@pytest.mark.slow
def test_encode_waveform_is_deterministic_and_accepts_1d_input(audio_vae) -> None:
    waveform = synthetic_waveform(1.6, LATENT_SAMPLE_RATE)

    first = encode_waveform(audio_vae, waveform)
    second = encode_waveform(audio_vae, waveform.squeeze(0))

    assert torch.equal(first, second)


@pytest.mark.slow
def test_encoded_latent_survives_the_cache_roundtrip(tmp_path, audio_vae) -> None:
    meta = make_meta()
    latent = encode_waveform(audio_vae, synthetic_waveform(2.0, LATENT_SAMPLE_RATE))

    with LatentCacheWriter(tmp_path / "cache", meta) as writer:
        writer.write("sine-2s", latent)
    with LatentCacheReader(tmp_path / "cache", expect=meta) as reader:
        restored = reader.read("sine-2s")

    assert restored.shape == latent.shape
    assert torch.allclose(restored, latent, atol=1e-2, rtol=1e-2)
