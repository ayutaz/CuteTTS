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

"""speaker embedding cache のテスト。

``test_embed_waveform_*`` だけが ``model/CuteTTS/weights/speaker_encoder`` の
実weightを必要とし、無い環境では skip する（``@pytest.mark.slow``）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from cutetts.training.latents import PREPROCESSING_VERSION, CacheMetaMismatch
from cutetts.training.speaker_cache import (
    SPEAKER_EMBEDDING_DIM,
    SPEAKER_SAMPLE_RATE,
    SpeakerCacheMeta,
    SpeakerEmbeddingCacheReader,
    SpeakerEmbeddingCacheWriter,
    embed_waveform,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEAKER_ENCODER_DIR = REPO_ROOT / "model" / "CuteTTS" / "weights" / "speaker_encoder"

ENCODER_SHA = "1" * 64


def make_meta(**overrides) -> SpeakerCacheMeta:
    payload = {
        "speaker_encoder_sha256": ENCODER_SHA,
        "preprocessing_version": PREPROCESSING_VERSION,
        "sample_rate": SPEAKER_SAMPLE_RATE,
        "embedding_dim": SPEAKER_EMBEDDING_DIM,
        "dtype": "float32",
    }
    payload.update(overrides)
    return SpeakerCacheMeta(**payload)


def make_embedding(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(SPEAKER_EMBEDDING_DIM, generator=generator)
    return vector / vector.norm()


# --- SpeakerCacheMeta --------------------------------------------------------


def test_meta_roundtrip_through_json() -> None:
    meta = make_meta()

    assert SpeakerCacheMeta.from_json(meta.to_json()) == meta


def test_meta_from_json_ignores_unknown_keys() -> None:
    payload = {**make_meta().to_json(), "cache_format": "shard-v1", "row_dim": 256}

    assert SpeakerCacheMeta.from_json(payload) == make_meta()


def test_meta_from_json_rejects_missing_keys() -> None:
    payload = make_meta().to_json()
    payload.pop("embedding_dim")

    with pytest.raises(KeyError):
        SpeakerCacheMeta.from_json(payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"speaker_encoder_sha256": ""},
        {"preprocessing_version": ""},
        {"sample_rate": -1},
        {"embedding_dim": 0},
        {"dtype": "float64"},
    ],
)
def test_meta_rejects_invalid_fields(overrides) -> None:
    with pytest.raises(ValueError):
        make_meta(**overrides)


# --- writer / reader roundtrip ----------------------------------------------


def test_write_then_read_roundtrips_exactly_in_float32(tmp_path) -> None:
    meta = make_meta()
    embedding = make_embedding(1)

    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", meta) as writer:
        writer.write("utt-0001", embedding)

    with SpeakerEmbeddingCacheReader(tmp_path / "cache", expect=meta) as reader:
        restored = reader.read("utt-0001")

    assert restored.shape == (SPEAKER_EMBEDDING_DIM,)
    assert restored.dtype == torch.float32
    assert torch.equal(restored, embedding)


def test_float16_cache_roundtrips_within_fp16_precision(tmp_path) -> None:
    meta = make_meta(dtype="float16")
    embedding = make_embedding(2)

    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", meta) as writer:
        writer.write("utt", embedding)
    with SpeakerEmbeddingCacheReader(tmp_path / "cache") as reader:
        assert torch.equal(reader.read("utt"), embedding.half().float())


def test_writer_accepts_a_batched_row(tmp_path) -> None:
    meta = make_meta()
    embedding = make_embedding(3)

    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", meta) as writer:
        writer.write("utt", embedding.unsqueeze(0))
    with SpeakerEmbeddingCacheReader(tmp_path / "cache") as reader:
        assert torch.equal(reader.read("utt"), embedding)


def test_reader_reports_meta_membership_and_keys(tmp_path) -> None:
    meta = make_meta()
    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", meta) as writer:
        for index in range(5):
            writer.write(f"utt-{index}", make_embedding(index))

    with SpeakerEmbeddingCacheReader(tmp_path / "cache") as reader:
        assert reader.meta == meta
        assert len(reader) == 5
        assert sorted(reader.keys()) == [f"utt-{index}" for index in range(5)]
        assert "utt-3" in reader
        assert "utt-8" not in reader
        for index in range(5):
            assert torch.equal(reader.read(f"utt-{index}"), make_embedding(index))


def test_read_of_unknown_id_raises_key_error(tmp_path) -> None:
    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", make_meta()) as writer:
        writer.write("known", make_embedding(4))

    with SpeakerEmbeddingCacheReader(tmp_path / "cache") as reader:
        with pytest.raises(KeyError):
            reader.read("missing")


def test_writing_the_same_input_twice_reads_back_identically(tmp_path) -> None:
    meta = make_meta()
    embedding = make_embedding(5)

    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", meta) as writer:
        writer.write("utt", embedding)
        writer.write("utt", embedding)

    with SpeakerEmbeddingCacheReader(tmp_path / "cache") as reader:
        assert len(reader) == 1
        assert torch.equal(reader.read("utt"), reader.read("utt"))
        assert torch.equal(reader.read("utt"), embedding)


def test_append_duplicate_mode_lets_the_last_write_win(tmp_path) -> None:
    meta = make_meta()
    first = make_embedding(6)
    second = make_embedding(7)

    with SpeakerEmbeddingCacheWriter(
        tmp_path / "cache", meta, on_duplicate="append"
    ) as writer:
        writer.write("utt", first)
        writer.write("utt", second)

    with SpeakerEmbeddingCacheReader(tmp_path / "cache") as reader:
        assert torch.equal(reader.read("utt"), second)


def test_reopening_a_matching_cache_appends(tmp_path) -> None:
    meta = make_meta()
    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", meta) as writer:
        writer.write("utt-a", make_embedding(8))
    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", meta) as writer:
        assert "utt-a" in writer
        writer.write("utt-b", make_embedding(9))

    with SpeakerEmbeddingCacheReader(tmp_path / "cache") as reader:
        assert len(reader) == 2


def test_records_roll_over_into_multiple_shards(tmp_path) -> None:
    meta = make_meta()
    root = tmp_path / "cache"
    # 1 record = 256 dim * 4 byte = 1024 byte -> 1シャード1record
    with SpeakerEmbeddingCacheWriter(root, meta, max_shard_bytes=1024) as writer:
        for index in range(3):
            writer.write(f"utt-{index}", make_embedding(index))

    shards = sorted(root.glob("shard-*.bin"))
    # 上限ちょうどで切り替わるので、最後に空のシャードが1本先行して作られる。
    assert [path.name for path in shards] == [
        "shard-00000.bin",
        "shard-00001.bin",
        "shard-00002.bin",
        "shard-00003.bin",
    ]
    assert [path.stat().st_size for path in shards] == [1024, 1024, 1024, 0]

    with SpeakerEmbeddingCacheReader(root) as reader:
        assert len(reader) == 3
        for index in range(3):
            assert torch.equal(reader.read(f"utt-{index}"), make_embedding(index))


def test_reopening_after_an_exact_boundary_roll_keeps_appending(tmp_path) -> None:
    """空の先行シャードがある状態で再開しても、既存recordが読めること。"""
    meta = make_meta()
    root = tmp_path / "cache"
    with SpeakerEmbeddingCacheWriter(root, meta, max_shard_bytes=1024) as writer:
        writer.write("utt-a", make_embedding(20))
    with SpeakerEmbeddingCacheWriter(root, meta, max_shard_bytes=1024) as writer:
        assert "utt-a" in writer
        writer.write("utt-b", make_embedding(21))

    with SpeakerEmbeddingCacheReader(root) as reader:
        assert len(reader) == 2
        assert torch.equal(reader.read("utt-a"), make_embedding(20))
        assert torch.equal(reader.read("utt-b"), make_embedding(21))


# --- meta mismatch -----------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"speaker_encoder_sha256": "e" * 64},
        {"preprocessing_version": "p1e-v2"},
        {"sample_rate": 24000},
        {"dtype": "float16"},
    ],
)
def test_writer_rejects_a_cache_built_with_other_settings(tmp_path, overrides) -> None:
    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", make_meta()) as writer:
        writer.write("utt", make_embedding(10))

    with pytest.raises(CacheMetaMismatch):
        SpeakerEmbeddingCacheWriter(tmp_path / "cache", make_meta(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"speaker_encoder_sha256": "e" * 64},
        {"preprocessing_version": "p1e-v2"},
        {"sample_rate": 24000},
        {"dtype": "float16"},
    ],
)
def test_reader_rejects_a_cache_that_does_not_match_expect(tmp_path, overrides) -> None:
    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", make_meta()) as writer:
        writer.write("utt", make_embedding(11))

    with pytest.raises(CacheMetaMismatch):
        SpeakerEmbeddingCacheReader(tmp_path / "cache", expect=make_meta(**overrides))


def test_reader_rejects_a_latent_cache_directory(tmp_path) -> None:
    """latent cache の root を speaker readerで開いたら弾かれること。"""
    from cutetts.training.latents import CacheMeta, LatentCacheWriter

    latent_meta = CacheMeta(
        vae_checkpoint_sha256="0" * 64,
        preprocessing_version=PREPROCESSING_VERSION,
        sample_rate=24000,
        latent_dim=64,
        dtype="float16",
    )
    with LatentCacheWriter(tmp_path / "cache", latent_meta) as writer:
        writer.write("utt", torch.zeros(3, 64))

    with pytest.raises(CacheMetaMismatch):
        SpeakerEmbeddingCacheReader(tmp_path / "cache")


def test_reader_rejects_a_missing_meta_file(tmp_path) -> None:
    root = tmp_path / "cache"
    root.mkdir()

    with pytest.raises(CacheMetaMismatch):
        SpeakerEmbeddingCacheReader(root)


def test_reader_rejects_an_unknown_cache_format(tmp_path) -> None:
    root = tmp_path / "cache"
    with SpeakerEmbeddingCacheWriter(root, make_meta()) as writer:
        writer.write("utt", make_embedding(12))
    payload = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    payload["cache_format"] = "shard-v999"
    (root / "meta.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CacheMetaMismatch):
        SpeakerEmbeddingCacheReader(root)


def test_writer_rejects_a_wrong_embedding_dim(tmp_path) -> None:
    with pytest.raises(ValueError):
        SpeakerEmbeddingCacheWriter(tmp_path / "cache", make_meta(embedding_dim=192))


# --- input validation --------------------------------------------------------


@pytest.mark.parametrize("shape", [(128,), (2, 256), (1, 1, 256), (1, 255)])
def test_writer_rejects_wrong_shapes(tmp_path, shape) -> None:
    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", make_meta()) as writer:
        with pytest.raises(ValueError):
            writer.write("utt", torch.zeros(*shape))


@pytest.mark.parametrize("bad_id", ["", "with\ttab", "with\nnewline"])
def test_writer_rejects_unusable_utterance_ids(tmp_path, bad_id) -> None:
    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", make_meta()) as writer:
        with pytest.raises(ValueError):
            writer.write(bad_id, make_embedding(13))


def test_writer_rejects_non_finite_embeddings(tmp_path) -> None:
    embedding = make_embedding(14)
    embedding[7] = float("inf")

    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", make_meta()) as writer:
        with pytest.raises(ValueError):
            writer.write("utt", embedding)


# --- real Speaker Encoder weights -------------------------------------------


def synthetic_waveform(seconds: float, sample_rate: int) -> torch.Tensor:
    """220 Hz + 660 Hz の混合 + 小さなノイズ。決定的に作る。"""
    generator = torch.Generator().manual_seed(4321)
    samples = int(round(seconds * sample_rate))
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    tone = 0.4 * torch.sin(2 * math.pi * 220.0 * time)
    tone = tone + 0.2 * torch.sin(2 * math.pi * 660.0 * time)
    noise = 0.02 * torch.randn(samples, generator=generator)
    return (tone + noise).unsqueeze(0)


@pytest.fixture(scope="module")
def speaker_encoder():
    if not SPEAKER_ENCODER_DIR.is_dir():
        pytest.skip(f"Speaker encoder weights are not available at {SPEAKER_ENCODER_DIR}")
    from safetensors.torch import load_file

    from cutetts.audio_codec.model.speaker_encoder import FbankECAPAStudent

    config = json.loads((SPEAKER_ENCODER_DIR / "config.json").read_text(encoding="utf-8"))
    config.pop("component", None)
    encoder = FbankECAPAStudent(**config)
    missing, unexpected = encoder.load_state_dict(
        load_file(str(SPEAKER_ENCODER_DIR / "model.safetensors")), strict=True
    )
    assert not missing and not unexpected
    return encoder.float().eval()


@pytest.mark.slow
def test_embed_waveform_returns_256_dims_with_real_weights(speaker_encoder) -> None:
    waveform = synthetic_waveform(8.0, SPEAKER_SAMPLE_RATE)

    embedding = embed_waveform(speaker_encoder, waveform)

    assert embedding.shape == (SPEAKER_EMBEDDING_DIM,)
    assert embedding.dtype == torch.float32
    assert torch.isfinite(embedding).all()
    # forward は F.normalize 済みの "embedding" を返す。
    assert embedding.norm().item() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.slow
def test_embed_waveform_is_deterministic_and_accepts_1d_input(speaker_encoder) -> None:
    waveform = synthetic_waveform(3.0, SPEAKER_SAMPLE_RATE)

    first = embed_waveform(speaker_encoder, waveform)
    second = embed_waveform(speaker_encoder, waveform.squeeze(0))

    assert torch.equal(first, second)


@pytest.mark.slow
def test_embed_waveform_refuses_a_training_mode_encoder(speaker_encoder) -> None:
    speaker_encoder.train()
    try:
        with pytest.raises(ValueError):
            embed_waveform(speaker_encoder, synthetic_waveform(2.0, SPEAKER_SAMPLE_RATE))
    finally:
        speaker_encoder.eval()


@pytest.mark.slow
def test_embedding_survives_the_cache_roundtrip(tmp_path, speaker_encoder) -> None:
    meta = make_meta()
    embedding = embed_waveform(speaker_encoder, synthetic_waveform(8.0, SPEAKER_SAMPLE_RATE))

    with SpeakerEmbeddingCacheWriter(tmp_path / "cache", meta) as writer:
        writer.write("sine-8s", embedding)
    with SpeakerEmbeddingCacheReader(tmp_path / "cache", expect=meta) as reader:
        restored = reader.read("sine-8s")

    assert torch.equal(restored, embedding)
