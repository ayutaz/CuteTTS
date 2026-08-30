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

"""Speaker embedding cache（P1e）。

08-execution-plan.md P1e のとおり、latent cache と **同じ1パス** で
Speaker Encoder（ECAPA student、16 kHz 入力 / 256次元）の embedding も書き出す。
発話あたり 256 dim x fp32 = 1 KB なので gol全体（7,405,094 発話）でも約 7.4 GB。

用途は2つある。

* 学習時の speaker conditioning（``CuteTTSModel.lm_speaker_linear`` と
  DiT の adaLN-Zero へ入る 256次元）
* P1d の voice clustering。gol の ``speaker_id`` は
  ``SHA-256(キャラクター表示名)[:32]`` で **声の識別子ではない** ため、
  zero-shot split はこのembedding由来のclusterで切る必要がある。

on-disk format は :mod:`cutetts.training.latents` のシャードストアをそのまま使う
（``[1, 256]`` の1行recordとして書く）。file配置・再開・index探索の性質は latents.py の
module docstring を参照。latent cache とは **別のroot** に置くこと（meta.json が別物なので）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from cutetts.training.latents import (
    CACHE_FORMAT_VERSION,
    DEFAULT_MAX_SHARD_BYTES,
    PREPROCESSING_VERSION,
    CacheMetaMismatch,
    _read_meta_json,
    _ShardStoreReader,
    _ShardStoreWriter,
    _as_mono_batch,
    _STORAGE_DTYPES,
)

__all__ = [
    "CacheMetaMismatch",
    "PREPROCESSING_VERSION",
    "SPEAKER_EMBEDDING_DIM",
    "SPEAKER_SAMPLE_RATE",
    "SpeakerCacheMeta",
    "SpeakerEmbeddingCacheReader",
    "SpeakerEmbeddingCacheWriter",
    "embed_waveform",
]

#: ``FbankECAPAStudent`` の出力次元。``config.json`` の ``lm_speaker_embedding_dim`` と一致。
SPEAKER_EMBEDDING_DIM: int = 256

#: Speaker Encoder の入力 sample rate。``forward`` は他のrateを渡すと ValueError を投げる。
SPEAKER_SAMPLE_RATE: int = 16000


@dataclass(frozen=True)
class SpeakerCacheMeta:
    """speaker embedding cacheの素性。

    :class:`cutetts.training.latents.CacheMeta` と同型だが、照合する checkpoint が
    Audio VAE ではなく Speaker Encoder なので別のdataclassにしてある
    （``vae_checkpoint_sha256`` という名前で speaker encoder の値を持たせない）。
    """

    speaker_encoder_sha256: str
    preprocessing_version: str
    sample_rate: int
    embedding_dim: int
    dtype: str

    def __post_init__(self) -> None:
        if not self.speaker_encoder_sha256:
            raise ValueError("speaker_encoder_sha256 must be a non-empty string.")
        if not self.preprocessing_version:
            raise ValueError("preprocessing_version must be a non-empty string.")
        if int(self.sample_rate) <= 0:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate!r}")
        if int(self.embedding_dim) <= 0:
            raise ValueError(f"Invalid embedding_dim: {self.embedding_dim!r}")
        if self.dtype not in _STORAGE_DTYPES:
            raise ValueError(
                f"Unsupported dtype {self.dtype!r}; expected one of {sorted(_STORAGE_DTYPES)}."
            )

    def to_json(self) -> dict:
        return {
            "speaker_encoder_sha256": str(self.speaker_encoder_sha256),
            "preprocessing_version": str(self.preprocessing_version),
            "sample_rate": int(self.sample_rate),
            "embedding_dim": int(self.embedding_dim),
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_json(cls, obj: dict) -> "SpeakerCacheMeta":
        """``to_json`` の逆。未知のkeyは無視する（前方互換のため）。"""
        if not isinstance(obj, dict):
            raise TypeError(f"Expected a JSON object, got {type(obj).__name__}.")
        required = (
            "speaker_encoder_sha256",
            "preprocessing_version",
            "sample_rate",
            "embedding_dim",
            "dtype",
        )
        missing = [key for key in required if key not in obj]
        if missing:
            raise KeyError(f"SpeakerCacheMeta is missing required keys: {missing}")
        return cls(
            speaker_encoder_sha256=str(obj["speaker_encoder_sha256"]),
            preprocessing_version=str(obj["preprocessing_version"]),
            sample_rate=int(obj["sample_rate"]),
            embedding_dim=int(obj["embedding_dim"]),
            dtype=str(obj["dtype"]),
        )

    @property
    def storage_dtype(self) -> Any:
        return _STORAGE_DTYPES[self.dtype]


def _speaker_meta_payload(meta: SpeakerCacheMeta) -> dict:
    return {
        **meta.to_json(),
        "cache_format": CACHE_FORMAT_VERSION,
        "record_kind": "speaker_embedding",
        "row_dim": int(meta.embedding_dim),
    }


class SpeakerEmbeddingCacheWriter:
    """Speaker embedding（``[256]``）をシャードへ追記する。

    :class:`cutetts.training.latents.LatentCacheWriter` と同じインターフェース。
    ``root`` が既にあり ``meta.json`` が不一致なら :class:`CacheMetaMismatch`。
    """

    def __init__(
        self,
        root: str | Path,
        meta: SpeakerCacheMeta,
        *,
        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
        on_duplicate: str = "skip",
    ) -> None:
        if int(meta.embedding_dim) != SPEAKER_EMBEDDING_DIM:
            raise ValueError(
                f"speaker cache expects embedding_dim={SPEAKER_EMBEDDING_DIM}, "
                f"got {meta.embedding_dim}."
            )
        self.meta = meta
        self._store = _ShardStoreWriter(
            root,
            _speaker_meta_payload(meta),
            row_dim=int(meta.embedding_dim),
            storage_dtype=meta.storage_dtype,
            max_shard_bytes=max_shard_bytes,
            on_duplicate=on_duplicate,
        )

    def write(self, utterance_id: str, embedding) -> None:
        """``embedding``（``[256]`` または ``[1, 256]``）を書く。

        既に同じ ``utterance_id`` があり ``on_duplicate="skip"``（既定）なら何もしない。
        """
        self._store.write_record(utterance_id, _as_row(embedding))

    def __contains__(self, utterance_id: str) -> bool:
        return utterance_id in self._store

    def __len__(self) -> int:
        return len(self._store)

    def flush(self) -> None:
        self._store.flush()

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "SpeakerEmbeddingCacheWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class SpeakerEmbeddingCacheReader:
    """:class:`SpeakerEmbeddingCacheWriter` が書いたcacheを読む。"""

    def __init__(self, root: str | Path, *, expect: SpeakerCacheMeta | None = None) -> None:
        self._root = Path(root).expanduser()
        payload = _read_meta_json(self._root)
        try:
            self.meta = SpeakerCacheMeta.from_json(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise CacheMetaMismatch(
                f"Cache metadata at {self._root} is not a valid SpeakerCacheMeta: {error}"
            ) from error
        if expect is not None and self.meta != expect:
            raise CacheMetaMismatch(
                f"Cache at {self._root} was produced with {self.meta}, expected {expect}."
            )
        stored_format = payload.get("cache_format")
        if stored_format != CACHE_FORMAT_VERSION:
            raise CacheMetaMismatch(
                f"Cache at {self._root} uses format {stored_format!r}, "
                f"this build reads {CACHE_FORMAT_VERSION!r}."
            )
        self._store = _ShardStoreReader(
            self._root,
            row_dim=int(self.meta.embedding_dim),
            storage_dtype=self.meta.storage_dtype,
        )

    def __contains__(self, utterance_id: str) -> bool:
        return utterance_id in self._store

    def read(self, utterance_id: str) -> "torch.Tensor":
        """``[256]`` の float32 tensorを返す。未登録なら :class:`KeyError`。"""
        record = self._store.read_record(utterance_id)
        if record.shape[0] != 1:
            raise RuntimeError(
                f"Speaker embedding for {utterance_id!r} has {record.shape[0]} rows, expected 1."
            )
        return torch.from_numpy(record[0])

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> Iterator[str]:
        return self._store.keys()

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "SpeakerEmbeddingCacheReader":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _as_row(embedding) -> "torch.Tensor":
    """``[256]`` / ``[1, 256]`` を ``[1, 256]`` の float32 に揃える。"""
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.as_tensor(embedding, dtype=torch.float32)
    embedding = embedding.detach().to(dtype=torch.float32)
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    if embedding.dim() != 2 or embedding.size(0) != 1:
        raise ValueError(
            f"Speaker embedding must be [{SPEAKER_EMBEDDING_DIM}] or "
            f"[1, {SPEAKER_EMBEDDING_DIM}], got shape {tuple(embedding.shape)}."
        )
    if embedding.size(1) != SPEAKER_EMBEDDING_DIM:
        raise ValueError(
            f"Speaker embedding must have {SPEAKER_EMBEDDING_DIM} dims, got {embedding.size(1)}."
        )
    return embedding.cpu()


def embed_waveform(speaker_encoder, waveform_16k) -> "torch.Tensor":
    """16 kHz mono waveform を Speaker Encoder の 256次元 embedding にする。

    Args:
        speaker_encoder: :class:`cutetts.audio_codec.model.speaker_encoder.FbankECAPAStudent`。
            **eval() 済みであること**（``layer1``〜``layer4`` に BatchNorm1d があり、
            train モードのまま呼ぶと running stats が汚れて以後のcacheが非決定的になる）。
        waveform_16k: ``[1, samples]`` または ``[samples]`` の float32 mono。
            推論側 ``prepare_reference_audio`` は先頭8秒へcropし、2秒未満はrepeatで伸ばす。
            cache生成側でも同じ前処理を通したwaveformを渡すこと。

    Returns:
        ``[256]`` の float32 tensor（CPU）。``F.normalize`` 済みのL2正規化ベクトル
        （``forward`` が返す ``"embedding"``。``"embedding_raw"`` ではない）。
    """
    if getattr(speaker_encoder, "training", False):
        raise ValueError(
            "speaker_encoder must be in eval() mode before caching embeddings "
            "(BatchNorm running stats would be updated otherwise)."
        )
    waveform = _as_mono_batch(waveform_16k, "waveform_16k")
    sample_rate = int(getattr(speaker_encoder, "sample_rate", SPEAKER_SAMPLE_RATE))
    try:
        parameter = next(speaker_encoder.parameters())
        device, dtype = parameter.device, parameter.dtype
    except (AttributeError, StopIteration):
        device, dtype = torch.device("cpu"), torch.float32

    with torch.no_grad():
        output = speaker_encoder(waveform.to(device=device, dtype=dtype), sample_rate)
    embedding = output["embedding"]
    if embedding.dim() != 2 or embedding.size(0) != 1:
        raise RuntimeError(f"Unexpected speaker embedding shape: {tuple(embedding.shape)}")
    if embedding.size(-1) != SPEAKER_EMBEDDING_DIM:
        raise RuntimeError(
            f"Expected speaker embedding dim {SPEAKER_EMBEDDING_DIM}, got {embedding.size(-1)}."
        )
    return embedding[0].detach().to(device="cpu", dtype=torch.float32).contiguous()
