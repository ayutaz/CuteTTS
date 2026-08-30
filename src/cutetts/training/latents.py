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

"""Audio VAE latent cache（P1e）。

``docs/japanese-training/08-execution-plan.md`` の P1e を実装する。
全音声を1パスで読み、Audio VAE latent（12.5 Hz x 64 dim）を fp16 で保存しておくと、
学習時に元音声（gol全体で 7 TB）が不要になる。fp16 latentなら gol 10,654 h で約 61 GB。

保存形式（決定事項 / ``cache_format`` として meta.json に記録する）
--------------------------------------------------------------------

**シャード化した単一バイナリ + 追記型index** を採用する。

.. code-block:: text

    <root>/
      meta.json            … CacheMeta + cache_format
      shard-00000.bin      … float16 の生バイト列を連結しただけのもの（header無し）
      shard-00000.idx      … "<utterance_id>\\t<offset bytes>\\t<frames>\\n" の追記型TSV
      shard-00001.bin
      shard-00001.idx
      ...

1発話1ファイルにしない理由:
gol-dataset は 7,405,094 発話ある。1発話1ファイルだと NTFS 上に 740万ファイルが並び、
ディレクトリ列挙・バックアップ・削除のいずれもが実用的でなくなる（MFTエントリだけで数GB、
1発話平均 8 KB に対しクラスタ 4 KB 単位の内部断片化も効く）。
シャード1本を既定 1 GiB にすると gol全体でも ``.bin`` 61本 + ``.idx`` 61本 = 122ファイルに収まる。

``.idx`` を「追記型」にしてあるのは、602本のtarを1本ずつ処理する前処理パスが途中で落ちても
**再開できる** ようにするため。``.bin`` へ書いてから ``.idx`` へ1行追記するので、
その間で落ちても ``.bin`` 末尾に参照されないゴミが残るだけで、index側は壊れない
（reader は改行で終わっていない末尾行を「書きかけ」として捨てる）。

index は reader 側で **sorted bytes array + binary search** に畳む。
``dict[str, tuple]`` だと 740万件で 2 GB 超になるが、この形式なら
id 40 byte + shard/offset/frames の int配列で 740万件でも 500 MB 程度に収まる。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

__all__ = [
    "CACHE_FORMAT_VERSION",
    "CacheMeta",
    "CacheMetaMismatch",
    "DEFAULT_MAX_SHARD_BYTES",
    "LATENT_DIM",
    "LATENT_FRAME_RATE",
    "LATENT_HOP_LENGTH",
    "LATENT_SAMPLE_RATE",
    "LatentCacheReader",
    "LatentCacheWriter",
    "PREPROCESSING_VERSION",
    "encode_waveform",
    "expected_latent_frames",
    "resample_to",
]

#: LM が消費する acoustic latent の frame rate（24 kHz / 1920 = 12.5 Hz）。
LATENT_FRAME_RATE: float = 12.5

#: acoustic latent の次元。``model/CuteTTS/config.json`` の ``acoustic_latent_dim``。
LATENT_DIM: int = 64

#: このmoduleが書くcacheの前処理仕様version。encode手順を変えたら必ず上げる。
PREPROCESSING_VERSION: str = "p1e-v1"

#: Audio VAE の入力 sample rate。
LATENT_SAMPLE_RATE: int = 24000

#: ``AudioVAE.preprocess`` が右paddingで揃える単位（24000 / 12.5）。
LATENT_HOP_LENGTH: int = 1920

#: on-disk format の version。file配置やrecord並びを変えたら上げる。
CACHE_FORMAT_VERSION: str = "shard-v1"

#: 1シャードの上限byte数。gol全体（約61 GB）でも ``.bin`` 61本に収まる。
DEFAULT_MAX_SHARD_BYTES: int = 1 << 30

_META_FILENAME = "meta.json"
_SHARD_PREFIX = "shard-"
_SHARD_STEM = _SHARD_PREFIX + "{:05d}"
_SHARD_SUFFIX = ".bin"
_INDEX_SUFFIX = ".idx"
_INDEX_GLOB = _SHARD_PREFIX + "*" + _INDEX_SUFFIX

_STORAGE_DTYPES: dict[str, Any] = {
    "float16": np.float16,
    "float32": np.float32,
}


class CacheMetaMismatch(RuntimeError):
    """既存cacheのmetaが期待と食い違うときに投げる。

    「別のVAE checkpointで作ったlatentを黙って学習に混ぜる」事故を防ぐための例外。
    08-execution-plan.md P1e のゴール「不一致のcacheをloadすると例外になる」に対応する。
    """


@dataclass(frozen=True)
class CacheMeta:
    """latent cacheの素性。``root/meta.json`` に書かれ、load時に照合される。"""

    vae_checkpoint_sha256: str
    preprocessing_version: str
    sample_rate: int
    latent_dim: int
    dtype: str

    def __post_init__(self) -> None:
        if not self.vae_checkpoint_sha256:
            raise ValueError("vae_checkpoint_sha256 must be a non-empty string.")
        if not self.preprocessing_version:
            raise ValueError("preprocessing_version must be a non-empty string.")
        if int(self.sample_rate) <= 0:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate!r}")
        if int(self.latent_dim) <= 0:
            raise ValueError(f"Invalid latent_dim: {self.latent_dim!r}")
        if self.dtype not in _STORAGE_DTYPES:
            raise ValueError(
                f"Unsupported dtype {self.dtype!r}; expected one of {sorted(_STORAGE_DTYPES)}."
            )

    def to_json(self) -> dict:
        return {
            "vae_checkpoint_sha256": str(self.vae_checkpoint_sha256),
            "preprocessing_version": str(self.preprocessing_version),
            "sample_rate": int(self.sample_rate),
            "latent_dim": int(self.latent_dim),
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_json(cls, obj: dict) -> "CacheMeta":
        """``to_json`` の逆。未知のkeyは無視する（前方互換のため）。"""
        if not isinstance(obj, dict):
            raise TypeError(f"Expected a JSON object, got {type(obj).__name__}.")
        required = (
            "vae_checkpoint_sha256",
            "preprocessing_version",
            "sample_rate",
            "latent_dim",
            "dtype",
        )
        missing = [key for key in required if key not in obj]
        if missing:
            raise KeyError(f"CacheMeta is missing required keys: {missing}")
        return cls(
            vae_checkpoint_sha256=str(obj["vae_checkpoint_sha256"]),
            preprocessing_version=str(obj["preprocessing_version"]),
            sample_rate=int(obj["sample_rate"]),
            latent_dim=int(obj["latent_dim"]),
            dtype=str(obj["dtype"]),
        )

    @property
    def storage_dtype(self) -> Any:
        """``dtype`` 文字列に対応する numpy dtype。"""
        return _STORAGE_DTYPES[self.dtype]


def expected_latent_frames(num_samples: int) -> int:
    """24 kHz の ``num_samples`` サンプルが何 latent frame になるか。

    ``AudioVAE.preprocess`` が右paddingで ``hop_length`` の倍数へ揃えるので
    ``ceil(num_samples / 1920)`` になる。collator側のpadding計算で使う。
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    return -(-int(num_samples) // LATENT_HOP_LENGTH)


# --- on-disk shard store -----------------------------------------------------
#
# latent cache と speaker embedding cache は record shape だけが違うので、
# 実体はこの2クラスを共有する（``cutetts.training.speaker_cache`` がここからimportする）。


def _validate_key(utterance_id: str) -> bytes:
    """utterance_id をindex行へ安全に書けるか検査し、utf-8 bytesにする。"""
    if not isinstance(utterance_id, str):
        raise TypeError(f"utterance_id must be str, got {type(utterance_id).__name__}.")
    if not utterance_id:
        raise ValueError("utterance_id must not be empty.")
    if any(char in utterance_id for char in ("\t", "\n", "\r")):
        raise ValueError(f"utterance_id must not contain tab/newline: {utterance_id!r}")
    return utterance_id.encode("utf-8")


def _parse_index_file(path: Path) -> tuple[list[bytes], list[int], list[int]]:
    """1シャード分の ``.idx`` を読む。書きかけの末尾行（改行なし）は捨てる。"""
    keys: list[bytes] = []
    offsets: list[int] = []
    frames: list[int] = []
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                # 前処理パスが書き込み中に落ちた痕跡。どこからも参照されないので捨てる。
                break
            fields = line[:-1].split("\t")
            if len(fields) != 3:
                raise CacheMetaMismatch(
                    f"Corrupted cache index {path} at line {line_no}: expected 3 fields."
                )
            keys.append(fields[0].encode("utf-8"))
            offsets.append(int(fields[1]))
            frames.append(int(fields[2]))
    return keys, offsets, frames


class _KeyIndex:
    """utterance_id -> (shard, offset, frames) の省メモリ索引。

    dictではなく「sortしたbytes配列 + int配列」で持ち、``np.searchsorted`` で引く。
    同じidが複数回書かれている場合は **最後の書き込みが勝つ**。
    """

    __slots__ = ("_keys", "_shards", "_offsets", "_frames")

    def __init__(
        self,
        keys: np.ndarray,
        shards: np.ndarray,
        offsets: np.ndarray,
        frames: np.ndarray,
    ) -> None:
        self._keys = keys
        self._shards = shards
        self._offsets = offsets
        self._frames = frames

    @classmethod
    def build(cls, root: Path) -> "_KeyIndex":
        """``root`` 配下の全 ``.idx`` を読み、shard順（=書き込み順）に畳む。"""
        chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for index_path in sorted(root.glob(_INDEX_GLOB)):
            shard_id = int(index_path.name[len(_SHARD_PREFIX) : -len(_INDEX_SUFFIX)])
            keys, offsets, frames = _parse_index_file(index_path)
            if not keys:
                continue
            chunks.append(
                (
                    np.array(keys, dtype="S"),
                    np.full(len(keys), shard_id, dtype=np.int32),
                    np.asarray(offsets, dtype=np.int64),
                    np.asarray(frames, dtype=np.int64),
                )
            )
        if not chunks:
            return cls(
                np.empty(0, dtype="S1"),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        itemsize = max(chunk[0].dtype.itemsize for chunk in chunks)
        key_dtype = np.dtype(f"S{itemsize}")
        all_keys = np.concatenate([chunk[0].astype(key_dtype) for chunk in chunks])
        all_shards = np.concatenate([chunk[1] for chunk in chunks])
        all_offsets = np.concatenate([chunk[2] for chunk in chunks])
        all_frames = np.concatenate([chunk[3] for chunk in chunks])

        # stable sort なので、同じidの中では「後で書いた行」が後ろに残る。
        order = np.argsort(all_keys, kind="stable")
        sorted_keys = all_keys[order]
        last = np.empty(sorted_keys.shape[0], dtype=bool)
        last[-1] = True
        if sorted_keys.shape[0] > 1:
            last[:-1] = sorted_keys[:-1] != sorted_keys[1:]
        rows = order[last]
        return cls(
            sorted_keys[last],
            all_shards[rows],
            all_offsets[rows],
            all_frames[rows],
        )

    def position(self, utterance_id: str) -> int | None:
        key = _validate_key(utterance_id)
        if self._keys.shape[0] == 0 or len(key) > self._keys.dtype.itemsize:
            return None
        pos = int(np.searchsorted(self._keys, np.bytes_(key)))
        if pos >= self._keys.shape[0] or self._keys[pos] != key:
            return None
        return pos

    def __contains__(self, utterance_id: str) -> bool:
        return self.position(utterance_id) is not None

    def __len__(self) -> int:
        return int(self._keys.shape[0])

    def keys(self) -> Iterator[str]:
        for key in self._keys:
            yield bytes(key).decode("utf-8")

    def locate(self, pos: int) -> tuple[int, int, int]:
        return (
            int(self._shards[pos]),
            int(self._offsets[pos]),
            int(self._frames[pos]),
        )


def _read_meta_json(root: Path) -> dict:
    path = root / _META_FILENAME
    if not path.is_file():
        raise CacheMetaMismatch(f"Cache metadata not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CacheMetaMismatch(f"Cache metadata is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise CacheMetaMismatch(f"Cache metadata must be a JSON object: {path}")
    return payload


class _ShardStoreWriter:
    """追記型シャードへ ``[rows, row_dim]`` のrecordを書く（latent / speaker共用）。"""

    def __init__(
        self,
        root: str | Path,
        meta_payload: dict,
        *,
        row_dim: int,
        storage_dtype: Any,
        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
        on_duplicate: str = "skip",
    ) -> None:
        if on_duplicate not in {"skip", "append"}:
            raise ValueError(f"on_duplicate must be 'skip' or 'append', got {on_duplicate!r}.")
        if int(max_shard_bytes) <= 0:
            raise ValueError("max_shard_bytes must be positive.")

        self._root = Path(root).expanduser()
        self._row_dim = int(row_dim)
        self._dtype = np.dtype(storage_dtype)
        self._max_shard_bytes = int(max_shard_bytes)
        self._on_duplicate = on_duplicate

        if self._root.exists() and (self._root / _META_FILENAME).is_file():
            existing = _read_meta_json(self._root)
            differences = {
                key: (existing.get(key), value)
                for key, value in meta_payload.items()
                if existing.get(key) != value
            }
            if differences:
                raise CacheMetaMismatch(
                    f"Existing cache at {self._root} was produced with different settings: "
                    f"{differences}"
                )
        else:
            if self._root.exists() and any(self._root.glob(_INDEX_GLOB)):
                raise CacheMetaMismatch(
                    f"Cache directory {self._root} has shards but no {_META_FILENAME}."
                )
            self._root.mkdir(parents=True, exist_ok=True)
            (self._root / _META_FILENAME).write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        self._existing = _KeyIndex.build(self._root)
        self._written: set[str] = set()
        self._shard_id = self._latest_shard_id()
        self._binary: Any = None
        self._index: Any = None
        self._offset = 0
        self._closed = False
        self._open_shard()
        # 再開時に最後のシャードが既に上限を超えていたら、書く前に次へ送る。
        self._roll_shard_if_needed()

    # -- shard handling -------------------------------------------------------

    def _shard_paths(self, shard_id: int) -> tuple[Path, Path]:
        stem = _SHARD_STEM.format(shard_id)
        return self._root / (stem + _SHARD_SUFFIX), self._root / (stem + _INDEX_SUFFIX)

    def _latest_shard_id(self) -> int:
        ids = [
            int(path.name[len(_SHARD_PREFIX) : -len(_INDEX_SUFFIX)])
            for path in self._root.glob(_INDEX_GLOB)
        ]
        return max(ids) if ids else 0

    def _open_shard(self) -> None:
        binary_path, index_path = self._shard_paths(self._shard_id)
        self._offset = binary_path.stat().st_size if binary_path.exists() else 0
        self._binary = binary_path.open("ab")
        self._index = index_path.open("a", encoding="utf-8", newline="\n")

    def _roll_shard_if_needed(self) -> None:
        if self._offset < self._max_shard_bytes:
            return
        self._close_handles()
        self._shard_id += 1
        self._open_shard()

    def _close_handles(self) -> None:
        for handle in (self._index, self._binary):
            if handle is None:
                continue
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self._index = None
        self._binary = None

    # -- record handling ------------------------------------------------------

    def _to_storage_array(self, value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(
                f"Expected a 2-D record [rows, {self._row_dim}], got shape {tuple(array.shape)}."
            )
        if array.shape[0] < 1:
            raise ValueError("Record must contain at least one row.")
        if array.shape[1] != self._row_dim:
            raise ValueError(f"Expected last dimension {self._row_dim}, got {array.shape[1]}.")
        if not np.isfinite(array).all():
            raise ValueError("Record contains NaN or Inf; refusing to cache it.")
        return np.ascontiguousarray(array, dtype=self._dtype)

    def write_record(self, utterance_id: str, value: Any) -> None:
        if self._closed:
            raise RuntimeError("Cache writer is closed.")
        _validate_key(utterance_id)
        if self._on_duplicate == "skip" and utterance_id in self:
            return
        array = self._to_storage_array(value)
        payload = array.tobytes(order="C")

        offset = self._offset
        self._binary.write(payload)
        self._offset += len(payload)
        # .bin を書いてから .idx を書く。途中で落ちても .bin 末尾のゴミが残るだけで済む。
        self._index.write(f"{utterance_id}\t{offset}\t{array.shape[0]}\n")
        self._index.flush()
        self._written.add(utterance_id)
        self._roll_shard_if_needed()

    def __contains__(self, utterance_id: str) -> bool:
        return utterance_id in self._written or utterance_id in self._existing

    def __len__(self) -> int:
        return len(self._existing) + sum(
            1 for key in self._written if key not in self._existing
        )

    def flush(self) -> None:
        if self._binary is not None:
            self._binary.flush()
        if self._index is not None:
            self._index.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._close_handles()
        self._closed = True

    def __enter__(self) -> "_ShardStoreWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class _ShardStoreReader:
    """:class:`_ShardStoreWriter` が書いたシャードを読む（latent / speaker共用）。"""

    def __init__(self, root: str | Path, *, row_dim: int, storage_dtype: Any) -> None:
        self._root = Path(root).expanduser()
        self._row_dim = int(row_dim)
        self._dtype = np.dtype(storage_dtype)
        self._index = _KeyIndex.build(self._root)
        self._handles: dict[int, Any] = {}

    def _handle(self, shard_id: int) -> Any:
        handle = self._handles.get(shard_id)
        if handle is None:
            path = self._root / (_SHARD_STEM.format(shard_id) + _SHARD_SUFFIX)
            handle = path.open("rb")
            self._handles[shard_id] = handle
        return handle

    def read_record(self, utterance_id: str) -> np.ndarray:
        pos = self._index.position(utterance_id)
        if pos is None:
            raise KeyError(utterance_id)
        shard_id, offset, frames = self._index.locate(pos)
        nbytes = frames * self._row_dim * self._dtype.itemsize
        handle = self._handle(shard_id)
        handle.seek(offset)
        raw = handle.read(nbytes)
        if len(raw) != nbytes:
            raise RuntimeError(
                f"Cache shard {shard_id} is truncated for {utterance_id!r}: "
                f"expected {nbytes} bytes at offset {offset}, got {len(raw)}."
            )
        flat = np.frombuffer(raw, dtype=self._dtype)
        return flat.reshape(frames, self._row_dim).astype(np.float32)

    def __contains__(self, utterance_id: str) -> bool:
        return utterance_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def keys(self) -> Iterator[str]:
        return self._index.keys()

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "_ShardStoreReader":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __getstate__(self) -> dict:
        # DataLoader worker（Windowsは spawn）へ渡せるよう、開いたfile handleは落とす。
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._handles = {}


def _latent_meta_payload(meta: CacheMeta) -> dict:
    return {
        **meta.to_json(),
        "cache_format": CACHE_FORMAT_VERSION,
        "record_kind": "acoustic_latent",
        "row_dim": int(meta.latent_dim),
        "frame_rate": LATENT_FRAME_RATE,
    }


class LatentCacheWriter:
    """Audio VAE latent（``[T, 64]``）をシャードへ追記する。

    ``root`` が既にあり ``meta.json`` が不一致なら :class:`CacheMetaMismatch` を投げる。
    一致していれば **追記して再開する**（前処理パスをtar単位で再実行できるように）。

    Args:
        root: cacheディレクトリ。無ければ作る。
        meta: このcacheの素性。``latent_dim`` は 64 でなければならない。
        max_shard_bytes: ``.bin`` 1本の上限。既定 1 GiB。
        on_duplicate: ``"skip"``（既定）なら既存idへの ``write`` は何もしない。
            ``"append"`` なら追記し、読み出しは最後に書いたものになる（再encode用）。
    """

    def __init__(
        self,
        root: str | Path,
        meta: CacheMeta,
        *,
        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
        on_duplicate: str = "skip",
    ) -> None:
        if int(meta.latent_dim) != LATENT_DIM:
            raise ValueError(
                f"latent cache expects latent_dim={LATENT_DIM}, got {meta.latent_dim}."
            )
        self.meta = meta
        self._store = _ShardStoreWriter(
            root,
            _latent_meta_payload(meta),
            row_dim=int(meta.latent_dim),
            storage_dtype=meta.storage_dtype,
            max_shard_bytes=max_shard_bytes,
            on_duplicate=on_duplicate,
        )

    def write(self, utterance_id: str, latent) -> None:
        """``latent``（``[T, 64]`` の torch.Tensor / ndarray）を書く。

        既に同じ ``utterance_id`` があり ``on_duplicate="skip"``（既定）なら何もしない。
        """
        self._store.write_record(utterance_id, latent)

    def __contains__(self, utterance_id: str) -> bool:
        return utterance_id in self._store

    def __len__(self) -> int:
        return len(self._store)

    def flush(self) -> None:
        self._store.flush()

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "LatentCacheWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class LatentCacheReader:
    """:class:`LatentCacheWriter` が書いたcacheを読む。

    ``expect`` を渡すと、cacheの ``meta.json`` がそれと違う時点で
    :class:`CacheMetaMismatch` を投げる（別VAEのlatentで学習しないため）。
    """

    def __init__(self, root: str | Path, *, expect: CacheMeta | None = None) -> None:
        self._root = Path(root).expanduser()
        payload = _read_meta_json(self._root)
        try:
            self.meta = CacheMeta.from_json(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise CacheMetaMismatch(
                f"Cache metadata at {self._root} is not a valid CacheMeta: {error}"
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
            row_dim=int(self.meta.latent_dim),
            storage_dtype=self.meta.storage_dtype,
        )

    def __contains__(self, utterance_id: str) -> bool:
        return utterance_id in self._store

    def read(self, utterance_id: str) -> "torch.Tensor":
        """``[T, 64]`` の float32 tensorを返す。未登録なら :class:`KeyError`。"""
        return torch.from_numpy(self._store.read_record(utterance_id))

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> Iterator[str]:
        return self._store.keys()

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "LatentCacheReader":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# --- encode helpers ----------------------------------------------------------


def _as_mono_batch(waveform, name: str) -> "torch.Tensor":
    """``[samples]`` / ``[1, samples]`` を ``[1, samples]`` の float32 に揃える。"""
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(
            f"{name} must be [samples] or [1, samples], got shape {tuple(waveform.shape)}."
        )
    if waveform.size(0) != 1:
        raise ValueError(f"{name} must be mono with a single row, got {waveform.size(0)}.")
    if waveform.size(1) == 0:
        raise ValueError(f"{name} is empty.")
    return waveform.float()


def resample_to(waveform, source_rate: int, target_rate: int) -> "torch.Tensor":
    """``torchaudio.functional.resample`` のラッパ。同じrateなら素通し。

    gol は 48 kHz（2:1）、moe は 44.1 kHz（147:80）から 24 kHz / 16 kHz へ落とす。
    推論側 ``cutetts.runtime._resample`` と同じ実装を使い、cacheと推論で系を揃える。
    """
    if int(source_rate) <= 0 or int(target_rate) <= 0:
        raise ValueError(f"Invalid rates: {source_rate} -> {target_rate}")
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if int(source_rate) == int(target_rate):
        return waveform
    import torchaudio.functional as audio_functional

    return audio_functional.resample(waveform, int(source_rate), int(target_rate))


def encode_waveform(vae, waveform_24k) -> "torch.Tensor":
    """24 kHz mono waveform を Audio VAE の posterior mean へ落とす。

    Args:
        vae: :class:`cutetts.modeling.audio_adapter.AudioAcousticVAEAdapter`。
        waveform_24k: ``[1, samples]`` または ``[samples]`` の float32 mono。

    Returns:
        ``[T, 64]`` の float32 tensor（CPU）。``T = ceil(samples / 1920)``。

    Note:
        **``speech_scaling_factor`` / ``speech_bias_factor`` による正規化はここで適用しない。**
        正規化は :meth:`cutetts.modeling.model.CuteTTSModel.forward_speech_features` が
        forward時に ``(latent + speech_bias_factor) * speech_scaling_factor`` として自分で行う。
        cacheへ正規化後の値を入れると

        1. forward側と二重に適用されてしまう、
        2. cacheがTTS checkpoint側のbufferに縛られる
           （fine-tuneでbufferを更新した瞬間に 61 GB のcacheが無効になる）、
        3. VAE decodeによる再構成評価（P1c）へ再利用できなくなる、

        の3点で不利になる。生のVAE latentを保存し、正規化は学習・推論のforwardに任せる。

        ``AudioVAE`` の posterior は ``posterior_type="sigma"`` で logvar 側にのみ
        ``randn`` を使うため、``mode()`` が返す mean は決定的。
    """
    waveform = _as_mono_batch(waveform_24k, "waveform_24k")
    try:
        parameter = next(vae.parameters())
        device, dtype = parameter.device, parameter.dtype
    except (AttributeError, StopIteration):
        device, dtype = torch.device("cpu"), torch.float32

    with torch.no_grad():
        output = vae.encode(waveform.to(device=device, dtype=dtype))
    latent = output.mean
    if latent.dim() != 3 or latent.size(0) != 1:
        raise RuntimeError(f"Unexpected VAE latent shape: {tuple(latent.shape)}")
    if latent.size(-1) != LATENT_DIM:
        raise RuntimeError(
            f"Expected latent dim {LATENT_DIM}, got {latent.size(-1)}; wrong VAE checkpoint?"
        )
    return latent[0].detach().to(device="cpu", dtype=torch.float32).contiguous()
