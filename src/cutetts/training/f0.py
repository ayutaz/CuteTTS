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

"""F0 をフレーム単位の条件にする（M4c）。

**抑揚は11通りの手段で動かなかった。** 学習設定8軸・データ量19倍・計算量4倍・
参照の与え方（M3b）・テキストへの記号（M4a）のすべてで、人間との一致は
動かない。天井は測ってあり（M2: 輪郭 +0.382 / アクセント 64.5%、現行は
+0.091 / 43.6%）、**伸びしろは実在する**。

診断は「**モデルに韻律を受け取る入口が無い**」。入口は3つしかなく、

* テキストtoken … **従うことは分かった**（核を偽の位置にすると対辞書が
  -5.95pt 動く。R-046）が、**書ける中身が辞書しかない**（人間と 44.8% 一致）
* speaker embedding … 256次元・**時間不変**。構造上、輪郭は運べない
* 参照音声のlatent … **声質と声域は取るが輪郭は取らない**（M3b / R-041）

そこで**時間方向に変化する条件**を作る。

**要点は latent が 12.5 Hz であること。** F0 も 12.5 Hz で取れば latent frame と
1対1に対応するので、**モーラの強制アラインメントが要らない**
（23万発話ぶんは重すぎる）。

    波形 ─(pyworld harvest+stonemask, 10 ms)─> F0 ─(8フレームを中央値)─> 12.5 Hz
                                                                    │
                        話者内で正規化（log2比・±1オクターブで打ち切り）
                                                                    ▼
                                    [有声フラグ, 正規化log F0] の2次元 × patch
                                                                    ▼
                                    :class:`F0Conditioner`（zero-init）→ LM入力へ加算

**zero-init なので学習開始時点は現行と同じ挙動**から始まる（公開checkpointからの
継続学習が壊れない）。

正規化は**発話内の中央値**で行う。声域は speaker embedding が既に持っているので、
ここで持たせると条件が二重になる。**運びたいのは輪郭だけ。**
"""

from __future__ import annotations

import numpy as np
from torch import Tensor, nn

__all__ = [
    "F0_CLIP_OCTAVES",
    "CacheMetaMismatch",
    "F0_ESTIMATOR",
    "F0CacheMeta",
    "F0CacheReader",
    "F0CacheWriter",
    "default_f0_meta",
    "F0_FEATURE_DIM",
    "F0_FRAMES_PER_LATENT",
    "F0_LOOKAHEAD_PATCHES",
    "lookahead_features",
    "F0Conditioner",
    "f0_features",
    "features_from_waveform",
    "load_f0_conditioner",
    "roundtrip_waveform",
    "step_embedding_hook",
    "frame_f0",
    "patch_features",
    "reference_hz",
]

#: 1 latent frame（80 ms）に入る `prosody.FRAME_PERIOD_MS`（10 ms）フレーム数。
F0_FRAMES_PER_LATENT = 8

#: 1フレームの特徴量の次元。``[有声フラグ, 正規化した log2 F0]``。
F0_FEATURE_DIM = 2

#: 正規化した log2 F0 の打ち切り幅（オクターブ）。
#:
#: 人間の実音声60件（有声3,449フレーム）で、発話内の中央値からの隔たりを測った。
#:
#: | パーセンタイル | 隔たり |
#: |---|---:|
#: | 50 | 0.247 |
#: | 90 | 0.697 |
#: | 95 | 0.951 |
#: | **99** | **1.859** |
#: | 最大 | 2.360 |
#:
#: **裾が重い。** 99パーセンタイルが 1.86 オクターブで最大が 2.36 なのは、
#: F0推定のオクターブ誤り（倍音・分数調波に乗る）がほとんどだと見られる。
#: ±1.0 で打ち切ると **4.64%** が当たるが、**打ち切りはその誤りの害を
#: 抑える側に働く**ので広げない（±1.5 なら 2.17%、±0.5 なら 20.93%）。
F0_CLIP_OCTAVES = 1.0


def frame_f0(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """**latent frame（12.5 Hz）ごとの F0**（Hz）。無声は 0。

    `prosody.track_f0`（harvest + stonemask + 局所ジャンプ補正）で 10 ms 間隔を
    取り、8フレームずつ**有声だけの中央値**へ畳む。

    **`dio` を使わない。** 1コアあたり 25.6× 実時間で `harvest` の 5.8 倍速いが、
    測定側（`prosody.track_f0`）と推定器が変わると条件づけと測定で
    別のものを見ることになる。
    """
    from cutetts.training.prosody import track_f0

    dense = np.asarray(track_f0(waveform, sample_rate), dtype=np.float64)
    frames = -(-dense.size // F0_FRAMES_PER_LATENT)
    out = np.zeros(frames, dtype=np.float64)
    for index in range(frames):
        window = dense[index * F0_FRAMES_PER_LATENT:(index + 1) * F0_FRAMES_PER_LATENT]
        voiced = window[window > 0]
        if voiced.size:
            out[index] = float(np.median(voiced))
    return out


def reference_hz(f0_hz: np.ndarray) -> float:
    """正規化の基準（発話内の有声フレームの中央値）。無声だけなら 0。"""
    array = np.asarray(f0_hz, dtype=np.float64)
    voiced = array[array > 0]
    return float(np.median(voiced)) if voiced.size else 0.0


def f0_features(f0_hz: np.ndarray, center_hz: float | None = None) -> np.ndarray:
    """``[T, 2]`` の特徴量（``float32``）。``[有声フラグ, 正規化 log2 F0]``。

    無声フレームは ``[0, 0]``。**有声で中央値と同じ高さのフレームも
    ``[1, 0]`` になる**ので、フラグが無いと無声と区別できない。
    """
    array = np.asarray(f0_hz, dtype=np.float64)
    center = reference_hz(array) if center_hz is None else float(center_hz)
    out = np.zeros((array.size, F0_FEATURE_DIM), dtype=np.float32)
    if center <= 0.0:
        return out
    voiced = array > 0
    out[voiced, 0] = 1.0
    ratio = np.log2(np.maximum(array[voiced], 1e-6) / center)
    out[voiced, 1] = np.clip(ratio, -F0_CLIP_OCTAVES, F0_CLIP_OCTAVES)
    return out


def features_from_waveform(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """波形から ``[T, 2]`` を作る（``T`` は latent frame 数）。"""
    return f0_features(frame_f0(waveform, sample_rate))


def patch_features(features: np.ndarray, patch_size: int, num_patches: int
                   ) -> np.ndarray:
    """フレーム単位の特徴量を **patch 単位**（``[N, patch_size * 2]``）へ。

    足りない分は 0 で埋める（無声と同じ扱い）。余った分は捨てる。
    **patch は latent と同じ並べ方**なので、collator の並びにそのまま乗る。
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != F0_FEATURE_DIM:
        raise ValueError(
            f"features must be [T, {F0_FEATURE_DIM}], got {array.shape}")
    need = int(num_patches) * int(patch_size)
    if array.shape[0] < need:
        array = np.concatenate(
            [array, np.zeros((need - array.shape[0], F0_FEATURE_DIM),
                             dtype=np.float32)], axis=0)
    return array[:need].reshape(int(num_patches), int(patch_size) * F0_FEATURE_DIM)


#: 条件に含める**先読み**の patch 数（M4c の2回目）。
#:
#: **1回目は「その patch の F0」だけを渡して失敗した。** teacher forcing では
#: patch i-1 の真の latent が入力に入っているので、**F0_i は履歴からほぼ
#: 予測できる**。実測で、真の F0 を渡しても flow loss は 0.2% しか下がらず
#: （0.8167 → 0.8149）、patch の順を入れ替えても同じだった（0.8159）。
#: **条件が持つ限界情報が小さく、使う動機が生まれない。**
#:
#: 先読みは履歴に無いので、使う動機が生まれる。
#: 4 patch = 0.64 秒。日本語のアクセント句より短く、モーラ数個ぶん。
F0_LOOKAHEAD_PATCHES = 4


def lookahead_features(patches: np.ndarray, lookahead: int = F0_LOOKAHEAD_PATCHES
                       ) -> np.ndarray:
    """patch ごとの特徴量に**この先 `lookahead` patch ぶん**を連結する。

    ``patches`` は ``[N, patch_size * 2]``。返り値は
    ``[N, patch_size * 2 * lookahead]``。末尾は 0 で埋める（無声と同じ扱い）。

    **i 行目は patch i, i+1, …, i+lookahead-1 の順**に並ぶ。
    ここを逆にすると、過去を渡すことになって意味が反転する。
    """
    array = np.asarray(patches, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"patches must be [N, D], got {array.shape}")
    if lookahead < 1:
        raise ValueError(f"lookahead must be >= 1, got {lookahead}")
    count, width = array.shape
    out = np.zeros((count, width * lookahead), dtype=np.float32)
    for offset in range(lookahead):
        usable = count - offset
        if usable <= 0:
            break
        out[:usable, offset * width:(offset + 1) * width] = array[offset:]
    return out


class F0Conditioner(nn.Module):
    """patch 単位の F0 特徴量を LM の隠れ次元へ写す **zero-init** の線形層。

    **zero-init が要点。** 学習開始時点の出力が 0 なので、
    公開checkpointからの継続学習が「現行と同じ挙動」から始まる。
    普通の初期化にすると、学習の最初に条件が雑音として入って
    既にできている部分を壊す。
    """

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.proj = nn.Linear(self.feature_dim, self.hidden_dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.dim() != 2 or features.size(-1) != self.feature_dim:
            raise ValueError(
                f"features must be [N, {self.feature_dim}], "
                f"got {tuple(features.shape)}")
        return self.proj(features.to(self.proj.weight.dtype))


# --- cache ------------------------------------------------------------------
#
# latent cache / speaker cache と同じ shard store を使う（`row_dim` だけ違う）。
# **latent と同じ並び・同じ長さ**で持つので、collator でそのまま添えられる。

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Iterator  # noqa: E402

import torch  # noqa: E402

from cutetts.training.latents import (  # noqa: E402
    CACHE_FORMAT_VERSION,
    DEFAULT_MAX_SHARD_BYTES,
    PREPROCESSING_VERSION,
    CacheMetaMismatch,
    _read_meta_json,
    _ShardStoreReader,
    _ShardStoreWriter,
    _STORAGE_DTYPES,
)

#: F0 の推定器。**測定側（`prosody.track_f0`）と同じものを使う。**
F0_ESTIMATOR = "harvest+stonemask+jumpfix"


@dataclass(frozen=True)
class F0CacheMeta:
    """F0 cache の素性。

    **`vae_checkpoint_sha256` を持つ。** F0 は latent を decode した波形から
    取るので、**decoder が変われば別のもの**になる。latent cache と同じ値に
    しておくことで、食い違いを load 時に見つけられる。
    """

    vae_checkpoint_sha256: str
    preprocessing_version: str
    feature_dim: int
    frame_period_ms: int
    estimator: str
    dtype: str

    def __post_init__(self) -> None:
        if not self.vae_checkpoint_sha256:
            raise ValueError("vae_checkpoint_sha256 must be a non-empty string.")
        if not self.preprocessing_version:
            raise ValueError("preprocessing_version must be a non-empty string.")
        if int(self.feature_dim) != F0_FEATURE_DIM:
            raise ValueError(
                f"f0 cache expects feature_dim={F0_FEATURE_DIM}, got {self.feature_dim}.")
        if int(self.frame_period_ms) <= 0:
            raise ValueError(f"Invalid frame_period_ms: {self.frame_period_ms!r}")
        if not self.estimator:
            raise ValueError("estimator must be a non-empty string.")
        if self.dtype not in _STORAGE_DTYPES:
            raise ValueError(
                f"Unsupported dtype {self.dtype!r}; "
                f"expected one of {sorted(_STORAGE_DTYPES)}.")

    def to_json(self) -> dict:
        return {
            "vae_checkpoint_sha256": str(self.vae_checkpoint_sha256),
            "preprocessing_version": str(self.preprocessing_version),
            "feature_dim": int(self.feature_dim),
            "frame_period_ms": int(self.frame_period_ms),
            "estimator": str(self.estimator),
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_json(cls, obj: dict) -> "F0CacheMeta":
        if not isinstance(obj, dict):
            raise TypeError(f"Expected a JSON object, got {type(obj).__name__}.")
        required = ("vae_checkpoint_sha256", "preprocessing_version", "feature_dim",
                    "frame_period_ms", "estimator", "dtype")
        missing = [key for key in required if key not in obj]
        if missing:
            raise KeyError(f"F0CacheMeta is missing required keys: {missing}")
        return cls(
            vae_checkpoint_sha256=str(obj["vae_checkpoint_sha256"]),
            preprocessing_version=str(obj["preprocessing_version"]),
            feature_dim=int(obj["feature_dim"]),
            frame_period_ms=int(obj["frame_period_ms"]),
            estimator=str(obj["estimator"]),
            dtype=str(obj["dtype"]),
        )

    @property
    def storage_dtype(self) -> Any:
        return _STORAGE_DTYPES[self.dtype]


def default_f0_meta(vae_checkpoint_sha256: str, *, dtype: str = "float16"
                    ) -> F0CacheMeta:
    """latent cache と対になる F0 cache の meta。"""
    return F0CacheMeta(
        vae_checkpoint_sha256=vae_checkpoint_sha256,
        preprocessing_version=PREPROCESSING_VERSION,
        feature_dim=F0_FEATURE_DIM,
        frame_period_ms=80,
        estimator=F0_ESTIMATOR,
        dtype=dtype,
    )


def _f0_meta_payload(meta: F0CacheMeta) -> dict:
    return {
        **meta.to_json(),
        "cache_format": CACHE_FORMAT_VERSION,
        "record_kind": "f0_features",
        "row_dim": int(meta.feature_dim),
    }


class F0CacheWriter:
    """F0 特徴量（``[T, 2]``）をシャードへ追記する。"""

    def __init__(self, root: str | Path, meta: F0CacheMeta, *,
                 max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
                 on_duplicate: str = "skip") -> None:
        self.meta = meta
        self._store = _ShardStoreWriter(
            root,
            _f0_meta_payload(meta),
            row_dim=int(meta.feature_dim),
            storage_dtype=meta.storage_dtype,
            max_shard_bytes=max_shard_bytes,
            on_duplicate=on_duplicate,
        )

    def write(self, utterance_id: str, features) -> None:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != F0_FEATURE_DIM:
            raise ValueError(
                f"features must be [T, {F0_FEATURE_DIM}], got {array.shape}")
        self._store.write_record(utterance_id, array)

    def __contains__(self, utterance_id: str) -> bool:
        return utterance_id in self._store

    def __len__(self) -> int:
        return len(self._store)

    def flush(self) -> None:
        self._store.flush()

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "F0CacheWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class F0CacheReader:
    """:class:`F0CacheWriter` が書いたcacheを読む。"""

    def __init__(self, root: str | Path, *, expect: F0CacheMeta | None = None) -> None:
        self._root = Path(root).expanduser()
        payload = _read_meta_json(self._root)
        try:
            self.meta = F0CacheMeta.from_json(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise CacheMetaMismatch(
                f"Cache metadata at {self._root} is not a valid F0CacheMeta: {error}"
            ) from error
        if expect is not None and self.meta != expect:
            raise CacheMetaMismatch(
                f"Cache at {self._root} was produced with {self.meta}, expected {expect}.")
        stored_format = payload.get("cache_format")
        if stored_format != CACHE_FORMAT_VERSION:
            raise CacheMetaMismatch(
                f"Cache at {self._root} uses format {stored_format!r}, "
                f"this build reads {CACHE_FORMAT_VERSION!r}.")
        self._store = _ShardStoreReader(
            self._root,
            row_dim=int(self.meta.feature_dim),
            storage_dtype=self.meta.storage_dtype,
        )

    def __contains__(self, utterance_id: str) -> bool:
        return utterance_id in self._store

    def read(self, utterance_id: str) -> "torch.Tensor":
        """``[T, 2]`` の float32 tensor。未登録なら :class:`KeyError`。"""
        return torch.from_numpy(self._store.read_record(utterance_id))

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> Iterator[str]:
        return self._store.keys()

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "F0CacheReader":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# --- 推論側 ------------------------------------------------------------------


def load_f0_conditioner(model_dir: str | Path, device: str = "cpu"
                        ) -> "F0Conditioner | None":
    """`inference/f0_conditioner.safetensors` を読む。無ければ ``None``。

    **`inference/weights` には混ぜない。** あちらは `strict=True` で読まれるので、
    公開checkpointに無いモジュールを置くと読めなくなる。
    """
    from safetensors.torch import load_file

    path = Path(model_dir) / "f0_conditioner.safetensors"
    if not path.is_file():
        return None
    state = load_file(str(path))
    weight = state["proj.weight"]
    module = F0Conditioner(int(weight.shape[1]), int(weight.shape[0]))
    module.load_state_dict(state, strict=True)
    return module.to(device).eval()


def roundtrip_waveform(vae, waveform: np.ndarray) -> np.ndarray:
    """波形を VAE で往復させる。

    **学習側の F0 は latent を decode した波形から取っている。**
    推論で元の波形から取ると、有声判定が 14% 食い違い 26% のフレームが
    1半音以上ずれる（実測）。**条件の分布を揃えるために往復させる。**
    """
    import torch

    from cutetts.training.latents import encode_waveform

    with torch.no_grad():
        wave = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        latent = encode_waveform(vae, wave)
        device = next(vae.parameters()).device
        decoded = vae.decode(latent.T.unsqueeze(0).to(device))
    return decoded.squeeze().float().cpu().numpy().astype(np.float64)


def step_embedding_hook(conditioner, patches, *, device: str = "cpu"):
    """patch ごとの条件を返す hook（`api.generate(extra_step_embedding=)` 用）。

    ``patches`` は ``[N, patch_size * 2]``。範囲を越えた step は ``None`` を
    返して素通りさせる（**条件を繰り返すと、無い高さを指定し続けることになる**）。
    """
    import torch

    if conditioner is None or patches is None or len(patches) == 0:
        return None
    tensor = torch.as_tensor(np.asarray(patches, dtype=np.float32), device=device)

    def hook(step: int):
        if step < 0 or step >= tensor.shape[0]:
            return None
        with torch.no_grad():
            return conditioner(tensor[step:step + 1])

    return hook
