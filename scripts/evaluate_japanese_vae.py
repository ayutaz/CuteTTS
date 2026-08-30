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

"""P1c: 公式Audio VAE（24 kHz / 12.5 Hz / 64-dim）の日本語再構成評価。

docs/japanese-training/08-execution-plan.md の P1c と
docs/japanese-training/06-evaluation-plan.md 第2節を実装する。

TTS本体は読まない。AudioAcousticVAEAdapter の encode -> decode だけで完結し、
「公式VAEをfreezeしたまま日本語へ進んでよいか / S4（Japanese VAE）を検討すべきか」を
判断するための実測値を出す。

測るもの:

* multi-resolution log-mel L1 距離（original vs reconstruction）
* 波形SNRとスペクトル収束（magnitude STFT）
* 公式Speaker Encoder（16 kHz -> 256-dim）でのcosine類似度
* 発話長bucket別・音韻マーク（促音/撥音/長音）別の傾向
* streaming_decode() と offline decode() の差（S1以降のstreaming評価の前提）
* 潜在の統計と、checkpointの (latent + bias) * scaling を当てた後の分布

測らないもの（この環境に依存が無いため。report と open issues に明記する）:

* PESQ / STOI / UTMOS などの品質予測
* ASR CER（日本語ASRを別途用意する必要がある）
* 人手聴取（無声化の欠落判定はここでは自動化できない）

使い方::

    .venv/Scripts/python.exe scripts/evaluate_japanese_vae.py \
        --config configs/japanese/vae-reconstruction.yaml

出力は artifacts/p1c/<timestamp>/ に run.json / env.json / inputs.json /
metrics.json / report.md / samples/ として残る（artifacts/ はgitignore済み。
samples/ は元音声そのものを含むため公開しないこと）。
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

from cutetts.audio_codec.model.speaker_encoder import FbankECAPAStudent
from cutetts.modeling.audio_adapter import AudioAcousticVAEAdapter
from cutetts.runtime import resolve_device
from cutetts.training.artifacts import (
    file_checksum,
    new_run_dir,
    write_metrics,
    write_run_metadata,
)

#: repository root（scripts/ の1階層上）。config内の相対pathはここから解決する。
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Speaker Encoderの入力sample rate（checkpoint configと一致させること）。
SPEAKER_SAMPLE_RATE = 16000

#: 推論側の reference 前処理規約（cutetts.runtime.prepare_reference_audio）。
SPEAKER_MAX_SECONDS = 8.0
MIN_REFERENCE_SECONDS = 2.0

#: 音韻マーク。無声化は表記から判定できないので人手聴取に回す。
PHONETIC_MARKS: dict[str, tuple[str, ...]] = {
    "sokuon": ("っ", "ッ"),
    "hatsuon": ("ん", "ン"),
    "chouon": ("ー", "〜", "～"),
}


# --- 設定 ---------------------------------------------------------------------


@dataclass(frozen=True)
class MelResolution:
    """log-mel距離を測る1解像度分の設定。"""

    n_fft: int
    hop_length: int
    n_mels: int

    @property
    def name(self) -> str:
        return f"n_fft{self.n_fft}_mels{self.n_mels}"


@dataclass(frozen=True)
class CheckpointConfig:
    audio_vae: Path
    speaker_encoder: Path
    tts_weights: Path | None


@dataclass(frozen=True)
class DataConfig:
    moe_zip_dir: Path
    max_speakers: int
    utterances_per_speaker: int
    min_duration_sec: float
    max_duration_sec: float
    duration_bucket_edges_sec: tuple[float, ...]


@dataclass(frozen=True)
class MetricConfig:
    mel_resolutions: tuple[MelResolution, ...]
    stft_n_fft: int
    stft_hop_length: int
    streaming_chunk_latent_frames: int


@dataclass(frozen=True)
class SampleConfig:
    max_pairs: int
    prefer_phonetic_marks: bool


@dataclass(frozen=True)
class Thresholds:
    """判定閾値（すべて提案値）。"""

    mel_l1_max: float = 1.0
    speaker_cosine_min: float = 0.85
    streaming_snr_db_min: float = 40.0
    short_utterance_mel_ratio_max: float = 1.5


@dataclass(frozen=True)
class EvalConfig:
    phase: str
    seed: int
    device: str
    checkpoints: CheckpointConfig
    data: DataConfig
    metrics: MetricConfig
    samples: SampleConfig
    thresholds: Thresholds
    artifact_root: Path
    source_path: Path | None = None


def _resolve_path(value: str | Path, root: Path = REPO_ROOT) -> Path:
    """相対pathを root（既定は repository root）から解決する。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path)


def config_from_mapping(
    payload: dict,
    *,
    root: Path = REPO_ROOT,
    source_path: Path | None = None,
) -> EvalConfig:
    """dict -> EvalConfig（テストからも使う純粋な変換）。未指定keyは既定値で埋める。"""
    checkpoints = dict(payload.get("checkpoints") or {})
    data = dict(payload.get("data") or {})
    metrics = dict(payload.get("metrics") or {})
    samples = dict(payload.get("samples") or {})
    thresholds = dict(payload.get("thresholds") or {})
    output = dict(payload.get("output") or {})

    tts_weights = checkpoints.get("tts_weights")
    resolutions = metrics.get("mel_resolutions") or [
        {"n_fft": 1024, "hop_length": 256, "n_mels": 80}
    ]
    edges = tuple(float(x) for x in data.get("duration_bucket_edges_sec", (2.0, 4.0, 8.0)))
    if list(edges) != sorted(edges):
        raise ValueError("duration_bucket_edges_sec must be sorted ascending.")

    return EvalConfig(
        phase=str(payload.get("phase", "p1c")),
        seed=int(payload.get("seed", 0)),
        device=str(payload.get("device", "auto")),
        checkpoints=CheckpointConfig(
            audio_vae=_resolve_path(
                checkpoints.get("audio_vae", "model/CuteTTS/weights/audio_vae"), root
            ),
            speaker_encoder=_resolve_path(
                checkpoints.get("speaker_encoder", "model/CuteTTS/weights/speaker_encoder"),
                root,
            ),
            tts_weights=_resolve_path(tts_weights, root) if tts_weights else None,
        ),
        data=DataConfig(
            moe_zip_dir=_resolve_path(data.get("moe_zip_dir", "data/raw/moe"), root),
            max_speakers=int(data.get("max_speakers", 16)),
            utterances_per_speaker=int(data.get("utterances_per_speaker", 12)),
            min_duration_sec=float(data.get("min_duration_sec", 1.0)),
            max_duration_sec=float(data.get("max_duration_sec", 20.0)),
            duration_bucket_edges_sec=edges,
        ),
        metrics=MetricConfig(
            mel_resolutions=tuple(
                MelResolution(
                    n_fft=int(item["n_fft"]),
                    hop_length=int(item["hop_length"]),
                    n_mels=int(item["n_mels"]),
                )
                for item in resolutions
            ),
            stft_n_fft=int(metrics.get("stft_n_fft", 1024)),
            stft_hop_length=int(metrics.get("stft_hop_length", 256)),
            streaming_chunk_latent_frames=int(
                metrics.get("streaming_chunk_latent_frames", 2)
            ),
        ),
        samples=SampleConfig(
            max_pairs=int(samples.get("max_pairs", 20)),
            prefer_phonetic_marks=bool(samples.get("prefer_phonetic_marks", True)),
        ),
        thresholds=Thresholds(
            mel_l1_max=float(thresholds.get("mel_l1_max", 1.0)),
            speaker_cosine_min=float(thresholds.get("speaker_cosine_min", 0.85)),
            streaming_snr_db_min=float(thresholds.get("streaming_snr_db_min", 40.0)),
            short_utterance_mel_ratio_max=float(
                thresholds.get("short_utterance_mel_ratio_max", 1.5)
            ),
        ),
        artifact_root=_resolve_path(output.get("artifact_root", "artifacts"), root),
        source_path=source_path,
    )


def load_config(path: str | Path, root: Path = REPO_ROOT) -> EvalConfig:
    """YAMLを EvalConfig にする。"""
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {config_path}.")
    return config_from_mapping(payload, root=root, source_path=config_path)


# --- metric primitives（modelに依存しない純粋関数） -----------------------------


def align_lengths(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """末尾を切って長さを揃える。VAEはhop単位でpaddingするので原音より長くなる。"""
    length = min(int(a.shape[-1]), int(b.shape[-1]))
    return a[..., :length], b[..., :length]


def snr_db(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """波形SNR[dB]。完全一致なら +inf、referenceが無音なら -inf。"""
    ref, est = align_lengths(reference.reshape(-1), estimate.reshape(-1))
    ref = ref.double()
    est = est.double()
    signal_power = float((ref * ref).sum())
    noise = ref - est
    noise_power = float((noise * noise).sum())
    if noise_power <= 0.0:
        return math.inf
    if signal_power <= 0.0:
        return -math.inf
    return 10.0 * math.log10(signal_power / noise_power)


def magnitude_spectrogram(
    waveform: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """|STFT|。短い入力でも落ちないよう center padding は constant にする。"""
    signal = waveform.reshape(-1).float()
    window = torch.hann_window(n_fft, device=signal.device, dtype=signal.dtype)
    spec = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    return spec.abs()


def spectral_convergence(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> float:
    """位相に依存しない振幅一致度 ||R - E||_F / ||R||_F。完全一致なら0。"""
    ref, est = align_lengths(reference.reshape(-1), estimate.reshape(-1))
    ref_mag = magnitude_spectrogram(ref, n_fft, hop_length).double()
    est_mag = magnitude_spectrogram(est, n_fft, hop_length).double()
    denominator = float(torch.linalg.norm(ref_mag))
    if denominator <= 0.0:
        return math.inf
    return float(torch.linalg.norm(ref_mag - est_mag)) / denominator


def build_mel_transform(
    resolution: MelResolution,
    sample_rate: int,
    device: torch.device | str = "cpu",
) -> torchaudio.transforms.MelSpectrogram:
    """log-mel距離用の MelSpectrogram を作る。"""
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=resolution.n_fft,
        win_length=resolution.n_fft,
        hop_length=resolution.hop_length,
        n_mels=resolution.n_mels,
        power=2.0,
        center=True,
        pad_mode="constant",
    ).to(device)


def log_mel_l1(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    mel: torchaudio.transforms.MelSpectrogram,
    eps: float = 1e-5,
) -> float:
    """log-mel の平均L1距離。同一波形なら厳密に0。"""
    ref, est = align_lengths(reference.reshape(-1), estimate.reshape(-1))
    device = mel.mel_scale.fb.device
    ref_mel = torch.log(mel(ref.float().to(device)).clamp_min(eps))
    est_mel = torch.log(mel(est.float().to(device)).clamp_min(eps))
    return float((ref_mel - est_mel).abs().mean())


def multi_resolution_log_mel_l1(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    mels: dict[str, torchaudio.transforms.MelSpectrogram],
) -> dict[str, float]:
    """解像度ごとのlog-mel L1と、その平均（キー ``mean``）。"""
    per_resolution = {
        name: log_mel_l1(reference, estimate, mel) for name, mel in mels.items()
    }
    if per_resolution:
        per_resolution["mean"] = float(
            sum(per_resolution.values()) / len(per_resolution)
        )
    return per_resolution


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """2つのembeddingのcosine類似度。"""
    return float(
        torch.nn.functional.cosine_similarity(
            a.reshape(1, -1).double(), b.reshape(1, -1).double(), dim=-1
        )
    )


def pairwise_cosine_baselines(
    embeddings: torch.Tensor,
    speakers: Sequence[str],
) -> dict[str, list[float]]:
    """original同士のcosineを同一話者ペア / 別話者ペアに分けて返す。

    original vs reconstruction のcosineは、この2つと並べて初めて意味を持つ
    （0.94が高いのか低いのかは、別話者ペアの水準を知らないと判断できない）。
    """
    if embeddings.dim() != 2:
        raise ValueError(f"Expected [N, D] embeddings, got {tuple(embeddings.shape)}.")
    if embeddings.shape[0] != len(speakers):
        raise ValueError("embeddings and speakers must have the same length.")
    normalized = torch.nn.functional.normalize(embeddings.double(), dim=-1)
    similarity = normalized @ normalized.T
    intra: list[float] = []
    inter: list[float] = []
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            value = float(similarity[i, j])
            (intra if speakers[i] == speakers[j] else inter).append(value)
    return {"intra_speaker": intra, "inter_speaker": inter}


def worst_cases(
    records: Sequence[dict],
    field: str,
    count: int,
    largest: bool,
) -> list[dict[str, Any]]:
    """failure例（人手聴取に回す候補）を field の上位/下位から取る。"""
    usable = [record for record in records if isinstance(record.get(field), float)]
    ordered = sorted(usable, key=lambda record: record[field], reverse=largest)
    return [
        {
            "utterance_id": record["utterance_id"],
            "duration_sec": record["duration_sec"],
            "mel_l1": record["mel_l1"],
            "speaker_cosine": record["speaker_cosine"],
            "snr_db": record["snr_db"],
        }
        for record in ordered[:count]
    ]


def rms_db(waveform: torch.Tensor) -> float:
    """RMS[dBFS]。無音は -inf。"""
    signal = waveform.reshape(-1).double()
    power = float((signal * signal).mean()) if signal.numel() else 0.0
    if power <= 0.0:
        return -math.inf
    return 10.0 * math.log10(power)


# --- 集計 ---------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """線形補間のpercentile（q は 0..100）。numpyと同じ定義。"""
    if not values:
        raise ValueError("percentile() requires at least one value.")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be within [0, 100].")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Iterable[float]) -> dict[str, Any]:
    """有限値だけで count/mean/std/min/p50/p95/max を作る。非有限は数だけ残す。"""
    raw = [float(v) for v in values]
    finite = [v for v in raw if math.isfinite(v)]
    summary: dict[str, Any] = {
        "count": len(raw),
        "finite_count": len(finite),
        "non_finite_count": len(raw) - len(finite),
    }
    if not finite:
        summary.update({"mean": None, "std": None, "min": None, "p50": None, "p95": None, "max": None})
        return summary
    mean = sum(finite) / len(finite)
    variance = (
        sum((v - mean) ** 2 for v in finite) / (len(finite) - 1) if len(finite) > 1 else 0.0
    )
    summary.update(
        {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(finite),
            "p50": percentile(finite, 50.0),
            "p95": percentile(finite, 95.0),
            "max": max(finite),
        }
    )
    return summary


def duration_bucket(seconds: float, edges: Sequence[float]) -> str:
    """発話長bucket名。edges=[2,4,8] なら lt2.0 / 2.0-4.0 / 4.0-8.0 / ge8.0。"""
    if not edges:
        return "all"
    if seconds < edges[0]:
        return f"lt{edges[0]:g}"
    for low, high in zip(edges, edges[1:]):
        if seconds < high:
            return f"{low:g}-{high:g}"
    return f"ge{edges[-1]:g}"


def bucket_order(edges: Sequence[float]) -> list[str]:
    """duration_bucket が返しうる名前を短い順に並べる。"""
    if not edges:
        return ["all"]
    names = [f"lt{edges[0]:g}"]
    names.extend(f"{low:g}-{high:g}" for low, high in zip(edges, edges[1:]))
    names.append(f"ge{edges[-1]:g}")
    return names


def phonetic_marks(text: str) -> dict[str, bool]:
    """促音・撥音・長音の有無。無声化は表記から判定できないので含めない。"""
    return {
        name: any(mark in text for mark in marks) for name, marks in PHONETIC_MARKS.items()
    }


def group_summary(
    records: Sequence[dict],
    key: Callable[[dict], str],
    fields: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """recordsを key でまとめ、fields ごとに summarize する。"""
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(key(record), []).append(record)
    result: dict[str, dict[str, Any]] = {}
    for name, items in groups.items():
        entry: dict[str, Any] = {"utterances": len(items)}
        for field_name in fields:
            entry[field_name] = summarize(
                item[field_name] for item in items if item.get(field_name) is not None
            )
        result[name] = entry
    return result


# --- 判定 ---------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdCheck:
    name: str
    value: float | None
    threshold: float
    comparison: str  # "<=" または ">="
    passed: bool
    blocking: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "passed": self.passed,
            "blocking": self.blocking,
        }


def _check(
    name: str,
    value: float | None,
    threshold: float,
    comparison: str,
    blocking: bool,
) -> ThresholdCheck:
    if value is None or not math.isfinite(value):
        passed = False
    elif comparison == "<=":
        passed = value <= threshold
    elif comparison == ">=":
        passed = value >= threshold
    else:
        raise ValueError(f"Unknown comparison: {comparison!r}")
    return ThresholdCheck(name, value, threshold, comparison, passed, blocking)


def run_threshold_checks(
    *,
    mel_l1: float | None,
    speaker_cosine: float | None,
    streaming_snr_db: float | None,
    short_bucket_mel_ratio: float | None,
    thresholds: Thresholds,
) -> list[ThresholdCheck]:
    """判定に使う4つのcheckを作る。純粋関数なのでテストで固定できる。

    streaming SNRが +inf（offline と完全一致）の場合は「合格」とみなす。
    """
    streaming_value = streaming_snr_db
    if streaming_value is not None and streaming_value == math.inf:
        streaming_value = thresholds.streaming_snr_db_min
    return [
        _check("mel_l1_mean", mel_l1, thresholds.mel_l1_max, "<=", blocking=True),
        _check(
            "speaker_cosine_mean",
            speaker_cosine,
            thresholds.speaker_cosine_min,
            ">=",
            blocking=True,
        ),
        _check(
            "streaming_vs_offline_snr_db",
            streaming_value,
            thresholds.streaming_snr_db_min,
            ">=",
            blocking=True,
        ),
        _check(
            "short_bucket_mel_ratio",
            short_bucket_mel_ratio,
            thresholds.short_utterance_mel_ratio_max,
            "<=",
            blocking=False,
        ),
    ]


def verdict_from_checks(checks: Sequence[ThresholdCheck]) -> str:
    """blocking checkが全部通れば freeze_ok、非blockingだけ落ちたら freeze_with_caveats。"""
    if any(not check.passed and check.blocking for check in checks):
        return "consider_s4"
    if any(not check.passed for check in checks):
        return "freeze_with_caveats"
    return "freeze_ok"


VERDICT_LABEL_JA = {
    "freeze_ok": "公式Audio VAEをfreezeしたままS0へ進んでよい",
    "freeze_with_caveats": "freezeで進めてよいが、条件付き（下記の注意を残す）",
    "consider_s4": "freezeのまま進めるのは危険。S4（Japanese VAE）の検討が必要",
}


# --- データ選択 ---------------------------------------------------------------


@dataclass(frozen=True)
class UtteranceRef:
    """moe-speech-plusのzip内1発話への参照。"""

    speaker: str
    zip_path: Path
    wav_member: str
    duration_sec: float
    text: str
    speech_mos: float | None

    @property
    def stem(self) -> str:
        return Path(self.wav_member).stem

    @property
    def utterance_id(self) -> str:
        return f"{self.speaker}/{self.stem}"


def list_speaker_archives(zip_dir: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """読めるzipと、壊れている（DL途中の）zipを分けて返す。"""
    usable: list[Path] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(Path(zip_dir).glob("*.zip")):
        try:
            with zipfile.ZipFile(path) as archive:
                if not any(name.endswith(".wav") for name in archive.namelist()):
                    skipped.append({"zip": path.name, "reason": "no wav member"})
                    continue
        except (zipfile.BadZipFile, OSError) as error:
            skipped.append({"zip": path.name, "reason": f"{type(error).__name__}: {error}"})
            continue
        usable.append(path)
    return usable, skipped


def read_archive_manifest(zip_path: Path) -> list[UtteranceRef]:
    """zip内の {uuid}_NNN.json / .wav ペアを読み、発話一覧を作る（wavは読まない）。"""
    speaker = zip_path.stem
    references: list[UtteranceRef] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for json_member in sorted(name for name in names if name.endswith(".json")):
            wav_member = json_member[: -len(".json")] + ".wav"
            if wav_member not in names:
                continue
            try:
                payload = json.loads(archive.read(json_member).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, zipfile.BadZipFile, OSError):
                continue
            duration = payload.get("duration")
            if duration is None:
                continue
            text = str(
                payload.get("parakeet_jp_transcription")
                or payload.get("anime_whisper_transcription")
                or ""
            )
            mos = payload.get("speechMOS")
            references.append(
                UtteranceRef(
                    speaker=speaker,
                    zip_path=zip_path,
                    wav_member=wav_member,
                    duration_sec=float(duration),
                    text=text,
                    speech_mos=float(mos) if mos is not None else None,
                )
            )
    return references


def stratified_sample(
    items: Sequence[Any],
    bucket_of: Callable[[Any], str],
    count: int,
    rng: random.Random,
) -> list[Any]:
    """bucketをround-robinしながら count 件選ぶ。同じrng・同じ入力順なら決定的。

    bucket名の昇順で回り、各bucket内はrngでshuffleした順に取る。
    ある bucket が尽きたら残りのbucketだけで続ける（総数が優先）。
    """
    if count <= 0:
        return []
    buckets: dict[str, list[Any]] = {}
    for item in items:
        buckets.setdefault(bucket_of(item), []).append(item)
    for values in buckets.values():
        rng.shuffle(values)

    selected: list[Any] = []
    names = sorted(buckets)
    while len(selected) < count and any(buckets[name] for name in names):
        for name in names:
            if not buckets[name]:
                continue
            selected.append(buckets[name].pop())
            if len(selected) >= count:
                break
    return selected


def select_utterances(
    archives: Sequence[Path],
    config: DataConfig,
    seed: int,
) -> tuple[list[UtteranceRef], list[dict[str, Any]]]:
    """話者ごとに発話長でstratified samplingする。seedだけで結果が決まる。"""
    selected: list[UtteranceRef] = []
    per_speaker: list[dict[str, Any]] = []
    for index, zip_path in enumerate(sorted(archives)[: config.max_speakers]):
        references = read_archive_manifest(zip_path)
        eligible = [
            ref
            for ref in references
            if config.min_duration_sec <= ref.duration_sec <= config.max_duration_sec
            and ref.text.strip()
        ]
        rng = random.Random(f"{seed}:{zip_path.stem}")
        picked = stratified_sample(
            sorted(eligible, key=lambda ref: ref.wav_member),
            lambda ref: duration_bucket(ref.duration_sec, config.duration_bucket_edges_sec),
            config.utterances_per_speaker,
            rng,
        )
        picked.sort(key=lambda ref: ref.wav_member)
        selected.extend(picked)
        per_speaker.append(
            {
                "speaker": zip_path.stem,
                "zip": zip_path.name,
                "utterances_in_archive": len(references),
                "eligible": len(eligible),
                "selected": len(picked),
                "index": index,
            }
        )
    return selected, per_speaker


# --- モデル -------------------------------------------------------------------


def load_speaker_encoder(folder: Path, device: torch.device) -> FbankECAPAStudent:
    """runtime.load_runtime と同じ手順でSpeaker Encoderを読む（componentキーをpop）。"""
    from safetensors.torch import load_file

    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if config.pop("component", None) != "speaker_encoder":
        raise ValueError(f"Invalid speaker encoder component: {folder}")
    encoder = FbankECAPAStudent(**config)
    missing, unexpected = encoder.load_state_dict(
        load_file(str(folder / "model.safetensors")), strict=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Speaker encoder did not load strictly: missing={missing}, unexpected={unexpected}"
        )
    return encoder.float().to(device).eval()


def read_latent_normalization(weights_path: Path) -> dict[str, float] | None:
    """tts weightから speech_scaling_factor / speech_bias_factor だけ読む。"""
    try:
        from safetensors import safe_open

        with safe_open(str(weights_path), framework="pt") as handle:
            keys = set(handle.keys())
            if not {"speech_scaling_factor", "speech_bias_factor"} <= keys:
                return None
            return {
                "speech_scaling_factor": float(handle.get_tensor("speech_scaling_factor")),
                "speech_bias_factor": float(handle.get_tensor("speech_bias_factor")),
            }
    except (OSError, ValueError, KeyError):
        return None


def prepare_speaker_input(waveform_24k: torch.Tensor) -> torch.Tensor:
    """推論と同じ規約でSpeaker Encoder入力を作る（先頭8秒 -> 16 kHz、2秒未満はrepeat）。"""
    signal = waveform_24k.reshape(1, -1)
    signal = signal[..., : int(SPEAKER_MAX_SECONDS * 24000)]
    resampled = torchaudio.functional.resample(signal, 24000, SPEAKER_SAMPLE_RATE)
    threshold = int(MIN_REFERENCE_SECONDS * SPEAKER_SAMPLE_RATE)
    if resampled.shape[-1] == 0:
        raise ValueError("Empty waveform for the speaker encoder.")
    if resampled.shape[-1] < threshold:
        repeats = math.ceil((threshold + 1) / resampled.shape[-1])
        resampled = resampled.repeat(1, repeats)
    return resampled.contiguous().float()


def read_waveform_24k(reference: UtteranceRef) -> tuple[torch.Tensor, int]:
    """zipから直接wavを読み、mono float32 / 24 kHz にして返す。"""
    with zipfile.ZipFile(reference.zip_path) as archive:
        payload = archive.read(reference.wav_member)
    data, source_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T, dtype=np.float32)).mean(
        dim=0, keepdim=True
    )
    resampled = torchaudio.functional.resample(waveform, int(source_rate), 24000)
    return resampled.contiguous(), int(source_rate)


def streaming_decode_waveform(
    vae: AudioAcousticVAEAdapter,
    latent: torch.Tensor,
    chunk_frames: int,
) -> torch.Tensor:
    """streaming_decode() で chunk_frames ずつdecodeし、結合した波形を返す。"""
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive.")
    chunks: list[torch.Tensor] = []
    with vae.streaming_decode() as decoder:
        for start in range(0, int(latent.shape[1]), chunk_frames):
            chunk = decoder.decode_chunk(latent[:, start : start + chunk_frames])
            chunks.append(chunk.detach())
    if not chunks:
        return latent.new_zeros((1, 0))
    return torch.cat(chunks, dim=-1)


# --- 1発話の評価 ---------------------------------------------------------------


def evaluate_utterance(
    reference: UtteranceRef,
    *,
    vae: AudioAcousticVAEAdapter,
    speaker_encoder: FbankECAPAStudent,
    mels: dict[str, torchaudio.transforms.MelSpectrogram],
    config: EvalConfig,
    device: torch.device,
    normalization: dict[str, float] | None,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """1発話をencode -> decodeし、record と (original, reconstruction, 話者embedding) を返す。"""
    waveform, source_rate = read_waveform_24k(reference)
    original = waveform.to(device)

    with torch.no_grad():
        latent = vae.encode(original.unsqueeze(0)).mean
        offline = vae.decode(latent)
        offline_wave = offline.squeeze(1) if offline.dim() == 3 else offline
        streaming = streaming_decode_waveform(
            vae, latent, config.metrics.streaming_chunk_latent_frames
        )
        streaming_wave = streaming.squeeze(1) if streaming.dim() == 3 else streaming

        original_embedding = speaker_encoder(
            prepare_speaker_input(original).to(device), SPEAKER_SAMPLE_RATE
        )["embedding"]
        recon_embedding = speaker_encoder(
            prepare_speaker_input(offline_wave).to(device), SPEAKER_SAMPLE_RATE
        )["embedding"]

    original_flat = original.reshape(-1)
    recon_flat = offline_wave.reshape(-1)
    streaming_flat = streaming_wave.reshape(-1)
    aligned_offline, aligned_streaming = align_lengths(recon_flat, streaming_flat)

    mel_distances = multi_resolution_log_mel_l1(original_flat, recon_flat, mels)
    latent_values = latent.detach().float()
    record: dict[str, Any] = {
        "utterance_id": reference.utterance_id,
        "speaker": reference.speaker,
        "wav_member": reference.wav_member,
        "source_sample_rate": source_rate,
        "duration_sec": float(original_flat.shape[-1]) / 24000.0,
        "metadata_duration_sec": reference.duration_sec,
        "duration_bucket": duration_bucket(
            reference.duration_sec, config.data.duration_bucket_edges_sec
        ),
        "text_length": len(reference.text),
        "speech_mos": reference.speech_mos,
        "latent_frames": int(latent.shape[1]),
        "latent_mean": float(latent_values.mean()),
        "latent_std": float(latent_values.std()),
        "latent_abs_max": float(latent_values.abs().max()),
        "mel_l1": float(mel_distances["mean"]),
        "snr_db": snr_db(original_flat, recon_flat),
        "spectral_convergence": spectral_convergence(
            original_flat,
            recon_flat,
            config.metrics.stft_n_fft,
            config.metrics.stft_hop_length,
        ),
        "speaker_cosine": cosine_similarity(original_embedding, recon_embedding),
        "rms_db_original": rms_db(original_flat),
        "rms_db_reconstruction": rms_db(recon_flat),
        "streaming_snr_db": snr_db(recon_flat, streaming_flat),
        "streaming_max_abs_diff": float((aligned_offline - aligned_streaming).abs().max()),
        "streaming_length_delta": int(streaming_flat.shape[-1]) - int(recon_flat.shape[-1]),
    }
    record.update({f"mel_l1_{name}": value for name, value in mel_distances.items() if name != "mean"})
    record.update(phonetic_marks(reference.text))
    if normalization is not None:
        normalized = (
            latent_values + normalization["speech_bias_factor"]
        ) * normalization["speech_scaling_factor"]
        record["normalized_latent_mean"] = float(normalized.mean())
        record["normalized_latent_std"] = float(normalized.std())
        record["normalized_latent_abs_max"] = float(normalized.abs().max())
    return (
        record,
        original_flat.detach().cpu(),
        recon_flat.detach().cpu(),
        original_embedding.detach().reshape(-1).float().cpu(),
    )


# --- samples ------------------------------------------------------------------


def choose_sample_records(
    records: Sequence[dict],
    max_pairs: int,
    prefer_phonetic_marks: bool,
) -> list[str]:
    """samples/ に残すutterance idを決める（話者をround-robinし、音韻マーク持ちを優先）。"""
    if max_pairs <= 0:
        return []

    def rank(record: dict) -> tuple[int, str]:
        marks = sum(bool(record.get(name)) for name in PHONETIC_MARKS)
        priority = -marks if prefer_phonetic_marks else 0
        return (priority, record["utterance_id"])

    by_speaker: dict[str, list[dict]] = {}
    for record in records:
        by_speaker.setdefault(record["speaker"], []).append(record)
    for items in by_speaker.values():
        items.sort(key=rank)

    chosen: list[str] = []
    speakers = sorted(by_speaker)
    while len(chosen) < max_pairs and any(by_speaker[name] for name in speakers):
        for name in speakers:
            if not by_speaker[name]:
                continue
            chosen.append(by_speaker[name].pop(0)["utterance_id"])
            if len(chosen) >= max_pairs:
                break
    return chosen


def write_sample_pair(
    directory: Path,
    utterance_id: str,
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    sample_rate: int = 24000,
) -> dict[str, str]:
    """original / reconstruction を 24 kHz PCM_16 で書き出す。"""
    directory.mkdir(parents=True, exist_ok=True)
    stem = utterance_id.replace("/", "_")
    paths = {}
    for suffix, waveform in (("original", original), ("reconstruction", reconstruction)):
        path = directory / f"{stem}_{suffix}.wav"
        data = waveform.reshape(-1).clamp(-1.0, 1.0).numpy()
        sf.write(str(path), data, sample_rate, subtype="PCM_16")
        paths[suffix] = path.name
    return paths


# --- report -------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isinf(value):
            return "+inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def _summary_row(name: str, summary: dict[str, Any], digits: int = 4) -> str:
    return (
        f"| {name} | {summary.get('finite_count')} | {_fmt(summary.get('mean'), digits)} | "
        f"{_fmt(summary.get('std'), digits)} | {_fmt(summary.get('min'), digits)} | "
        f"{_fmt(summary.get('p50'), digits)} | {_fmt(summary.get('p95'), digits)} | "
        f"{_fmt(summary.get('max'), digits)} |"
    )


def render_report(metrics: dict[str, Any], config: EvalConfig) -> str:
    """metrics.json と同じ内容から report.md を作る。"""
    overall = metrics["overall"]
    dataset = metrics["dataset"]
    verdict = metrics["verdict"]
    lines: list[str] = []
    add = lines.append

    add("# P1c: 公式Audio VAE 日本語再構成評価")
    add("")
    add(f"- phase: `{metrics['phase']}` / seed: `{metrics['seed']}` / device: `{metrics['device']}`")
    add(f"- config: `{metrics.get('config_path')}`")
    add(
        f"- 対象: moe-speech-plus {dataset['speakers']} 話者 / "
        f"{dataset['utterances']} 発話 / 合計 {_fmt(dataset['total_duration_sec'] / 60.0, 2)} 分"
    )
    add(f"- VAE: `{metrics['checkpoints']['audio_vae']}`（24 kHz / 12.5 Hz / 64-dim、freeze）")
    add("- 元音声は 44.1 kHz 16-bit mono。24 kHz へリサンプルしてから encode している。")
    if dataset["skipped_archives"]:
        add(
            "- skipしたzip（ダウンロード途中など）: "
            + ", ".join(
                f"`{item['zip']}` ({item['reason']})" for item in dataset["skipped_archives"]
            )
        )
    add("")

    add("## 判定")
    add("")
    add(f"**{VERDICT_LABEL_JA[verdict['verdict']]}**（`{verdict['verdict']}`）")
    add("")
    add("| check | 実測 | 閾値 | 判定 | blocking |")
    add("|---|---|---|---|---|")
    for check in verdict["checks"]:
        mark = "OK" if check["passed"] else "NG"
        add(
            f"| {check['name']} | {_fmt(check['value'])} | "
            f"{check['comparison']} {_fmt(check['threshold'])} | {mark} | "
            f"{'yes' if check['blocking'] else 'no'} |"
        )
    add("")
    add("閾値は `configs/japanese/vae-reconstruction.yaml` の `thresholds:` にあり、")
    add("すべて **提案値**（この実測が初回なので基準線がまだ無い）。")
    add("")

    add("## 全体")
    add("")
    add("| metric | n | mean | std | min | p50 | p95 | max |")
    add("|---|---|---|---|---|---|---|---|")
    add(_summary_row("multi-resolution log-mel L1", overall["mel_l1"]))
    add(_summary_row("waveform SNR [dB]", overall["snr_db"], 2))
    add(_summary_row("spectral convergence", overall["spectral_convergence"]))
    add(_summary_row("speaker cosine", overall["speaker_cosine"]))
    add(_summary_row("RMS delta [dB]", overall["rms_delta_db"], 3))
    add("")
    reference_block = metrics.get("speaker_reference") or {}
    if reference_block:
        intra = reference_block["intra_speaker"]
        inter = reference_block["inter_speaker"]
        add(
            "speaker cosine の読み方: original同士のcosineは "
            f"**同一話者ペア {_fmt(intra['mean'])}**（n={intra['finite_count']}）、"
            f"**別話者ペア {_fmt(inter['mean'])}**（n={inter['finite_count']}）。"
        )
        recon_mean = overall["speaker_cosine"]["mean"]
        if recon_mean is not None and intra["mean"] is not None and inter["mean"] is not None:
            closer = "同一話者" if abs(recon_mean - intra["mean"]) <= abs(recon_mean - inter["mean"]) else "別話者"
            add(
                f"再構成の {_fmt(recon_mean)} は{closer}ペア側に寄っている"
                f"（同一話者との差 {_fmt(recon_mean - intra['mean'])}、"
                f"別話者との差 {_fmt(recon_mean - inter['mean'])}）。"
            )
        add("")
    add("解像度別 log-mel L1:")
    add("")
    add("| resolution | n | mean | std | min | p50 | p95 | max |")
    add("|---|---|---|---|---|---|---|---|")
    for name, summary in overall["mel_l1_per_resolution"].items():
        add(_summary_row(name, summary))
    add("")

    add("## streaming decode と offline decode の一致")
    add("")
    add(
        f"latent {config.metrics.streaming_chunk_latent_frames} frame ずつ "
        "`streaming_decode()` に流し、offline `decode()` の出力と比較した"
        "（推論は1 patch = 2 latent frame ずつ流す）。"
    )
    add("")
    add("| metric | n | mean | std | min | p50 | p95 | max |")
    add("|---|---|---|---|---|---|---|---|")
    add(_summary_row("streaming vs offline SNR [dB]", overall["streaming_snr_db"], 2))
    add(_summary_row("streaming max abs diff", overall["streaming_max_abs_diff"], 6))
    add("")
    add(
        f"波形長の不一致（streaming - offline）: "
        f"{metrics['streaming']['length_mismatch_count']} / {dataset['utterances']} 発話"
    )
    add("")

    add("## 話者別")
    add("")
    add("| speaker | n | mel L1 | SNR [dB] | speaker cos |")
    add("|---|---|---|---|---|")
    for name in sorted(metrics["by_speaker"]):
        entry = metrics["by_speaker"][name]
        add(
            f"| `{name}` | {entry['utterances']} | {_fmt(entry['mel_l1']['mean'])} | "
            f"{_fmt(entry['snr_db']['mean'], 2)} | {_fmt(entry['speaker_cosine']['mean'])} |"
        )
    add("")

    add("## 発話長別")
    add("")
    add("| bucket [s] | n | mel L1 | SNR [dB] | speaker cos |")
    add("|---|---|---|---|---|")
    for name in metrics["bucket_order"]:
        entry = metrics["by_duration"].get(name)
        if entry is None:
            continue
        add(
            f"| {name} | {entry['utterances']} | {_fmt(entry['mel_l1']['mean'])} | "
            f"{_fmt(entry['snr_db']['mean'], 2)} | {_fmt(entry['speaker_cosine']['mean'])} |"
        )
    add("")
    duration_summary = overall["duration_sec"]
    empty_buckets = [
        name for name in metrics["bucket_order"] if name not in metrics["by_duration"]
    ]
    add(
        f"観測された発話長は {_fmt(duration_summary['min'], 2)} 〜 "
        f"{_fmt(duration_summary['max'], 2)} 秒、"
        f"判定に使った最短bucketは `{verdict.get('short_bucket')}`。"
    )
    if empty_buckets:
        add(
            "空だったbucket: "
            + ", ".join(f"`{name}`" for name in empty_buckets)
            + "。**この長さの発話はこのrunでは評価していない。**"
        )
    add("")

    add("## 音韻マーク別（表記ベース）")
    add("")
    add("| mark | n | mel L1 | SNR [dB] | speaker cos |")
    add("|---|---|---|---|---|")
    for name in sorted(metrics["by_phonetic_mark"]):
        entry = metrics["by_phonetic_mark"][name]
        add(
            f"| {name} | {entry['utterances']} | {_fmt(entry['mel_l1']['mean'])} | "
            f"{_fmt(entry['snr_db']['mean'], 2)} | {_fmt(entry['speaker_cosine']['mean'])} |"
        )
    add("")
    add(
        "促音（っ）・撥音（ん）・長音（ー）は転写テキストの表記から判定している。"
        "**無声化は表記から判定できない**ため、この表には無い。"
    )
    add("")

    add("## 潜在の統計")
    add("")
    add("| metric | n | mean | std | min | p50 | p95 | max |")
    add("|---|---|---|---|---|---|---|---|")
    add(_summary_row("latent std（raw）", overall["latent_std"]))
    add(_summary_row("latent abs max（raw）", overall["latent_abs_max"], 3))
    normalization = metrics.get("latent_normalization")
    if normalization and "normalized_latent_std" in overall:
        add(_summary_row("正規化後 std", overall["normalized_latent_std"]))
        add(_summary_row("正規化後 abs max", overall["normalized_latent_abs_max"], 3))
    add("")
    if normalization:
        add(
            "checkpointの `speech_scaling_factor` = "
            f"{_fmt(normalization['speech_scaling_factor'], 6)}、"
            "`speech_bias_factor` = "
            f"{_fmt(normalization['speech_bias_factor'], 6)}。"
        )
        add(
            "学習側は `(latent + bias) * scaling` を再現する必要がある"
            "（`src/cutetts/modeling/model.py` の `forward_speech_features`）。"
            "正規化後 std が 1 から大きく外れる場合、日本語データの潜在スケールが"
            "公開checkpointの想定と違うことになる。"
        )
    add("")

    worst = metrics.get("worst_cases") or {}
    if worst:
        add("## 失敗例（人手聴取に回す候補）")
        add("")
        add("speaker cosine が低い順:")
        add("")
        add("| utterance | 長さ[s] | mel L1 | speaker cos | SNR [dB] |")
        add("|---|---|---|---|---|")
        for item in worst.get("lowest_speaker_cosine", []):
            add(
                f"| `{item['utterance_id']}` | {_fmt(item['duration_sec'], 2)} | "
                f"{_fmt(item['mel_l1'])} | {_fmt(item['speaker_cosine'])} | "
                f"{_fmt(item['snr_db'], 2)} |"
            )
        add("")
        add("mel L1 が大きい順:")
        add("")
        add("| utterance | 長さ[s] | mel L1 | speaker cos | SNR [dB] |")
        add("|---|---|---|---|---|")
        for item in worst.get("highest_mel_l1", []):
            add(
                f"| `{item['utterance_id']}` | {_fmt(item['duration_sec'], 2)} | "
                f"{_fmt(item['mel_l1'])} | {_fmt(item['speaker_cosine'])} | "
                f"{_fmt(item['snr_db'], 2)} |"
            )
        add("")
        add("これらは `samples/` の選択（話者round-robin + 音韻マーク優先）とは別。")
        add("必要なら `metrics.json` の `utterances` から該当発話を引き当てて聴くこと。")
        add("")

    add("## この評価に含まれていないもの")
    add("")
    add("- PESQ / STOI / UTMOS: 依存（pesq, pystoi, UTMOS）がこの環境に無いため未実装。")
    add("- ASR CER: 日本語ASRを別途固定する必要がある（06章 第2節の指標のうち唯一の未実施）。")
    add("- 人手聴取: 促音・撥音・長音・無声化の欠落、metallic/phase artifact、breath や語尾は")
    add("  `samples/` のペアを日本語話者が聴いて確認すること（08章 P1c のゴール）。")
    add("- 収録条件の分散: 現時点の入力は moe-speech-plus（アニメ調・スタジオ収録）のみ。")
    add("  帯域制限・noise付きのsubsetや gol-dataset は含まれていない。")
    add("")

    add("## artifact")
    add("")
    add("- `metrics.json`: 全発話のper-utterance値と集計")
    add(f"- `samples/`: original / reconstruction のペア {metrics['samples']['pairs']} 件")
    add("  **MoeSpeech LICENSEにより、この音声は1ファイルでも公開してはならない。**")
    add("")
    return "\n".join(lines) + "\n"


# --- main ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="評価設定YAML")
    parser.add_argument("--device", default=None, help="configのdeviceを上書き")
    parser.add_argument("--seed", type=int, default=None, help="configのseedを上書き")
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--utterances-per-speaker", type=int, default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="話者・発話の選択だけ行い、modelを読まずに終了する",
    )
    return parser


def apply_overrides(config: EvalConfig, args: argparse.Namespace) -> EvalConfig:
    """CLIの上書きをconfigへ反映する（frozen dataclassなので作り直す）。"""
    data = config.data
    if args.max_speakers is not None or args.utterances_per_speaker is not None:
        data = DataConfig(
            moe_zip_dir=data.moe_zip_dir,
            max_speakers=args.max_speakers if args.max_speakers is not None else data.max_speakers,
            utterances_per_speaker=(
                args.utterances_per_speaker
                if args.utterances_per_speaker is not None
                else data.utterances_per_speaker
            ),
            min_duration_sec=data.min_duration_sec,
            max_duration_sec=data.max_duration_sec,
            duration_bucket_edges_sec=data.duration_bucket_edges_sec,
        )
    return EvalConfig(
        phase=config.phase,
        seed=args.seed if args.seed is not None else config.seed,
        device=args.device or config.device,
        checkpoints=config.checkpoints,
        data=data,
        metrics=config.metrics,
        samples=config.samples,
        thresholds=config.thresholds,
        artifact_root=(
            _resolve_path(args.artifact_root) if args.artifact_root else config.artifact_root
        ),
        source_path=config.source_path,
    )


def collect_inputs(
    config: EvalConfig,
    archives: Sequence[Path],
    skipped: Sequence[dict[str, str]],
) -> dict[str, Any]:
    """inputs.json の中身（checkpointと入力データの同定情報）。"""
    audio_vae = config.checkpoints.audio_vae
    speaker_encoder = config.checkpoints.speaker_encoder
    return {
        "config_path": str(config.source_path) if config.source_path else None,
        "config_sha256": (
            file_checksum(config.source_path) if config.source_path else None
        ),
        "audio_vae": {
            "path": str(audio_vae),
            "config_sha256": file_checksum(audio_vae / "config.json"),
            "weights_sha256": file_checksum(audio_vae / "model.safetensors"),
        },
        "speaker_encoder": {
            "path": str(speaker_encoder),
            "config_sha256": file_checksum(speaker_encoder / "config.json"),
            "weights_sha256": file_checksum(speaker_encoder / "model.safetensors"),
        },
        "tts_weights": (
            {
                "path": str(config.checkpoints.tts_weights),
                "size_bytes": config.checkpoints.tts_weights.stat().st_size,
            }
            if config.checkpoints.tts_weights and config.checkpoints.tts_weights.exists()
            else None
        ),
        "moe_zip_dir": str(config.data.moe_zip_dir),
        "archives": [
            {
                "zip": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_checksum(path),
            }
            for path in archives
        ],
        "skipped_archives": list(skipped),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = apply_overrides(load_config(args.config), args)

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    archives, skipped = list_speaker_archives(config.data.moe_zip_dir)
    if not archives:
        raise SystemExit(f"No readable speaker archives in {config.data.moe_zip_dir}")
    archives = archives[: config.data.max_speakers]
    references, per_speaker = select_utterances(archives, config.data, config.seed)
    print(
        f"[p1c] archives={len(archives)} skipped={len(skipped)} "
        f"utterances={len(references)}",
        flush=True,
    )
    for item in skipped:
        print(f"[p1c] skip {item['zip']}: {item['reason']}", flush=True)
    if args.dry_run:
        for reference in references:
            print(f"  {reference.utterance_id} {reference.duration_sec:.2f}s")
        return 0
    if not references:
        raise SystemExit("No utterance matched the selection filters.")

    device = resolve_device(config.device)
    print(f"[p1c] device={device}", flush=True)
    vae = AudioAcousticVAEAdapter(config.checkpoints.audio_vae).to(device).eval()
    if int(vae.sample_rate) != 24000:
        raise SystemExit(f"Expected a 24 kHz audio VAE, got {vae.sample_rate}.")
    speaker_encoder = load_speaker_encoder(config.checkpoints.speaker_encoder, device)
    normalization = (
        read_latent_normalization(config.checkpoints.tts_weights)
        if config.checkpoints.tts_weights and config.checkpoints.tts_weights.exists()
        else None
    )
    mels = {
        resolution.name: build_mel_transform(resolution, 24000, device)
        for resolution in config.metrics.mel_resolutions
    }

    run_dir = new_run_dir(config.phase, config.artifact_root)
    write_run_metadata(
        run_dir,
        phase=config.phase,
        command=[sys.executable, *sys.argv],
        seed=config.seed,
        inputs=collect_inputs(config, archives, skipped),
        extra={
            "config_path": str(config.source_path),
            "device": str(device),
            "speakers": [item["speaker"] for item in per_speaker],
            "utterances": len(references),
        },
    )
    print(f"[p1c] run_dir={run_dir}", flush=True)

    records: list[dict[str, Any]] = []
    waveforms: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    embeddings: list[torch.Tensor] = []
    for index, reference in enumerate(references, start=1):
        record, original, reconstruction, embedding = evaluate_utterance(
            reference,
            vae=vae,
            speaker_encoder=speaker_encoder,
            mels=mels,
            config=config,
            device=device,
            normalization=normalization,
        )
        records.append(record)
        waveforms[record["utterance_id"]] = (original, reconstruction)
        embeddings.append(embedding)
        if index % 10 == 0 or index == len(references):
            print(f"[p1c] {index}/{len(references)} evaluated", flush=True)

    metrics = aggregate(
        records,
        per_speaker,
        skipped,
        config,
        device,
        normalization,
        torch.stack(embeddings),
    )

    sample_ids = choose_sample_records(
        records, config.samples.max_pairs, config.samples.prefer_phonetic_marks
    )
    sample_entries = []
    for utterance_id in sample_ids:
        original, reconstruction = waveforms[utterance_id]
        paths = write_sample_pair(run_dir / "samples", utterance_id, original, reconstruction)
        record = next(item for item in records if item["utterance_id"] == utterance_id)
        source = next(item for item in references if item.utterance_id == utterance_id)
        sample_entries.append(
            {
                "utterance_id": utterance_id,
                "files": paths,
                "text": source.text,
                "duration_sec": record["duration_sec"],
                "mel_l1": record["mel_l1"],
                "speaker_cosine": record["speaker_cosine"],
                "sokuon": record["sokuon"],
                "hatsuon": record["hatsuon"],
                "chouon": record["chouon"],
            }
        )
    if sample_entries:
        (run_dir / "samples" / "samples.json").write_text(
            json.dumps(
                {
                    "note": "MoeSpeech LICENSE: この音声は1ファイルでも公開しないこと。",
                    "pairs": sample_entries,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    metrics["samples"] = {"pairs": len(sample_entries), "directory": "samples"}

    write_metrics(run_dir, metrics)
    (run_dir / "report.md").write_text(render_report(metrics, config), encoding="utf-8")

    verdict = metrics["verdict"]
    print(f"[p1c] verdict={verdict['verdict']} ({VERDICT_LABEL_JA[verdict['verdict']]})")
    for check in verdict["checks"]:
        print(
            f"[p1c]   {check['name']}: {_fmt(check['value'])} "
            f"{check['comparison']} {_fmt(check['threshold'])} -> "
            f"{'OK' if check['passed'] else 'NG'}"
        )
    print(f"[p1c] wrote {run_dir / 'metrics.json'} and {run_dir / 'report.md'}")
    return 0


def aggregate(
    records: Sequence[dict],
    per_speaker: Sequence[dict],
    skipped: Sequence[dict],
    config: EvalConfig,
    device: torch.device,
    normalization: dict[str, float] | None,
    original_embeddings: torch.Tensor | None = None,
) -> dict[str, Any]:
    """per-utterance record から metrics.json の中身を作る。"""
    for record in records:
        record["rms_delta_db"] = record["rms_db_reconstruction"] - record["rms_db_original"]

    fields = ["mel_l1", "snr_db", "spectral_convergence", "speaker_cosine"]
    overall: dict[str, Any] = {
        name: summarize(record[name] for record in records)
        for name in [
            *fields,
            "rms_delta_db",
            "streaming_snr_db",
            "streaming_max_abs_diff",
            "latent_std",
            "latent_abs_max",
            "duration_sec",
        ]
    }
    overall["mel_l1_per_resolution"] = {
        resolution.name: summarize(record[f"mel_l1_{resolution.name}"] for record in records)
        for resolution in config.metrics.mel_resolutions
    }
    if normalization is not None:
        for name in ("normalized_latent_std", "normalized_latent_abs_max"):
            overall[name] = summarize(record[name] for record in records)

    by_speaker = group_summary(records, lambda record: record["speaker"], fields)
    by_duration = group_summary(records, lambda record: record["duration_bucket"], fields)
    by_mark: dict[str, dict[str, Any]] = {}
    for mark in PHONETIC_MARKS:
        for flag in (True, False):
            subset = [record for record in records if bool(record[mark]) is flag]
            if not subset:
                continue
            label = f"{mark}={'yes' if flag else 'no'}"
            by_mark[label] = group_summary(subset, lambda _record: label, fields)[label]

    order = bucket_order(config.data.duration_bucket_edges_sec)
    shortest = next((name for name in order if name in by_duration), None)
    overall_mel = overall["mel_l1"]["mean"]
    short_ratio = None
    if shortest is not None and overall_mel:
        short_mean = by_duration[shortest]["mel_l1"]["mean"]
        if short_mean is not None:
            short_ratio = short_mean / overall_mel

    speaker_reference: dict[str, Any] = {}
    if original_embeddings is not None and len(records) > 1:
        baselines = pairwise_cosine_baselines(
            original_embeddings, [record["speaker"] for record in records]
        )
        speaker_reference = {
            "note": (
                "original同士のcosine。original vs reconstruction のcosineは "
                "intra_speaker に近く inter_speaker から離れているほどよい。"
            ),
            "intra_speaker": summarize(baselines["intra_speaker"]),
            "inter_speaker": summarize(baselines["inter_speaker"]),
        }

    checks = run_threshold_checks(
        mel_l1=overall["mel_l1"]["mean"],
        speaker_cosine=overall["speaker_cosine"]["mean"],
        streaming_snr_db=overall["streaming_snr_db"]["mean"],
        short_bucket_mel_ratio=short_ratio,
        thresholds=config.thresholds,
    )

    return {
        "phase": config.phase,
        "seed": config.seed,
        "device": str(device),
        "config_path": str(config.source_path) if config.source_path else None,
        "checkpoints": {
            "audio_vae": str(config.checkpoints.audio_vae),
            "speaker_encoder": str(config.checkpoints.speaker_encoder),
        },
        "dataset": {
            "source": "moe-speech-plus (data/raw/moe/*.zip)",
            "speakers": len({record["speaker"] for record in records}),
            "utterances": len(records),
            "total_duration_sec": sum(record["duration_sec"] for record in records),
            "per_speaker": list(per_speaker),
            "skipped_archives": list(skipped),
            "selection": {
                "seed": config.seed,
                "utterances_per_speaker": config.data.utterances_per_speaker,
                "min_duration_sec": config.data.min_duration_sec,
                "max_duration_sec": config.data.max_duration_sec,
                "duration_bucket_edges_sec": list(config.data.duration_bucket_edges_sec),
                "strategy": "duration-stratified round robin, seeded per speaker",
            },
        },
        "latent_normalization": normalization,
        "speaker_reference": speaker_reference,
        "worst_cases": {
            "lowest_speaker_cosine": worst_cases(records, "speaker_cosine", 5, largest=False),
            "highest_mel_l1": worst_cases(records, "mel_l1", 5, largest=True),
        },
        "overall": overall,
        "by_speaker": by_speaker,
        "by_duration": by_duration,
        "bucket_order": order,
        "by_phonetic_mark": by_mark,
        "streaming": {
            "chunk_latent_frames": config.metrics.streaming_chunk_latent_frames,
            "length_mismatch_count": sum(
                1 for record in records if record["streaming_length_delta"] != 0
            ),
        },
        "verdict": {
            "verdict": verdict_from_checks(checks),
            "label_ja": VERDICT_LABEL_JA[verdict_from_checks(checks)],
            "checks": [check.to_dict() for check in checks],
            "short_bucket": shortest,
            "not_measured": [
                "PESQ",
                "STOI",
                "UTMOS",
                "ASR CER",
                "human listening (無声化/artifact)",
            ],
        },
        "thresholds": {
            "mel_l1_max": config.thresholds.mel_l1_max,
            "speaker_cosine_min": config.thresholds.speaker_cosine_min,
            "streaming_snr_db_min": config.thresholds.streaming_snr_db_min,
            "short_utterance_mel_ratio_max": config.thresholds.short_utterance_mel_ratio_max,
            "status": "提案（初回実測のため基準線なし）",
        },
        "utterances": list(records),
    }


if __name__ == "__main__":
    raise SystemExit(main())
