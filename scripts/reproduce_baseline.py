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

"""P0: 推論ベースライン再現スクリプト。

``docs/japanese-training/08-execution-plan.md`` の **P0** を実行する。
このforkと公開checkpointがこの環境で正しく動くことを確認し、以降のすべての比較の
基準線（音声・速度・メモリ）を ``artifacts/p0/<timestamp>/`` に残す。

実行する内容:

* ``{存在するcheckpoint} × {tts, voice_clone} × {offline, streaming}`` の生成
* 各出力の破損検査（NaN / 無音 / 長さ0 / decode上限に張り付いた途中切れ）
* 同一text・同一reference・同一seedでの2回実行による再現性（波形差）
* streaming出力とoffline出力の波形差
* TTFA / RTF / chunk間隔 / peak VRAM の計測
* variant固有制約（distillは ``diffusion_steps ∈ {1,2,4}`` / sway不可 / ``cfg_strength ≤ 5``）
  の実機確認
* sampler compile mode（``full-sampler`` の ``torch.compile`` 経路）が動くかの実機probe
* 日本語textでのcrash有無確認（この時点のcheckpointは日本語未学習なので品質は基準外）

計測は再実装せず :class:`cutetts.demo.metrics.MetricsRecorder` を流用する。
chunk間隔だけはMetricsRecorderが持たないので、chunk受信時刻を別途記録して統計を取る。

sampler compile modeについて（この環境での実測）:
``CuteTTS.from_pretrained`` は非MPSで ``full-sampler`` を選ぶが、
``AudioDiTHead.sample`` の compiled 経路は ``sway_sampling_coefficient == 0.0`` のときだけ
使われる。したがって sway を使う base は eager 相当で動き、sway を使えない **distill は
必ず compiled 経路に入る**。inductor backendはtritonを要求するため、tritonの無い環境
（Windows + torch 2.5.1）ではdistillが全滅する。``--sampler-compile-mode auto``（既定）は
tritonの有無を見て自動的に ``eager`` へ落とし、その事実をmetrics/reportに残す。

使い方::

    .venv/Scripts/python.exe scripts/reproduce_baseline.py
    .venv/Scripts/python.exe scripts/reproduce_baseline.py --max-decode-length 200

**注意**: 公式READMEの「約40 ms / 約9倍 real time」はdistill側・RTX 4090・warm serviceの
公式報告値。ローカル値がこれと異なっても異常とは限らない。比較対象は公式値ではなく、
このP0の自己計測値である。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

__all__ = [
    "CaseSpec",
    "CheckpointSpec",
    "DEFAULT_CHECKPOINT_IDS",
    "TEXTS",
    "build_case_matrix",
    "chunk_interval_stats",
    "discover_checkpoints",
    "inspect_waveform",
    "read_hf_revision",
    "render_report",
    "resolve_sampler_compile_mode",
    "samples_per_patch",
    "waveform_diff",
]

#: ``model/`` 直下で探すcheckpointディレクトリ名（実行計画のbase / distill）。
DEFAULT_CHECKPOINT_IDS: tuple[str, ...] = ("CuteTTS", "CuteTTS-distill")

#: 生成に使うtext。``en`` が基準線、``ja`` はcrash確認用（品質は基準外）。
TEXTS: dict[str, str] = {
    "en": (
        "This recording is the local baseline for CuteTTS inference, "
        "captured before any Japanese continual training begins."
    ),
    "ja": "こんにちは。これは日本語継続学習を始める前の基準線の録音です。",
}

#: 無音判定の閾値。実音声のpeakは 0.1〜0.9 程度なので十分な余裕がある。
SILENCE_PEAK_THRESHOLD = 1e-3
SILENCE_RMS_THRESHOLD = 1e-4

#: clipping判定に使う振幅。
CLIP_THRESHOLD = 0.999

#: 末尾の無音を見る窓（秒）。途中切れの傍証として記録する。
TAIL_WINDOW_SECONDS = 0.2

#: ``--sampler-compile-mode`` に渡せる値。``auto`` 以外は
#: :func:`cutetts.modeling.sampling.set_sampler_compile_mode` へそのまま渡す。
SAMPLER_COMPILE_CHOICES = ("auto", "eager", "euler-only", "full-sampler")

#: probeのerror messageをmetricsへ入れる際の上限文字数（tracebackが非常に長いため）。
PROBE_MESSAGE_LIMIT = 400


# --------------------------------------------------------------------------
# sampler compile mode の解決
# --------------------------------------------------------------------------


def _detect_triton() -> bool:
    """inductor backendが使えるか（＝tritonがあるか）。判定できなければ ``False``。"""
    try:
        from torch.utils._triton import has_triton
    except Exception:  # noqa: BLE001 - 私的APIなのでtorch版によっては無い
        try:
            import triton  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True
    try:
        return bool(has_triton())
    except Exception:  # noqa: BLE001
        return False


def resolve_sampler_compile_mode(
    requested: str,
    device_type: str,
    *,
    has_triton: Callable[[], bool] | None = None,
) -> tuple[str, str]:
    """実際に使うsampler compile modeと、その理由を返す。

    ``CuteTTS.from_pretrained`` は非MPSで ``full-sampler`` を選ぶが、
    その compiled 経路（``AudioDiTHead.sample``）はinductor＝tritonを要求する。
    tritonが無い環境では **sway を使わない生成が全滅する**（distillは常にそこを通る）ので、
    ``auto`` のときは自動で ``eager`` へ落とす。

    Args:
        requested: CLIで指定された値（``auto`` を含む）。
        device_type: ``torch.device.type``（``cuda`` / ``mps`` / ``cpu``）。
        has_triton: triton検出関数。テストから差し替えられるようにしてある。

    Returns:
        ``(mode, reason)``。``reason`` はreportにそのまま載せる日本語の説明。
    """
    if requested not in SAMPLER_COMPILE_CHOICES:
        raise ValueError(
            f"Unsupported sampler compile mode {requested!r}; "
            f"expected one of {list(SAMPLER_COMPILE_CHOICES)}."
        )
    if requested != "auto":
        return requested, "CLIで明示指定された"
    if device_type == "mps":
        return "eager", "MPSではupstream（api.from_pretrained）もeagerに落とす"
    if device_type == "cpu":
        return "eager", "CPUではinductorのGPU kernelを使わないためeagerで統一する"
    detector = _detect_triton if has_triton is None else has_triton
    if detector():
        return "full-sampler", "tritonが利用可能なのでupstream既定のfull-samplerを使う"
    return (
        "eager",
        "tritonが無くinductor backendが使えないためeagerへfallback"
        "（sway=0の生成＝distill全体がfull-samplerでは失敗する）",
    )


# --------------------------------------------------------------------------
# checkpoint探索と入力の記録
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointSpec:
    """1つのcheckpointディレクトリ。"""

    checkpoint_id: str
    path: Path
    variant: str


def read_hf_revision(root: str | Path) -> str | None:
    """``hf download`` が残す ``.cache/huggingface/download/*.metadata`` からcommit hashを読む。

    metadata fileの1行目がrepositoryのcommit hash。存在しない/形式が違う場合は ``None``。
    ``--revision`` を明示せずdownloadしていても、実際に取得したcommitはここから分かる。
    """
    meta = Path(root) / ".cache" / "huggingface" / "download" / "config.json.metadata"
    try:
        first_line = meta.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError, UnicodeDecodeError):
        return None
    if len(first_line) != 40:
        return None
    try:
        int(first_line, 16)
    except ValueError:
        return None
    return first_line


def _read_config(root: Path) -> dict | None:
    """``config.json`` を読む。読めなければ ``None``（呼び出し側でskip理由にする）。"""
    try:
        value = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def samples_per_patch(config: dict) -> int:
    """LM 1 patchあたりのwaveform sample数。

    ``speech_compress_rate``（24 kHz → 12.5 Hz なら 1920）× ``locenc_patch_size``（2）。
    公開baseでは 3840 sample = 0.16 秒。途中切れ判定の分母に使う。
    """
    compress = int(config["processor"]["speech_compress_rate"])
    patch = int(config["architecture"]["locenc_patch_size"])
    if compress <= 0 or patch <= 0:
        raise ValueError("speech_compress_rate and locenc_patch_size must be positive.")
    return compress * patch


def discover_checkpoints(
    model_root: str | Path,
    checkpoint_ids: Sequence[str] = DEFAULT_CHECKPOINT_IDS,
) -> tuple[list[CheckpointSpec], list[dict]]:
    """存在するcheckpointだけを返し、欠けているものは理由つきでskip listへ入れる。

    Returns:
        ``(found, skipped)``。``skipped`` の各要素は ``{"checkpoint_id", "path", "reason"}``。
        **黙って飛ばさない** ため、呼び出し側はskippedをmetrics/reportへ必ず出す。
    """
    root = Path(model_root)
    found: list[CheckpointSpec] = []
    skipped: list[dict] = []
    for checkpoint_id in checkpoint_ids:
        path = root / checkpoint_id
        if not path.is_dir():
            skipped.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "path": str(path),
                    "reason": "ディレクトリが存在しない（未取得）",
                }
            )
            continue
        config = _read_config(path)
        if config is None:
            skipped.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "path": str(path),
                    "reason": "config.json が読めない",
                }
            )
            continue
        missing = [
            rel
            for rel in (
                "weights/tts/model.safetensors",
                "weights/audio_vae/model.safetensors",
                "weights/speaker_encoder/model.safetensors",
            )
            if not (path / rel).is_file()
        ]
        if missing:
            skipped.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "path": str(path),
                    "reason": f"weightが欠けている: {', '.join(missing)}",
                }
            )
            continue
        found.append(
            CheckpointSpec(
                checkpoint_id=checkpoint_id,
                path=path,
                variant=str(config.get("variant", "unknown")),
            )
        )
    return found, skipped


def checkpoint_inputs(spec: CheckpointSpec) -> dict:
    """再現に要る入力情報（revision + 主要fileのchecksum）を集める。"""
    from cutetts.training.artifacts import file_checksum

    relative = (
        "config.json",
        "tokenizer/tokenizer.model",
        "weights/tts/model.safetensors",
        "weights/audio_vae/model.safetensors",
        "weights/speaker_encoder/model.safetensors",
    )
    checksums: dict[str, dict] = {}
    for rel in relative:
        path = spec.path / rel
        if not path.is_file():
            continue
        checksums[rel] = {
            "sha256": file_checksum(path),
            "bytes": path.stat().st_size,
        }
    return {
        "checkpoint_id": spec.checkpoint_id,
        "path": str(spec.path),
        "variant": spec.variant,
        "hf_revision": read_hf_revision(spec.path),
        "files": checksums,
    }


# --------------------------------------------------------------------------
# 波形の検査（torchにのみ依存、modelは不要）
# --------------------------------------------------------------------------


def inspect_waveform(
    waveform,
    sample_rate: int,
    *,
    max_decode_length: int,
    patch_samples: int,
) -> dict:
    """1本の出力波形を破損（無音・NaN・長さ0・途中切れ）について検査する。

    Args:
        waveform: ``[S]`` / ``[1,S]`` / ``[1,1,S]`` のいずれかのtorch tensor。
        sample_rate: 波形のsample rate。
        max_decode_length: 生成に使ったdecode上限（patch数）。
        patch_samples: 1 patchあたりのsample数（:func:`samples_per_patch`）。

    Returns:
        検査結果のdict。``ok`` が ``True`` のときだけ「破損なし」と扱う。
        ``hit_decode_limit`` は stop predictor が発火せず上限で打ち切られたことを示し、
        **途中切れの疑い** として ``ok`` を落とす。
    """
    import torch

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if patch_samples <= 0:
        raise ValueError("patch_samples must be positive.")
    if max_decode_length <= 0:
        raise ValueError("max_decode_length must be positive.")

    tensor = waveform.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    samples = int(tensor.numel())
    if samples == 0:
        return {
            "samples": 0,
            "duration_seconds": 0.0,
            "finite": True,
            "peak_amplitude": 0.0,
            "rms": 0.0,
            "tail_rms": 0.0,
            "clipped_fraction": 0.0,
            "decoded_patches": 0.0,
            "hit_decode_limit": False,
            "silent": True,
            "empty": True,
            "ok": False,
            "problems": ["empty"],
        }

    finite = bool(torch.isfinite(tensor).all().item())
    if finite:
        peak = float(tensor.abs().max().item())
        rms = float(tensor.pow(2).mean().sqrt().item())
        tail = tensor[-max(1, int(TAIL_WINDOW_SECONDS * sample_rate)) :]
        tail_rms = float(tail.pow(2).mean().sqrt().item())
        clipped = float((tensor.abs() > CLIP_THRESHOLD).float().mean().item())
    else:
        peak = rms = tail_rms = clipped = float("nan")

    silent = finite and (peak < SILENCE_PEAK_THRESHOLD or rms < SILENCE_RMS_THRESHOLD)
    decoded_patches = samples / patch_samples
    hit_limit = decoded_patches >= max_decode_length - 0.5

    problems: list[str] = []
    if not finite:
        problems.append("non_finite")
    if silent:
        problems.append("silent")
    if hit_limit:
        problems.append("hit_decode_limit")

    return {
        "samples": samples,
        "duration_seconds": samples / sample_rate,
        "finite": finite,
        "peak_amplitude": peak,
        "rms": rms,
        "tail_rms": tail_rms,
        "clipped_fraction": clipped,
        "decoded_patches": decoded_patches,
        "hit_decode_limit": hit_limit,
        "silent": silent,
        "empty": False,
        "ok": not problems,
        "problems": problems,
    }


def waveform_diff(left, right) -> dict:
    """2本の波形の差。長さが違う場合は短い方までを比較し、長さ差も残す。"""
    import torch

    a = left.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    b = right.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    length = min(int(a.numel()), int(b.numel()))
    if length == 0:
        return {
            "left_samples": int(a.numel()),
            "right_samples": int(b.numel()),
            "length_delta_samples": int(a.numel()) - int(b.numel()),
            "compared_samples": 0,
            "identical": bool(a.numel() == b.numel() == 0),
            "max_abs_diff": None,
            "rms_diff": None,
        }
    delta = a[:length] - b[:length]
    return {
        "left_samples": int(a.numel()),
        "right_samples": int(b.numel()),
        "length_delta_samples": int(a.numel()) - int(b.numel()),
        "compared_samples": length,
        "identical": bool(a.numel() == b.numel() and torch.equal(a, b)),
        "max_abs_diff": float(delta.abs().max().item()),
        "rms_diff": float(delta.pow(2).mean().sqrt().item()),
    }


def chunk_interval_stats(timestamps: Sequence[float]) -> dict:
    """streaming chunkの受信時刻からchunk間隔の統計を作る。

    ``MetricsRecorder`` はTTFA / RTF / patch数を持つがchunk間隔は持たないため、
    ここだけ受信時刻の差分から算出する（TTFA・RTFは再実装しない）。
    """
    ordered = [float(value) for value in timestamps]
    if len(ordered) < 2:
        return {
            "chunk_count": len(ordered),
            "interval_count": 0,
            "mean_seconds": None,
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "p95_seconds": None,
        }
    intervals = [second - first for first, second in zip(ordered, ordered[1:])]
    ranked = sorted(intervals)
    index = min(len(ranked) - 1, int(round(0.95 * (len(ranked) - 1))))
    return {
        "chunk_count": len(ordered),
        "interval_count": len(intervals),
        "mean_seconds": statistics.fmean(intervals),
        "median_seconds": statistics.median(intervals),
        "min_seconds": min(intervals),
        "max_seconds": max(intervals),
        "p95_seconds": ranked[index],
    }


# --------------------------------------------------------------------------
# 実行するcaseの定義
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseSpec:
    """1回の生成。"""

    checkpoint_id: str
    mode: str
    transport: str
    text_id: str
    run_index: int = 1
    purpose: str = "matrix"

    @property
    def case_id(self) -> str:
        return (
            f"{self.checkpoint_id}__{self.mode}__{self.transport}"
            f"__{self.text_id}__run{self.run_index}"
        )


def build_case_matrix(
    checkpoint_id: str,
    *,
    repeat_for_reproducibility: bool = True,
    include_japanese: bool = True,
) -> list[CaseSpec]:
    """1 checkpoint分の実行caseを組み立てる。

    ``{tts, voice_clone} × {offline, streaming}`` の4本が実行計画のゴール。
    そこへ再現性確認のためのoffline 2回目と、日本語crash確認の1本を足す。
    """
    cases: list[CaseSpec] = []
    for mode in ("tts", "voice_clone"):
        for transport in ("offline", "streaming"):
            cases.append(
                CaseSpec(
                    checkpoint_id=checkpoint_id,
                    mode=mode,
                    transport=transport,
                    text_id="en",
                )
            )
    if repeat_for_reproducibility:
        for mode in ("tts", "voice_clone"):
            cases.append(
                CaseSpec(
                    checkpoint_id=checkpoint_id,
                    mode=mode,
                    transport="offline",
                    text_id="en",
                    run_index=2,
                    purpose="reproducibility",
                )
            )
    if include_japanese:
        cases.append(
            CaseSpec(
                checkpoint_id=checkpoint_id,
                mode="tts",
                transport="offline",
                text_id="ja",
                purpose="japanese_smoke",
            )
        )
    return cases


# --------------------------------------------------------------------------
# 生成の実行
# --------------------------------------------------------------------------


@dataclass
class CaseOutcome:
    """1 caseの記録と、比較に使う波形（失敗時は ``None``）。"""

    record: dict
    waveform: Any = None


def _synchronize(device) -> None:
    """計測境界でdevice queueを空にする（非同期実行で時間が歪まないように）。"""
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _reset_peak_memory(device) -> int:
    """peak統計をリセットし、その時点の常駐allocationを返す（CUDA以外は0）。"""
    import torch

    if device.type != "cuda":
        return 0
    torch.cuda.reset_peak_memory_stats()
    return int(torch.cuda.memory_allocated())


def _peak_memory(device) -> int | None:
    import torch

    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated())


def _generate_kwargs(spec: CaseSpec, options: dict) -> dict:
    return {
        "mode": spec.mode,
        "reference_audio": options["reference_audio"] if spec.mode == "voice_clone" else None,
        "cfg_strength": options["cfg_strength"],
        "diffusion_steps": options["diffusion_steps"],
        "diffusion_sway_coefficient": options["diffusion_sway_coefficient"],
        "max_decode_length": options["max_decode_length"],
        "seed": options["seed"],
        "show_progress": False,
    }


def _run_offline(model, spec: CaseSpec, text: str, options: dict) -> tuple[Any, dict]:
    """``CuteTTS.generate`` を1回。TTFA/RTFは MetricsRecorder に計算させる。

    offlineでは途中でPCMが出ないので、完成した波形を1 chunkとして
    ``MetricsRecorder`` に渡す。結果として ``ttfa_seconds`` は
    「最初の音が手に入るまで＝生成全体の時間」になる（定義上そうなる）。
    非有限sampleは ``MetricsRecorder.add_chunk`` が ``ValueError`` で弾く。
    """
    from cutetts.demo.metrics import MetricsRecorder

    device = model.runtime.model.device
    _synchronize(device)
    t0 = time.perf_counter()
    result = model.generate(text, **_generate_kwargs(spec, options))
    _synchronize(device)
    t_end = time.perf_counter()

    recorder = MetricsRecorder(t0, result.sample_rate)
    recorder.add_chunk(result.waveform, timestamp=t_end)
    metrics = recorder.finish().to_dict()
    metrics["wall_seconds"] = t_end - t0
    metrics["ttfa_note"] = "offlineは完成まで音が出ないため ttfa == generation time"
    metrics["chunk_intervals"] = chunk_interval_stats([t_end])
    return recorder.waveform(), metrics


def _run_streaming(model, spec: CaseSpec, text: str, options: dict) -> tuple[Any, dict]:
    """``CuteTTS.generate_stream`` を1回。chunk受信時刻からTTFAとchunk間隔を出す。"""
    from cutetts.demo.metrics import MetricsRecorder

    device = model.runtime.model.device
    _synchronize(device)
    t0 = time.perf_counter()
    recorder = MetricsRecorder(t0, model.runtime.sample_rate)
    timestamps: list[float] = []
    for chunk in model.generate_stream(text, **_generate_kwargs(spec, options)):
        now = time.perf_counter()
        timestamps.append(now)
        recorder.add_chunk(chunk.waveform, timestamp=now)
    _synchronize(device)
    t_end = time.perf_counter()

    metrics = recorder.finish().to_dict()
    metrics["wall_seconds"] = t_end - t0
    metrics["chunk_intervals"] = chunk_interval_stats(timestamps)
    return recorder.waveform(), metrics


def _save_wave(path: Path, waveform, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform.detach().cpu().float().reshape(-1).numpy(), int(sample_rate))


def run_case(
    model,
    spec: CaseSpec,
    options: dict,
    *,
    samples_dir: Path,
    patch_samples: int,
) -> CaseOutcome:
    """1 caseを実行する。例外は握りつぶさず ``record["error"]`` に残して返す。"""
    device = model.runtime.model.device
    resident = _reset_peak_memory(device)
    record: dict = {
        "case_id": spec.case_id,
        "checkpoint_id": spec.checkpoint_id,
        "variant": model.variant,
        "mode": spec.mode,
        "transport": spec.transport,
        "text_id": spec.text_id,
        "run_index": spec.run_index,
        "purpose": spec.purpose,
        "status": "error",
        "resident_vram_bytes": resident,
    }
    text = options["texts"][spec.text_id]
    runner: Callable[..., tuple[Any, dict]] = (
        _run_streaming if spec.transport == "streaming" else _run_offline
    )
    try:
        waveform, metrics = runner(model, spec, text, options)
    except BaseException as error:  # noqa: BLE001 - 原因を残して次のcaseへ進む
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        record["peak_vram_bytes"] = _peak_memory(device)
        return CaseOutcome(record=record)

    record["peak_vram_bytes"] = _peak_memory(device)
    record["metrics"] = metrics
    record["inspection"] = inspect_waveform(
        waveform,
        model.runtime.sample_rate,
        max_decode_length=options["max_decode_length"],
        patch_samples=patch_samples,
    )
    record["status"] = "ok" if record["inspection"]["ok"] else "corrupt"
    sample_path = samples_dir / f"{spec.case_id}.wav"
    _save_wave(sample_path, waveform, model.runtime.sample_rate)
    record["sample_file"] = f"samples/{sample_path.name}"
    return CaseOutcome(record=record, waveform=waveform)


def check_variant_constraints(model, checkpoint_id: str, options: dict) -> list[dict]:
    """variant固有の入力制約を実機で確認する（P0の判断ゲート）。

    ``api.generate`` はパラメータ検証を重い処理の前に行うので、
    ここでの呼び出しは即座に ``ValueError`` になり計算コストは無視できる。
    """
    text = options["texts"]["en"]
    if model.variant == "distill":
        probes = [
            ("diffusion_steps=3 は拒否される", {"diffusion_steps": 3}),
            ("sway sampling は拒否される", {"diffusion_sway_coefficient": -0.8}),
            ("cfg_strength=5.5 は拒否される", {"cfg_strength": 5.5}),
        ]
    else:
        probes = [
            ("sway が定義域外なら拒否される", {"diffusion_sway_coefficient": -2.0}),
            ("diffusion_steps=0 は拒否される", {"diffusion_steps": 0}),
        ]
    base_kwargs = {
        "mode": "tts",
        "cfg_strength": options["cfg_strength"],
        "diffusion_steps": options["diffusion_steps"],
        "diffusion_sway_coefficient": options["diffusion_sway_coefficient"],
        "max_decode_length": options["max_decode_length"],
        "seed": options["seed"],
        "show_progress": False,
    }
    results: list[dict] = []
    for name, override in probes:
        kwargs = dict(base_kwargs)
        kwargs.update(override)
        entry = {
            "checkpoint_id": checkpoint_id,
            "variant": model.variant,
            "check": name,
            "override": dict(override),
            "raised_value_error": False,
            "message": None,
        }
        try:
            model.generate(text, **kwargs)
        except ValueError as error:
            entry["raised_value_error"] = True
            entry["message"] = str(error)
        except BaseException as error:  # noqa: BLE001
            entry["message"] = f"{type(error).__name__}: {error}"
        results.append(entry)
    return results


def probe_full_sampler(
    model,
    checkpoint_id: str,
    options: dict,
    *,
    restore_mode: str,
) -> dict:
    """``full-sampler`` の ``torch.compile`` 経路が実際に動くかを1本だけ試す。

    compiled 経路は ``sway_sampling_coefficient == 0.0`` のときだけ選ばれるので、
    baseでも sway を 0.0 に固定して **必ず compiled 経路へ入れる**。
    結果（成功 / 例外の型とmessage）を記録し、最後に ``restore_mode`` へ戻す。

    これは計測ではなく環境判定なので、失敗しても run 全体は続行する。
    """
    from cutetts.modeling.sampling import set_sampler_compile_mode

    entry: dict = {
        "checkpoint_id": checkpoint_id,
        "variant": model.variant,
        "probed_mode": "full-sampler",
        "forced_sway": 0.0,
        "succeeded": False,
        "seconds": None,
        "error_type": None,
        "error_message": None,
    }
    set_sampler_compile_mode("full-sampler")
    t0 = time.perf_counter()
    try:
        model.generate(
            options["texts"]["en"],
            mode="tts",
            cfg_strength=options["cfg_strength"],
            diffusion_steps=options["diffusion_steps"],
            diffusion_sway_coefficient=0.0,
            max_decode_length=options["max_decode_length"],
            seed=options["seed"],
            show_progress=False,
        )
    except BaseException as error:  # noqa: BLE001 - 環境判定なので握って記録する
        entry["error_type"] = type(error).__name__
        entry["error_message"] = " ".join(str(error).split())[:PROBE_MESSAGE_LIMIT]
    else:
        entry["succeeded"] = True
        entry["seconds"] = time.perf_counter() - t0
    finally:
        set_sampler_compile_mode(restore_mode)
    return entry


def run_distill_step_sweep(
    model,
    checkpoint_id: str,
    options: dict,
    *,
    samples_dir: Path,
    patch_samples: int,
) -> list[dict]:
    """distillの ``diffusion_steps ∈ {1,2,4}`` で速度基準を取る（P0の判断ゲート）。"""
    device = model.runtime.model.device
    text = options["texts"]["en"]
    entries: list[dict] = []
    for steps in (1, 2, 4):
        _reset_peak_memory(device)
        _synchronize(device)
        t0 = time.perf_counter()
        try:
            result = model.generate(
                text,
                mode="tts",
                cfg_strength=options["cfg_strength"],
                diffusion_steps=steps,
                max_decode_length=options["max_decode_length"],
                seed=options["seed"],
                show_progress=False,
            )
        except BaseException as error:  # noqa: BLE001
            entries.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "diffusion_steps": steps,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        _synchronize(device)
        elapsed = time.perf_counter() - t0
        inspection = inspect_waveform(
            result.waveform,
            result.sample_rate,
            max_decode_length=options["max_decode_length"],
            patch_samples=patch_samples,
        )
        name = f"{checkpoint_id}__steps{steps}__tts__offline__en.wav"
        _save_wave(samples_dir / name, result.waveform, result.sample_rate)
        duration = inspection["duration_seconds"]
        entries.append(
            {
                "checkpoint_id": checkpoint_id,
                "diffusion_steps": steps,
                "generation_seconds": elapsed,
                "audio_duration_seconds": duration,
                "rtf": (elapsed / duration) if duration > 0 else None,
                "peak_vram_bytes": _peak_memory(device),
                "ok": inspection["ok"],
                "problems": inspection["problems"],
                "sample_file": f"samples/{name}",
            }
        )
    return entries


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _gib(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value / (1024 ** 3):.2f}"


def render_report(payload: dict) -> str:
    """``metrics.json`` 相当のpayloadから ``report.md`` の本文を作る。"""
    settings = payload.get("settings", {})
    lines: list[str] = []
    lines.append("# P0: 推論ベースライン再現 report")
    lines.append("")
    lines.append(f"生成時刻: {payload.get('generated_at', '—')}")
    lines.append("")
    lines.append("## この文書の読み方（重要）")
    lines.append("")
    lines.append(
        "ここに並ぶ TTFA / RTF / peak VRAM は **この環境の自己計測値** であり、"
        "以降のすべての比較の基準線です。"
    )
    lines.append("")
    lines.append(
        "公式READMEの「約40 ms / 約9倍 real time」は distill 側・RTX 4090・warm service の"
        "公式報告値です。**ローカル値がこれと異なっても異常ではありません。**"
        "比較対象は公式値ではなく、このP0の自己計測値です。"
    )
    lines.append("")
    lines.append(
        "cold start（各checkpointの1本目）はCUDA kernelのwarmupを含むため遅く出ます。"
        "同一checkpoint内では後半のcaseほど速いのが正常です。"
    )
    lines.append("")

    lines.append("## 設定")
    lines.append("")
    lines.append("| 項目 | 値 |")
    lines.append("|---|---|")
    for key in (
        "device",
        "seed",
        "max_decode_length",
        "cfg_strength",
        "diffusion_steps",
        "diffusion_sway_coefficient",
        "sampler_compile_mode_request",
        "sampler_compile_mode",
        "reference_audio",
    ):
        lines.append(f"| {key} | {_fmt(settings.get(key))} |")
    for text_id, text in (settings.get("texts") or {}).items():
        lines.append(f"| text[{text_id}] | {text} |")
    lines.append("")
    if settings.get("sampler_compile_mode_reason"):
        lines.append(
            f"sampler compile mode の選択理由: {settings['sampler_compile_mode_reason']}"
        )
        lines.append("")

    lines.append("## checkpoint")
    lines.append("")
    lines.append("| checkpoint | variant | hf revision | load(s) |")
    lines.append("|---|---|---|---:|")
    load_times = payload.get("model_load_seconds") or {}
    for entry in (payload.get("checkpoints") or {}).values():
        checkpoint_id = entry.get("checkpoint_id")
        lines.append(
            f"| `{checkpoint_id}` | {entry.get('variant')} | "
            f"`{entry.get('hf_revision') or '—'}` | "
            f"{_fmt(load_times.get(checkpoint_id), 1)} |"
        )
    lines.append("")

    skipped = payload.get("skipped_checkpoints") or []
    lines.append("### skipしたcheckpoint")
    lines.append("")
    if skipped:
        for entry in skipped:
            lines.append(
                f"- `{entry['checkpoint_id']}` (`{entry['path']}`): {entry['reason']} "
                "→ **このrunでは未検証**。P0ゲートはこのcheckpointについて未達です。"
            )
    else:
        lines.append("なし（対象checkpointはすべて検証しました）。")
    lines.append("")

    lines.append("## 生成case（破損検査つき）")
    lines.append("")
    lines.append(
        "| case | mode | transport | text | status | 長さ(s) | TTFA(s) | RTF | "
        "peak VRAM(GiB) | chunk数 | chunk間隔中央値(s) | 問題 |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for case in payload.get("cases", []):
        metrics = case.get("metrics") or {}
        inspection = case.get("inspection") or {}
        intervals = metrics.get("chunk_intervals") or {}
        if case["status"] == "error":
            note = f"ERROR: {(case.get('error') or {}).get('type')}"
        else:
            note = ", ".join(inspection.get("problems") or []) or "なし"
        lines.append(
            "| `{case}` | {mode} | {transport} | {text} | {status} | {dur} | {ttfa} | "
            "{rtf} | {vram} | {chunks} | {median} | {note} |".format(
                case=case["case_id"],
                mode=case["mode"],
                transport=case["transport"],
                text=case["text_id"],
                status=case["status"],
                dur=_fmt(inspection.get("duration_seconds"), 2),
                ttfa=_fmt(metrics.get("ttfa_seconds")),
                rtf=_fmt(metrics.get("rtf")),
                vram=_gib(case.get("peak_vram_bytes")),
                chunks=_fmt(intervals.get("chunk_count")),
                median=_fmt(intervals.get("median_seconds")),
                note=note,
            )
        )
    lines.append("")
    lines.append(
        "- offline case の TTFA は「完成した波形が手に入るまで」＝生成全体の時間です"
        "（途中でPCMが出ないため定義上そうなります）。streaming case の TTFA だけが"
        "本来の first-audio latency です。"
    )
    lines.append(
        "- `hit_decode_limit` は stop predictor が発火せず `--max-decode-length` に"
        "張り付いたこと（＝途中切れの疑い）を意味します。"
    )
    lines.append("")

    repro = payload.get("reproducibility") or []
    lines.append("## 再現性（同一text・同一reference・同一seedで2回）")
    lines.append("")
    if repro:
        lines.append("| checkpoint | mode | 完全一致 | 長さ差(sample) | max abs diff | rms diff |")
        lines.append("|---|---|---|---:|---:|---:|")
        for entry in repro:
            diff = entry.get("diff") or {}
            lines.append(
                f"| `{entry['checkpoint_id']}` | {entry['mode']} | "
                f"{_fmt(diff.get('identical'))} | {_fmt(diff.get('length_delta_samples'))} | "
                f"{_fmt(diff.get('max_abs_diff'), 8)} | {_fmt(diff.get('rms_diff'), 8)} |"
            )
    else:
        lines.append("比較できるペアがありませんでした。")
    lines.append("")

    cross = payload.get("streaming_vs_offline") or []
    lines.append("## streaming 出力 と offline 出力の差")
    lines.append("")
    if cross:
        lines.append("| checkpoint | mode | 完全一致 | 長さ差(sample) | max abs diff | rms diff |")
        lines.append("|---|---|---|---:|---:|---:|")
        for entry in cross:
            diff = entry.get("diff") or {}
            lines.append(
                f"| `{entry['checkpoint_id']}` | {entry['mode']} | "
                f"{_fmt(diff.get('identical'))} | {_fmt(diff.get('length_delta_samples'))} | "
                f"{_fmt(diff.get('max_abs_diff'), 8)} | {_fmt(diff.get('rms_diff'), 8)} |"
            )
        lines.append("")
        lines.append(
            "`voice_clone` は offline でも `decode_each_patch=True` で streaming VAE decoder を"
            "通るため、offline と streaming は同じ decode path になります。"
            "`tts` の offline だけが全系列一括 decode なので、ここだけ causal streaming decode との"
            "差が出ます。"
        )
    else:
        lines.append("比較できるペアがありませんでした。")
    lines.append("")

    constraints = payload.get("constraint_checks") or []
    if constraints:
        lines.append("## variant固有制約の実機確認")
        lines.append("")
        lines.append("| checkpoint | variant | 確認内容 | ValueError | message |")
        lines.append("|---|---|---|---|---|")
        for entry in constraints:
            lines.append(
                f"| `{entry['checkpoint_id']}` | {entry['variant']} | {entry['check']} | "
                f"{_fmt(entry['raised_value_error'])} | {entry.get('message') or '—'} |"
            )
        lines.append("")

    probes = payload.get("sampler_compile_probes") or []
    if probes:
        lines.append("## sampler compile mode の実機probe")
        lines.append("")
        lines.append(
            "`AudioDiTHead.sample` の `full-sampler`（`torch.compile` + inductor）経路は "
            "`sway_sampling_coefficient == 0.0` のときだけ選ばれます。"
            "base は既定で sway=-0.8 なので eager 相当、**distill は sway を使えないので必ず "
            "compiled 経路** に入ります。ここでは sway を 0.0 に固定して両方を compiled 経路へ"
            "入れ、実際に動くかを確認しています。"
        )
        lines.append("")
        lines.append("| checkpoint | variant | full-sampler | 所要(s) | 例外 |")
        lines.append("|---|---|---|---:|---|")
        for entry in probes:
            detail = "—"
            if not entry["succeeded"]:
                detail = f"`{entry['error_type']}`: {entry.get('error_message') or ''}"
            lines.append(
                f"| `{entry['checkpoint_id']}` | {entry['variant']} | "
                f"{_fmt(entry['succeeded'])} | {_fmt(entry.get('seconds'))} | {detail} |"
            )
        lines.append("")

    sweep = payload.get("distill_step_sweep") or []
    if sweep:
        lines.append("## distill `diffusion_steps` 速度基準（以降の速度比較プロトコル）")
        lines.append("")
        lines.append("| checkpoint | steps | 生成(s) | 音声長(s) | RTF | peak VRAM(GiB) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for entry in sweep:
            if "error" in entry:
                lines.append(
                    f"| `{entry['checkpoint_id']}` | {entry['diffusion_steps']} | ERROR | — | — | "
                    f"{entry.get('error')} |"
                )
                continue
            lines.append(
                f"| `{entry['checkpoint_id']}` | {entry['diffusion_steps']} | "
                f"{_fmt(entry['generation_seconds'])} | "
                f"{_fmt(entry['audio_duration_seconds'], 2)} | {_fmt(entry['rtf'])} | "
                f"{_gib(entry['peak_vram_bytes'])} |"
            )
        lines.append("")

    summary = payload.get("summary") or {}
    lines.append("## まとめ")
    lines.append("")
    lines.append(f"- case総数: {summary.get('total_cases')}")
    lines.append(
        f"- ok: {summary.get('ok_cases')} / corrupt: {summary.get('corrupt_cases')}"
        f" / error: {summary.get('error_cases')}"
    )
    lines.append(f"- run全体のpeak VRAM: {_gib(summary.get('peak_vram_bytes_overall'))} GiB")
    lines.append(
        f"- 実行計画の8通りmatrix（`{{checkpoint}} × {{tts, voice_clone}} × "
        f"{{offline, streaming}}`）で失敗したcase: "
        f"{summary.get('matrix_failures') or 'なし'}"
    )
    lines.append(f"- P0ゲート通過: {_fmt(summary.get('gate_passed'))}")
    lines.append(
        f"- 計測時の sampler compile mode: `{settings.get('sampler_compile_mode')}`"
        "（この値が違うとRTFは比較できません）"
    )
    lines.append("")
    lines.append("### 日本語textについて")
    lines.append("")
    lines.append(
        "`text[ja]` のcaseは **crashしないことと出力の様子** を記録するためのものです。"
        "この時点のcheckpointは日本語未学習なので、品質は評価対象外です。"
    )
    lines.append("")
    lines.append("### artifactの取り扱い")
    lines.append("")
    lines.append(
        "`samples/` の音声は `artifacts/` 配下にあり、gitには入りません"
        "（08-execution-plan.md の「artifactの公開制限」）。"
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P0: 公開checkpointの推論ベースラインをこの環境で再現し計測する。"
    )
    parser.add_argument("--model-root", default="model", help="checkpointの親ディレクトリ")
    parser.add_argument(
        "--checkpoint",
        action="append",
        dest="checkpoints",
        help="対象checkpointディレクトリ名（複数指定可、既定は CuteTTS と CuteTTS-distill）",
    )
    parser.add_argument("--reference-audio", default="assets/default_reference.wav")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--diffusion-sway-coefficient", type=float, default=None)
    parser.add_argument(
        "--max-decode-length",
        type=int,
        default=750,
        help="decode上限patch数。既定750は約120秒。遅い環境では小さくして完走させる。",
    )
    parser.add_argument(
        "--sampler-compile-mode",
        choices=SAMPLER_COMPILE_CHOICES,
        default="auto",
        help=(
            "DiT samplerの実行mode。既定 auto は triton の有無を見て決める"
            "（tritonが無い環境では eager へ落とす）。"
        ),
    )
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp", default=None, help="run directory名を固定する")
    parser.add_argument(
        "--skip-compile-probe",
        action="store_true",
        help="full-sampler経路が動くかの実機probeを省く",
    )
    parser.add_argument(
        "--skip-step-sweep",
        action="store_true",
        help="distillの diffusion_steps 1/2/4 速度計測を省く",
    )
    parser.add_argument(
        "--skip-japanese",
        action="store_true",
        help="日本語textのsmoke caseを省く",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    from cutetts import CuteTTS
    from cutetts.modeling.sampling import set_sampler_compile_mode
    from cutetts.runtime import resolve_device
    from cutetts.training.artifacts import (
        file_checksum,
        new_run_dir,
        write_metrics,
        write_run_metadata,
    )

    model_root = Path(args.model_root).expanduser().resolve()
    reference_audio = Path(args.reference_audio).expanduser().resolve()
    if not reference_audio.is_file():
        raise SystemExit(f"reference audio not found: {reference_audio}")

    checkpoint_ids = tuple(args.checkpoints) if args.checkpoints else DEFAULT_CHECKPOINT_IDS
    found, skipped = discover_checkpoints(model_root, checkpoint_ids)
    for entry in skipped:
        print(f"[skip] {entry['checkpoint_id']}: {entry['reason']}", flush=True)
    if not found:
        raise SystemExit(f"No usable checkpoint under {model_root}.")

    run_dir = new_run_dir("p0", root=args.artifact_root, timestamp=args.timestamp)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] artifacts -> {run_dir}", flush=True)

    options = {
        "reference_audio": reference_audio,
        "cfg_strength": float(args.cfg_strength),
        "diffusion_steps": args.diffusion_steps,
        "diffusion_sway_coefficient": args.diffusion_sway_coefficient,
        "max_decode_length": int(args.max_decode_length),
        "seed": int(args.seed),
        "texts": dict(TEXTS),
    }

    print("[inputs] hashing checkpoint files ...", flush=True)
    checkpoint_inputs_map = {spec.checkpoint_id: checkpoint_inputs(spec) for spec in found}
    device = resolve_device(args.device)
    sampler_mode, sampler_reason = resolve_sampler_compile_mode(
        args.sampler_compile_mode, device.type
    )
    print(f"[sampler] compile mode = {sampler_mode} ({sampler_reason})", flush=True)
    write_run_metadata(
        run_dir,
        phase="p0",
        command=list(sys.argv),
        seed=int(args.seed),
        inputs={
            "checkpoints": checkpoint_inputs_map,
            "skipped_checkpoints": skipped,
            "reference_audio": {
                "path": str(reference_audio),
                "sha256": file_checksum(reference_audio),
                "bytes": reference_audio.stat().st_size,
            },
            "texts": dict(TEXTS),
            "resolved_device": str(device),
        },
        extra={
            "max_decode_length": int(args.max_decode_length),
            "cfg_strength": float(args.cfg_strength),
            "diffusion_steps": args.diffusion_steps,
            "diffusion_sway_coefficient": args.diffusion_sway_coefficient,
            "device_request": args.device,
            "sampler_compile_mode_request": args.sampler_compile_mode,
            "sampler_compile_mode": sampler_mode,
            "sampler_compile_mode_reason": sampler_reason,
        },
    )

    cases: list[dict] = []
    repro: list[dict] = []
    cross: list[dict] = []
    constraints: list[dict] = []
    sweep: list[dict] = []
    compile_probes: list[dict] = []
    load_times: dict[str, float] = {}

    for spec in found:
        print(f"[load] {spec.checkpoint_id} ({spec.variant})", flush=True)
        t_load = time.perf_counter()
        model = CuteTTS.from_pretrained(spec.path, device=args.device)
        # from_pretrained は非MPSで full-sampler を選ぶので、解決済みmodeで上書きする。
        set_sampler_compile_mode(sampler_mode)
        load_times[spec.checkpoint_id] = time.perf_counter() - t_load
        print(f"[load] done in {load_times[spec.checkpoint_id]:.1f}s", flush=True)

        config = _read_config(spec.path) or {}
        patch_samples = samples_per_patch(config)
        checkpoint_inputs_map[spec.checkpoint_id]["samples_per_patch"] = patch_samples

        waveforms: dict[str, Any] = {}
        for case_spec in build_case_matrix(
            spec.checkpoint_id, include_japanese=not args.skip_japanese
        ):
            print(f"[case] {case_spec.case_id} ...", flush=True)
            outcome = run_case(
                model,
                case_spec,
                options,
                samples_dir=samples_dir,
                patch_samples=patch_samples,
            )
            cases.append(outcome.record)
            if outcome.waveform is not None:
                waveforms[case_spec.case_id] = outcome.waveform
            if outcome.record["status"] == "error":
                error = outcome.record["error"]
                print(
                    f"[case] {case_spec.case_id} ERROR {error['type']}: {error['message']}",
                    flush=True,
                )
                print(error["traceback"], flush=True)
            else:
                metrics = outcome.record["metrics"]
                inspection = outcome.record["inspection"]
                print(
                    f"[case] {case_spec.case_id} {outcome.record['status']} "
                    f"dur={inspection['duration_seconds']:.2f}s "
                    f"ttfa={metrics['ttfa_seconds']:.3f}s rtf={metrics['rtf']:.3f} "
                    f"problems={inspection['problems'] or 'none'}",
                    flush=True,
                )

        for mode in ("tts", "voice_clone"):
            first = waveforms.get(f"{spec.checkpoint_id}__{mode}__offline__en__run1")
            second = waveforms.get(f"{spec.checkpoint_id}__{mode}__offline__en__run2")
            streamed = waveforms.get(f"{spec.checkpoint_id}__{mode}__streaming__en__run1")
            if first is not None and second is not None:
                repro.append(
                    {
                        "checkpoint_id": spec.checkpoint_id,
                        "mode": mode,
                        "text_id": "en",
                        "diff": waveform_diff(first, second),
                    }
                )
            if first is not None and streamed is not None:
                cross.append(
                    {
                        "checkpoint_id": spec.checkpoint_id,
                        "mode": mode,
                        "text_id": "en",
                        "diff": waveform_diff(streamed, first),
                    }
                )

        constraints.extend(check_variant_constraints(model, spec.checkpoint_id, options))
        if model.variant == "distill" and not args.skip_step_sweep:
            sweep.extend(
                run_distill_step_sweep(
                    model,
                    spec.checkpoint_id,
                    options,
                    samples_dir=samples_dir,
                    patch_samples=patch_samples,
                )
            )
        # compile probeは計測を汚さないよう、そのcheckpointの計測がすべて終わってから走らせる。
        if not args.skip_compile_probe:
            probe = probe_full_sampler(
                model, spec.checkpoint_id, options, restore_mode=sampler_mode
            )
            compile_probes.append(probe)
            print(
                f"[probe] {spec.checkpoint_id} full-sampler "
                f"{'ok' if probe['succeeded'] else 'FAILED: ' + str(probe['error_type'])}",
                flush=True,
            )

        del model
        waveforms.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    peak_overall = max(
        (int(case["peak_vram_bytes"]) for case in cases if case.get("peak_vram_bytes")),
        default=0,
    )
    ok_cases = sum(1 for case in cases if case["status"] == "ok")
    corrupt_cases = sum(1 for case in cases if case["status"] == "corrupt")
    error_cases = sum(1 for case in cases if case["status"] == "error")
    matrix_failures = [
        case["case_id"]
        for case in cases
        if case["purpose"] == "matrix" and case["status"] != "ok"
    ]
    payload = {
        "phase": "p0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "settings": {
            "device": str(device),
            "device_request": args.device,
            "seed": int(args.seed),
            "max_decode_length": int(args.max_decode_length),
            "cfg_strength": float(args.cfg_strength),
            "diffusion_steps": args.diffusion_steps,
            "diffusion_sway_coefficient": args.diffusion_sway_coefficient,
            "reference_audio": str(reference_audio),
            "texts": dict(TEXTS),
            "sampler_compile_mode_request": args.sampler_compile_mode,
            "sampler_compile_mode": sampler_mode,
            "sampler_compile_mode_reason": sampler_reason,
        },
        "checkpoints": checkpoint_inputs_map,
        "skipped_checkpoints": skipped,
        "model_load_seconds": load_times,
        "cases": cases,
        "reproducibility": repro,
        "streaming_vs_offline": cross,
        "constraint_checks": constraints,
        "sampler_compile_probes": compile_probes,
        "distill_step_sweep": sweep,
        "summary": {
            "total_cases": len(cases),
            "ok_cases": ok_cases,
            "corrupt_cases": corrupt_cases,
            "error_cases": error_cases,
            "matrix_failures": matrix_failures,
            "peak_vram_bytes_overall": peak_overall or None,
            "gate_passed": not matrix_failures and not skipped,
        },
    }
    write_metrics(run_dir, payload)
    (run_dir / "report.md").write_text(render_report(payload), encoding="utf-8")

    print(f"[done] {run_dir}", flush=True)
    print(
        f"[done] ok={ok_cases} corrupt={corrupt_cases} error={error_cases} "
        f"peak_vram={peak_overall / (1024 ** 3):.2f} GiB",
        flush=True,
    )
    if skipped:
        print(
            "[done] skipped checkpoints: "
            + ", ".join(entry["checkpoint_id"] for entry in skipped),
            flush=True,
        )
    return 0 if not matrix_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
