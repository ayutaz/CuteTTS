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

"""P1e: 前処理パス。全音声を1回だけ読み、latent と speaker embedding を同時に作る。

P1dのvoiceクラスタリングとP2のlatent cacheはどちらも全音声を1回読む必要があるため、
別々に実施すると同じI/Oを2回払う。この1パスで両方を生成する（08章 P1e）。

* Audio VAE は 24 kHz 入力 → ``[T, 64]`` latent
* Speaker Encoder は 16 kHz 入力 → ``[256]`` embedding

したがって**リサンプルは2系統**必要。書庫（tar / zip）は1本ずつ開いて処理し、
元音声をローカルに常駐させない。既にcacheにあるIDはスキップするので再開可能。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import soundfile as sf
import torch

from cutetts.audio_codec.model.speaker_encoder import FbankECAPAStudent
from cutetts.modeling.audio_adapter import AudioAcousticVAEAdapter
from cutetts.training import artifacts
from cutetts.training.latents import (
    CacheMeta,
    LatentCacheWriter,
    PREPROCESSING_VERSION,
    encode_waveform,
    resample_to,
)
from cutetts.training.manifest import Utterance, load_manifest, split_audio_ref
from cutetts.training.speaker_cache import (
    SPEAKER_SAMPLE_RATE,
    SpeakerCacheMeta,
    SpeakerEmbeddingCacheWriter,
    embed_waveform,
)

# gol全体の規模。外挿の基準（data-inventory.md の実測値）。
GOL_TOTAL_HOURS = 10654.32
GOL_TOTAL_UTTERANCES = 7405094


def _load_speaker_encoder(folder: Path, device: torch.device) -> FbankECAPAStudent:
    from safetensors.torch import load_file

    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    config.pop("component", None)
    model = FbankECAPAStudent(**config)
    model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
    return model.float().to(device).eval()


def _group_by_container(records: list[Utterance]) -> dict[str, list[tuple[str, Utterance]]]:
    grouped: dict[str, list[tuple[str, Utterance]]] = defaultdict(list)
    for record in records:
        container, member = split_audio_ref(record.audio_ref)
        if member is None:
            grouped[container].append((container, record))
        else:
            grouped[container].append((member, record))
    return grouped


def _iter_members(container: Path, wanted: list[tuple[str, Utterance]]) -> Iterator[tuple[Utterance, bytes]]:
    """書庫を1回だけ走査して、必要なメンバのバイト列を返す。"""
    want = {member: record for member, record in wanted}
    if container.suffix == ".tar":
        with tarfile.open(container) as archive:
            for info in archive:
                record = want.get(info.name)
                if record is None:
                    continue
                payload = archive.extractfile(info)
                if payload is not None:
                    yield record, payload.read()
    elif container.suffix == ".zip":
        with zipfile.ZipFile(container) as archive:
            for name in archive.namelist():
                record = want.get(name)
                if record is not None:
                    yield record, archive.read(name)
    else:  # 素のファイル
        for _member, record in wanted:
            yield record, container.read_bytes()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P1e: latent と speaker embedding の生成")
    parser.add_argument("--manifest", default="data/manifests/all.jsonl")
    parser.add_argument("--model-dir", default="model/CuteTTS")
    parser.add_argument("--latent-cache", default="data/cache/latents")
    parser.add_argument("--speaker-cache", default="data/cache/speaker")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    parser.add_argument("--limit", type=int, help="処理する発話数の上限（パイロット用）")
    parser.add_argument("--speaker-reference-seconds", type=float, default=8.0,
                        help="speaker embedding に使う先頭秒数（推論の prepare_reference_audio と同じ既定）")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model_dir = Path(args.model_dir)
    run_dir = artifacts.new_run_dir("p1e", args.artifact_root, timestamp=args.timestamp)

    records = list(load_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]
    print(f"manifest {args.manifest}: {len(records):,} 発話")

    vae_weights = model_dir / "weights" / "audio_vae" / "model.safetensors"
    spk_weights = model_dir / "weights" / "speaker_encoder" / "model.safetensors"
    latent_meta = CacheMeta(
        vae_checkpoint_sha256=artifacts.file_checksum(vae_weights),
        preprocessing_version=PREPROCESSING_VERSION,
        sample_rate=24000, latent_dim=64, dtype="float16",
    )
    speaker_meta = SpeakerCacheMeta(
        speaker_encoder_sha256=artifacts.file_checksum(spk_weights),
        preprocessing_version=PREPROCESSING_VERSION,
        sample_rate=SPEAKER_SAMPLE_RATE, embedding_dim=256, dtype="float16",
    )

    vae = AudioAcousticVAEAdapter(model_dir / "weights" / "audio_vae").to(device).eval()
    speaker_encoder = _load_speaker_encoder(model_dir / "weights" / "speaker_encoder", device)
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None

    grouped = _group_by_container(records)
    done = skipped = failed = 0
    audio_seconds = 0.0
    latent_frames = 0
    started = time.perf_counter()
    decode_s = resample_s = vae_s = spk_s = 0.0

    with LatentCacheWriter(args.latent_cache, latent_meta) as lat_writer, \
         SpeakerEmbeddingCacheWriter(args.speaker_cache, speaker_meta) as spk_writer:
        for container_str, wanted in sorted(grouped.items()):
            container = Path(container_str)
            pending = [(m, r) for m, r in wanted
                       if r.utterance_id not in lat_writer or r.utterance_id not in spk_writer]
            skipped += len(wanted) - len(pending)
            if not pending:
                continue
            if not container.exists():
                print(f"  [warn] 書庫が無い: {container}", file=sys.stderr)
                failed += len(pending)
                continue
            print(f"  {container.name}: {len(pending):,} 発話")
            for record, payload in _iter_members(container, pending):
                try:
                    t0 = time.perf_counter()
                    data, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
                    wave = torch.from_numpy(data.T).mean(dim=0, keepdim=True)
                    decode_s += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    wave24 = resample_to(wave, rate, 24000).to(device)
                    head = wave[..., : int(args.speaker_reference_seconds * rate)]
                    wave16 = resample_to(head, rate, SPEAKER_SAMPLE_RATE).to(device)
                    resample_s += time.perf_counter() - t0

                    with torch.inference_mode():
                        t0 = time.perf_counter()
                        latent = encode_waveform(vae, wave24)
                        vae_s += time.perf_counter() - t0
                        t0 = time.perf_counter()
                        embedding = embed_waveform(speaker_encoder, wave16)
                        spk_s += time.perf_counter() - t0

                    lat_writer.write(record.utterance_id, latent)
                    spk_writer.write(record.utterance_id, embedding)
                    done += 1
                    audio_seconds += wave24.shape[-1] / 24000.0
                    latent_frames += int(latent.shape[0])
                except Exception as error:
                    failed += 1
                    print(f"  [warn] {record.utterance_id}: {type(error).__name__} {error}",
                          file=sys.stderr)
        lat_writer.flush()
        spk_writer.flush()

    elapsed = time.perf_counter() - started
    latent_bytes = sum(p.stat().st_size for p in Path(args.latent_cache).rglob("*") if p.is_file())
    speaker_bytes = sum(p.stat().st_size for p in Path(args.speaker_cache).rglob("*") if p.is_file())

    per_utt = elapsed / done if done else 0.0
    per_audio_second = elapsed / audio_seconds if audio_seconds else 0.0
    latent_bytes_per_audio_second = latent_bytes / audio_seconds if audio_seconds else 0.0

    metrics = {
        "phase": "p1e",
        "device": str(device),
        "processed_utterances": done,
        "skipped_utterances": skipped,
        "failed_utterances": failed,
        "processed_audio_seconds": audio_seconds,
        "processed_audio_hours": audio_seconds / 3600.0,
        "latent_frames": latent_frames,
        "measured_latent_frame_rate": latent_frames / audio_seconds if audio_seconds else None,
        "elapsed_seconds": elapsed,
        "seconds_per_utterance": per_utt,
        "seconds_per_audio_second": per_audio_second,
        "realtime_factor": 1.0 / per_audio_second if per_audio_second else None,
        "stage_seconds": {
            "decode": decode_s, "resample": resample_s,
            "vae_encode": vae_s, "speaker_encode": spk_s,
        },
        "latent_cache_bytes": latent_bytes,
        "speaker_cache_bytes": speaker_bytes,
        "latent_bytes_per_audio_second": latent_bytes_per_audio_second,
        "peak_vram_bytes": (torch.cuda.max_memory_allocated() if device.type == "cuda" else None),
        "extrapolation_gol_full": {
            "total_hours": GOL_TOTAL_HOURS,
            "total_utterances": GOL_TOTAL_UTTERANCES,
            "latent_cache_gb": latent_bytes_per_audio_second * GOL_TOTAL_HOURS * 3600 / 1e9,
            "speaker_cache_gb": (
                (speaker_bytes / done) * GOL_TOTAL_UTTERANCES / 1e9 if done else None
            ),
            "compute_hours": per_audio_second * GOL_TOTAL_HOURS,
            "plan_estimate_latent_gb": 61.0,
        },
    }

    artifacts.write_run_metadata(
        run_dir, phase="p1e", command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=None,
        inputs={
            "manifest": str(args.manifest),
            "vae_sha256": latent_meta.vae_checkpoint_sha256,
            "speaker_encoder_sha256": speaker_meta.speaker_encoder_sha256,
            "preprocessing_version": PREPROCESSING_VERSION,
        },
    )
    artifacts.write_metrics(run_dir, metrics)

    ex = metrics["extrapolation_gol_full"]
    print(f"\n処理 {done:,} 発話 / {audio_seconds/3600:.2f} h / {elapsed:.1f} 秒 "
          f"(スキップ {skipped:,} / 失敗 {failed:,})")
    print(f"latent {latent_bytes/1e6:.1f} MB, speaker {speaker_bytes/1e6:.1f} MB, "
          f"実測 frame rate {metrics['measured_latent_frame_rate']:.3f} Hz")
    print(f"スループット {1.0/per_audio_second:.1f}x realtime, "
          f"peak VRAM {(metrics['peak_vram_bytes'] or 0)/1e9:.2f} GB")
    print(f"gol全体(10,654 h)への外挿: latent {ex['latent_cache_gb']:.1f} GB "
          f"(計画見積 61 GB) / 所要 {ex['compute_hours']:.1f} GPU時間")
    print(f"\n完了: {run_dir}")


if __name__ == "__main__":
    main()
