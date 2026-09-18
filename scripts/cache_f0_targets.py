#!/usr/bin/env python
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

"""latent から F0 の条件を作る（M4c の前処理）。

**音声そのものは置いていない**（MoeSpeech のライセンス上も置かない）。
学習に使えるのは latent cache だけなので、**decode してから F0 を取る**。

    latent ─(VAE decoder / GPU)─> 波形 ─(pyworld / CPU)─> F0 @12.5 Hz ─> cache

GPU（decode）と CPU（F0）は別の資源なので**並べて流す**。
F0 は 1コアあたり 4.4× 実時間（実測）なので、コア数が効く。

    uv run --no-sync python scripts/cache_f0_targets.py \\
      --latent-cache data/s1v2/latents-v2 --out data/s1v2/f0-v2 \\
      --model-dir model/CuteTTS --device cuda --workers 8

`--manifest` を渡すとその発話だけに絞る（部分集合での検証用）。
既にある id は飛ばすので**途中で落ちても再開できる**。
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cutetts.modeling.audio_adapter import AudioAcousticVAEAdapter  # noqa: E402
from cutetts.training import artifacts  # noqa: E402
from cutetts.training.f0 import (  # noqa: E402
    F0CacheWriter,
    default_f0_meta,
    features_from_waveform,
)
from cutetts.training.latents import LATENT_SAMPLE_RATE, LatentCacheReader  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--out", required=True, help="F0 cache の出力先")
    parser.add_argument("--model-dir", default="model/CuteTTS",
                        help="Audio VAE の weights を持つディレクトリ")
    parser.add_argument("--manifest", help="この manifest の発話だけに絞る")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4,
                        help="F0 を取る thread 数（`pyworld` はGILを離す）")
    parser.add_argument("--limit", type=int, help="先頭 N 件だけ")
    parser.add_argument("--report-every", type=int, default=500)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def target_ids(reader: LatentCacheReader, manifest: str | None,
               limit: int | None) -> list[str]:
    """cache にある id のうち、manifest（あれば）に載っているものだけ。"""
    keys = list(reader.keys())
    if manifest:
        from cutetts.training.manifest import load_manifest

        wanted = {record.utterance_id for record in load_manifest(manifest)}
        keys = [key for key in keys if key in wanted]
    keys.sort()
    return keys[:limit] if limit else keys


def main() -> int:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("m4c-f0-cache", args.artifact_root,
                                    timestamp=args.timestamp)
    device = resolve_device(args.device)

    model_dir = Path(args.model_dir)
    vae_dir = model_dir / "weights" / "audio_vae"
    if not vae_dir.is_dir():
        raise SystemExit(f"Audio VAE が無い: {vae_dir}")
    vae_sha = artifacts.file_checksum(vae_dir / "model.safetensors")

    reader = LatentCacheReader(args.latent_cache)
    # **latent cache と同じ VAE で作る。** 食い違うと F0 が別のものになる
    if reader.meta.vae_checkpoint_sha256 != vae_sha:
        raise SystemExit(
            "latent cache と VAE が違う\n"
            f"  latent cache: {reader.meta.vae_checkpoint_sha256}\n"
            f"  --model-dir : {vae_sha}")

    keys = target_ids(reader, args.manifest, args.limit)
    meta = default_f0_meta(vae_sha)
    writer = F0CacheWriter(args.out, meta)
    pending = [key for key in keys if key not in writer]
    print(f"{len(keys):,} 件中 {len(pending):,} 件が未処理  device={device}  "
          f"workers={args.workers}")
    if not pending:
        print("すべて済み")
        return 0

    vae = AudioAcousticVAEAdapter(vae_dir).to(device).eval()

    # decode（GPU）と F0（CPU）を並べる。間は有界キューで繋ぐ
    work: "queue.Queue[tuple[str, np.ndarray] | None]" = queue.Queue(maxsize=64)
    done: "queue.Queue[tuple[str, np.ndarray] | None]" = queue.Queue(maxsize=256)
    errors: list[tuple[str, str]] = []
    lock = threading.Lock()

    def f0_worker() -> None:
        while True:
            item = work.get()
            if item is None:
                work.task_done()
                return
            key, wave = item
            try:
                features = features_from_waveform(wave, LATENT_SAMPLE_RATE)
                done.put((key, features))
            except Exception as error:               # 1件の失敗で止めない
                with lock:
                    errors.append((key, f"{type(error).__name__}: {error}"))
            finally:
                work.task_done()

    workers = [threading.Thread(target=f0_worker, daemon=True)
               for _ in range(max(1, args.workers))]
    for thread in workers:
        thread.start()

    written = 0
    seconds = 0.0
    started = time.perf_counter()
    decode_s = 0.0

    def drain(block: bool = False) -> None:
        nonlocal written
        while True:
            try:
                item = done.get(timeout=0.5) if block else done.get_nowait()
            except queue.Empty:
                return
            if item is None:
                return
            key, features = item
            writer.write(key, features)
            written += 1
            if written % args.report_every == 0:
                elapsed = time.perf_counter() - started
                print(f"  {written:,}/{len(pending):,}  "
                      f"音声 {seconds / 3600:.2f} h  "
                      f"decode {decode_s:.0f}s  "
                      f"全体 {seconds / max(elapsed, 1e-9):.1f}× 実時間",
                      flush=True)

    with torch.no_grad():
        for key in pending:
            latent = reader.read(key)                       # [T, 64]
            start = time.perf_counter()
            wave = vae.decode(latent.T.unsqueeze(0).to(device))
            wave = wave.squeeze(0).squeeze(0).float().cpu().numpy()
            decode_s += time.perf_counter() - start
            seconds += wave.size / LATENT_SAMPLE_RATE
            work.put((key, wave.astype(np.float64)))
            drain()

    work.join()
    for _ in workers:
        work.put(None)
    drain(block=True)
    while not done.empty():
        drain()

    writer.flush()
    writer.close()
    elapsed = time.perf_counter() - started
    rate = seconds / max(elapsed, 1e-9)
    print(f"\n  書いた {written:,} 件 / 音声 {seconds / 3600:.2f} h")
    print(f"  {elapsed / 60:.1f} 分 → **{rate:.1f}× 実時間**"
          f"（decode だけなら {seconds / max(decode_s, 1e-9):.1f}×）")
    if errors:
        print(f"  **失敗 {len(errors)} 件**")
        for key, detail in errors[:5]:
            print(f"    {key}: {detail}")

    artifacts.write_run_metadata(
        run_dir, phase="m4c-f0-cache",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=None,
        inputs={"latent_cache": args.latent_cache, "out": args.out,
                "model_dir": args.model_dir, "manifest": args.manifest},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "m4c-f0-cache",
        "written": written,
        "audio_hours": seconds / 3600,
        "elapsed_minutes": elapsed / 60,
        "realtime_factor": rate,
        "decode_realtime_factor": seconds / max(decode_s, 1e-9),
        "errors": errors[:50],
        "error_count": len(errors),
        "meta": meta.to_json(),
    })
    print(f"  {run_dir}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
