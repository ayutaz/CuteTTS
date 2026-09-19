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

"""テキスト → 韻律の輪郭 の予測器を学習して、辞書と比べる（M4b）。

**TTS を通さずに測れる。** 「テキストから韻律はどこまで予測できるか」は
予測器と実測の輪郭を直接比べれば分かるので、生成の雑音が入らない。

比べる相手は3つ。

| 相手 | 意味 |
|---|---|
| **辞書**（`dictionary_contour`） | アクセント句から作る理想化した段。**これを超えられるか** |
| 平坦（全部 0） | 何も予測しない場合。相関は 0 |
| **学習した予測器** | コーパスの傾向を学べるぶんだけ上へ行けるはず |

    uv run --no-sync python scripts/train_f0_predictor.py \\
      --f0-cache data/s1v2/f0-17.9h --manifest data/s1v2/subset-17.9h.jsonl \\
      --frontend accent --epochs 8 --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from cutetts.training import artifacts  # noqa: E402
from cutetts.training.f0 import F0CacheReader  # noqa: E402
from cutetts.training.f0_predictor import (  # noqa: E402
    CONTOUR_POINTS,
    F0Predictor,
    contour_from_f0,
    dictionary_contour,
    encode_text,
    text_vocabulary,
)
from cutetts.training.manifest import load_manifest  # noqa: E402
from cutetts.training.yomi import frontend_text  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0-cache", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frontend", default="accent")
    parser.add_argument("--max-text", type=int, default=96)
    parser.add_argument("--limit", type=int, default=20000,
                        help="学習に使う発話数の上限")
    parser.add_argument("--dev-limit", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="checkpoints/f0-predictor")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def collect(records, reader, frontend: str, limit: int, points: int):
    """(テキスト, 実測の輪郭, 辞書の輪郭) を集める。"""
    rows = []
    for record in records:
        if len(rows) >= limit:
            break
        if record.utterance_id not in reader:
            continue
        features = reader.read(record.utterance_id).numpy()
        # `[T, 2]` の 0 列目が有声フラグ、1 列目が正規化 log2 F0。
        # **輪郭は log2 のまま扱う**（半音へ直しても相関は変わらない）
        voiced = features[:, 0] > 0.5
        if int(voiced.sum()) < 10:
            continue
        pseudo = np.where(voiced, np.exp2(features[:, 1]) * 100.0, 0.0)
        contour = contour_from_f0(pseudo, points)
        if contour is None:
            continue
        text = frontend_text(record.text_raw, frontend)
        if not text:
            continue
        rows.append((text, contour, dictionary_contour(text, points)))
    return rows


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("m4b-predictor", args.artifact_root,
                                    timestamp=args.timestamp)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    reader = F0CacheReader(args.f0_cache)
    records = list(load_manifest(args.manifest))
    train_rows = collect([r for r in records if r.split == "train"], reader,
                         args.frontend, args.limit, CONTOUR_POINTS)
    dev_rows = collect([r for r in records if r.split == "dev-seen"], reader,
                       args.frontend, args.dev_limit, CONTOUR_POINTS)
    print(f"train {len(train_rows):,} / dev {len(dev_rows):,}")
    if len(train_rows) < 100 or len(dev_rows) < 50:
        raise SystemExit("データが足りない")

    vocab = text_vocabulary([text for text, _, _ in train_rows])
    print(f"語彙 {len(vocab):,} 文字")

    def tensors(rows):
        ids = np.stack([encode_text(t, vocab, args.max_text) for t, _, _ in rows])
        target = np.stack([c for _, c, _ in rows])
        return (torch.from_numpy(ids), torch.from_numpy(target))

    train_ids, train_target = tensors(train_rows)
    dev_ids, dev_target = tensors(dev_rows)

    model = F0Predictor(len(vocab) + 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()
    count = train_ids.shape[0]
    started = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(count)
        total = 0.0
        for start in range(0, count, args.batch_size):
            index = order[start:start + args.batch_size]
            predicted = model(train_ids[index].to(device))
            loss = loss_fn(predicted, train_target[index].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(index)
        model.eval()
        with torch.no_grad():
            dev_pred = model(dev_ids.to(device)).cpu().numpy()
        scores = [correlation(dev_pred[i], dev_target[i].numpy())
                  for i in range(len(dev_rows))]
        scores = np.array([s for s in scores if s == s])
        print(f"  epoch {epoch + 1}  L1={total / count:.4f}  "
              f"dev 相関 {scores.mean():+.3f}", flush=True)

    # ---------------------------------------------------------------- 比較
    model.eval()
    with torch.no_grad():
        dev_pred = model(dev_ids.to(device)).cpu().numpy()

    def summarize(name: str, values: list[float]) -> dict:
        array = np.array([v for v in values if v == v])
        print(f"  {name:<20} 平均 {array.mean():+.3f}  "
              f"中央 {np.median(array):+.3f}  n={array.size}")
        return {"name": name, "mean": float(array.mean()),
                "median": float(np.median(array)), "n": int(array.size)}

    truth = [row[1] for row in dev_rows]
    dictionary = [row[2] for row in dev_rows]
    print()
    rows = [
        summarize("**学習した予測器**", [correlation(dev_pred[i], truth[i])
                                    for i in range(len(truth))]),
        summarize("辞書", [correlation(dictionary[i], truth[i])
                         if dictionary[i] is not None else float("nan")
                         for i in range(len(truth))]),
        summarize("別の文の実測（床）", [correlation(truth[(i + 1) % len(truth)],
                                            truth[i])
                                   for i in range(len(truth))]),
    ]
    print()
    print("  **辞書を超えなければ、テキストから韻律を取る道は細い。**")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                "points": CONTOUR_POINTS, "frontend": args.frontend,
                "max_text": args.max_text}, out / "f0_predictor.pt")

    artifacts.write_run_metadata(
        run_dir, phase="m4b-predictor",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"f0_cache": args.f0_cache, "manifest": args.manifest})
    artifacts.write_metrics(run_dir, {
        "phase": "m4b-predictor", "frontend": args.frontend,
        "train": len(train_rows), "dev": len(dev_rows),
        "epochs": args.epochs, "vocab": len(vocab),
        "comparison": rows,
        "elapsed_minutes": (time.perf_counter() - started) / 60,
    })
    print(f"  {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
