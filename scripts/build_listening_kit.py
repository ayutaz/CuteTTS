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

"""聴取用の音声を生成する（自然性・アクセントの定性確認）。

**S1のゴールのうち自然性とアクセントだけが未測定で、しかも日本語向けの
信頼できる自動指標が存在しない。** 聴取で代替するしかない。
19回の学習を CER だけで判断してきたので、**CERが知覚と対応しているかすら
未検証**である。

盲検にするためファイル名は連番にし、どちらがどの条件かは別のkeyへ書く。
人間の実音声（`data/eval/asr_floor/`）を無告知で混ぜる（アンカー）。
聞き分けられなければ、そこが到達点だという意味になる。

    python scripts/build_listening_kit.py \\
        --model-dir checkpoints/s1v2-fp32-30k/inference --out artifacts/listen-kit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import soundfile as sf  # noqa: E402

from cutetts import CuteTTS  # noqa: E402
from cutetts.training.listening_page import render  # noqa: E402
from cutetts.training.reading import expand_kanji_numerals  # noqa: E402


def pick(items: list, count: int, seed: int) -> list:
    """seed と内容のハッシュで決定的に選ぶ。"""
    def key(item) -> str:
        text = item["text"] if isinstance(item, dict) else str(item)
        return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return sorted(items, key=key)[:count]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="聴取用の音声を生成する")
    parser.add_argument("--model-dir", help="推論用export。--html-only なら不要")
    parser.add_argument("--html-only", action="store_true",
                        help="音声を作り直さず index.html だけ書き直す")
    parser.add_argument("--base-dir", default="model/CuteTTS",
                        help="比較対象の未学習checkpoint")
    parser.add_argument("--eval-set", default="data/eval/eval_set_v3.json")
    parser.add_argument("--numeral-set", default="data/eval/numeral_eval_set.json")
    parser.add_argument("--reference-audio", default="assets/default_reference.wav")
    parser.add_argument("--out", default="artifacts/listen-kit")
    parser.add_argument("--conversational", type=int, default=14)
    parser.add_argument("--phonetic", type=int, default=6)
    parser.add_argument("--numerals", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def write_page(out: Path) -> None:
    """manifest と アンカー音源から index.html を書く。"""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    anchors = manifest.get("anchors") or []
    if not anchors:
        anchors = add_local_anchors(out, manifest["seed"])
        manifest["anchors"] = anchors
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "index.html").write_text(render(manifest, anchors), encoding="utf-8")
    print(f"index.html を書きました（アンカー {len(anchors)}本）")


def add_local_anchors(out: Path, seed: int, count: int = 4) -> list[dict]:
    """人間の実音声をアンカーとして取り込む。ASR床10.4%を測った音源。"""
    floor = Path("data/eval/asr_floor")
    if not (floor / "items.json").is_file():
        return []
    items = json.loads((floor / "items.json").read_text(encoding="utf-8"))
    items = items if isinstance(items, list) else items.get("items", [])
    anchors: list[dict] = []
    for item in pick([i for i in items if i.get("group") == "plain"], count, seed):
        source = floor / item["wav"]
        if not source.is_file():
            continue
        data, rate = sf.read(source)
        name = f"human_{len(anchors):02d}.wav"
        sf.write(out / "audio" / name, data, rate)
        anchors.append({"file": name, "model": "human",
                        "group": "アンカー", "text": item["text"]})
    return anchors


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    if args.html_only:
        write_page(out)
        return
    if not args.model_dir:
        raise SystemExit("--model-dir が要る（--html-only なら不要）")

    payload = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    numerals = json.loads(Path(args.numeral_set).read_text(encoding="utf-8"))

    plan: list[dict] = []
    for item in pick(payload["subsets"]["in_domain"], args.conversational, args.seed):
        plan.append({"text": item["text"], "group": "会話文", "expand": False})
    for item in pick(payload["subsets"]["phonetic"], args.phonetic, args.seed):
        plan.append({"text": item["text"], "group": "音韻", "expand": False})
    # 数詞は J2 の有無で聴き比べる（同じ文の2条件）
    for item in pick([i for i in numerals["subsets"]["numerals"] if i["expanded"]],
                     args.numerals, args.seed):
        plan.append({"text": item["text"], "group": "数詞", "expand": True})

    print(f"生成する文: {len(plan)}（会話文 {args.conversational} / "
          f"音韻 {args.phonetic} / 数詞 {args.numerals}）")

    models = {"base": args.base_dir, "trained": args.model_dir}
    entries: list[dict] = []
    for name, directory in models.items():
        model = CuteTTS.from_pretrained(directory, device=args.device)
        print(f"\n=== {name}: {directory} ===")
        for index, spec in enumerate(plan):
            variants = [("", False)]
            if spec["expand"]:
                variants = [("_raw", False), ("_j2", True)]
            for suffix, expand in variants:
                text = expand_kanji_numerals(spec["text"]) if expand else spec["text"]
                result = model.generate(
                    text, mode="voice_clone", reference_audio=args.reference_audio,
                    seed=42, max_decode_length=400, show_progress=False,
                )
                stem = f"{name}_{index:02d}{suffix}"
                sf.write(out / "audio" / f"{stem}.wav",
                         result.waveform.squeeze(0).float().numpy(), result.sample_rate)
                entries.append({
                    "file": f"{stem}.wav", "index": index, "model": name,
                    "group": spec["group"], "text": spec["text"],
                    "spoken": text if text != spec["text"] else None,
                    "j2": expand,
                    "seconds": result.waveform.shape[-1] / result.sample_rate,
                })
            print(f"  [{index:02d}] {spec['group']} {spec['text'][:30]}")

    # 人間の実音声をアンカーとして混ぜる（無告知）
    floor = Path("data/eval/asr_floor")
    anchors: list[dict] = []
    if (floor / "items.json").is_file():
        items = json.loads((floor / "items.json").read_text(encoding="utf-8"))
        items = items if isinstance(items, list) else items.get("items", [])
        for item in pick([i for i in items if i.get("group") == "plain"], 4, args.seed):
            source = floor / item["wav"]
            if not source.is_file():
                continue
            data, rate = sf.read(source)
            stem = f"human_{len(anchors):02d}"
            sf.write(out / "audio" / f"{stem}.wav", data, rate)
            anchors.append({"file": f"{stem}.wav", "model": "human",
                            "group": "アンカー", "text": item["text"]})
    print(f"\nアンカー（人間の実音声）: {len(anchors)}件")

    rng = random.Random(args.seed)
    manifest = {
        "seed": args.seed,
        "models": models,
        "entries": entries,
        "anchors": anchors,
        # 盲検A/Bで左右どちらに学習後を置くか。項目ごとに固定
        "left_is_trained": {str(i): rng.random() < 0.5 for i in range(len(plan))},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_page(out)
    print(f"完了: {out}  音声 {len(list((out / 'audio').glob('*.wav')))} 本")


if __name__ == "__main__":
    main()
