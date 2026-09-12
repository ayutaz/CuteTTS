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

"""日本語継続学習したcheckpointで音声を作る。

**upstream の推論pathには触らない。** `cutetts` CLI と `api.py` はそのままで、
ここは「日本語向けのtext前処理を掛けてから公開APIを呼ぶ」薄い層にすぎない。

前処理はふたつ:

* **漢数字の読み展開（J2 / D-008）** — `千二百八十円` → `せんにひゃくはちじゅう円`。
  学習コーパスに複合漢数字は 0.32% しかなく桁の合成規則を学べないが、
  仮名なら既に読める。専用評価set 200文で **-11.80pt**
  （95%CI [-17.45, -6.00]、有意）。**再学習を要しない。**

  既定で有効。数詞を含まない文には何もしないので常時掛けてよい
  （in_domain 600文で悪化しないことを確認済み）。無効にするには `--raw-text`。
* **短いreferenceの延長（R-026）** — 3〜4秒のreferenceでは声質と抑揚が崩れる。
  同一話者・同一文で人間と聴き比べると、劣る側の最大3.9秒 < 近い側の最小8.2秒で
  境界が重ならなかった。学習時は平均9.61秒。既定の下限は8秒で、
  `--min-reference-seconds 0` で無効化できる。

    python scripts/synthesize_japanese.py \\
        --model-dir checkpoints/s1v2-fp32-30000 \\
        --text "価格は千二百八十円、消費税込みです。" \\
        --reference-audio assets/default_reference.wav \\
        --output out.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import soundfile as sf  # noqa: E402

from cutetts import CuteTTS  # noqa: E402
from cutetts.training.reading import expand_kanji_numerals  # noqa: E402
from cutetts.training.reference import (  # noqa: E402
    DEFAULT_MINIMUM_SECONDS,
    duration_seconds,
    ensure_minimum_duration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="日本語checkpointで合成する（漢数字の読み展開つき）")
    parser.add_argument("--model-dir", required=True,
                        help="推論用export（例: checkpoints/s1v2-fp32-30000）")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--mode", default="voice_clone", choices=("tts", "voice_clone"))
    parser.add_argument("--reference-audio", default="assets/default_reference.wav",
                        help="voice_clone のとき必須")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decode-length", type=int, default=400,
                        help="400 patch = 64.0秒。張り付くと停止に失敗している（R-021）")
    parser.add_argument("--raw-text", action="store_true",
                        help="読み展開を行わない。素のtextをそのまま渡す")
    parser.add_argument("--min-reference-seconds", type=float,
                        default=DEFAULT_MINIMUM_SECONDS,
                        help="referenceがこれより短ければ繰り返して伸ばす（R-026）。"
                             "0 で無効")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    spoken = args.text if args.raw_text else expand_kanji_numerals(args.text)
    if spoken != args.text:
        print(f"読み展開: {args.text}\n        → {spoken}")

    # **短いreferenceは伸ばす。** 同一話者・同一文で人間と聴き比べると、
    # 3〜4秒のreferenceでは声質と抑揚が崩れた（R-026）。学習時は平均9.61秒。
    reference = args.reference_audio
    if args.mode == "voice_clone" and args.min_reference_seconds > 0:
        before = duration_seconds(reference)
        reference = str(ensure_minimum_duration(
            reference, minimum_seconds=args.min_reference_seconds))
        if reference != str(Path(args.reference_audio).expanduser().resolve()):
            print(f"reference を延長: {before:.2f}秒 → "
                  f"{duration_seconds(reference):.2f}秒（下限 "
                  f"{args.min_reference_seconds:g}秒）")

    model = CuteTTS.from_pretrained(args.model_dir, device=args.device)
    result = model.generate(
        spoken, mode=args.mode,
        reference_audio=reference if args.mode == "voice_clone" else None,
        seed=args.seed, max_decode_length=args.max_decode_length,
    )

    seconds = result.waveform.shape[-1] / result.sample_rate
    limit = args.max_decode_length / 6.25          # LMのtoken rate 6.25 patch/s
    if seconds >= limit - 0.5:
        print(f"⚠ 生成が上限 {limit:.1f} 秒に張り付いた。停止に失敗している可能性がある（R-021）")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, result.waveform.squeeze(0).float().numpy(), result.sample_rate)
    print(f"完了: {output}  {seconds:.2f}秒  {result.sample_rate} Hz")


if __name__ == "__main__":
    main()
