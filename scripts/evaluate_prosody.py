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

"""抑揚を測る（M1）。**CERでは測れないものを測る。**

聴取で残った指摘は読み間違い45% / **抑揚40%** / アクセント30%。
CERはASRの転写を見るので、抑揚が平坦でも転写が合えば誤りにならない。

同一文・同一話者の**人間の実音声**を基準に、次を測る。

* **抑揚の幅**（`semitone_range`）— 「棒読み」は小さい
* **輪郭の一致**（`contour_similarity`）— 人間と同じ上下をしているか

* **アクセント核の位置**（`observed_nucleus`）— `箸` と `橋` を読み分けているか

アクセントは **モーラ単位の強制アラインメント**（`MMS_FA`）で音声とテキストを
対応付けてから測る。基準は2つ置く。

* **辞書**（`pyopenjtalk` の full-context label）との一致率。測定器が信号を
  拾えているかの確認。人間の実音声で **46.1%**（固定回答36.8% / 当てずっぽう25.1%）
* **人間の実音声**との一致率。**こちらが本来見たい値。** モデルが人間と
  同じところで下げているか。辞書が正しいかどうかに依存しない

    python scripts/evaluate_prosody.py \\
      --model-dir checkpoints/s1v2-fp32-30000 \\
      --eval-set data/eval/prosody_eval_set_v2.json --label trained --device cuda

## 並列で回す

**生成は `batch=1` の自己回帰なのでGPUが埋まらない**（実測: 使用率15〜71%、
消費電力45〜82W / TDP285W）。1プロセスが使うGPUメモリは 4.2 GiB なので、
**3分割して同時に走らせると約3倍**になる。

    for k in 1 2 3; do
      python scripts/evaluate_prosody.py --shard $k/3 --label trained-s$k ... &
    done; wait
    python scripts/evaluate_prosody.py --merge <s1>,<s2>,<s3> --label trained

`--merge` は生成をやり直さず、行を結合して集計だけを作る（GPU不要）。
集計は `cutetts.training.prosody.summarize_run` の1箇所にあるので、
**分割して測っても結果は同じになる**。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from cutetts import CuteTTS  # noqa: E402
from cutetts.training import artifacts  # noqa: E402
from cutetts.training.alignment import MoraAligner, phrase_plan  # noqa: E402
from cutetts.training.prosody import (  # noqa: E402
    contour_similarity,
    summarize_run,
    measure,
    mora_pitches,
    observed_nucleus,
    semitone_contour,
    track_f0,
)
from cutetts.training.yomi import FRONTEND_MODES, apply_frontend, frontend_text  # noqa: E402


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(str(path), dtype="float64")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sample_rate


def _cached_f0(cache, key, waveform, sample_rate):
    """同じファイルのF0を使い回す。referenceは話者ごとに1つ固定なので効く。"""
    if key not in cache:
        cache[key] = track_f0(waveform, sample_rate)
    return cache[key]


def _nuclei(aligner, text, waveform, sample_rate, f0):
    """アクセント句ごとの核の位置を読む。失敗したら空リスト。

    `f0` は計算済みのものを渡す。**`track_f0` は1発話で約2秒かかるので、
    同じ音声に対して呼び直さない。**
    """
    if aligner is None:
        return []
    phrases = phrase_plan(text)
    spans = aligner.align(waveform, sample_rate, text)
    if len(spans) != sum(len(p.moras) for p in phrases):
        return []
    pitches = mora_pitches(f0, spans)
    out = []
    offset = 0
    for phrase in phrases:
        size = len(phrase.moras)
        out.append(observed_nucleus(pitches[offset:offset + size]))
        offset += size
    return out


def _accent(aligner, text, human_wave, human_rate, human_f0,
            model_wave, model_rate, model_f0):
    """辞書・人間・モデルの核を並べて返す。"""
    if aligner is None:
        return {}
    expected = [p.internal_nucleus for p in phrase_plan(text)]
    human = _nuclei(aligner, text, human_wave, human_rate, human_f0)
    model = _nuclei(aligner, text, model_wave, model_rate, model_f0)
    if len(human) != len(expected) or len(model) != len(expected):
        return {"accent": None}
    return {"accent": {"expected": expected, "human": human, "model": model}}


def _f0_hook(path: Path, conditioner, vae, patch_size: int, device: str):
    """1発話ぶんの F0 条件を作る（M4c）。

    **`num_patches` は与える音声の長さから決める。** 生成がそれより長く
    続いた step は `None` を返して素通りさせる（条件を繰り返すと、
    存在しない高さを指定し続けることになる）。
    """
    from cutetts.training.f0 import (
        features_from_waveform,
        patch_features,
        roundtrip_waveform,
        step_embedding_hook,
    )
    from cutetts.training.latents import LATENT_SAMPLE_RATE

    wave, rate = read_audio(path)
    if rate != LATENT_SAMPLE_RATE:
        import torchaudio

        wave = torchaudio.functional.resample(
            torch.from_numpy(wave).float(), rate, LATENT_SAMPLE_RATE).numpy()
    if vae is not None:
        wave = roundtrip_waveform(vae, wave)
    features = features_from_waveform(np.asarray(wave, dtype=np.float64),
                                      LATENT_SAMPLE_RATE)
    num_patches = max(1, -(-len(features) // patch_size))
    patches = patch_features(features, patch_size=patch_size,
                             num_patches=num_patches)
    return step_embedding_hook(conditioner, patches, device=device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抑揚を測る（M1）")
    parser.add_argument("--model-dir")
    parser.add_argument("--eval-set", default="data/eval/prosody_eval_set_v2.json")
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decode-length", type=int, default=400)
    parser.add_argument("--expand-numerals", action="store_true",
                        help="J2（漢数字の読み展開）を掛けてから合成する")
    parser.add_argument("--frontend", choices=FRONTEND_MODES,
                        help="frontend をまとめて指定する（M4a）。"
                             "指定すると --expand-numerals / --assign-yomi より優先する")
    parser.add_argument("--assign-yomi", action="store_true",
                        help="J3（語の読み付与）を掛けてから合成する")
    parser.add_argument("--f0-source", default="none",
                        choices=("none", "oracle", "transfer", "mismatch"),
                        help="F0 の条件（M4c）。oracle はその文自身の人間音声、"
                             "transfer は同じ台詞の別テイク（天井setのみ）、"
                             "**mismatch は同一話者の別の文**（対照。効果が"
                             "特異的かを見る）")
    parser.add_argument("--no-f0-roundtrip", action="store_true",
                        help="F0 を取る前に VAE で往復させない。**既定は往復させる**"
                             "（学習側の F0 は decode 由来なので分布を揃える）")
    parser.add_argument("--no-accent", action="store_true",
                        help="アクセント核の測定を行わない（強制アラインメントを"
                             "省く）。初回は 1.18 GB のモデルを取得する")
    parser.add_argument("--save-samples", type=int, default=0,
                        help="生成音声を残す件数。**artifacts配下の音声は公開しない**")
    parser.add_argument("--no-warmup", action="store_true",
                        help="捨て生成を省く。**プロセス内の最初の生成だけ結果が"
                             "違う**ので、既定では1回捨てて揃える（--shard で"
                             "分けた結果を一括と一致させるのに要る）")
    parser.add_argument("--shard", metavar="K/N",
                        help="評価setを N 分割して K 番目だけを測る（1始まり）。"
                             "**生成はbatch=1でGPUが埋まらないので、分割して"
                             "同時に走らせると速い**")
    parser.add_argument("--merge", metavar="PATHS",
                        help="shardの metrics.json をカンマ区切りで渡すと、"
                             "生成をやり直さず行を結合して集計だけ作る（GPU不要）")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def parse_shard(text: str, total: int) -> list[int]:
    """`K/N` を、担当する index の一覧へ。

    **飛び飛びに取る**（`items[K-1::N]`）。発話の長さがまちまちなので、
    前半・後半で切ると担当の重さが偏る。
    """
    part, _, count = text.partition("/")
    index, size = int(part), int(count)
    if not (1 <= index <= size):
        raise SystemExit(f"--shard の値が範囲外: {text}")
    return list(range(index - 1, total, size))


def merge_rows(paths: str) -> list[dict]:
    """shardの metrics.json から行を集める。**index で重複を弾く。**"""
    seen: dict[int, dict] = {}
    for path in paths.split(","):
        payload = json.loads(Path(path.strip()).read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            key = row.get("index")
            if key in seen:
                raise SystemExit(f"index {key} が複数のshardにある: {path}")
            seen[key] = row
    return [seen[key] for key in sorted(seen)]


def write_metrics(run_dir, args, summary: dict, rows: list[dict]) -> None:
    """metrics.json を書く。**shardでも結合でも同じ形にする。**"""

    artifacts.write_run_metadata(
        run_dir, phase="prosody",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"model_dir": args.model_dir, "eval_set": args.eval_set},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "prosody", "label": args.label, "model_dir": str(args.model_dir),
        "eval_set": str(args.eval_set),
        "eval_set_sha256": artifacts.file_checksum(args.eval_set),
        "settings": {"seed": args.seed, "expand_numerals": args.expand_numerals,
                     "assign_yomi": args.assign_yomi,
                     "frontend": args.frontend,
                     "f0_source": args.f0_source,
                     "f0_roundtrip": not args.no_f0_roundtrip},
        "shard": args.shard, "merged_from": args.merge,
        "summary": summary, "rows": rows,
    })



def report(summary: dict, run_dir) -> None:
    """集計を表示する。"""
    print("\n=== 抑揚 ===")
    print(f"  n={summary['n']}  生成失敗 {summary['n_error']}")
    if not summary.get("n"):
        print("  **測定できた発話が無い**（有声フレームが足りないか、生成が全滅）")
        print(f"\n完了: {run_dir}")
        return
    print(f"  半音の幅（5〜95%）  人間 {summary['human_semitone_range']:5.2f}  "
          f"→ モデル {summary['model_semitone_range']:5.2f}")
    print(f"  半音sd              人間 {summary['human_semitone_sd']:5.2f}  "
          f"→ モデル {summary['model_semitone_sd']:5.2f}")
    print(f"  長さ(秒)            人間 {summary['human_seconds']:5.2f}  "
          f"→ モデル {summary['model_seconds']:5.2f}")
    if summary["contour_similarity_mean"] is not None:
        print(f"  輪郭の相関          平均 {summary['contour_similarity_mean']:5.2f}  "
              f"中央値 {summary['contour_similarity_median']:5.2f}")
    print(f"  人間より平坦だった文 {summary['n_flatter_than_human']}/{summary['n']}")
    if "above_floor" in summary:
        above = summary["above_floor"]
        print(f"\n  床（同一話者・別の文）平均 {summary['floor_similarity_mean']:+.3f}")
        print(f"  床との差 {above['difference']:+.3f} "
              f"95%CI [{above['low']:+.3f}, {above['high']:+.3f}]  "
              f"{'**有意に上回る**' if above['significant'] else '**床と区別できない**'}")
    accent = summary.get("accent")
    if accent:
        print("\n=== アクセント核 ===")
        print(f"  {accent['n_utterances']} 発話 / {accent['n_phrases']} アクセント句")
        for key, label in (("model_vs_human", "**モデル 対 人間**"),
                           ("human_vs_dictionary", "人間 対 辞書"),
                           ("model_vs_dictionary", "モデル 対 辞書")):
            entry = accent[key]
            if entry["rate"] is None:
                continue
            print(f"  {label:18s} {entry['rate']:6.1%}  (n={entry['n']})")
        print(f"  当てずっぽう {accent['chance']:6.1%}   "
              f"固定回答 {accent['constant']:6.1%}")
    else:
        print("\n  アクセント核は測っていない（--no-accent）")
    print(f"\n完了: {run_dir}")




def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("prosody", args.artifact_root,
                                    timestamp=args.timestamp)
    device = resolve_device(args.device)

    payload = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    audio_dir = artifacts.as_local_path(
        payload.get("audio_dir", "data/eval/prosody_audio"))
    items = payload["items"]

    if args.merge:
        # **生成をやり直さない。** 行を結合して集計だけ作る
        rows = merge_rows(args.merge)
        summary = summarize_run(rows)
        print(f"{len(rows)} 行を結合（shard {len(args.merge.split(','))} 個）")
        write_metrics(run_dir, args, summary, rows)
        report(summary, run_dir)
        return

    if args.model_dir is None:
        raise SystemExit("--model-dir が要る（--merge のときは不要）")
    # **話者は speaker_key（game×speaker）で数える。** golの話者IDは表示名の
    # SHA-256 なので、game をまたぐと別人でも同じIDになりうる。
    indices = (parse_shard(args.shard, len(items)) if args.shard
               else list(range(len(items))))
    keys = {items[i].get("speaker_key") or items[i]["speaker"] for i in indices}
    shard_note = f"  shard {args.shard}" if args.shard else ""
    print(f"{len(indices)}/{len(items)} 文 / {len(keys)} 話者  "
          f"device={device}{shard_note}")

    model = CuteTTS.from_pretrained(args.model_dir, device=str(device))
    assigner = None
    if args.assign_yomi or args.frontend == "yomi":
        from cutetts.training.yomi import ReadingAssigner

        assigner = ReadingAssigner.from_model_dir(args.model_dir)

    # M4c: F0 の条件。**conditioner が無ければ黙って素通りさせない**
    f0_conditioner = None
    f0_vae = None
    patch_size = 2
    if args.f0_source != "none":
        from cutetts.training.f0 import load_f0_conditioner

        f0_conditioner = load_f0_conditioner(args.model_dir, device=str(device))
        if f0_conditioner is None:
            raise SystemExit(
                f"--f0-source={args.f0_source} だが "
                f"{args.model_dir}/f0_conditioner.safetensors が無い")
        if not args.no_f0_roundtrip:
            from cutetts.modeling.audio_adapter import AudioAcousticVAEAdapter

            f0_vae = AudioAcousticVAEAdapter(
                Path(args.model_dir) / "weights" / "audio_vae").to(device).eval()
        patch_size = int(json.loads(
            (Path(args.model_dir) / "config.json").read_text(encoding="utf-8")
        )["architecture"]["locenc_patch_size"])
        print(f"F0 条件: {args.f0_source}"
              f"（往復 {'なし' if args.no_f0_roundtrip else 'あり'} / patch {patch_size}）")

    aligner = None
    if not args.no_accent:
        # アラインメントのモデルは初回のみ 1.18 GB を取得する
        aligner = MoraAligner(device=str(device))

    samples_dir = run_dir / "samples"
    if args.save_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)

    # **プロセス内の最初の生成だけ結果が違う。** 実測で、同じ文でも
    # 1件目に生成したときと2件目以降で幅が 21.99 / 13.60 と変わった
    # （seedは呼び出しごとに設定しているのに）。初回の遅延初期化が乱数列を
    # ずらしていると見られる。**捨て生成を1回入れて揃える。**
    # これをしないと --shard で分けた結果が一括と一致しない。
    if not args.no_warmup:
        model.generate(items[indices[0]]["text"], mode="voice_clone",
                       reference_audio=str(audio_dir / items[indices[0]]["reference_wav"]),
                       seed=args.seed, max_decode_length=args.max_decode_length,
                       show_progress=False)

    rows: list[dict] = []
    saved = 0
    # referenceは話者ごとに同じファイルなので、**ファイル名で使い回す**
    f0_cache: dict[str, np.ndarray] = {}
    for index in indices:
        item = items[index]
        text = item["text"]
        if args.frontend:
            spoken = frontend_text(text, args.frontend, assigner=assigner)
        else:
            # **J3 → J2 の順**（逆にすると数詞が壊れる。`yomi.apply_frontend`）
            spoken = apply_frontend(text, assigner=assigner,
                                    expand_numerals=args.expand_numerals)
        # **J3 をもう一度掛けてはいけない。** `frontend_text` / `apply_frontend`
        # の中で既に掛かっている。2回掛けると J2 の出力を再解釈して漢数字が
        # 戻る（`マコトニ` → `マコト二`、`さんじゅうご` → `さんジュウゴ`）。
        # 実測で prosody set 240文のうち2文（0.8%）が壊れていた。
        reference = audio_dir / item["reference_wav"]
        f0_hook = None
        if f0_conditioner is not None:
            source_key = {"oracle": "human_wav",
                          "transfer": "take_b_wav",
                          "mismatch": "reference_wav"}[args.f0_source]
            if not item.get(source_key):
                rows.append({"index": index, "text": text, "status": "error",
                             "detail": f"{source_key} が無い（--f0-source"
                                       f"={args.f0_source}）"})
                continue
            f0_hook = _f0_hook(audio_dir / item[source_key], f0_conditioner,
                               f0_vae, patch_size, str(device))
        try:
            result = model.generate(
                spoken, mode="voice_clone", reference_audio=str(reference),
                seed=args.seed, max_decode_length=args.max_decode_length,
                show_progress=False, extra_step_embedding=f0_hook,
            )
        except Exception as error:                 # 失敗も記録して先へ進む
            rows.append({"index": index, "text": text, "status": "error",
                         "detail": f"{type(error).__name__}: {error}"[:200]})
            continue

        model_wave = result.waveform.squeeze(0).float().numpy().astype(np.float64)
        human_wave, human_rate = read_audio(audio_dir / item["human_wav"])
        reference_wave, reference_rate = read_audio(reference)

        # **F0は1音声につき1回だけ計算する。** `track_f0` は1発話で約2秒
        # かかるので、呼び直すと1件あたり7回＝14秒になる（実測で1.5分/件）。
        human_f0 = _cached_f0(f0_cache, item["human_wav"], human_wave, human_rate)
        reference_f0 = _cached_f0(f0_cache, item["reference_wav"],
                                  reference_wave, reference_rate)
        model_f0 = track_f0(model_wave, result.sample_rate)

        model_stats = measure(model_wave, result.sample_rate, f0=model_f0)
        human_stats = measure(human_wave, human_rate, f0=human_f0)
        human_contour = semitone_contour(human_f0)
        similarity = contour_similarity(
            human_contour, semitone_contour(model_f0))
        # **床を同じ文ごとに測る。** reference は同一話者の**別の文**なので、
        # 内容を共有しないときの相関になる（240文で平均 -0.001 / sd 0.152。
        # **真の床はほぼゼロ**。n=67 のときの +0.070 はノイズだった）。
        # モデルがこれを有意に上回らなければ、抑揚を再現できていない。
        floor = contour_similarity(human_contour, semitone_contour(reference_f0))
        accent = _accent(aligner, text, human_wave, human_rate, human_f0,
                         model_wave, result.sample_rate, model_f0)
        rows.append({
            "index": index, "text": text, "speaker": item["speaker"],
            "speaker_key": item.get("speaker_key") or item["speaker"],
            "group": item.get("group"), "status": "ok",
            "human": vars(human_stats), "model": vars(model_stats),
            "contour_similarity": None if np.isnan(similarity) else float(similarity),
            "floor_similarity": None if np.isnan(floor) else float(floor),
            "spoken": None if spoken == text else spoken,
            **accent,
        })
        if saved < args.save_samples:
            sf.write(samples_dir / f"{index:03d}.wav",
                     result.waveform.squeeze(0).float().numpy(), result.sample_rate)
            saved += 1
        print(f"  [{index:3d}] 幅 人{human_stats.semitone_range:5.1f} "
              f"/ 模{model_stats.semitone_range:5.1f}  相関 {similarity:5.2f}")

    summary = summarize_run(rows)
    write_metrics(run_dir, args, summary, rows)
    report(summary, run_dir)


if __name__ == "__main__":
    main()
