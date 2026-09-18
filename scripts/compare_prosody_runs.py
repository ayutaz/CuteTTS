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

"""抑揚・アクセントを2実行で比べる（M1 / M4）。

**これまで比較のたびに手で書いていた。** 同じ検定を何度も書くとずれるので
1箇所にする（`summarize_eval_runs.py --compare` の抑揚版）。

対応付けは**テキスト**で行う（indexではない。評価setが差し替わると
別の文どうしを比べてしまう。R-022）。

    python scripts/compare_prosody_runs.py m4a-accent-prosody m4a-shuffled-prosody

指標:

``contour_similarity``
    輪郭の相関。文ごとに1つ。
``model_vs_human`` / ``model_vs_dictionary`` / ``human_vs_dictionary``
    アクセント核の一致率。**文ごとの率**にしてから対応のある検定に入れる
    （句ごとに入れると文の長さで重みが変わる）。
``model_seconds``
    長さ。**記号を足すと伸びることがある**ので見る。
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutetts.training.evalstats import paired_compare  # noqa: E402


def find_run(label: str, root: str) -> dict:
    """ラベルで metrics.json を探す。**同じラベルが複数あれば最も新しいもの。**"""
    found: list[tuple[str, dict]] = []
    for path in sorted(glob.glob(f"{root}/prosody/*/metrics.json")):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("label") == label:
            found.append((path, payload))
    if not found:
        raise SystemExit(f"ラベルが見つからない: {label}（{root}/prosody/*）")
    return found[-1][1]


def agreement(pairs: list[tuple[int, int]]) -> float | None:
    """核の一致率。**判定できなかった句（負）は分母から外す。**"""
    usable = [(e, o) for e, o in pairs if e >= 0 and o >= 0]
    if not usable:
        return None
    return sum(1 for e, o in usable if e == o) / len(usable)


def rates(row: dict) -> dict[str, float | None]:
    """1文ぶんの一致率。`accent` が無い行は None。"""
    accent = row.get("accent")
    if not accent:
        return {"model_vs_human": None, "model_vs_dictionary": None,
                "human_vs_dictionary": None}
    expected = accent.get("expected") or []
    human = accent.get("human") or []
    model = accent.get("model") or []
    if not (len(expected) == len(human) == len(model)):
        return {"model_vs_human": None, "model_vs_dictionary": None,
                "human_vs_dictionary": None}
    return {
        "model_vs_human": agreement(list(zip(human, model))),
        "model_vs_dictionary": agreement(list(zip(expected, model))),
        "human_vs_dictionary": agreement(list(zip(expected, human))),
    }


def usable_rows(payload: dict) -> dict[str, dict]:
    """テキスト → 行。**status が ok の行だけ。**"""
    out: dict[str, dict] = {}
    for row in payload.get("rows", []):
        if row.get("status") != "ok":
            continue
        out[row["text"]] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs=2, metavar=("A", "B"))
    parser.add_argument("--artifact-root", default="artifacts")
    args = parser.parse_args()

    left, right = (find_run(label, args.artifact_root) for label in args.labels)
    rows_a, rows_b = usable_rows(left), usable_rows(right)
    common = sorted(set(rows_a) & set(rows_b))
    if len(common) < 2:
        raise SystemExit(f"共通の文が足りない（{len(common)}文）")

    print(f"\n=== {args.labels[0]}  →  {args.labels[1]} ===")
    print(f"  共通 {len(common)} 文（A {len(rows_a)} / B {len(rows_b)}。"
          "テキストで対応付け）")
    for payload, label in ((left, args.labels[0]), (right, args.labels[1])):
        errors = sum(1 for row in payload.get("rows", [])
                     if row.get("status") == "error")
        print(f"  {label}: 生成失敗 {errors} / frontend="
              f"{payload.get('frontend') or (payload.get('settings') or {}).get('frontend')}")

    metrics = [
        ("輪郭の相関", lambda row: row.get("contour_similarity"), "{:+.3f}"),
        ("モデル対人間", lambda row: rates(row)["model_vs_human"], "{:+.2%}"),
        ("モデル対辞書", lambda row: rates(row)["model_vs_dictionary"], "{:+.2%}"),
        ("長さ(秒)", lambda row: (row.get("model") or {}).get("seconds"), "{:+.3f}"),
    ]

    print()
    print(f"  {'指標':<14} {'A':>9} {'B':>9} {'差':>10}  "
          f"{'95%CI':>22}  判定")
    for name, getter, fmt in metrics:
        values_a: list[float] = []
        values_b: list[float] = []
        for text in common:
            left_value, right_value = getter(rows_a[text]), getter(rows_b[text])
            if left_value is None or right_value is None:
                continue
            if left_value != left_value or right_value != right_value:
                continue                         # NaN
            values_a.append(float(left_value))
            values_b.append(float(right_value))
        if len(values_a) < 2:
            print(f"  {name:<14} 測れない（n={len(values_a)}）")
            continue
        comparison = paired_compare(values_a, values_b)
        mean_a = sum(values_a) / len(values_a)
        mean_b = sum(values_b) / len(values_b)
        verdict = "**有意**" if comparison.significant else "有意差なし"
        scale = 1.0
        if fmt.endswith("%}"):
            shown_a, shown_b = f"{mean_a:.1%}", f"{mean_b:.1%}"
            diff = f"{comparison.difference * 100:+.2f}pt"
            interval = (f"[{comparison.low * 100:+.2f}, "
                        f"{comparison.high * 100:+.2f}] pt")
        else:
            shown_a, shown_b = f"{mean_a:+.3f}", f"{mean_b:+.3f}"
            diff = f"{comparison.difference * scale:+.3f}"
            interval = f"[{comparison.low:+.3f}, {comparison.high:+.3f}]"
        print(f"  {name:<14} {shown_a:>9} {shown_b:>9} {diff:>10}  "
              f"{interval:>22}  {verdict}  n={len(values_a)}")

    print()
    print("  **点推定の順位ではなく信頼区間で判断する。**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
