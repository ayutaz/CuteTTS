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

"""アクセント最小対の評価setを作る（A1）。

**`箸` と `橋` は仮名にすると両方 `ハシ` になる。** J3（読み付与）は語を
仮名へ置き換えるので、この区別が入力から消える。M4a の核記号は
`ハ'シ` / `ハシ'` として残す。**どちらがどれだけ効いているかを測れるように
する。**

作るのは「**片仮名は完全に同一で、核の位置だけが違う文の対**」。

    その箸を見ていました。 → ソノハ'シオミテイマシタ。
    その橋を見ていました。 → ソノハシ'オミテイマシタ。

この対に対して frontend ごとに合成し、

* ``none``   漢字が残るので原理上は区別できる
* ``yomi``   語が仮名に置き換わると区別が消える（**置き換わるかは語による**）
* ``accent`` 核記号で区別が明示される

**モデルの音声が実際に区別しているか**を `observed_nucleus` で読む。
評価は `scripts/evaluate_accent_pairs.py`。

コーパスから語を拾うので、**実際に出てくる語だけ**が対象になる。

    python scripts/build_accent_pair_set.py --limit 200000 --pairs 60
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cutetts.training.accent import accent_marked_text  # noqa: E402
from cutetts.training.alignment import phrase_plan  # noqa: E402
from cutetts.training.yomi import frontend_text  # noqa: E402

#: 対象語を差し込む文型。
#:
#: **名詞がひとつのアクセント句になるものを選んだ。** 助詞が続くと句が
#: 伸びて核の位置が動くので、対の両側で同じだけ動くように文型は共有する。
CARRIERS = (
    "その{}を見ていました。",
    "ここに{}があります。",
    "新しい{}を買いました。",
    "{}のことを考えていた。",
    "{}はどこにありますか。",
)

#: 対象語のモーラ数。短すぎると核の位置が1通りしかなく、長いと辞書が外れる。
MORA_RANGE = (2, 4)


def is_orthographic_variant(left: str, right: str) -> bool:
    """同じ語の**書き方の違い**か（`父様` と `父さま`）。

    辞書は別語として別のアクセントを与えることがあるが、実際には同じ語で
    同じアクセント。**対に混ぜると正解が定義できない。**

    漢字だけを残して、一方が他方の先頭一致になるものを落とす
    （`父様`→`父様` と `父さま`→`父`。`自信`/`自身` は残る）。
    """
    def kanji_only(text: str) -> str:
        return "".join(ch for ch in text if "一" <= ch <= "鿿")

    a, b = kanji_only(left), kanji_only(right)
    if not a or not b:
        return True                        # 片方が仮名だけなら書き方の違い
    return a.startswith(b) or b.startswith(a)


def has_kanji(text: str) -> bool:
    """漢字を含むか。**`none` で区別できる対だけを残すために要る。**"""
    return any("一" <= ch <= "鿿" for ch in text)


def collect_words(corpus: Path, limit: int) -> dict[tuple[str, str, int], int]:
    """コーパスから (表記, 読み, 核の位置) ごとの出現数を集める。"""
    import pyopenjtalk

    counter: collections.Counter[tuple[str, str, int]] = collections.Counter()
    with corpus.open(encoding="utf-8", errors="replace") as handle:
        handle.readline()
        for line in itertools.islice(handle, limit):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            text = fields[2]
            if not (6 <= len(text) <= 60):
                continue
            try:
                features = pyopenjtalk.run_frontend(text)
            except Exception:
                continue
            for item in features:
                if item.get("pos") != "名詞":
                    continue
                # **固有名詞と助数詞を落とす。** 人名は辞書のアクセントが
                # 当てにならず、助数詞は文型に載せると意味をなさない
                # （「ここに枚があります」）
                if item.get("pos_group1") != "一般":
                    continue
                surface = item.get("string") or ""
                pron = item.get("pron") or ""
                if not surface or not pron or not has_kanji(surface):
                    continue
                size = int(item.get("mora_size") or 0)
                if not MORA_RANGE[0] <= size <= MORA_RANGE[1]:
                    continue
                counter[(surface, pron, int(item.get("acc") or 0))] += 1
    return dict(counter)


def find_candidates(counts: dict[tuple[str, str, int], int], *, min_count: int
                    ) -> list[tuple[str, tuple[str, int, int], tuple[str, int, int]]]:
    """読みが同じで**核の位置が違う**表記の対を探す。"""
    by_pron: dict[str, list[tuple[str, int, int]]] = collections.defaultdict(list)
    for (surface, pron, acc), count in counts.items():
        if count < min_count:
            continue
        by_pron[pron].append((surface, acc, count))

    pairs = []
    for pron, entries in by_pron.items():
        # 同じ表記で核が揺れるものは辞書の曖昧性なので落とす
        surfaces = {surface for surface, _, _ in entries}
        if len(surfaces) < 2:
            continue
        best: dict[str, tuple[str, int, int]] = {}
        for surface, acc, count in entries:
            if surface not in best or count > best[surface][2]:
                best[surface] = (surface, acc, count)
        ranked = sorted(best.values(), key=lambda item: -item[2])
        for left, right in itertools.combinations(ranked, 2):
            if left[1] == right[1]:
                continue
            if is_orthographic_variant(left[0], right[0]):
                continue
            pairs.append((pron, left, right))
            break
    return pairs


def build_item(pron: str, left: tuple[str, int, int], right: tuple[str, int, int]
               ) -> dict | None:
    """文型を順に試して、**核の位置だけが違う対**になるものを1つ返す。"""
    for carrier in CARRIERS:
        text_left = carrier.format(left[0])
        text_right = carrier.format(right[0])
        marked_left = accent_marked_text(text_left)
        marked_right = accent_marked_text(text_right)
        if marked_left.unknown or marked_right.unknown:
            continue
        # 片仮名は同一でなければならない（読みが同じなので普通は成り立つ）
        bare_left = marked_left.text.replace("'", "")
        bare_right = marked_right.text.replace("'", "")
        if bare_left != bare_right:
            continue
        if marked_left.text == marked_right.text:
            continue                       # 文脈で核が揃ってしまった

        plan_left = phrase_plan(text_left)
        plan_right = phrase_plan(text_right)
        if len(plan_left) != len(plan_right):
            continue
        if [len(p.moras) for p in plan_left] != [len(p.moras) for p in plan_right]:
            continue
        differing = [index for index, (a, b) in enumerate(zip(plan_left, plan_right))
                     if a.internal_nucleus != b.internal_nucleus]
        if len(differing) != 1:
            continue                       # 差が1句に収まらないと測れない
        index = differing[0]

        variants = []
        for surface, acc, plan in ((left[0], left[1], plan_left),
                                   (right[0], right[1], plan_right)):
            text = carrier.format(surface)
            variants.append({
                "surface": surface,
                "dictionary_accent": acc,
                "text": text,
                "text_yomi": frontend_text(text, "yomi"),
                "text_accent": frontend_text(text, "accent"),
                "expected_nucleus": plan[index].internal_nucleus,
            })
        return {
            "pair_id": hashlib.sha256(
                f"{pron}:{left[0]}:{right[0]}".encode("utf-8")).hexdigest()[:12],
            "reading": pron,
            "kana": bare_left,
            "carrier": carrier,
            "phrase_index": index,
            "phrase_moras": len(plan_left[index].moras),
            # **J3 で区別が消えるか。** 語が仮名へ置き換わるかは語による
            "yomi_identical": variants[0]["text_yomi"] == variants[1]["text_yomi"],
            "variants": variants,
        }
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path,
                        default=Path("data/raw/gol/metadata.tsv"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/eval/accent_pair_set_v1.json"))
    parser.add_argument("--limit", type=int, default=200000,
                        help="走査するコーパス行数")
    parser.add_argument("--min-count", type=int, default=3,
                        help="語の最小出現数")
    parser.add_argument("--pairs", type=int, default=60, help="採用する対の数")
    args = parser.parse_args()

    print(f"コーパスを走査する（{args.limit:,} 行）...", flush=True)
    counts = collect_words(args.corpus, args.limit)
    print(f"  名詞 {len(counts):,} 種")

    candidates = find_candidates(counts, min_count=args.min_count)
    print(f"  読みが同じで核が違う対: {len(candidates):,} 組")

    items = []
    dropped = 0
    for pron, left, right in sorted(candidates,
                                    key=lambda item: -min(item[1][2], item[2][2])):
        item = build_item(pron, left, right)
        if item is None:
            dropped += 1
            continue
        items.append(item)
        if len(items) >= args.pairs:
            break

    if not items:
        print("対が作れなかった", file=sys.stderr)
        return 1

    yomi_lost = sum(1 for item in items if item["yomi_identical"])
    payload = {
        "version": "accent_pair_v1",
        "carriers": list(CARRIERS),
        "pairs": len(items),
        "yomi_identical": yomi_lost,
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()

    print(f"\n  採用 {len(items)} 対 / 文型に載らず落とした {dropped} 組")
    print(f"  **J3 で区別が消える対: {yomi_lost}/{len(items)}**"
          f"（{yomi_lost / len(items):.0%}）")
    print(f"  {args.out}  sha256={digest[:16]}…")
    print("\n  例:")
    for item in items[:6]:
        a, b = item["variants"]
        print(f"    {a['surface']} / {b['surface']}  （{item['kana']}）")
        print(f"      {a['text_accent']}   核 {a['expected_nucleus']}")
        print(f"      {b['text_accent']}   核 {b['expected_nucleus']}")
        if item["yomi_identical"]:
            print(f"      J3: {a['text_yomi']}  ← **両方これになる**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
