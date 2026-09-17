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

"""抑揚・アクセントの**天井**を測るためのsetを作る（M2）。

M1 では「同一文・同一話者の別テイクが無いので天井が測れない」と書いた。
gol の `metadata.tsv` を全走査すると、**同じ game・同じ話者・同じ台詞で
長さの違う録音が 27,367組（557 game / 4,138話者）ある**（2〜15秒・語彙的
内容ありに限った数）。人間 対 人間で測れば、そのまま指標の天井になる。

**出るのは天井の下限。** 同じ台詞でも感情や文脈は違いうるし、
別テイクは「同じ読み方をしようとした2回」ではない。

規約は `build_prosody_set.py` に合わせる（対象2〜15秒 / 語彙的内容あり /
1話者あたりの上限あり / 選定は seed とパスだけで決まる）。

**ファイル名の規約も合わせる**（`{game8}_{speaker8}_{もとの名前}`）ので、
`fetch_prosody_audio.py --eval-set <このJSON>` で音声を揃えられる。

    python scripts/build_retake_set.py --games <id>,<id> --count 180
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutetts.training import artifacts, text_rules  # noqa: E402

#: 対象テイクの長さの範囲（秒）。`build_prosody_set.py` と同じ。
MIN_TAKE_SECONDS = 2.0
MAX_TAKE_SECONDS = 15.0

#: 床（同一話者・別の文）に使う音声の最短の長さ（秒）。
MIN_FLOOR_SECONDS = 3.0

#: 1話者から採る対象の上限。少数の話者に偏らせない。
MAX_PER_SPEAKER = 4

_STRIP = re.compile(r"[\s、。「」『』・…‥！？!?,.\-―ー~〜\"'()（）]")


def has_lexical_content(text: str) -> bool:
    """語彙的内容を持つか。`build_prosody_set.py` と同じ規則。"""
    stripped = _STRIP.sub("", unicodedata.normalize("NFKC", text))
    if len(stripped) < 10:
        return False
    if re.search(r"(.)\1{3,}", stripped):
        return False
    if len(set(stripped)) / len(stripped) < 0.45:
        return False
    return bool(re.search(r"[ァ-ヶ一-龥]", stripped))


def usable_take(row: dict) -> bool:
    """テイクとして使えるか（長さとテキストの性質だけで決める）。"""
    if not (MIN_TAKE_SECONDS <= row["seconds"] <= MAX_TAKE_SECONDS):
        return False
    text = row["text"]
    if text_rules.is_punctuation_only(text) or text_rules.contains_markup(text):
        return False
    if text_rules.has_name_placeholder(text):
        return False
    return has_lexical_content(text)


def local_name(game_id: str, speaker_id: str, file_path: str) -> str:
    """抽出後のファイル名。`build_prosody_set.py` と同じ規約。"""
    return f"{game_id[:8]}_{speaker_id[:8]}_{Path(file_path).name}"


def choose_pairs(rows: list[dict], *, seed: int, max_per_speaker: int = MAX_PER_SPEAKER):
    """別テイク対を選ぶ。**同じ入力なら必ず同じ結果になる。**

    Args:
        rows: `game_id` / `speaker_id` / `text` / `file_path` / `seconds` を持つ辞書の列。

    Returns:
        `(take_a, take_b, 同一話者の別の文)` の組の列。
        別の文が無い話者は**落とす**（床を同じ素材で測れないため）。
    """
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    by_speaker: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["game_id"], row["speaker_id"])
        by_speaker[key].append(row)
        groups[(row["game_id"], row["speaker_id"], row["text"])].append(row)

    def order(row: dict) -> str:
        return hashlib.sha256(
            f"{seed}:{row['file_path']}".encode("utf-8")).hexdigest()

    pairs: list[tuple[dict, dict, dict]] = []
    for (game_id, speaker_id, text), takes in sorted(groups.items()):
        paths = {row["file_path"] for row in takes}
        lengths = {round(row["seconds"], 2) for row in takes}
        # **長さが同じものは重複登録の可能性がある。** 同じ録音を2回
        # 数えると天井が不当に高く出る（相関が 1.0 になる）
        if len(paths) < 2 or len(lengths) < 2:
            continue
        usable = [row for row in takes if usable_take(row)]
        if len({row["file_path"] for row in usable}) < 2:
            continue
        take_a, take_b = sorted(usable, key=order)[:2]
        # 床は**同一話者の別の文**（`build_prosody_set.py` の reference と同じ考え方）
        others = [row for row in by_speaker[(game_id, speaker_id)]
                  if row["text"] != text and row["seconds"] >= MIN_FLOOR_SECONDS]
        if not others:
            continue
        floor = max(others, key=lambda r: r["seconds"])
        pairs.append((take_a, take_b, floor))

    picked: list[tuple[dict, dict, dict]] = []
    per_speaker: dict[tuple[str, str], int] = defaultdict(int)
    for take_a, take_b, floor in sorted(pairs, key=lambda p: order(p[0])):
        key = (take_a["game_id"], take_a["speaker_id"])
        if per_speaker[key] >= max_per_speaker:
            continue
        per_speaker[key] += 1
        picked.append((take_a, take_b, floor))
    return picked


def to_item(take_a: dict, take_b: dict, floor: dict) -> dict:
    """評価setの1件へ。**キー名は `prosody_eval_set_v2` に合わせる。**

    `human_wav` がテイクA、`take_b_wav` がテイクB、`reference_wav` が床
    （同一話者の別の文）。こうしておくと `fetch_prosody_audio.py` と
    `summarize_run` をそのまま使える。
    """
    game_id, speaker_id = take_a["game_id"], take_a["speaker_id"]
    return {
        "text": take_a["text"],
        "speaker": speaker_id,
        "speaker_key": f"{game_id[:8]}:{speaker_id[:8]}",
        "game_id": game_id,
        "human_wav": local_name(game_id, speaker_id, take_a["file_path"]),
        "take_b_wav": local_name(game_id, speaker_id, take_b["file_path"]),
        "reference_wav": local_name(floor["game_id"], speaker_id, floor["file_path"]),
        "human_seconds": take_a["seconds"],
        "take_b_seconds": take_b["seconds"],
        "reference_seconds": floor["seconds"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="別テイクのsetを作る（M2）")
    parser.add_argument("--gol-metadata", default="data/raw/gol/metadata.tsv")
    parser.add_argument("--games", required=True,
                        help="対象 game_id をカンマ区切りで。tar は game 単位なので"
                             "**game を絞るとダウンロード量が決まる**")
    parser.add_argument("--audio-dir", default="data/eval/prosody_ceiling_audio")
    parser.add_argument("--out", default="data/eval/prosody_ceiling_set_v1.json")
    parser.add_argument("--count", type=int, default=180)
    parser.add_argument("--max-per-speaker", type=int, default=MAX_PER_SPEAKER)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--timestamp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = artifacts.new_run_dir("prosody-ceiling-set", args.artifact_root,
                                    timestamp=args.timestamp)

    games = {g.strip() for g in args.games.split(",") if g.strip()}
    rows: list[dict] = []
    with Path(args.gol_metadata).open(encoding="utf-8", newline="") as handle:
        handle.readline()
        for line in handle:
            if line[:32] not in games:
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 5:
                continue
            game_id, speaker_id, text, file_path, duration = fields[:5]
            try:
                seconds = float(duration)
            except ValueError:
                continue
            rows.append({"game_id": game_id, "speaker_id": speaker_id,
                         "text": text.strip(), "file_path": file_path,
                         "seconds": seconds})
    print(f"{len(rows):,} 発話 / {len(games)} game を読んだ")

    pairs = choose_pairs(rows, seed=args.seed, max_per_speaker=args.max_per_speaker)
    picked = pairs[:args.count]
    if len(picked) < args.count:
        print(f"**候補が {len(picked)} 組しかない**（要求 {args.count}）")
    items = [to_item(*pair) for pair in picked]
    speakers = {item["speaker_key"] for item in items}
    print(f"{len(items)} 組 / {len(speakers)} 話者")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": 1,
        "seed": args.seed,
        "created_for": "M2 / 抑揚とアクセントの天井",
        "audio_dir": Path(args.audio_dir).as_posix(),
        "note": (
            "同じ game・同じ話者・同じ台詞の**別テイク**の組。"
            "human_wav がテイクA、take_b_wav がテイクB、reference_wav が"
            "**同一話者の別の文**（床）。人間 対 人間で測ると指標の天井になる。"
            "**出るのは天井の下限**（別テイクは感情や文脈が違いうる）。"
            "**結果を見てから変更しないこと。**"
        ),
        "items": items,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for item in items[:3]:
        print(f"  {item['speaker_key']} {item['human_seconds']:5.2f}s / "
              f"{item['take_b_seconds']:5.2f}s  {item['text'][:30]}")

    artifacts.write_run_metadata(
        run_dir, phase="prosody-ceiling-set",
        command=[Path(sys.argv[0]).name] + sys.argv[1:], seed=args.seed,
        inputs={"gol_metadata": args.gol_metadata, "games": sorted(games)},
    )
    artifacts.write_metrics(run_dir, {
        "phase": "prosody-ceiling-set", "count": len(items),
        "speakers": len(speakers), "games": sorted(games),
        "output": str(out), "sha256": artifacts.file_checksum(out),
    })
    print(f"\n完了: {out}  sha256 {artifacts.file_checksum(out)[:16]}...")


if __name__ == "__main__":
    main()
