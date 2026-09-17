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

"""M2（天井）の別テイクsetづくり（`scripts/build_retake_set.py`）。

**同じ録音を2回数えると天井が 1.0 に近づいて無意味になる。**
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from build_retake_set import choose_pairs, to_item  # noqa: E402

TEXT = "今日はいい天気ですね、出かけましょう"
OTHER = "また明日の朝に集まりましょうか"


def row(path: str, seconds: float, text: str = TEXT, speaker: str = "S1") -> dict:
    return {"game_id": "G" * 32, "speaker_id": speaker, "text": text,
            "file_path": f"{'G' * 32}/{speaker}/{path}", "seconds": seconds}


def base_rows() -> list[dict]:
    return [row("a.wav", 3.0), row("b.wav", 3.4), row("floor.wav", 9.0, text=OTHER)]


def test_別テイクの組を選ぶ():
    pairs = choose_pairs(base_rows(), seed=1)
    assert len(pairs) == 1
    take_a, take_b, floor = pairs[0]
    assert {Path(take_a["file_path"]).name, Path(take_b["file_path"]).name} == {"a.wav", "b.wav"}
    assert Path(floor["file_path"]).name == "floor.wav"


def test_長さが同じ組は落とす():
    """**同一長は同じ録音の重複登録でありうる。**"""
    rows = [row("a.wav", 3.0), row("b.wav", 3.0), row("floor.wav", 9.0, text=OTHER)]
    assert choose_pairs(rows, seed=1) == []


def test_テイクが1本しかなければ落とす():
    rows = [row("a.wav", 3.0), row("floor.wav", 9.0, text=OTHER)]
    assert choose_pairs(rows, seed=1) == []


def test_床になる別の文が無ければ落とす():
    rows = [row("a.wav", 3.0), row("b.wav", 3.4)]
    assert choose_pairs(rows, seed=1) == []


def test_短すぎるテイクは使わない():
    rows = [row("a.wav", 1.0), row("b.wav", 1.2), row("floor.wav", 9.0, text=OTHER)]
    assert choose_pairs(rows, seed=1) == []


def test_語彙的内容の無いテキストは落とす():
    rows = [row("a.wav", 3.0, text="ああああああああああ"),
            row("b.wav", 3.4, text="ああああああああああ"),
            row("floor.wav", 9.0, text=OTHER)]
    assert choose_pairs(rows, seed=1) == []


def test_1話者あたりの上限が効く():
    rows = [row("floor.wav", 9.0, text=OTHER)]
    for index in range(6):
        text = f"{TEXT}{index}"
        rows += [row(f"a{index}.wav", 3.0, text=text),
                 row(f"b{index}.wav", 3.5, text=text)]
    assert len(choose_pairs(rows, seed=1, max_per_speaker=4)) == 4


def test_同じseedなら同じ組になる():
    rows = base_rows()
    assert choose_pairs(rows, seed=9) == choose_pairs(rows, seed=9)


def test_話者が違えば別の組として数える():
    rows = base_rows()
    rows += [row("c.wav", 3.0, speaker="S2"), row("d.wav", 3.6, speaker="S2"),
             row("floor2.wav", 9.0, text=OTHER, speaker="S2")]
    pairs = choose_pairs(rows, seed=1)
    assert len(pairs) == 2
    assert len({p[0]["speaker_id"] for p in pairs}) == 2


def test_項目のキー名は抑揚setと同じ():
    """`fetch_prosody_audio.py` と `summarize_run` をそのまま使うため。"""
    take_a, take_b, floor = choose_pairs(base_rows(), seed=1)[0]
    item = to_item(take_a, take_b, floor)
    assert set(item) >= {"text", "speaker", "speaker_key", "game_id",
                         "human_wav", "take_b_wav", "reference_wav"}
    assert item["human_wav"].startswith("GGGGGGGG_S1_")
