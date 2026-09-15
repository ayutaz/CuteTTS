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

"""凍結済みの抑揚評価setに必要な人間音声だけを gol から取り出す。

**評価setは作り直さない。** `build_prosody_set.py` は選定からやり直すので、
新しいインスタンスで使うと別のsetになる。こちらは **JSONに書かれている
wavだけ**を取り出すので、setは凍結したまま音声を揃えられる。

**音声は転送・公開しない。** 学習データのライセンス上、抽出した音声を
リポジトリへ入れたり配布したりしてはならない。必要な機械の上で毎回
取り出す（`data/` はgitignore済み）。

    HF_TOKEN=<read権限> python scripts/fetch_prosody_audio.py

gol の tar は **`_partN` に分割されていることがある**（S1で2 gameを丸ごと
取りこぼした）。game_id で始まるファイルをすべて拾う。
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutetts.training import artifacts  # noqa: E402

GOL_REPO = "midralab/gol-dataset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抑揚評価setの音声を取り出す")
    parser.add_argument("--eval-set", default="data/eval/prosody_eval_set_v2.json")
    parser.add_argument("--tar-dir", default="data/raw/gol/tars")
    parser.add_argument("--repo", default=GOL_REPO)
    parser.add_argument("--keep-tars", action="store_true",
                        help="取り出したあともtarを残す（既定は消して容量を空ける）")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    items = payload["items"]
    audio_dir = artifacts.as_local_path(
        payload.get("audio_dir", "data/eval/prosody_audio"))
    audio_dir.mkdir(parents=True, exist_ok=True)

    # 必要な (game_id, 話者id, ファイル名) を集める。
    # ファイル名は `{game8}_{speaker8}_{もとの名前}` の形で保存されている。
    wanted: dict[str, dict[str, str]] = defaultdict(dict)
    for item in items:
        for key in ("human_wav", "reference_wav"):
            name = item[key]
            game8, speaker8, original = name.split("_", 2)
            if game8 != item["game_id"][:8]:
                raise SystemExit(f"game_idと名前が合わない: {name}")
            # tar内のpathは `{話者idの全体}/{もとの名前}`。話者idは item にある
            wanted[item["game_id"]][f"{item['speaker']}/{original}"] = name

    have = {path.name for path in audio_dir.glob("*.wav")}
    need = {game: {k: v for k, v in members.items() if v not in have}
            for game, members in wanted.items()}
    need = {game: members for game, members in need.items() if members}
    total = sum(len(v) for v in need.values())
    print(f"{len(items)} 文 / 音声 {sum(len(v) for v in wanted.values())} 本 "
          f"（未取得 {total} 本 / {len(need)} game）")
    if not need:
        print("すべて揃っている")
        return

    from huggingface_hub import hf_hub_download, list_repo_files

    available = [name for name in list_repo_files(args.repo, repo_type="dataset")
                 if name.endswith(".tar")]
    tar_dir = Path(args.tar_dir)
    tar_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    for game, members in sorted(need.items()):
        # **`_partN` に分割されていることがある。** 名前をそのまま使うと
        # エラーも出さずに game が丸ごと落ちる（S1で170時間・52%を失った）
        files = [name for name in available
                 if Path(name).name.startswith(game)]
        if not files:
            raise SystemExit(f"{game} のtarがrepoに無い")
        print(f"  {game[:8]}: tar {len(files)} 本 / 音声 {len(members)} 本")
        for name in sorted(files):
            local = Path(hf_hub_download(args.repo, name, repo_type="dataset",
                                         local_dir=str(tar_dir)))
            with tarfile.open(local) as archive:
                for member in archive:
                    target = members.get(member.name)
                    if target is None:
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        continue
                    (audio_dir / target).write_bytes(source.read())
                    extracted += 1
            if not args.keep_tars:
                local.unlink(missing_ok=True)

    missing = [item[key] for item in items for key in ("human_wav", "reference_wav")
               if not (audio_dir / item[key]).is_file()]
    print(f"\n取り出した {extracted} 本。不足 {len(set(missing))} 本")
    if missing:
        raise SystemExit(f"音声が足りない: {sorted(set(missing))[:5]}")
    print(f"完了: {audio_dir}")


if __name__ == "__main__":
    main()
