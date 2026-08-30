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

"""日本語継続学習（fork独自）の学習・前処理コード。

upstream の ``src/cutetts/`` は推論専用で、このpackageだけがfork側の追加物。
既存の推論pathからは何もimportされないので、ここを触っても推論は壊れない。

このmoduleは他のP1担当が共通で使う契約だけを再exportする:

* :mod:`cutetts.training.artifacts` — run directory / env snapshot / checksum
* :mod:`cutetts.training.manifest`  — JSONL manifestのschema・validator・集計
"""

from __future__ import annotations

from cutetts.training.artifacts import (
    env_snapshot,
    file_checksum,
    new_run_dir,
    write_metrics,
    write_run_metadata,
)
from cutetts.training.manifest import (
    DATASET_IDS,
    VALIDATION_CODES,
    SplitCounts,
    Utterance,
    ValidationIssue,
    load_manifest,
    manifest_checksum,
    summarize,
    validate,
    write_manifest,
)

__all__ = [
    "DATASET_IDS",
    "SplitCounts",
    "Utterance",
    "VALIDATION_CODES",
    "ValidationIssue",
    "env_snapshot",
    "file_checksum",
    "load_manifest",
    "manifest_checksum",
    "new_run_dir",
    "summarize",
    "validate",
    "write_manifest",
    "write_metrics",
    "write_run_metadata",
]
