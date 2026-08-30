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

"""学習側の prompt 構築。**推論と完全に同じ並びを作る。**

推論の `CuteTTSProcessor._reference_prompt_segments` は voice_clone で次を並べる。

    [ "Transform the text into speech output, utilizing the distinct voice
       of the provided speech sample.\\nvoice reference:\\n<|im_start|>" ]
    [ speaker slot (1 token) ]
    [ "<|im_end|>\\n<|im_start|>" ]
    [ reference speech patches ]
    [ "<|im_end|>\\ntext input:\\n{target_text}\\n<|endofprompt|>" ]

tts（reference なし）は次の1本だけ。

    "Transform the text into speech output.\\ntext input:\\n{text}\\n<|endofprompt|>"

**ここがずれると学習と推論で条件が食い違い、S0のゲートが意味を失う。**
そのため文字列は推論実装からコピーせず、`CuteTTSProcessor` の
メソッドを直接呼んで組み立てる。テンプレートが変わっても自動で追従する。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from cutetts.modeling.processor import CuteTTSProcessor


@dataclass(frozen=True)
class PromptLayout:
    """prefix の並び。speech / speaker slot の位置は collator が埋める。"""

    leading_ids: Tensor
    """speaker slot より前に来る text token。"""
    middle_ids: Tensor
    """speaker slot と reference speech の間に来る text token。"""
    trailing_ids: Tensor
    """reference speech の後に来る text token（target text を含む）。"""
    has_speaker_slot: bool

    @property
    def text_token_count(self) -> int:
        return int(self.leading_ids.shape[0] + self.middle_ids.shape[0]
                   + self.trailing_ids.shape[0])


def build_voice_clone_prompt(processor: CuteTTSProcessor, target_text: str) -> PromptLayout:
    """voice_clone の prompt を推論と同じ並びで作る。"""
    tokenizer = processor.tokenizer
    leading = tokenizer.encode(
        "Transform the text into speech output, utilizing the distinct voice "
        "of the provided speech sample.\nvoice reference:\n<|im_start|>"
    )
    middle = tokenizer.encode("<|im_end|>\n<|im_start|>")
    trailing = tokenizer.encode(
        f"<|im_end|>\ntext input:\n{target_text}\n{processor.text_suffix_token}"
    )
    return PromptLayout(
        leading_ids=torch.tensor(leading, dtype=torch.long),
        middle_ids=torch.tensor(middle, dtype=torch.long),
        trailing_ids=torch.tensor(trailing, dtype=torch.long),
        has_speaker_slot=True,
    )


def build_tts_prompt(processor: CuteTTSProcessor, text: str) -> PromptLayout:
    """reference なしの prompt。speaker slot も reference speech も無い。"""
    ids = processor.tokenizer.encode(processor._text_only_prompt(text))
    empty = torch.zeros(0, dtype=torch.long)
    return PromptLayout(
        leading_ids=empty,
        middle_ids=empty,
        trailing_ids=torch.tensor(ids, dtype=torch.long),
        has_speaker_slot=False,
    )
