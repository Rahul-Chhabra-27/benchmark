# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What every sub-call did and cost, so a run can be inspected decision by decision.

One CallRecord per llm_query: which turn, how much it read, at which factor, how its
KV split between GPU and CPU (offloaded), what it cost, whether it was refused, and
-- once the question ends -- whether compression was needed and whether the read fed
the final answer. `summarize` folds them into per-question numbers for metrics.json.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

PREVIEW_CHARS = 200


@dataclass(frozen=True)
class Placement:
    """Where one read's KV lives: kept = read / F; up to B on the GPU, the rest offloaded."""

    kept: int
    gpu: int
    offloaded: int

    @staticmethod
    def of(read_tokens: int, factor: int, budget_tokens: Optional[int]) -> "Placement":
        kept = math.ceil(read_tokens / factor)
        if budget_tokens is None:  # no budget: nothing is ever offloaded
            return Placement(kept, kept, 0)
        return Placement(kept, min(kept, budget_tokens), max(0, kept - budget_tokens))

    def cost(self, gpu_token: float, offloaded_token: float) -> float:
        return self.gpu * gpu_token + self.offloaded * offloaded_token


@dataclass(frozen=True)
class CallRecord:
    step: int
    question: str
    chunks: Optional[int]  # chunks passed, when the root read a chunk list
    read_tokens: int
    kept_tokens: int  # KV kept after compression (= read_tokens at F = 1)
    limit_tokens: int  # the most a read at this factor may carry
    budget_tokens: Optional[int]  # B, the KV one read may keep on the GPU; None without a budget
    factor: int
    seconds: float
    gpu_tokens: int = 0
    offloaded_tokens: int = 0
    cost: float = 0.0
    refused: str = ""  # why the read was not made, else ""
    answer: str = ""
    # Filled when the question ends:
    compression_needed: Optional[bool] = None  # F > 1 only: would the read NOT have fit in B uncompressed?
    contributed: Optional[bool] = None  # does the read's answer share a content word with FINAL's?

    def to_dict(self) -> dict:
        return asdict(self)


def preview(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1] + "…"


_WORD = re.compile(r"[a-z0-9]+")
_FILLER = frozenset("the a an of in on at to and or is was are were for by with from as that this it its".split())


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(str(text).lower()) if w not in _FILLER}


def contributed(read_answer: str, final_answer: Optional[str]) -> bool:
    """A read fed the answer if they share a content word. Crude, but it separates reads
    that found something from the "nothing relevant here" reads, which is the question."""
    return bool(final_answer) and bool(content_words(read_answer) & content_words(final_answer or ""))


def summarize(calls: Iterable[CallRecord]) -> dict:
    """Per-question numbers: how many reads, how big, at which factors, what they cost."""
    calls = list(calls)
    made = [c for c in calls if not c.refused]
    out = {
        "sub_calls": len(made),
        "refused_calls": len(calls) - len(made),
        "read_tokens": sum(c.read_tokens for c in made),
        "kept_tokens_max": max((c.kept_tokens for c in made), default=0),
        "offloaded_tokens": sum(c.offloaded_tokens for c in made),
        "offloaded_calls": sum(1 for c in made if c.offloaded_tokens),
        "cost": sum(c.cost for c in made),
        "factors": dict(sorted(Counter(f"x{c.factor}" for c in made).items())),
        # Compressed reads that would have fit uncompressed: compression bought nothing there.
        "compressed_unneeded": sum(1 for c in made if c.compression_needed is False),
        "contributing_calls": sum(1 for c in made if c.contributed),
        "read_seconds": round(sum(c.seconds for c in made), 3),
    }
    if made:
        out["read_tokens_mean"] = out["read_tokens"] / len(made)
        # How much of what one read may carry the root actually used: near 0 means it
        # reads tiny slices and the compression budget buys nothing.
        out["read_fill_mean"] = sum(c.read_tokens / c.limit_tokens for c in made) / len(made)
    return out
