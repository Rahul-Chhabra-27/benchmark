# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What every sub-call did, so a run can be inspected decision by decision.

One CallRecord per llm_query: which turn, how much it read, how much KV it kept
against the budget, whether it was refused, how long it took, what came back.
`summarize` folds them into per-question numbers for metrics.json.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional

PREVIEW_CHARS = 200


@dataclass(frozen=True)
class CallRecord:
    step: int
    question: str
    chunks: Optional[int]  # chunks passed, when the root read a chunk list
    read_tokens: int
    kept_tokens: int
    limit_tokens: int  # the most one read may carry (N = B x F)
    budget_tokens: Optional[int]  # B, the KV one read may keep; None without a budget
    factor: int
    seconds: float
    refused: str = ""  # why the read was not made, else ""
    answer: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def preview(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1] + "…"


def summarize(calls: Iterable[CallRecord]) -> dict:
    """Per-question numbers: how many reads, how big, how full against their limit."""
    calls = list(calls)
    made = [c for c in calls if not c.refused]
    out = {
        "sub_calls": len(made),
        "refused_calls": len(calls) - len(made),
        "read_tokens": sum(c.read_tokens for c in made),
        "kept_tokens_max": max((c.kept_tokens for c in made), default=0),
        "read_seconds": round(sum(c.seconds for c in made), 3),
    }
    if made:
        out["read_tokens_mean"] = out["read_tokens"] / len(made)
        # How much of what one read may carry the root actually used: near 0 means it
        # reads tiny slices and the compression budget buys nothing.
        out["read_fill_mean"] = sum(c.read_tokens / c.limit_tokens for c in made) / len(made)
    return out
