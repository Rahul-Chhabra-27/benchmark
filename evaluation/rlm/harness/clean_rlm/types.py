# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Result, memory, scratchpad, and token-accounting types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .prompts import CHARS_PER_TOKEN_ESTIMATE


class TokenCounter:
    """Token counter: uses tiktoken cl100k_base if installed, else len//4."""

    def __init__(self, fn: Optional[Callable[[str], int]] = None):
        self._fn = fn
        self._enc = None
        if fn is None:
            try:  # pragma: no cover - depends on environment
                import tiktoken

                self._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._enc = None

    def count(self, text: str) -> int:
        if self._fn is not None:
            return self._fn(text)
        if self._enc is not None:
            try:  # pragma: no cover
                return len(self._enc.encode(text))
            except Exception:
                pass
        return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)

    def count_messages(self, messages: list[dict]) -> int:
        # +4 tokens/message is the usual chat-format overhead approximation.
        return sum(self.count(m.get("content", "")) + 4 for m in messages)


# --------------------------------------------------------------------------- #
# Memory budget
# --------------------------------------------------------------------------- #
@dataclass
class MemoryBudget:
    """Caps the ROOT model's context window (eviction-only).

    max_context_tokens : hard cap on the root prompt (system + begin + recent
                         turns). The headline knob to sweep.
    keep_recent_turns  : how many most-recent (assistant,observation) pairs to
                         try to keep verbatim before budget squeezing kicks in.

    Older turns that don't fit are simply DROPPED (evicted) — nothing is saved or
    summarized. The model must keep what it needs in persistent REPL variables.
    """

    max_context_tokens: int = 4096
    keep_recent_turns: int = 3


@dataclass
class Scratchpad:
    """Opt-in persistent notes for the root model (orthogonal to MemoryBudget).

    Enables a `note(text)` REPL tool; saved notes are re-injected into the
    root's view every turn (and survive budget eviction when a MemoryBudget is
    also set).

    max_notes_tokens : cap on the notes block injected into context; oldest
                       text is truncated away once exceeded.
    """

    max_notes_tokens: int = 1024


@dataclass
class RLMResult:
    answer: str | None
    steps: int
    finished: bool
    transcript: list = field(default_factory=list)
    end_reason: str = ""
    metrics: dict = field(default_factory=dict)
