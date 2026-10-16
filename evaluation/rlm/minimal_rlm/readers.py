# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How one llm_query reads its text: the `compression` option.

    HttpReader  plain reads through the vLLM server (no budget set)
    KVReader    reads through ONE in-process kvpress model, whose press is set per read:
                F = 1 is that model with no press (the control), F > 1 presses by 1 - 1/F.
                So the root can choose a different factor on every call, with no extra
                servers and one copy of the weights on the GPU.

Both count the tokens they read; the loop turns that into kept / GPU / offloaded
tokens and a cost (stats.Placement).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Protocol

from ..client import LLMClient
from .config import ExperimentConfig
from .prompts import CHARS_PER_TOKEN, SUB_PROMPT


@dataclass(frozen=True)
class Read:
    answer: str
    read_tokens: int  # tokens of document text the sub-model was given
    seconds: float


class Reader(Protocol):
    def count_tokens(self, text: str) -> int: ...

    def read(self, question: str, text: str, factor: int = 1) -> Read: ...


class HttpReader:
    """Dense sub-calls to the OpenAI-compatible server (factor 1 only)."""

    def __init__(self, client: Any):
        self.client = client

    def count_tokens(self, text: str) -> int:
        return math.ceil(len(text) / CHARS_PER_TOKEN)

    def read(self, question: str, text: str, factor: int = 1) -> Read:
        if factor != 1:
            raise ValueError("http reads are uncompressed; a factor > 1 needs compression.budget_mb")
        started = time.monotonic()
        answer = self.client.chat([{"role": "user", "content": f"{SUB_PROMPT.format(question=question)}\n\n{text}"}])
        return Read(str(answer or ""), self.count_tokens(text), time.monotonic() - started)


class KVReader:
    """Sub-calls through kvpress: the text is the compressed side, the question is not."""

    def __init__(self, sub: Any):
        self.sub = sub
        self.press = sub.press  # the KVzipPress, or None when no allowed factor compresses

    def count_tokens(self, text: str) -> int:
        return int(self.sub.token_len(text))

    def _use(self, factor: int) -> None:
        """Press the next read by exactly 1 - 1/F (no press at all for F = 1)."""
        if factor == 1:
            self.sub.press = None
        else:
            if self.press is None:
                raise ValueError(f"factor {factor} needs a press; the reader was built without one")
            self.press.compression_ratio = 1 - 1 / factor
            self.sub.press = self.press
            self.sub.press_min_tokens = 1  # every read at F > 1 is pressed, so F is exact
        self.sub.compression_factor = float(factor)

    def read(self, question: str, text: str, factor: int = 1) -> Read:
        self._use(factor)
        started = time.monotonic()
        answer = self.sub.chat_split(question=SUB_PROMPT.format(question=question), context=text)
        return Read(str(answer or ""), self.count_tokens(text), time.monotonic() - started)


def make_reader(config: ExperimentConfig) -> Reader:
    """The reader a config asks for. Loads a model on the GPU when a budget is set."""
    compression = config.compression
    if not compression.in_process:
        client = LLMClient(model=config.models.sub_model, base_url=config.models.base_url)
        if not config.models.thinking:
            client.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        return HttpReader(client)
    from ..kvzip_backend import KVzipSubClient  # torch and kvpress, only when needed

    largest = max(compression.factors)
    sub = KVzipSubClient(
        model=config.models.sub_model,
        press_name="kvzip" if largest > 1 else "no_press",
        memory_budget=compression.budget_mb,
        memory_budget_unit="MB",
        max_context_tokens=config.largest_read_tokens,
        compression_factor=largest,
    )
    return KVReader(sub)
