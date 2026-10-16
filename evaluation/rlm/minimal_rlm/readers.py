# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How one llm_query reads its text: the `compression` option.

    HttpReader  plain reads through the vLLM server (no budget set)
    KVReader    reads through the in-process kvpress model at factor F (budget set);
                F = 1 is the same model with no press, the control for every F > 1

Both count the tokens they read and keep, which is what the per-call statistics
(and later the cost model) are built from.
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
    kept_tokens: int  # tokens of KV it kept after compression (= read_tokens at F = 1)
    seconds: float


class Reader(Protocol):
    factor: int

    def count_tokens(self, text: str) -> int: ...

    def read(self, question: str, text: str) -> Read: ...


class HttpReader:
    """Dense sub-calls to the OpenAI-compatible server."""

    factor = 1

    def __init__(self, client: Any):
        self.client = client

    def count_tokens(self, text: str) -> int:
        return math.ceil(len(text) / CHARS_PER_TOKEN)

    def read(self, question: str, text: str) -> Read:
        started = time.monotonic()
        answer = self.client.chat([{"role": "user", "content": f"{SUB_PROMPT.format(question=question)}\n\n{text}"}])
        tokens = self.count_tokens(text)
        return Read(str(answer or ""), tokens, tokens, time.monotonic() - started)


class KVReader:
    """Sub-calls through kvpress: the text is the compressed side, the question is not."""

    def __init__(self, sub: Any, factor: int):
        self.sub = sub
        self.factor = factor

    def count_tokens(self, text: str) -> int:
        return int(self.sub.token_len(text))

    def read(self, question: str, text: str) -> Read:
        started = time.monotonic()
        answer = self.sub.chat_split(question=SUB_PROMPT.format(question=question), context=text)
        tokens = self.count_tokens(text)
        return Read(str(answer or ""), tokens, math.ceil(tokens / self.factor), time.monotonic() - started)


def make_reader(config: ExperimentConfig) -> Reader:
    """The reader a config asks for. Loads a model on the GPU when a budget is set."""
    compression = config.compression
    if not compression.in_process:
        client = LLMClient(model=config.models.sub_model, base_url=config.models.base_url)
        if not config.models.thinking:
            client.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        return HttpReader(client)
    from ..kvzip_backend import KVzipSubClient  # torch and kvpress, only when needed

    sub = KVzipSubClient(
        model=config.models.sub_model,
        press_name="kvzip" if compression.factor > 1 else "no_press",
        memory_budget=compression.budget_mb,
        memory_budget_unit="MB",
        max_context_tokens=config.read_tokens,
        compression_factor=compression.factor,
    )
    return KVReader(sub, compression.factor)
