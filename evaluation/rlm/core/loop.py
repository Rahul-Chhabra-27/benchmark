# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The RLM loop: the root writes code, the REPL runs it, until FINAL or a limit.

    turn:  root reply -> first ```python block -> run it -> show its output
    tools: context, [chunks], llm_query(question, text_or_chunks), FINAL(answer)
    ends:  final | max_steps | run_timeout | max_sub_calls reached (reads refused)

Nothing here knows which options are on: ordering arrives as a chunk list (or
none), compression as a Reader. Every sub-call leaves a CallRecord.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from ..retrieval import Hit
from . import prompts
from .readers import Reader
from .repl import Repl, extract_code
from .stats import CallRecord, preview


@dataclass
class Result:
    answer: Optional[str]
    steps: int
    end_reason: str  # final | max_steps | run_timeout | root_error
    calls: list[CallRecord] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    seconds: float = 0.0


@dataclass(frozen=True)
class Limits:
    max_steps: int = 30
    max_sub_calls: int = 40
    exec_timeout_s: float = 60.0
    run_timeout_s: float = 900.0
    output_chars: int = 4000
    read_tokens: int = 8192  # the most one llm_query may read
    budget_tokens: Optional[int] = None  # B, for the records


def is_context_overflow(exc: Exception) -> bool:
    """An OpenAI-compatible 400 saying the prompt is longer than the model's window."""
    message = str(exc).lower()
    return getattr(exc, "status_code", None) == 400 and ("context length" in message or "too long" in message)


class RLM:
    def __init__(self, root: Any, reader: Reader, limits: Limits, ordering: str = "none"):
        self.root = root
        self.reader = reader
        self.limits = limits
        self.ordering = ordering

    def run(
        self,
        document: str,
        question: str,
        chunks: Sequence[Hit] = (),
        answer_format: Optional[str] = None,
        list_answer: bool = False,
        chunk_chars: int = 0,
    ) -> Result:
        limits = self.limits
        started = time.monotonic()
        result = Result(answer=None, steps=0, end_reason="max_steps")
        repl = Repl({}, limits.exec_timeout_s)
        repl.namespace.update(self._tools(document, chunks, repl, result))

        system = prompts.system_prompt(
            question=question,
            context_chars=len(document),
            read_tokens=limits.read_tokens,
            max_steps=limits.max_steps,
            output_chars=limits.output_chars,
            ordering=self.ordering,
            n_chunks=len(chunks),
            chunk_chars=chunk_chars,
            answer_format=answer_format,
            list_answer=list_answer,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompts.BEGIN}]
        for step in range(1, limits.max_steps + 1):
            if time.monotonic() - started > limits.run_timeout_s:
                result.end_reason = "run_timeout"
                break
            result.steps = step
            reply = self._root_reply(messages)
            if reply is None:
                result.end_reason = "root_error"
                break
            code = extract_code(reply)
            output = prompts.NO_CODE_BLOCK if code is None else repl.run(code, limits.output_chars)
            result.transcript.append({"step": step, "reply": reply, "output": output})
            if result.answer is not None:
                result.end_reason = "final"
                break
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": prompts.REPL_OUTPUT.format(output=output)},
            ]
        result.seconds = time.monotonic() - started
        return result

    def _root_reply(self, messages: list[dict]) -> Optional[str]:
        """One root completion. On a window overflow, drop the oldest turn and retry."""
        while True:
            try:
                return self.root.chat(messages)
            except Exception as exc:  # noqa: BLE001 - the client raises the SDK's errors
                # messages = [system, begin, (assistant, output)*]: keep the first two.
                if not is_context_overflow(exc) or len(messages) <= 3:
                    return None
                del messages[2:4]

    def _tools(self, document: str, chunks: Sequence[Hit], repl: Repl, result: Result) -> dict:
        limits, reader = self.limits, self.reader

        def llm_query(question: str, text: Any = "") -> str:
            n_chunks = None
            if isinstance(text, (list, tuple)):
                n_chunks = len(text)
                text = "\n\n".join(getattr(c, "text", str(c)) for c in text)
            text = str(text)
            tokens = reader.count_tokens(text)
            record = dict(
                step=result.steps,
                question=preview(question),
                chunks=n_chunks,
                read_tokens=tokens,
                limit_tokens=limits.read_tokens,
                budget_tokens=limits.budget_tokens,
                factor=reader.factor,
            )
            made = sum(1 for c in result.calls if not c.refused)
            refusal = ""
            if made >= limits.max_sub_calls:
                refusal = f"[REFUSED] the limit of {limits.max_sub_calls} llm_query calls is reached."
            elif tokens > limits.read_tokens:
                refusal = (
                    f"[REFUSED] that text is {tokens:,} tokens; one llm_query reads at most "
                    f"{limits.read_tokens:,}. Split it into smaller pieces."
                )
            if refusal:
                result.calls.append(CallRecord(kept_tokens=0, seconds=0.0, refused=refusal, **record))
                return refusal
            with repl.outside():
                read = reader.read(str(question), text)
            result.calls.append(
                CallRecord(kept_tokens=read.kept_tokens, seconds=read.seconds, answer=preview(read.answer), **record)
            )
            return read.answer

        def FINAL(answer: Any) -> str:  # noqa: N802 - the name the prompt uses
            result.answer = str(answer)
            return result.answer

        tools: dict[str, Any] = {"context": document, "llm_query": llm_query, "FINAL": FINAL}
        if chunks:
            tools["chunks"] = list(chunks)
        return tools
