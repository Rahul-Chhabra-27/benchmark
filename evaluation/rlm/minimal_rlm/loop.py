# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The RLM loop: the root writes code, the REPL runs it, until FINAL or a limit.

    turn:  root reply -> first ```python block -> run it -> show its output
    tools: context, [chunks], llm_query(question, text_or_chunks[, factor=F]), FINAL(answer)
    ends:  final | max_steps | run_timeout | root_error

Nothing here knows which options are on: ordering arrives as a chunk list (or
none), compression as a Reader plus the factors and limits in `Limits`. Every
sub-call leaves a CallRecord with its GPU / offloaded split and its cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

from ..retrieval import Hit
from . import prompts
from .readers import Reader
from .repl import Repl, extract_code
from .stats import CallRecord, Placement, contributed, preview


@dataclass
class Result:
    answer: Optional[str]
    steps: int
    end_reason: str  # final | max_steps | run_timeout | root_error
    calls: list[CallRecord] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    # Each call's full sub-model answer (None if refused), for the end-of-question marks.
    read_answers: list[Optional[str]] = field(default_factory=list, repr=False)


@dataclass(frozen=True)
class Limits:
    max_steps: int = 30
    max_sub_calls: int = 40
    exec_timeout_s: float = 60.0
    run_timeout_s: float = 900.0
    output_chars: int = 4000
    read_tokens: int = 8192  # the most one read may carry, unless read_limits says otherwise
    budget_tokens: Optional[int] = None  # B: KV one read may keep on the GPU
    factors: tuple[int, ...] = (1,)  # factors llm_query accepts
    default_factor: int = 1
    read_limits: Optional[Mapping[int, int]] = None  # factor -> most tokens one read may carry
    offload: bool = False  # KV beyond B is offloaded (and costed) instead of refused
    gpu_token_cost: float = 0.0
    offloaded_token_cost: float = 100.0
    show_cost: bool = False

    def read_limit(self, factor: int) -> int:
        return (self.read_limits or {}).get(factor, self.read_tokens)


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
        system = self.system_prompt(question, len(document), len(chunks), chunk_chars, answer_format, list_answer)
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
        result.calls = [
            self._mark(call, answer, result.answer) for call, answer in zip(result.calls, result.read_answers)
        ]
        return result

    def system_prompt(
        self,
        question: str,
        context_chars: int,
        n_chunks: int = 0,
        chunk_chars: int = 0,
        answer_format: Optional[str] = None,
        list_answer: bool = False,
    ) -> str:
        """The root's prompt under these limits (also what `check --show-prompt` prints)."""
        limits = self.limits
        return prompts.system_prompt(
            question=question,
            context_chars=context_chars,
            read_tokens=limits.read_limit(limits.default_factor),
            max_steps=limits.max_steps,
            output_chars=limits.output_chars,
            ordering=self.ordering,
            n_chunks=n_chunks,
            chunk_chars=chunk_chars,
            answer_format=answer_format,
            list_answer=list_answer,
            budget_tokens=limits.budget_tokens,
            factors=limits.factors,
            default_factor=limits.default_factor,
            offload_cap=limits.read_limit(max(limits.factors)) if limits.offload else None,
            cost=(limits.gpu_token_cost, limits.offloaded_token_cost) if limits.show_cost else None,
        )

    @staticmethod
    def _mark(call: CallRecord, read_answer: Optional[str], final: Optional[str]) -> CallRecord:
        """Once the question ends: was compression needed, and did the read feed the answer?"""
        if call.refused or read_answer is None:
            return call
        needed = None
        if call.factor > 1 and call.budget_tokens is not None:
            needed = call.read_tokens > call.budget_tokens  # would it NOT have fit uncompressed?
        return replace(call, compression_needed=needed, contributed=contributed(read_answer, final))

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

    def _refusal(self, result: Result, factor: Any, tokens: int) -> str:
        limits = self.limits
        if sum(1 for c in result.calls if not c.refused) >= limits.max_sub_calls:
            return f"[REFUSED] the limit of {limits.max_sub_calls} llm_query calls is reached."
        if factor not in limits.factors:
            return f"[REFUSED] factor={factor!r} is not available; use one of {list(limits.factors)}."
        limit = limits.read_limit(factor)
        if tokens > limit:
            return (
                f"[REFUSED] that text is {tokens:,} tokens; one llm_query at factor {factor} reads at most "
                f"{limit:,}. Split it into smaller pieces."
            )
        return ""

    def _tools(self, document: str, chunks: Sequence[Hit], repl: Repl, result: Result) -> dict:
        limits, reader = self.limits, self.reader

        def llm_query(question: str, text: Any = "", factor: Any = None) -> str:
            factor = limits.default_factor if factor is None else factor
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
                budget_tokens=limits.budget_tokens,
            )
            refusal = self._refusal(result, factor, tokens)
            if refusal:
                limit = limits.read_limit(factor) if factor in limits.factors else 0
                factor_seen = factor if isinstance(factor, int) else 0
                result.calls.append(
                    CallRecord(
                        kept_tokens=0, limit_tokens=limit, factor=factor_seen, seconds=0.0, refused=refusal, **record
                    )
                )
                result.read_answers.append(None)
                return refusal
            placement = Placement.of(tokens, factor, limits.budget_tokens)
            cost = placement.cost(limits.gpu_token_cost, limits.offloaded_token_cost)
            with repl.outside():
                read = reader.read(str(question), text, factor)
            result.calls.append(
                CallRecord(
                    kept_tokens=placement.kept,
                    limit_tokens=limits.read_limit(factor),
                    factor=factor,
                    seconds=read.seconds,
                    gpu_tokens=placement.gpu,
                    offloaded_tokens=placement.offloaded,
                    cost=cost,
                    answer=preview(read.answer),
                    **record,
                )
            )
            result.read_answers.append(read.answer)
            if not limits.show_cost:
                return read.answer
            total = sum(c.cost for c in result.calls)
            line = prompts.COST_LINE.format(
                read=tokens, factor=factor, gpu=placement.gpu, offloaded=placement.offloaded, cost=cost, total=total
            )
            return f"{read.answer}\n{line}"

        def FINAL(answer: Any) -> str:  # noqa: N802 - the name the prompt uses
            result.answer = str(answer)
            return result.answer

        tools: dict[str, Any] = {"context": document, "llm_query": llm_query, "FINAL": FINAL}
        if chunks:
            tools["chunks"] = list(chunks)
        return tools
