# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct-context baseline for the OpenAI-compatible vLLM client."""

from __future__ import annotations

from typing import Callable, Optional

from .backend import VLLMClient, is_context_overflow
from .types import TokenCounter


def vanilla_answer(
    client: VLLMClient,
    context: str,
    task: str,
    char_limit: int = 400_000,
    answer_format: str | None = None,
    max_prompt_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    stats: Optional[dict] = None,
) -> str:
    """Baseline: stuff (possibly truncated) context directly into the prompt.

    `answer_format` is the benchmark's output contract (e.g. LOFT's
    "Final Answer: ['answer1', ...]"). It goes to BOTH arms -- the RLM gets the same
    string in ROOT_SYSTEM_PROMPT -- so neither arm is scored on an answer shape the
    other was never asked for. Without it this prompt said only "Answer concisely"
    while the RLM answered via `str(FINAL(x))`, and string-matching metrics then
    measured that style difference as if it were accuracy.

    `stats`, if given, is filled with how much context actually survived
    truncation. The caller needs this to separate "the model got it wrong" from
    "the answer was never in the prompt". Without it the vanilla arm's score is
    not comparable to a KVPress run, which always sees the full context: on a
    200k-char synthetic with a 100k limit vanilla is structurally capped, and a
    bare score column reads that cap as model accuracy.
    """
    truncated = context[:char_limit]
    fmt = f"\nFormat your answer as: {answer_format}" if answer_format else ""

    def build(body: str) -> str:
        note = "" if len(context) <= len(body) else "\n[NOTE: document truncated]"
        return f"Document:\n{body}{note}\n\nTask: {task}\nAnswer concisely.{fmt}"

    prompt = build(truncated)

    # A CHARACTER limit is not a token limit. `--vanilla-char-limit 100000` assumes
    # ~4 chars/token, but dense subsets tokenize far tighter -- RULER's `cwe`
    # (repeated word lists) and `niah_multikey_3` (UUID-like keys) run ~2.5, so 100k
    # chars is ~40k tokens and the server rejected EVERY request with a 400
    # "maximum context length is 40960 tokens". Those cells scored 0.0 for vanilla
    # from a harness error rather than from the model. Shrink until it really fits.
    if max_prompt_tokens:
        tok = TokenCounter(token_counter)
        while tok.count(prompt) > max_prompt_tokens and len(truncated) > 2000:
            truncated = truncated[: int(len(truncated) * 0.8)]
            prompt = build(truncated)

    # ...but the PREDICTIVE shrink above cannot be trusted on its own, because the
    # counter is not the server's tokenizer. TokenCounter uses tiktoken cl100k_base
    # when installed and `len // 4` when not -- and `len // 4` is exactly the 4
    # chars/token assumption this whole block exists to escape, so on a box without
    # tiktoken the loop is a silent no-op. Even with tiktoken, cl100k and Qwen3's
    # BPE disagree by well over the available headroom on the dense subsets.
    #
    # So treat the estimate as a first guess and let the SERVER be the authority:
    # shrink and retry whenever it rejects the prompt for length. This is tokenizer-
    # independent and cannot silently no-op.
    def record(retries: int) -> None:
        if stats is not None:
            stats.update(
                context_chars=len(context),
                context_chars_used=len(truncated),
                truncated=len(truncated) < len(context),
                shrink_retries=retries,
            )

    # The final attempt is inside the loop, not after it: a 13th call outside
    # the try raised uncaught on overflow, and record(12) then reported 12
    # retries for what was really a 13th attempt.
    last_attempt = 12
    for attempt in range(last_attempt + 1):
        try:
            out = client.chat([{"role": "user", "content": prompt}])
            record(attempt)
            return out
        except Exception as e:  # noqa: BLE001 - narrowed by the guard below
            if attempt == last_attempt or not is_context_overflow(e) or len(truncated) <= 2000:
                record(attempt)
                raise
            truncated = truncated[: int(len(truncated) * 0.8)]
            prompt = build(truncated)
    raise AssertionError("unreachable: the loop either returns or raises")
