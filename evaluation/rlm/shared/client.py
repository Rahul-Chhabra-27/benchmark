# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible client with retries, for a locally served model.

Points at any OpenAI-compatible endpoint — in practice a vLLM (or SGLang)
server on the same host or cluster node. Local servers don't authenticate,
so no API key is involved.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Sequence

from openai import APIError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    # The last call's prompt size, as the SERVER counted it. The running totals
    # above cannot answer "how big was the context at its peak", and the RLM's
    # cost axis is exactly that question -- previously answered with a local
    # estimate from a different tokenizer.
    last_prompt_tokens: int | None = None

    def add(self, resp) -> None:
        u = getattr(resp, "usage", None)
        if u:
            self.prompt_tokens += u.prompt_tokens or 0
            self.completion_tokens += u.completion_tokens or 0
            self.last_prompt_tokens = u.prompt_tokens or None
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMClient:
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    max_retries: int = 6
    timeout: float = 300.0
    temperature: float = 0.0
    max_tokens: int = 4096
    extra_body: dict | None = None
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        # Local OpenAI-compatible servers ignore the key but the SDK requires one.
        key = self.api_key or "EMPTY"
        self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=self.timeout)
        self._async_client = AsyncOpenAI(base_url=self.base_url, api_key=key, timeout=self.timeout)

    def _chat_params(self, messages: list[dict], **kw) -> dict:
        params = dict(
            model=self.model,
            messages=messages,
            temperature=kw.get("temperature", self.temperature),
            max_tokens=kw.get("max_tokens", self.max_tokens),
        )
        eb = kw.get("extra_body") or self.extra_body
        if eb:
            params["extra_body"] = eb
        return params

    @staticmethod
    def _retryable_api_error(exc: APIError) -> bool:
        status = getattr(exc, "status_code", None)
        return status is None or status >= 500

    def chat(self, messages: list[dict], **kw) -> str:
        """One chat completion with backoff. Returns assistant text."""
        params = self._chat_params(messages, **kw)
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**params)
                self.usage.add(resp)
                return resp.choices[0].message.content or ""
            except (RateLimitError, APITimeoutError):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 120)
            except APIError as e:
                # 5xx are retryable; 4xx (bad request, context too long) are not.
                # Connection errors (e.g. APIConnectionError) have no status_code
                # at all -- treat those as retryable too.
                if not self._retryable_api_error(e) or attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 120)
        raise RuntimeError("unreachable")

    def chat_batch(self, jobs: Sequence[Sequence[dict]], **kw) -> list[str]:
        """Concurrent chat completions so vLLM can engine-batch a SCAN.

        The OpenAI chat endpoint takes one conversation per HTTP request.
        Firing them together is how a served vLLM instance actually batches.
        """
        message_lists = [list(m) for m in jobs]
        if not message_lists:
            return []
        if len(message_lists) == 1:
            return [self.chat(message_lists[0], **kw)]
        return asyncio.run(self._chat_batch_async(message_lists, **kw))

    async def _chat_batch_async(self, message_lists: list[list[dict]], **kw) -> list[str]:
        async def one(messages: list[dict]) -> str:
            params = self._chat_params(messages, **kw)
            delay = 2.0
            for attempt in range(self.max_retries):
                try:
                    resp = await self._async_client.chat.completions.create(**params)
                    self.usage.add(resp)
                    return resp.choices[0].message.content or ""
                except (RateLimitError, APITimeoutError):
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 120)
                except APIError as e:
                    if not self._retryable_api_error(e) or attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 120)
            raise RuntimeError("unreachable")

        return list(await asyncio.gather(*[one(m) for m in message_lists]))
