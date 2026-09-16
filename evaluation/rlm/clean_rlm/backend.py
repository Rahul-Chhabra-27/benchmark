# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit client for a local OpenAI-compatible vLLM server."""

from dataclasses import dataclass
from urllib.parse import urlparse

from ..client import LLMClient


def is_context_overflow(exc: Exception) -> bool:
    """Whether an OpenAI-compatible 400 reports an oversized prompt."""
    if getattr(exc, "status_code", None) != 400:
        return False
    message = str(exc).lower()
    return "context length" in message or "context_length" in message or "too long" in message


@dataclass
class VLLMClient(LLMClient):
    """OpenAI-compatible HTTP client used by the clean RLM.

    The distinct name makes the supported transport explicit. There is no
    backend selector and no split-context protocol in this package.
    """

    def __post_init__(self) -> None:
        endpoint = urlparse(self.base_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
            raise ValueError("clean RLM clients require an explicit http(s) OpenAI-compatible vLLM endpoint")
        super().__post_init__()
