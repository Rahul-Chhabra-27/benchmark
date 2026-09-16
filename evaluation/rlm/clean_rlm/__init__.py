# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Modular RLM implementation for OpenAI-compatible local vLLM servers."""

from .backend import VLLMClient
from .rlm import RLM
from .types import MemoryBudget, RLMResult, Scratchpad, TokenCounter
from .vanilla import vanilla_answer

__all__ = [
    "MemoryBudget",
    "RLM",
    "RLMResult",
    "Scratchpad",
    "TokenCounter",
    "VLLMClient",
    "vanilla_answer",
]
