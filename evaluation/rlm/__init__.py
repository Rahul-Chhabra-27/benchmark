# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recursive Language Model evaluation support.

    shared/            building blocks: datasets, model client, KVzip sub-calls, sizing, BM25, bm25-graph
    harness/           the original runner (run_benchmark.py) and the RLMs it runs with --mode
    configurable_rlm/  a clean base RLM; ordering, tuning and compression are features on top
    chunk_sources/     chunks handed to the RLM by something else (oracle, query-free, notes)
    analysis/          reading finished runs (paired comparisons, B x F tables, hard questions, ...)
    campaigns/         tooling for finished campaigns, kept for reproducibility
"""

from .harness.original_rlm import OriginalRLM
from .harness.rlm import RLM, MemoryBudget, RLMResult, vanilla_answer
from .harness.skylight_rlm import SkylightResult, SkylightRLM
from .shared.client import LLMClient

__all__ = [
    "MemoryBudget",
    "LLMClient",
    "OriginalRLM",
    "RLM",
    "RLMResult",
    "SkylightRLM",
    "SkylightResult",
    "vanilla_answer",
]
