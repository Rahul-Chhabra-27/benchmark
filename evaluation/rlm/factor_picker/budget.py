# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The B x F arithmetic a picker needs, with no model loaded.

B is the KV one sub-call may keep, in tokens; F is how hard each read is pressed;
a read carries N = B x F tokens. F = 1 is the no_press control (plain dense reads).
"""

from __future__ import annotations

# The factors a picker chooses between. 1 runs with no press at all.
FACTORS: tuple[int, ...] = (1, 2, 4, 8)

# Qwen3-4B-Instruct-2507: 36 layers x 8 KV heads x 128 dims x 2 (K and V) x 2 bytes.
# The same number kvzip_backend reads off the loaded model; repeated here so a plan
# can be made on a CPU host before any model is loaded.
QWEN3_4B_KV_BYTES_PER_TOKEN = 147_456

# kvpress.pipeline.MEMORY_UNIT_TO_BYTES["MB"]: decimal, as is the lane's own `min_free` math.
BYTES_PER_MB = 1000**2


def budget_tokens(budget_mb: float, kv_bytes_per_token: int = QWEN3_4B_KV_BYTES_PER_TOKEN) -> int:
    """B in tokens for a budget in MB: 1024 MB -> 6,944 tokens on Qwen3-4B."""
    if budget_mb <= 0:
        raise ValueError(f"budget_mb must be positive, got {budget_mb}")
    tokens = int(budget_mb * BYTES_PER_MB // kv_bytes_per_token)
    if tokens < 1:
        raise ValueError(f"{budget_mb:g} MB holds less than one KV token")
    return tokens
