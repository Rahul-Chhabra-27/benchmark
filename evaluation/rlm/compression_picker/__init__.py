# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pick one KV compression factor per question, BEFORE the RLM runs.

At a fixed memory budget B, the factor F decides how much text each sub-call reads
(N = B x F tokens) and how hard it is compressed. F = 1 is plain dense reads with no
kvpress at all. Instead of one F for a whole run, a picker chooses F per question
from the question alone, and the run then uses it unchanged:

    1. plan       picker -> plan.jsonl + one ids file per (task, F)     (CPU, seconds)
    2. run        each ids file runs as an ordinary B x F cell          (GPU; run_picked.sh)
    3. assemble   merge, score, compare with fixed F and the oracle     (CPU)

Choose the picker on dev first (`calibrate`), then plan and report on test.
"""

from .budget import FACTORS, budget_tokens
from .features import QuestionFeatures, question_features
from .pickers import FixedPicker, PerTaskPicker, Pick, SpreadPicker, make_picker

__all__ = [
    "FACTORS",
    "FixedPicker",
    "PerTaskPicker",
    "Pick",
    "QuestionFeatures",
    "SpreadPicker",
    "budget_tokens",
    "make_picker",
    "question_features",
]
