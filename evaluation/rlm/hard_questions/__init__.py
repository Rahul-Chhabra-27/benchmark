# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which questions does an RLM need more than ordering for? See README.md in this folder.

    arms.py   find finished runs and label each from its config.yaml
    find.py   score every question in every arm, sort them into categories, write ids files

    python -m evaluation.rlm.hard_questions --results <runs root> --out <dir>
"""

from .arms import Arm, describe, discover
from .find import CATEGORIES, VALIDATION, classify, score_matrix, summarize, write

__all__ = ["Arm", "CATEGORIES", "VALIDATION", "classify", "describe", "discover", "score_matrix", "summarize", "write"]
