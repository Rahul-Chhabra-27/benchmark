# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared scoring boundary for independent evaluation backends."""

from typing import Any

import pandas as pd

from .registry import SCORER_REGISTRY


def score_prediction_frame(dataset_name: str, predictions: pd.DataFrame) -> dict[str, Any]:
    """Score a common prediction frame with the canonical benchmark scorer."""
    if dataset_name not in SCORER_REGISTRY:
        raise ValueError(f"No scorer found for {dataset_name!r}")
    frame = predictions.copy()
    if "predicted_answer" not in frame.columns:
        raise ValueError("Prediction frame requires a 'predicted_answer' column")
    frame["predicted_answer"] = frame["predicted_answer"].fillna("").astype(str)
    score = SCORER_REGISTRY[dataset_name](frame)
    return score if isinstance(score, dict) else {"score": score}
