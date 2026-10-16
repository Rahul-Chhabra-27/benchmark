# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Choose the picker on the DEV split, so the test split is only ever reported.

Input: fixed-F runs of the dev rows, one per factor. Every candidate picker is
scored on those rows by looking up, per question, the answer its factor produced:

    fixed x1 .. x8     one factor everywhere
    spread:<ratio>     for every ratio in the grid
    per-task           each task's best factor on dev (ties go to the lower F)

The candidate with the best task-average score is `chosen`, ties to the earlier
(simpler) one. All of these are in-sample on dev, per-task most of all: with ten
dev rows per LOFT task its table is the easiest to overfit, which is why the
simpler candidates win ties.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

from .features import QuestionFeatures
from .pickers import FixedPicker, PerTaskPicker, Picker, SpreadPicker, describe_picker

DEFAULT_RATIOS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def score_matrix(cells: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """id x factor -> official score, over the questions answered at EVERY factor."""
    common = set.intersection(*(set(cell.index) for cell in cells.values())) if cells else set()
    ids = sorted(common)
    matrix = pd.DataFrame({factor: cells[factor].loc[ids, "score"] for factor in sorted(cells)}, index=ids)
    matrix.insert(0, "task", [cells[min(cells)].loc[i, "task"] for i in ids])
    return matrix


def evaluate(picker: Picker, matrix: pd.DataFrame, features: Mapping[str, QuestionFeatures]) -> dict:
    """Score a picker on the matrix: per-task mean and the task average, in points."""
    chosen = [matrix.loc[i, picker.pick(features[i], matrix.loc[i, "task"]).factor] for i in matrix.index]
    per_task = pd.Series(chosen, index=matrix.index).groupby(matrix["task"]).mean() * 100
    return {"per_task": per_task.round(4).to_dict(), "score": float(per_task.mean())}


def best_per_task(matrix: pd.DataFrame) -> dict[str, int]:
    factors = [c for c in matrix.columns if c != "task"]
    means = matrix.groupby("task")[factors].mean()
    # idxmax returns the first maximum; columns are ascending, so ties go to the lower F.
    return {task: int(means.loc[task].idxmax()) for task in means.index}


def calibrate(
    cells: Mapping[int, pd.DataFrame],
    features: Mapping[str, QuestionFeatures],
    budget_tokens: int,
    ratios: Iterable[float] = DEFAULT_RATIOS,
) -> dict:
    """Every candidate's dev score, and the chosen one."""
    matrix = score_matrix(cells)
    if matrix.empty:
        raise ValueError("no dev question was answered at every factor; run each fixed-F dev cell first")
    missing = [i for i in matrix.index if i not in features]
    if missing:
        raise ValueError(f"no features for {len(missing)} dev rows, e.g. {missing[:3]}")
    factors = tuple(sorted(cells))
    candidates: list[Picker] = [FixedPicker(f) for f in factors]
    candidates += [SpreadPicker(budget_tokens, r, factors) for r in ratios]
    candidates.append(PerTaskPicker(best_per_task(matrix)))
    results = [{"name": c.name, **describe_picker(c), **evaluate(c, matrix, features)} for c in candidates]
    best = max(range(len(results)), key=lambda k: (results[k]["score"], -k))
    return {
        "rows": len(matrix),
        "factors": list(factors),
        "candidates": results,
        "chosen": describe_picker(candidates[best]),
        "chosen_name": candidates[best].name,
    }
