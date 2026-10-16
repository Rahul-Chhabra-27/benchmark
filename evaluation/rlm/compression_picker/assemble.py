# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 3: merge each question's answer at its picked F, score, compare.

The comparison is always on the SAME rows: a question counts only if it was
answered at its picked factor and in every fixed-F baseline being compared.
`oracle` is the best factor per question in hindsight -- not a method, the
ceiling any picker over these factors could reach.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import pandas as pd


def picked_scores(plan_rows: list[dict], picked_cells: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """One row per planned question, scored from the cell of its picked factor.

    A question whose cell has no answer for it yet keeps a null score: it is a
    missing measurement, reported, never a zero.
    """
    rows = []
    for plan in plan_rows:
        cell = picked_cells.get(plan["factor"])
        scored = cell.loc[plan["id"]] if cell is not None and plan["id"] in cell.index else None
        rows.append(
            {
                "id": plan["id"],
                "task": plan["task"],
                "factor": plan["factor"],
                "reason": plan["reason"],
                "metric": None if scored is None else scored["metric"],
                "score": None if scored is None else float(scored["score"]),
                "predicted_answer": None if scored is None else scored["predicted_answer"],
            }
        )
    return pd.DataFrame(rows)


def compare(picked: pd.DataFrame, baselines: Optional[dict[int, pd.DataFrame]] = None) -> pd.DataFrame:
    """Per task (and the task average): picker vs every fixed F vs oracle, on common rows."""
    baselines = baselines or {}
    answered = picked.dropna(subset=["score"])
    common = set(answered["id"])
    for cell in baselines.values():
        common &= set(cell.index)
    rows = []
    for task, group in answered[answered["id"].isin(common)].groupby("task"):
        ids = list(group["id"])
        planned = picked[picked["task"] == task]
        histogram = Counter(int(f) for f in group["factor"])
        row = {
            "task": task,
            "metric": group["metric"].iloc[0],
            "planned": len(planned),
            "compared": len(ids),
            "picks": " ".join(f"x{f}:{n}" for f, n in sorted(histogram.items())),
            "picked": 100 * group["score"].mean(),
        }
        for factor, cell in sorted(baselines.items()):
            row[f"fixed x{factor}"] = 100 * cell.loc[ids, "score"].mean()
        if baselines:
            best = pd.concat([cell.loc[ids, "score"] for cell in baselines.values()], axis=1).max(axis=1)
            row["oracle"] = 100 * best.mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table) > 1:
        numeric = [c for c in table.columns if c not in ("task", "metric", "picks")]
        average = {"task": "task average", "metric": "", "picks": ""}
        average.update({c: table[c].mean() if c not in ("planned", "compared") else table[c].sum() for c in numeric})
        table = pd.concat([table, pd.DataFrame([average])], ignore_index=True)
    return table


def render(table: pd.DataFrame) -> str:
    """A markdown table, scores to two decimals."""
    if table.empty:
        return "(no question was answered at its picked factor and in every baseline)"
    shown = table.copy()
    for column in shown.columns:
        if shown[column].dtype.kind == "f":
            shown[column] = shown[column].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    header = "| " + " | ".join(shown.columns) + " |"
    rule = "|" + "|".join("---" for _ in shown.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in shown.itertuples(index=False)]
    return "\n".join([header, rule, *body])
