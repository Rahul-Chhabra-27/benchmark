# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read finished B x F runs and score them row by row.

A cell is found by what its config.yaml says (ordered-chunk agentic run, this
chunk order, this budget, this factor), not by its directory name, so the same
code reads a live picked run, a lane sweep, or anything copied off a cluster.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml

from evaluation.benchmarks.results import score_prediction_frame

# The score a dataset is judged by, first match wins. LOFT reports F1 on its
# single-answer tasks and coverage on its list tasks (qampari, quest).
OFFICIAL_METRICS = ("coverage", "f1", "accuracy", "score", "subspan_em")


def _matches(config: dict, budget_mb: float, factor: int, chunk_order: str) -> bool:
    return (
        config.get("backend") == "rlm"
        and config.get("mode") == "rlm"
        and config.get("chunk_order") == chunk_order
        and config.get("sub_kv_memory_budget_unit") == "MB"
        and config.get("sub_kv_memory_budget") is not None
        and float(config["sub_kv_memory_budget"]) == float(budget_mb)
        and config.get("compression_factor") is not None
        and float(config["compression_factor"]) == float(factor)
    )


def _checkpoint_records(path: Path) -> Iterable[dict]:
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a torn last line from a killed run
            continue
        if not record.get("error"):
            yield record


def load_cell(results_root: Path, budget_mb: float, factor: int, chunk_order: str = "bm25-graph") -> dict[str, dict]:
    """Every answered question of one B x F cell under `results_root`, by id.

    Refuses an id answered by two different run directories: that is two
    configurations of the same cell (a different max_steps, say), and silently
    taking either would mix them.
    """
    records: dict[str, dict] = {}
    owner: dict[str, Path] = {}
    for config_path in sorted(Path(results_root).rglob("config.yaml")):
        checkpoint = config_path.parent / "checkpoint.jsonl"
        if not checkpoint.exists():
            continue
        config = yaml.safe_load(config_path.read_text()) or {}
        if not _matches(config, budget_mb, factor, chunk_order):
            continue
        for record in _checkpoint_records(checkpoint):
            known = owner.get(record["id"])
            if known is not None and known != config_path.parent:
                raise ValueError(
                    f"{record['id']} is answered by both {known} and {config_path.parent}; "
                    "point --results at a directory holding one run per cell"
                )
            owner[record["id"]] = config_path.parent
            records[record["id"]] = record  # retries append; the last one is the answer
    return records


def official_metric(metrics: dict) -> str:
    return next(name for name in OFFICIAL_METRICS if name in metrics)


def score_records(records: Iterable[dict], dataset: str = "loft") -> pd.DataFrame:
    """One row per question: its official score (and EM / subspan EM beside it).

    Scored one row at a time with the canonical scorer, so each number is exactly
    what that question contributes to the run's own metrics.json.
    """
    rows = []
    for record in records:
        frame = pd.DataFrame([record]).rename(columns={"pred": "predicted_answer"})
        metrics = score_prediction_frame(dataset, frame)
        metric = official_metric(metrics)
        rows.append(
            {
                "id": record["id"],
                "task": record.get("task"),
                "metric": metric,
                "score": float(metrics[metric]),
                "em": metrics.get("em"),
                "subspan_em": metrics.get("subspan_em"),
                "predicted_answer": record.get("pred"),
                "steps": record.get("steps"),
                "tokens": record.get("tokens"),
            }
        )
    return pd.DataFrame(
        rows, columns=["id", "task", "metric", "score", "em", "subspan_em", "predicted_answer", "steps", "tokens"]
    )


def load_scored_cells(
    results_root: Path,
    budget_mb: float,
    factors: Iterable[int],
    chunk_order: str = "bm25-graph",
    dataset: str = "loft",
    ids: Optional[set[str]] = None,
) -> dict[int, pd.DataFrame]:
    """factor -> per-question scores, for every factor that has any answered row."""
    cells = {}
    for factor in factors:
        records = load_cell(results_root, budget_mb, factor, chunk_order)
        if ids is not None:
            records = {key: value for key, value in records.items() if key in ids}
        if records:
            cells[factor] = score_records(records.values(), dataset).set_index("id")
    return cells
