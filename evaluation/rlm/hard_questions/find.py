# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score every question in every arm, and sort the questions by who can solve them.

A question is *solved* by an arm when its official score (F1, or coverage for list
tasks) reaches `solved` (default 0.5). Only questions answered by a vanilla arm AND
by an RLM arm without compression are compared. Each then lands in one category:

    compression_rescues  no uncompressed RLM arm solves it, a compressed arm does
    vanilla_only         vanilla solves it, no RLM arm does (compressed included)
    both_solve           vanilla and an uncompressed RLM arm both solve it
    rlm_only             an uncompressed RLM arm solves it, vanilla does not
    nobody               nothing solves it

"Uncompressed RLM" means the ordered arms when there are any (the question is what
fails AFTER ordering), else the plain RLM arms. `vanilla_only` plus
`compression_rescues` is the validation set: questions an RLM can only get right by
doing something beyond ordering -- reading wider through compression, or falling back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..compression_picker.results import score_records
from .arms import Arm

CATEGORIES = ("compression_rescues", "vanilla_only", "both_solve", "rlm_only", "nobody")
VALIDATION = ("compression_rescues", "vanilla_only")


def _records(run_dirs: Iterable[Path]) -> list[dict]:
    """Every answered (non-error) question of an arm; a retried id keeps its last answer."""
    by_id: dict[tuple, dict] = {}
    for run_dir in run_dirs:
        for line in (run_dir / "checkpoint.jsonl").read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:  # a torn last line from a killed run
                continue
            if not record.get("error"):
                by_id[(record.get("task"), record["id"])] = record
    return list(by_id.values())


def score_matrix(arms: dict[str, Arm]) -> pd.DataFrame:
    """(task, id) x arm label -> official score; NaN where an arm did not answer."""
    columns = {}
    for label, arm in arms.items():
        scored = score_records(_records(arm.run_dirs), arm.dataset)
        if len(scored):
            columns[label] = scored.set_index(["task", "id"])["score"]
    return pd.DataFrame(columns).sort_index()


def classify(matrix: pd.DataFrame, arms: dict[str, Arm], solved: float = 0.5) -> pd.DataFrame:
    """One row per compared question: each arm's score, the best per kind (best_*), its category."""
    kind_of = {label: arms[label].kind for label in matrix.columns}
    by_kind = {k: [label for label, kind in kind_of.items() if kind == k] for k in set(kind_of.values())}
    vanilla = by_kind.get("vanilla", [])
    uncompressed = by_kind.get("ordered") or by_kind.get("rlm", [])
    compressed = by_kind.get("compressed", [])
    if not vanilla or not uncompressed:
        raise ValueError(
            "need a vanilla arm and an uncompressed RLM arm (ordered, or plain rlm) to compare; "
            f"found kinds {sorted(by_kind)}"
        )
    compared = matrix.dropna(subset=vanilla, how="all").dropna(subset=uncompressed, how="all")

    def best(labels: list[str]) -> pd.Series:
        return compared[labels].max(axis=1) if labels else pd.Series(float("nan"), index=compared.index)

    table = pd.DataFrame(
        {
            "best_vanilla": best(vanilla),
            "best_rlm_uncompressed": best(uncompressed),
            "best_rlm_compressed": best(compressed),
        }
    )
    v, u = table["best_vanilla"] >= solved, table["best_rlm_uncompressed"] >= solved
    c = table["best_rlm_compressed"].fillna(-1) >= solved
    table["category"] = "nobody"
    table.loc[u & ~v, "category"] = "rlm_only"
    table.loc[u & v, "category"] = "both_solve"
    table.loc[v & ~u & ~c, "category"] = "vanilla_only"
    table.loc[~u & c, "category"] = "compression_rescues"
    return table.join(compared)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """task x category -> number of questions, with a total row."""
    counts = table.groupby(level="task")["category"].value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=list(CATEGORIES), fill_value=0)
    counts["validation_set"] = counts[list(VALIDATION)].sum(axis=1)
    counts["compared"] = counts[list(CATEGORIES)].sum(axis=1)
    counts.loc["total"] = counts.sum()
    return counts


def write(table: pd.DataFrame, out: Path) -> dict[str, Path]:
    """hard_questions.csv, summary.md, and ids/<task>__<category>.txt for the validation set."""
    out = Path(out)
    ids_dir = out / "ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    for stale in ids_dir.glob("*.txt"):
        stale.unlink()
    table.reset_index().to_csv(out / "hard_questions.csv", index=False)
    written = {}
    for task, group in table.groupby(level="task"):
        for name, wanted in [*((c, (c,)) for c in VALIDATION), ("validation_set", VALIDATION)]:
            ids = [i for (_, i), cat in group["category"].items() if cat in wanted]
            if ids:
                path = ids_dir / f"{task}__{name}.txt"
                path.write_text("\n".join(ids) + "\n")
                written[f"{task}__{name}"] = path
    return written
