# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 1: pick F for every question and write the plan, before any model runs.

A plan directory holds

    plan.json                 the picker, B, split and rows it was made for
    plan.jsonl                one line per question: id, task, factor, reason
    ids/<task>__x<F>.txt      the questions of one task that got factor F

and each ids file is exactly what `run_benchmark --ids-file` (or the lane's
IDS_FILE) needs to run that group as an ordinary B x F cell.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..datasets import load_examples
from .budget import budget_tokens as to_budget_tokens
from .features import question_features
from .pickers import Picker, describe_picker


def plan_questions(examples: Iterable[dict], picker: Picker, dataset: str = "loft") -> list[dict]:
    """One row per example: which factor the picker chose, and why."""
    rows = []
    for example in examples:
        task = example.get("task") or dataset
        features = question_features(example["question"], example["context"], dataset)
        pick = picker.pick(features, task)
        rows.append(
            {
                "id": example["id"],
                "task": task,
                "factor": pick.factor,
                "reason": pick.reason,
                "query": features.query,
            }
        )
    return rows


def load_rows(
    dataset: str,
    tasks: Sequence[str],
    split: Optional[str],
    limit: Optional[int],
) -> Iterable[dict]:
    for task in tasks:
        yield from load_examples(dataset, task, limit, split)


def write_plan(rows: list[dict], out_dir: Path, meta: dict) -> Path:
    """Write plan.json, plan.jsonl and one ids file per (task, factor) group."""
    out_dir = Path(out_dir)
    ids_dir = out_dir / "ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    for stale in ids_dir.glob("*.txt"):  # a re-plan must not leave an old group behind
        stale.unlink()
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["factor"])].append(row["id"])
    for (task, factor), ids in sorted(groups.items()):
        (ids_dir / f"{task}__x{factor}.txt").write_text("\n".join(ids) + "\n")
    (out_dir / "plan.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    meta = {**meta, "rows": len(rows), "groups": {f"{t}__x{f}": len(ids) for (t, f), ids in sorted(groups.items())}}
    (out_dir / "plan.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def read_plan(plan_dir: Path) -> tuple[dict, list[dict]]:
    plan_dir = Path(plan_dir)
    meta = json.loads((plan_dir / "plan.json").read_text())
    rows = [json.loads(line) for line in (plan_dir / "plan.jsonl").read_text().splitlines() if line.strip()]
    return meta, rows


def make_plan(
    picker: Picker,
    tasks: Sequence[str],
    budget_mb: float,
    out_dir: Path,
    dataset: str = "loft",
    split: Optional[str] = "test",
    limit: Optional[int] = None,
) -> tuple[Path, list[dict]]:
    """The whole of step 1: load the rows, pick, write. Returns the plan dir and its rows."""
    rows = plan_questions(load_rows(dataset, tasks, split, limit), picker, dataset)
    meta = {
        "picker": describe_picker(picker),
        "picker_name": picker.name,
        "dataset": dataset,
        "tasks": list(tasks),
        "split": split,
        "limit": limit,
        "budget_mb": budget_mb,
        "budget_tokens": to_budget_tokens(budget_mb),
    }
    return write_plan(rows, out_dir, meta), rows


def summarize(rows: list[dict]) -> str:
    """task -> how many questions got each factor, as a small text table."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row["task"]][row["factor"]] += 1
    factors = sorted({row["factor"] for row in rows})
    lines = ["task".ljust(16) + "".join(f"x{f}".rjust(6) for f in factors)]
    for task in sorted(counts):
        lines.append(task.ljust(16) + "".join(str(counts[task][f]).rjust(6) for f in factors))
    return "\n".join(lines)
