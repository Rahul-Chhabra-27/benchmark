# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pickers: each one chooses ONE compression factor per question, before the run.

    fixed:4             every question at x4 -- today's B x F cell, the baseline
    spread:0.3          per question, from how spread out its BM25 evidence is
    per-task:nq=8,...   per task, a table (usually written by `calibrate` from dev)
    calibrated:FILE     whichever of the above `calibrate` chose, with its settings

A picker never calls a model, so a whole plan is made in seconds on a CPU host.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .budget import FACTORS
from .features import QuestionFeatures

DEFAULT_SPREAD_RATIO = 0.5


@dataclass(frozen=True)
class Pick:
    factor: int
    reason: str


class Picker(Protocol):
    @property
    def name(self) -> str: ...

    def pick(self, features: QuestionFeatures, task: str) -> Pick: ...


def _check_factor(factor: int, factors: Sequence[int]) -> int:
    if factor not in factors:
        raise ValueError(f"factor {factor} is not one of {list(factors)}")
    return factor


@dataclass(frozen=True)
class FixedPicker:
    """The same factor for every question: a plain B x F cell."""

    factor: int

    @property
    def name(self) -> str:
        return f"fixed{self.factor}"

    def pick(self, features: QuestionFeatures, task: str) -> Pick:
        return Pick(self.factor, f"fixed x{self.factor}")


@dataclass(frozen=True)
class SpreadPicker:
    """The least compression whose read is wide enough to hold the evidence.

    The evidence is every chunk scoring at least `ratio` x the best one. If those
    chunks fit in B tokens, read them uncompressed (x1): pressing could only lose
    detail. Otherwise take the smallest F whose read, B x F tokens, covers them.
    `ratio` is the one setting, and the one `calibrate` tunes on dev.
    """

    budget_tokens: int
    ratio: float = DEFAULT_SPREAD_RATIO
    factors: tuple[int, ...] = FACTORS

    @property
    def name(self) -> str:
        return f"spread{self.ratio:g}"

    def pick(self, features: QuestionFeatures, task: str) -> Pick:
        chunks = features.evidence_chunks(self.ratio)
        need = chunks * features.chunk_tokens
        options = sorted(self.factors)
        factor = next((f for f in options if self.budget_tokens * f >= need), options[-1])
        return Pick(
            factor,
            f"{chunks} chunks score >= {self.ratio:g} x top (~{need:,.0f} tokens); "
            f"B={self.budget_tokens:,} -> x{factor}",
        )


@dataclass(frozen=True)
class PerTaskPicker:
    """One factor per task, from a table. Keys may be `hotpotqa_128k` or `hotpotqa`."""

    table: Mapping[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "per-task"

    def pick(self, features: QuestionFeatures, task: str) -> Pick:
        for key in (task, task.split("_", 1)[0]):
            if key in self.table:
                return Pick(int(self.table[key]), f"task {key} -> x{self.table[key]}")
        raise KeyError(f"per-task table has no entry for {task!r}; it covers {sorted(self.table)}")


def picker_from_dict(spec: dict, budget_tokens: int, factors: Sequence[int] = FACTORS) -> Picker:
    """Rebuild a picker from the JSON `calibrate` writes (or `describe_picker` returns)."""
    kind = spec.get("picker")
    if kind == "fixed":
        return FixedPicker(_check_factor(int(spec["factor"]), factors))
    if kind == "spread":
        return SpreadPicker(budget_tokens, float(spec["ratio"]), tuple(factors))
    if kind == "per-task":
        table = {task: _check_factor(int(f), factors) for task, f in spec["table"].items()}
        return PerTaskPicker(table)
    raise ValueError(f"unknown picker {kind!r} in {spec}")


def describe_picker(picker: Picker) -> dict:
    """The JSON form of a picker: written into every plan so a result names its rule."""
    if isinstance(picker, FixedPicker):
        return {"picker": "fixed", "factor": picker.factor}
    if isinstance(picker, SpreadPicker):
        return {"picker": "spread", "ratio": picker.ratio}
    if isinstance(picker, PerTaskPicker):
        return {"picker": "per-task", "table": dict(picker.table)}
    raise TypeError(f"cannot describe {picker!r}")


def make_picker(spec: str, budget_tokens: int, factors: Sequence[int] = FACTORS) -> Picker:
    """Parse a command-line picker: fixed:4 | spread | spread:0.3 | per-task:nq=8,quest=1 | calibrated:FILE."""
    kind, _, arg = spec.partition(":")
    if kind == "fixed":
        return picker_from_dict({"picker": "fixed", "factor": arg}, budget_tokens, factors)
    if kind == "spread":
        return picker_from_dict({"picker": "spread", "ratio": arg or DEFAULT_SPREAD_RATIO}, budget_tokens, factors)
    if kind == "per-task":
        try:
            table = dict(item.split("=", 1) for item in arg.split(",") if item)
        except ValueError:
            raise ValueError(f"per-task expects task=factor pairs, got {arg!r}") from None
        return picker_from_dict({"picker": "per-task", "table": table}, budget_tokens, factors)
    if kind == "calibrated":
        chosen = json.loads(Path(arg).read_text())["chosen"]
        return picker_from_dict(chosen, budget_tokens, factors)
    raise ValueError(f"unknown picker {spec!r}; use fixed:F, spread[:RATIO], per-task:TASK=F,... or calibrated:FILE")
