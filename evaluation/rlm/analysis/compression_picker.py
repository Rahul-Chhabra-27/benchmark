# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pick one KV compression factor per question, BEFORE the RLM runs.

At a fixed memory budget B, the factor F decides how much text each sub-call reads
(N = B x F tokens) and how hard it is compressed; F = 1 is plain reads with no kvpress.
A picker chooses F per question from the question alone, and the run then uses it:

    1. plan       picker -> plan.jsonl + one ids file per (task, F)     (CPU, seconds)
    2. run        each ids file runs as an ordinary B x F cell          (GPU; scripts/rlm/run_picked.sh)
    3. assemble   merge, score, compare with fixed F and the oracle     (CPU)

Pickers: fixed:F | spread[:RATIO] (how spread out the question's BM25 evidence is) |
per-task:T=F,... | calibrated:FILE (whatever `calibrate` chose on the dev split).

    python -m evaluation.rlm.analysis.compression_picker plan --tasks hotpotqa --budget-mb 1024 \
        --picker spread:0.3 --out P
    python -m evaluation.rlm.analysis.compression_picker calibrate --tasks ... --budget-mb 1024 \
        --dev-results R --out cal.json
    python -m evaluation.rlm.analysis.compression_picker assemble --plan P --results R [--baselines R]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Protocol, Sequence

import pandas as pd
import yaml

from evaluation.benchmarks.results import score_prediction_frame
from evaluation.rlm.harness.rlm import CHARS_PER_TOKEN_ESTIMATE
from evaluation.rlm.harness.run_benchmark import chunk_order_query
from evaluation.rlm.shared.datasets import load_examples
from evaluation.rlm.shared.organization import CONFIG as ORGANIZATION_CONFIG
from evaluation.rlm.shared.retrieval import get_index

# ======================================================================================
# B x F arithmetic
# ======================================================================================
# The B x F arithmetic a picker needs, with no model loaded.
#
# B is the KV one sub-call may keep, in tokens; F is how hard each read is pressed;
# a read carries N = B x F tokens. F = 1 is the no_press control (plain dense reads).


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


# ======================================================================================
# What a picker may look at
# ======================================================================================
# What a picker may look at before a run: the question and the document, never a model.
#
# The one signal is how SPREAD OUT the question's lexical evidence is. Every chunk of
# the bm25-graph grid is scored by BM25 against the bare question; a question whose
# strong matches sit in two chunks needs a narrow read, one whose matches are scattered
# over thirty needs a wide one. On LOFT-128k test rows the signal tracks the task the
# way one would hope: nq and hotpotqa questions concentrate in 1-3 chunks, quest and
# musique spread over far more.


@dataclass(frozen=True)
class QuestionFeatures:
    """BM25 evidence profile of one question over its document."""

    query: str
    n_chunks: int
    # Average tokens in one grid chunk, so a chunk count converts to a read size.
    chunk_tokens: float
    # Positive BM25 scores, best first. Kept whole so calibration can sweep the
    # threshold without re-scoring the document.
    scores: tuple[float, ...]

    def evidence_chunks(self, ratio: float) -> int:
        """Chunks scoring at least `ratio` x the best chunk.

        A question sharing no term with the document carries no evidence of where its
        answer is, so it counts as spread over everything: the widest read.
        """
        if not 0 < ratio <= 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        if not self.scores:
            return self.n_chunks
        floor = ratio * self.scores[0]
        return sum(1 for score in self.scores if score >= floor)

    def evidence_tokens(self, ratio: float) -> float:
        return self.evidence_chunks(ratio) * self.chunk_tokens


def question_features(
    question: str,
    document: str,
    dataset: str = "loft",
    chars_per_token: float = CHARS_PER_TOKEN_ESTIMATE,
) -> QuestionFeatures:
    """Score every grid chunk against the bare question.

    The grid is the organization's (2000 chars, no overlap): the same chunks a
    bm25-graph read is made of. The query is what the bm25-query arm ranks by, with
    LOFT's citation boilerplate stripped -- left in, "documents" and "answer" match
    every chunk and every question looks maximally spread.
    """
    index = get_index(document, ORGANIZATION_CONFIG["chunk_chars"], ORGANIZATION_CONFIG["overlap"])
    query = chunk_order_query(question, dataset)
    scores = tuple(hit.score for hit in index.rank_all(query) if hit.score > 0)
    return QuestionFeatures(
        query=query,
        n_chunks=index.n_chunks,
        chunk_tokens=len(document) / max(index.n_chunks, 1) / chars_per_token,
        scores=scores,
    )


# ======================================================================================
# Pickers
# ======================================================================================
# Pickers: each one chooses ONE compression factor per question, before the run.
#
#     fixed:4             every question at x4 -- today's B x F cell, the baseline
#     spread:0.3          per question, from how spread out its BM25 evidence is
#     per-task:nq=8,...   per task, a table (usually written by `calibrate` from dev)
#     calibrated:FILE     whichever of the above `calibrate` chose, with its settings
#
# A picker never calls a model, so a whole plan is made in seconds on a CPU host.


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


# ======================================================================================
# Reading finished runs
# ======================================================================================
# Read finished B x F runs and score them row by row.
#
# A cell is found by what its config.yaml says (ordered-chunk agentic run, this
# chunk order, this budget, this factor), not by its directory name, so the same
# code reads a live picked run, a lane sweep, or anything copied off a cluster.


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


# ======================================================================================
# Step 1: plan
# ======================================================================================
# Step 1: pick F for every question and write the plan, before any model runs.
#
# A plan directory holds
#
#     plan.json                 the picker, B, split and rows it was made for
#     plan.jsonl                one line per question: id, task, factor, reason
#     ids/<task>__x<F>.txt      the questions of one task that got factor F
#
# and each ids file is exactly what `run_benchmark --ids-file` (or the lane's
# IDS_FILE) needs to run that group as an ordinary B x F cell.


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
        "budget_tokens": budget_tokens(budget_mb),
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


# ======================================================================================
# Step 3: assemble and compare
# ======================================================================================
# Step 3: merge each question's answer at its picked F, score, compare.
#
# The comparison is always on the SAME rows: a question counts only if it was
# answered at its picked factor and in every fixed-F baseline being compared.
# `oracle` is the best factor per question in hindsight -- not a method, the
# ceiling any picker over these factors could reach.


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


# ======================================================================================
# Choosing the picker on dev
# ======================================================================================
# Choose the picker on the DEV split, so the test split is only ever reported.
#
# Input: fixed-F runs of the dev rows, one per factor. Every candidate picker is
# scored on those rows by looking up, per question, the answer its factor produced:
#
#     fixed x1 .. x8     one factor everywhere
#     spread:<ratio>     for every ratio in the grid
#     per-task           each task's best factor on dev (ties go to the lower F)
#
# The candidate with the best task-average score is `chosen`, ties to the earlier
# (simpler) one. All of these are in-sample on dev, per-task most of all: with ten
# dev rows per LOFT task its table is the easiest to overfit, which is why the
# simpler candidates win ties.


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


# ======================================================================================
# Command line
# ======================================================================================


def _tasks(names: list[str], dataset: str, length: str) -> list[str]:
    """LOFT subsets may be named bare (`hotpotqa`) like the lane does; add the length."""
    if dataset != "loft":
        return names
    return [name if "_" in name else f"{name}_{length}" for name in names]


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dataset", default="loft")
    ap.add_argument("--tasks", nargs="+", required=True, help="e.g. hotpotqa nq (LOFT: _128k is added)")
    ap.add_argument("--loft-length", default="128k")
    ap.add_argument("--budget-mb", type=float, required=True, help="B, the KV one read may keep")
    ap.add_argument("--factors", type=int, nargs="+", default=list(FACTORS))
    ap.add_argument("--chunk-order", default="bm25-graph")


def cmd_plan(args: argparse.Namespace) -> None:
    tasks = _tasks(args.tasks, args.dataset, args.loft_length)
    picker = make_picker(args.picker, budget_tokens(args.budget_mb), args.factors)
    out, rows = make_plan(picker, tasks, args.budget_mb, Path(args.out), args.dataset, args.split, args.limit)
    print(f"[plan] {picker.name}: {len(rows)} questions, B={args.budget_mb:g}MB -> {out}")
    print(summarize(rows))


def cmd_calibrate(args: argparse.Namespace) -> None:
    tasks = _tasks(args.tasks, args.dataset, args.loft_length)
    cells = load_scored_cells(Path(args.dev_results), args.budget_mb, args.factors, args.chunk_order, args.dataset)
    missing = [f for f in args.factors if f not in cells]
    if missing:
        raise SystemExit(f"no dev runs found for factors {missing} under {args.dev_results}")
    features = {}
    for task in tasks:
        for example in load_examples(args.dataset, task, None, "dev"):
            features[example["id"]] = question_features(example["question"], example["context"], args.dataset)
    cells = {f: cell[cell.index.isin(features)] for f, cell in cells.items()}
    result = calibrate(cells, features, budget_tokens(args.budget_mb), args.ratios)
    result.update(budget_mb=args.budget_mb, split="dev", tasks=tasks, chunk_order=args.chunk_order)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"[calibrate] {result['rows']} dev rows; task-average score per candidate:")
    for candidate in result["candidates"]:
        print(f"  {candidate['name']:<12} {candidate['score']:6.2f}")
    print(f"[calibrate] chosen: {result['chosen_name']} -> {args.out}")
    print(f"            use it with: --picker calibrated:{args.out}")


def cmd_assemble(args: argparse.Namespace) -> None:
    meta, rows = read_plan(Path(args.plan))
    budget_mb, dataset = meta["budget_mb"], meta["dataset"]
    ids = {row["id"] for row in rows}
    factors = sorted({row["factor"] for row in rows})
    picked_cells = load_scored_cells(Path(args.results), budget_mb, factors, args.chunk_order, dataset, ids)
    baselines = None
    if args.baselines:
        baselines = load_scored_cells(Path(args.baselines), budget_mb, args.factors, args.chunk_order, dataset, ids)
    picked = picked_scores(rows, picked_cells)
    table = compare(picked, baselines)
    out = Path(args.out or Path(args.plan) / "result")
    out.mkdir(parents=True, exist_ok=True)
    picked.to_csv(out / "predictions.csv", index=False)
    table.to_csv(out / "comparison.csv", index=False)
    (out / "comparison.md").write_text(render(table) + "\n")
    unanswered = int(picked["score"].isna().sum())
    (out / "metrics.json").write_text(
        json.dumps({"plan": meta, "answered": len(picked) - unanswered, "unanswered": unanswered}, indent=2) + "\n"
    )
    print(f"[assemble] {meta['picker_name']}, B={budget_mb:g}MB: {len(picked) - unanswered}/{len(picked)} answered")
    if unanswered:
        print(f"[assemble] {unanswered} questions have no answer at their picked factor yet (run step 2)")
    print(render(table))
    print(f"[assemble] -> {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m evaluation.rlm.analysis.compression_picker", description=__doc__.split("\n")[0]
    )
    sub = ap.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="pick F per question and write the plan")
    _common(plan)
    plan.add_argument(
        "--picker", default="spread", help="fixed:F | spread[:RATIO] | per-task:T=F,... | calibrated:FILE"
    )
    plan.add_argument("--split", default="test")
    plan.add_argument("--limit", type=int, default=None, help="first N rows of each task (same as the lane's LIMIT)")
    plan.add_argument("--out", required=True, help="plan directory to write")
    plan.set_defaults(func=cmd_plan)

    cal = sub.add_parser("calibrate", help="choose the picker on dev from fixed-F dev runs")
    _common(cal)
    cal.add_argument("--dev-results", required=True, help="root holding one dev run per factor")
    cal.add_argument("--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS))
    cal.add_argument("--out", required=True, help="calibration JSON to write")
    cal.set_defaults(func=cmd_calibrate)

    asm = sub.add_parser("assemble", help="merge, score and compare a plan's answers")
    asm.add_argument("--plan", required=True)
    asm.add_argument("--results", required=True, help="root holding the runs at the picked factors")
    asm.add_argument("--baselines", default=None, help="root holding full fixed-F runs to compare against")
    asm.add_argument("--factors", type=int, nargs="+", default=list(FACTORS), help="baseline factors to compare")
    asm.add_argument("--chunk-order", default="bm25-graph")
    asm.add_argument("--out", default=None, help="default: <plan>/result")
    asm.set_defaults(func=cmd_assemble)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
