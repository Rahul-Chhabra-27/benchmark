# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which questions does an RLM need more than ordering for?

Ordering gave the biggest jump. This finds the questions still wrong AFTER ordering:

    vanilla_only         vanilla (whole document in the window) solves it; no RLM arm does
    compression_rescues  RLM + ordering fails; an RLM with compressed (wider) reads solves it
    both_solve / rlm_only / nobody

validation_set = vanilla_only + compression_rescues, written as one ids file per task.
Run it with `python -m evaluation.rlm.configurable_rlm run <experiment> --set data.ids_file=<file>`
or `run_benchmark --ids-file <file>`.

Steps on a host with results (LOFT-128k, 55 test rows per task):

    1. vanilla:        run_benchmark --mode vanilla --vanilla-char-limit 600000 ...
                       (the default 400,000 cuts LOFT-128k documents, ~430,000 chars)
    2. RLM + ordering: LIMIT=55 BUDGETS_MB=1024 FACTORS="1 2 4" ARMS=bm25-graph bash scripts/rlm/run_bxf.sh
    3. python -m evaluation.rlm.analysis.hard_questions --results <results root> --out hard_set
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml

from .compression_picker import score_records

# ======================================================================================
# Finding finished runs and labelling them
# ======================================================================================
# Find finished runs and say what each one is, from its config.yaml alone.
#
# An arm is one configuration (possibly spread over one directory per task):
#
#     kind        what it is                                   example label
#     vanilla     the whole document in the model's window     vanilla
#     rlm         RLM, no ordering, no compression             rlm
#     ordered     RLM over ordered chunks, no compression      rlm+bm25-graph
#     compressed  RLM whose reads are KV-compressed (F > 1)    rlm+bm25-graph 1024MB x2
#     other       anything else (structured, skylight, ...)    structured
#
# Both the older harness (run_benchmark.py) and configurable_rlm write a config.yaml
# next to their checkpoint.jsonl; both are understood.


KINDS = ("vanilla", "rlm", "ordered", "compressed", "other")


@dataclass
class Arm:
    label: str
    kind: str
    dataset: str
    run_dirs: list[Path] = field(default_factory=list)


def describe(config: dict) -> Optional[tuple[str, str, str]]:
    """(label, kind, dataset) for a run's config, or None if it is not an RLM-family run."""
    if "ordering" in config and isinstance(config.get("compression"), dict):  # configurable_rlm
        compression = config["compression"]
        factor = max([int(compression.get("factor") or 1), *(int(f) for f in compression.get("choices") or [])])
        ordering = config.get("ordering") or "none"
        kind = "compressed" if factor > 1 else "ordered" if ordering != "none" else "rlm"
        return f"configurable_rlm:{config['name']}", kind, config["data"]["dataset"]
    if config.get("backend") != "rlm":
        return None
    mode, dataset = config.get("mode"), config.get("dataset")
    if mode == "vanilla":
        return "vanilla", "vanilla", dataset
    order = config.get("chunk_order")
    factor = config.get("compression_factor")
    budget = config.get("sub_kv_memory_budget")
    label = str(mode) + (f"+{order}" if order else "")
    if config.get("sub_backend") == "kvzip" and factor is not None and budget is not None:
        label += f" {float(budget):g}{config.get('sub_kv_memory_budget_unit') or 'MB'} x{float(factor):g}"
    if mode not in ("rlm", "clean-rlm"):
        return label, "other", dataset
    kind = "compressed" if (factor or 1) > 1 else "ordered" if order else "rlm"
    return label, kind, dataset


def discover(roots: Iterable[Path]) -> dict[str, Arm]:
    """Every arm under `roots`, keyed by label. One label may span several task directories."""
    arms: dict[str, Arm] = {}
    for root in roots:
        for config_path in sorted(Path(root).rglob("config.yaml")):
            if not (config_path.parent / "checkpoint.jsonl").exists():
                continue
            described = describe(yaml.safe_load(config_path.read_text()) or {})
            if described is None:
                continue
            label, kind, dataset = described
            arm = arms.setdefault(label, Arm(label, kind, dataset))
            arm.run_dirs.append(config_path.parent)
    return arms


# ======================================================================================
# Sorting questions into categories
# ======================================================================================
# Score every question in every arm, and sort the questions by who can solve them.
#
# A question is *solved* by an arm when its official score (F1, or coverage for list
# tasks) reaches `solved` (default 0.5). Only questions answered by a vanilla arm AND
# by an RLM arm without compression are compared. Each then lands in one category:
#
#     compression_rescues  no uncompressed RLM arm solves it, a compressed arm does
#     vanilla_only         vanilla solves it, no RLM arm does (compressed included)
#     both_solve           vanilla and an uncompressed RLM arm both solve it
#     rlm_only             an uncompressed RLM arm solves it, vanilla does not
#     nobody               nothing solves it
#
# "Uncompressed RLM" means the ordered arms when there are any (the question is what
# fails AFTER ordering), else the plain RLM arms. `vanilla_only` plus
# `compression_rescues` is the validation set: questions an RLM can only get right by
# doing something beyond ordering -- reading wider through compression, or falling back.


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


# ======================================================================================
# Command line
# ======================================================================================


def _markdown(frame) -> str:
    header = "| " + " | ".join(str(c) for c in [frame.index.name or "", *frame.columns]) + " |"
    rule = "|" + "---|" * (len(frame.columns) + 1)
    rows = ["| " + " | ".join(str(v) for v in [index, *row]) + " |" for index, row in frame.iterrows()]
    return "\n".join([header, rule, *rows])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m evaluation.rlm.analysis.hard_questions", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--results", type=Path, nargs="+", required=True, help="roots holding finished runs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--solved", type=float, default=0.5, help="official score at which a question counts as solved")
    ap.add_argument("--arms", nargs="*", default=None, help="only these arm labels (default: every arm found)")
    args = ap.parse_args(argv)

    arms = discover(args.results)
    if args.arms is not None:
        arms = {label: arm for label, arm in arms.items() if label in args.arms}
    print("arms found:")
    for label, arm in sorted(arms.items(), key=lambda item: (item[1].kind, item[0])):
        print(f"  {arm.kind:<11} {label}  ({len(arm.run_dirs)} task dirs)")
    table = classify(score_matrix(arms), arms, args.solved)
    counts = summarize(table)
    means = (100 * table.drop(columns=["category"]).groupby(level="task").mean()).round(2)
    written = write(table, args.out)
    report = (
        f"# Hard questions (solved = official score >= {args.solved:g})\n\n"
        "Questions per category. `validation_set` = vanilla_only + compression_rescues.\n\n"
        f"{_markdown(counts)}\n\nMean official score per arm on the compared questions:\n\n{_markdown(means)}\n"
    )
    (args.out / "summary.md").write_text(report)
    print("\n" + report)
    print(f"{len(written)} ids files in {args.out / 'ids'}; run one as a validation set with")
    print(
        "  python -m evaluation.rlm.configurable_rlm run <experiment> "
        f"--set data.ids_file={args.out / 'ids'}/<task>__validation_set.txt"
    )


if __name__ == "__main__":
    main()
