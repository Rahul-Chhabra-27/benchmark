# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command line: python -m evaluation.rlm.factor_picker {plan,calibrate,assemble} ...

    plan       pick F per question (no model), write plan dir + ids files
    calibrate  choose the picker on dev, from fixed-F dev runs
    assemble   merge answers at the picked F, score, compare with fixed F

See README.md in this folder for the full walkthrough.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..datasets import load_examples
from .assemble import compare, picked_scores, render
from .budget import FACTORS, budget_tokens
from .calibrate import DEFAULT_RATIOS, calibrate
from .features import question_features
from .pickers import make_picker
from .plan import make_plan, read_plan, summarize
from .results import load_scored_cells


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
    ap = argparse.ArgumentParser(prog="python -m evaluation.rlm.factor_picker", description=__doc__.split("\n")[0])
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
