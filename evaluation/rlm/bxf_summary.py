# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One table for a B x F results tree: what each cell scored, and whether it got its B and F.

    python -m evaluation.rlm.bxf_summary /tmp/$USER/bxf/results

Every run under the tree that read its chunks through a KVzip model (a B x F cell of an
ordered-chunk arm) becomes one row. Each is checked with the campaign gate's own per-run
check (rlm_order_gate.check_run): every read at most N tokens, pressed by exactly 1 - 1/F,
keeping at most B, and no sub-model error in any transcript. The exit status is 1 when any
cell fails, so a script can end on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from evaluation.compare import headline_score

from .rlm_order_gate import check_run

COLUMNS = ("B", "F", "arm", "subset", "rows", "score", "N", "kept_max", "B_tok", "F_real", "pressed", "check")


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def collect(results: Path) -> list[dict]:
    """One row per B x F run under `results`, with its score and its per-cell check."""
    rows = []
    for config_path in sorted(results.rglob("config.yaml")):
        config = yaml.safe_load(config_path.read_text()) or {}
        if not config.get("chunk_order") or config.get("sub_backend") != "kvzip":
            continue
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        runtime = metrics.get("runtime", {})
        problems, info = check_run(run_dir, config, None)
        if not metrics_path.exists():
            problems.append("no metrics.json: the run has not finished")
        _, score = headline_score(metrics, config.get("dataset"), config.get("data_dir"))
        budget, factor = config.get("sub_kv_memory_budget"), config.get("compression_factor")
        rows.append(
            dict(
                B=f"{budget:g}{config.get('sub_kv_memory_budget_unit')}" if budget is not None else None,
                F=f"{factor:g}" if factor is not None else None,
                arm=config.get("chunk_order"),
                subset=config.get("data_dir"),
                rows=len(info["ids"]),
                score=score,
                N=config.get("chunk_call_tokens"),
                kept_max=runtime.get("max_sub_retained_context_tokens"),
                B_tok=config.get("chunk_kv_budget_tokens"),
                F_real=runtime.get("realized_compression_factor"),
                pressed=runtime.get("sub_pressed_call_fraction"),
                check="FAIL" if problems else "PASS",
                problems=problems,
                review=info["review"],
                run=run_dir.name,
            )
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="the RESULTS directory run_bxf.sh wrote to")
    args = ap.parse_args()
    rows = collect(args.results)
    if not rows:
        print(f"no B x F runs under {args.results}")
        return 1
    table = [[_fmt(row[column]) for column in COLUMNS] for row in rows]
    widths = [max(len(column), *(len(line[i]) for line in table)) for i, column in enumerate(COLUMNS)]
    print("  ".join(column.ljust(width) for column, width in zip(COLUMNS, widths)))
    for line in table:
        print("  ".join(value.ljust(width) for value, width in zip(line, widths)))
    # REVIEW rows are behaviour (a guard ended the row, or it never read a chunk), not a failure.
    for row in rows:
        for why in row["review"]:
            print(f"REVIEW {row['run']}: {why}")
    failed = [row for row in rows if row["problems"]]
    for row in failed:
        for problem in row["problems"]:
            print(f"FAIL {row['run']}: {problem}")
    print(f"\n{len(rows) - len(failed)} of {len(rows)} cells passed their checks")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
