# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read a geometry sweep, and say whether a cell gained SIGNAL or gained DISTRACTORS.

Raising k does both at once. Five windows become ten, some of the five new ones
carry the answer and the rest are plausible neighbours the retriever liked, and a
single accuracy number cannot say which side moved. So the k-curve turning over is
usually reported as "more context hurts" when the measured claim is only "more of
BOTH hurts on net".

The same confound was shown to invert a published effect from the other direction:
naive middle-removal truncation conflates removing distractors with removing signal,
and separating them flipped the sign of the measured context-length effect
(arXiv 2608.03297). This walks the sweep in the mirror direction, using the
per-window `gold_windows_in_chunks` / `nongold_windows_in_chunks` counts
run_benchmark records beside `gold_in_retrieved`.

Both counts are the same LOOSE containment test `gold_in_retrieved` uses: a matching
answer string does not establish that the window carries sufficient evidence
(c448c97). Read a "gold window" as an upper bound on signal, never a certificate.

Usage:
    python -m evaluation.rlm.structured_decompose --results <tree> [--csv out.csv]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from evaluation.compare import headline_score, iter_runs


def _cell(config: dict) -> Optional[tuple]:
    """The sweep coordinates of one run, or None if it is not part of a sweep."""
    if config.get("mode") != "structured":
        return None
    task = config.get("data_dir")
    task = "__".join(task) if isinstance(task, list) else task
    return (
        task,
        config.get("chunks_source"),
        int(config.get("search_window_chars") or 0),
        int(config.get("search_overlap_chars") or 0),
        int(config.get("chunks_k") or 0),
        int(config.get("structured_map_chars") or 0),
        int(config.get("structured_hops") or 1),
        # Load-bearing, for the same reason chunks_source is: a `bare`-query cell and
        # a `task`-query cell sit at identical (window, k) coordinates, and a grid
        # keyed without this would silently render one on top of the other.
        config.get("chunks_query") or "task",
    )


def collect(results: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir, config, metrics in iter_runs(results):
        cell = _cell(config)
        if cell is None:
            continue
        task, source, window, overlap, k, map_chars, hops, query = cell
        runtime = metrics.get("runtime", {}) if isinstance(metrics.get("runtime"), dict) else {}
        metric_name, score = headline_score(metrics, config.get("dataset"), task)
        gold = runtime.get("average_gold_windows_in_chunks")
        nongold = runtime.get("average_nongold_windows_in_chunks")
        rows.append(
            {
                "run_dir": run_dir.name,
                "task": task,
                "chunks_source": source,
                "window": window,
                "overlap": overlap,
                "k": k,
                "map_chars": map_chars,
                "hops": hops,
                "chunks_query": query,
                "admitted_chars": k * window if not map_chars else k * min(window, map_chars),
                "metric": metric_name,
                "score": score,
                "em": metrics.get("em"),
                "scored": runtime.get("scored"),
                "errors": runtime.get("errors"),
                "abstained": runtime.get("abstained"),
                "gold_windows": gold,
                "nongold_windows": nongold,
                "gold_in_retrieved_fraction": runtime.get("gold_in_retrieved_fraction"),
                "chunks_with_findings": runtime.get("average_chunks_with_findings"),
                "map_calls": runtime.get("average_map_calls"),
                "peak_context_tokens": runtime.get("average_peak_context_tokens"),
                "latency_s": runtime.get("average_latency_s"),
                # The structured arm's version of `sub_pressed_call_fraction` near
                # zero: the pipeline ran and learned nothing, so the cell is not a
                # measurement of the geometry it names.
                # Errors, or the pipeline finding nothing DESPITE the gold being in
                # front of it. Findings near zero is only a defect when retrieval
                # worked: at k=2 on LOFT the chunks genuinely do not contain the
                # answer (gold_in_retrieved 0.00-0.15), and an abstention there is the
                # honest result, not a cell to throw away.
                "suspect": bool(runtime.get("errors"))
                or (
                    (runtime.get("average_chunks_with_findings") or 0) < 0.05
                    and (runtime.get("gold_in_retrieved_fraction") or 0) > 0.2
                ),
                "retrieval_starved": (runtime.get("gold_in_retrieved_fraction") or 0) < 0.2,
            }
        )
    return rows


def _fmt(value, spec="6.3f", blank="     -"):
    return blank if value is None else format(value, spec)


def _sweep_rows(rows: list[dict[str, Any]], task: str, query: str = "task") -> list[dict[str, Any]]:
    """Just the geometry sweep: BM25 chunks, one hop, no exposure cap.

    `chunks_source` is load-bearing here. The oracle arm is a `file` cell at the same
    window and k as a BM25 cell, so keying the grid on (window, k) alone let it
    OVERWRITE its own control -- nq's 2000/5 rendered as 0.800 (the oracle) where the
    BM25 cell is 0.545. Every filter that selects a comparison has to name the axis
    that distinguishes the arms, not just the axes being swept.
    """
    return [
        r
        for r in rows
        if r["task"] == task
        and r["chunks_source"] == "bm25"
        and r["hops"] == 1
        and not r["map_chars"]
        and r["chunks_query"] == query
    ]


def grid(rows: list[dict[str, Any]], task: str, query: str = "task") -> str:
    """Score by window (rows) x k (columns), the way the fixed-chunk grid reads."""
    cells = {(r["window"], r["k"]): r for r in _sweep_rows(rows, task, query)}
    if not cells:
        return ""
    windows = sorted({w for w, _ in cells})
    ks = sorted({k for _, k in cells})
    metric = next((r["metric"] for r in rows if r["task"] == task and r["metric"]), "score")
    asked = "the question" if query == "bare" else "the whole task string"
    out = [
        f"\n{task} -- {metric} by window (rows) x k (cols), BM25 asked {asked}",
        "  W\\k  " + "".join(f"{k:>8}" for k in ks),
    ]
    for window in windows:
        line = f"{window:>6} "
        for k in ks:
            row = cells.get((window, k))
            if row is None:
                line += "       -"
                continue
            mark = "*" if row["suspect"] else ("!" if row["retrieval_starved"] else " ")
            line += f"{_fmt(row['score'], '8.3f')}{mark}"
        out.append(line.rstrip())
    if any(row["suspect"] for row in cells.values()):
        out.append("  * suspect: errors, or nothing found although the gold was retrieved")
    if any(row["retrieval_starved"] for row in cells.values()):
        out.append("  ! retrieval-starved: the gold reached under 20% of examples -- a real")
        out.append("    result about the retriever, not about the pipeline reading it")
    return "\n".join(out)


def decompose(rows: list[dict[str, Any]], task: str, query: str = "task") -> str:
    """Walk k upward at each window and attribute the score change to the two counts."""
    by_window: dict[int, list[dict]] = defaultdict(list)
    for row in _sweep_rows(rows, task, query):
        if row["score"] is not None:
            by_window[row["window"]].append(row)
    if not by_window:
        return ""
    out = [
        f"\n{task} -- what each step up in k actually added",
        "     W   k_from   k_to    d_score   d_gold  d_nongold   per_gold  per_nongold",
    ]
    for window in sorted(by_window):
        series = sorted(by_window[window], key=lambda r: r["k"])
        for before, after in zip(series, series[1:]):
            d_score = after["score"] - before["score"]
            d_gold = _delta(after["gold_windows"], before["gold_windows"])
            d_nongold = _delta(after["nongold_windows"], before["nongold_windows"])
            out.append(
                f"{window:>6}{before['k']:>9}{after['k']:>7}{d_score:>11.3f}"
                f"{_fmt(d_gold, '9.2f', '        -')}{_fmt(d_nongold, '11.2f', '          -')}"
                # Score moved per window ADDED of each kind. When the two have
                # opposite signs the turnover is distractor load; when d_gold is
                # ~0 and the score still falls, the extra windows were pure noise.
                f"{_fmt(_ratio(d_score, d_gold), '11.3f', '          -')}"
                f"{_fmt(_ratio(d_score, d_nongold), '13.3f', '            -')}"
            )
    return "\n".join(out)


def _delta(after, before):
    return None if after is None or before is None else after - before


def _ratio(numerator, denominator):
    if denominator is None or abs(denominator) < 1e-9:
        return None
    return numerator / denominator


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="a results tree of --mode structured runs")
    ap.add_argument("--csv", default=None, help="also write every cell as one row here")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    rows = collect(Path(args.results))
    if not rows:
        raise SystemExit(f"no --mode structured runs with a metrics.json under {args.results}")

    for task in sorted({r["task"] for r in rows}):
        for query in sorted({r["chunks_query"] for r in rows if r["task"] == task}):
            print(grid(rows, task, query))
            print(decompose(rows, task, query))

    exposure = [r for r in rows if r["map_chars"]]
    if exposure:
        print("\nexposure sweep -- score by characters the map step SAW")
        print("  task                 map_chars   score   findings")
        for row in sorted(exposure, key=lambda r: (r["task"], r["map_chars"])):
            print(
                f"  {row['task']:<20}{row['map_chars']:>10}{_fmt(row['score'], '8.3f')}"
                f"{_fmt(row['chunks_with_findings'], '11.2f')}"
            )

    oracle = [r for r in rows if r["chunks_source"] == "file"]
    if oracle:
        print("\noracle chunks -- retrieval perfect by construction, so what is left is the reader")
        print("  task                 oracle    bm25 same cell    gap")
        for row in sorted(oracle, key=lambda r: r["task"]):
            control = next(
                (
                    c
                    for c in _sweep_rows(rows, row["task"], "task")
                    if c["window"] == row["window"] and c["k"] == row["k"]
                ),
                None,
            )
            gap = None if control is None or control["score"] is None else row["score"] - control["score"]
            print(
                f"  {row['task']:<20}{_fmt(row['score'], '8.3f')}{_fmt(control and control['score'], '18.3f')}"
                f"{_fmt(gap, '8.3f')}"
            )

    hopped = [r for r in rows if r["hops"] > 1]
    if hopped:
        print("\nsecond hop")
        print("  task                 hops   score   gold_in_retrieved")
        for row in sorted(hopped, key=lambda r: r["task"]):
            print(
                f"  {row['task']:<20}{row['hops']:>6}{_fmt(row['score'], '8.3f')}"
                f"{_fmt(row['gold_in_retrieved_fraction'], '19.3f')}"
            )

    suspect = [r["run_dir"] for r in rows if r["suspect"]]
    if suspect:
        print(f"\n{len(suspect)} SUSPECT cell(s) -- do not put these in a table:")
        for name in suspect:
            print(f"  {name}")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {len(rows)} rows to {args.json_out}")


if __name__ == "__main__":
    main()
