# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate the ordered-chunk agentic runs before any number from them is reported.

    python -m evaluation.rlm.rlm_order_gate <results_dir> [--expect N] [--structured-results DIR]

Every check is per EXAMPLE, because a run-level average can hide one bad row: no error,
a transcript, the whole document offered as the view, and every chunk read priced exactly
as the server counted it and within its budget. Across arms: the same rows; the
bm25-graph view is ONE order per LOFT corpus (the organization, not something rebuilt per
question), while the bm25-query view changes with the question. With
--structured-results, the rows and the documents must also be the graph-order campaign's.

Model BEHAVIOUR is reported, never failed: a root that never passed a chunk to llm_query,
a [SUB-CALL LIMIT] notice, an abstention, a refused over-budget read -- those are outcomes
of the arm, and failing the gate on them would drop exactly the rows that describe it.
Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ARMS = ("bm25-graph", "bm25-query")
BEHAVIOUR_NOTICES = ("[SUB-CALL LIMIT", "[TIME LIMIT")


def _rows(run_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    path = run_dir / "checkpoint.jsonl"
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[str(record["id"])] = record  # a retry appends; the last one counts
    return rows


def check_run(run_dir: Path, config: dict, expect: int | None) -> tuple[list[str], dict]:
    """Problems (failures) and a behaviour summary for one run directory."""
    problems: list[str] = []
    budget = config.get("chunk_call_tokens")
    rows = _rows(run_dir)
    if expect is not None and len(rows) != expect:
        problems.append(f"{len(rows)} examples, expected {expect}")
    behaviour: Counter = Counter()
    views: set = set()
    documents: set = set()
    for example_id, row in rows.items():
        tag = f"{example_id}:"
        if row.get("error"):
            problems.append(f"{tag} error {str(row['error'])[:160]}")
            continue
        m = row.get("metrics") or {}
        if not m.get("chunks_total"):
            problems.append(f"{tag} no chunk view recorded")
        calls = m.get("chunk_payload_calls", 0)
        local = m.get("reader_input_tokens")
        served = m.get("reader_served_prompt_tokens")
        if local is None or served is None or len(local) != calls or len(served) != calls:
            counts = f"{len(local or [])} local / {len(served or [])} served"
            problems.append(f"{tag} {calls} chunk reads but {counts} token counts")
        elif local != served:
            problems.append(f"{tag} chunk reads priced differently from the server: local {local} vs served {served}")
        if local and budget and max(local) > budget:
            problems.append(f"{tag} a {max(local)}-token chunk read reached the server over the {budget} budget")
        path = run_dir / "transcripts" / f"{example_id}.json"
        text = path.read_text() if path.exists() else ""
        if not text:
            problems.append(f"{tag} no transcript")
        for notice in BEHAVIOUR_NOTICES:
            if notice in text:
                behaviour[f"transcripts with {notice}"] += 1
        behaviour[f"end_reason={row.get('end_reason')}"] += 1
        if not calls:
            behaviour["never passed a chunk to llm_query"] += 1
        behaviour["refused over-budget reads"] += m.get("chunk_budget_refusals", 0)
        behaviour["root overflow evictions"] += m.get("overflow_evictions", 0)
        if m.get("organization_cache_hit") is False:
            behaviour["organization built here (not a cache hit)"] += 1
        views.add(m.get("chunk_view_sha256"))
        documents.add(m.get("document_sha256"))
    return problems, dict(behaviour=behaviour, views=views, documents=documents, ids=set(rows))


def _structured_runs(root: Path | None) -> dict[str, dict[str, Path]]:
    """{data_dir: {structured_order: run_dir}} for the graph-order campaign."""
    found: dict[str, dict[str, Path]] = {}
    if root is None:
        return found
    for config_path in sorted(root.glob("*/config.yaml")):
        config = yaml.safe_load(config_path.read_text()) or {}
        if config.get("structured_order"):
            found.setdefault(config["data_dir"], {})[config["structured_order"]] = config_path.parent
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--expect", type=int, default=None, help="examples each run must hold")
    ap.add_argument("--structured-results", type=Path, default=None, help="the graph-order campaign's results")
    args = ap.parse_args()

    by_subset: dict[tuple, dict[str, tuple[Path, dict]]] = {}
    for config_path in sorted(args.results.glob("*/config.yaml")):
        config = yaml.safe_load(config_path.read_text()) or {}
        if config.get("chunk_order"):
            key = (config["dataset"], config["data_dir"])
            by_subset.setdefault(key, {})[config["chunk_order"]] = (config_path.parent, config)
    if not by_subset:
        sys.exit(f"no ordered-chunk runs under {args.results}")
    structured = _structured_runs(args.structured_results)

    failures = 0
    for (dataset, subset), arms in sorted(by_subset.items()):
        info = {}
        for arm, (run_dir, config) in sorted(arms.items()):
            problems, info[arm] = check_run(run_dir, config, args.expect)
            status = "OK" if not problems else f"FAIL ({len(problems)})"
            print(f"{dataset}/{subset:16s} {arm:11s} {status}  ({len(info[arm]['ids'])} rows)")
            for problem in problems[:20]:
                print(f"    {problem}")
            for name, count in sorted(info[arm]["behaviour"].items()):
                if count:
                    print(f"    · {name}: {count}")
            failures += len(problems)
        if set(info) != set(ARMS):
            print(f"{dataset}/{subset:16s} MISSING an arm: have {sorted(info)}")
            failures += 1
            continue
        graph, query = info["bm25-graph"], info["bm25-query"]
        if graph["ids"] != query["ids"]:
            print(f"{dataset}/{subset:16s} the two arms ran different questions")
            failures += 1
        if dataset == "loft" and len(graph["views"]) != 1:
            # One corpus per LOFT subset, so one organization: several orders mean the view
            # was rebuilt from something other than the document.
            print(f"{dataset}/{subset:16s} bm25-graph used {len(graph['views'])} different orders on one corpus")
            failures += 1
        if len(query["ids"]) > 1 and len(query["views"]) < 2:
            print(f"{dataset}/{subset:16s} WARNING: bm25-query gave every question the same order")
        for s_arm, s_dir in sorted(structured.get(subset, {}).items()):
            s_rows = _rows(s_dir)
            if set(s_rows) != graph["ids"]:
                print(f"{dataset}/{subset:16s} rows differ from structured {s_arm} ({s_dir.name})")
                failures += 1
            s_docs = {(r.get("metrics") or {}).get("organization_sha256") for r in s_rows.values()} - {None}
            if s_docs and s_docs != graph["documents"] - {None}:
                print(f"{dataset}/{subset:16s} a different document from structured {s_arm}")
                failures += 1
    print("GATE PASSED" if not failures else f"GATE FAILED: {failures} problem(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
