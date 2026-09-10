# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate the ordered-group runs before any number from them is reported.

    python -m evaluation.rlm.graph_order_gate <results_dir> [--expect N]

Every check is per EXAMPLE, because a run-level average can hide one bad row:
no errors, full coverage, one map call per packed group, every reader request
within its budget and priced exactly as the server counted it, the reducer
within the context, no limit notices anywhere in a transcript, and the two arms
reading exactly the same source spans. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

NOTICES = ("[SUB-CALL LIMIT", "[TIME LIMIT", "[SUB-MODEL ERROR]")


def _rows(run_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in (run_dir / "checkpoint.jsonl").read_text().splitlines():
        if line.strip():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[str(record["id"])] = record  # a retry appends; the last one counts
    return rows


def _spans(run_dir: Path, example_id: str) -> list | None:
    path = run_dir / "transcripts" / f"{example_id}.json"
    if not path.exists():
        return None
    steps = json.loads(path.read_text()).get("transcript", [])
    return sorted(tuple(s) for step in steps for s in step.get("source_spans", []))


def check_run(run_dir: Path, config: dict, expect: int | None) -> tuple[list[str], dict]:
    problems: list[str] = []
    budget = config["structured_group_input_tokens"]
    context = config["structured_context_tokens"]
    reduce_cap = config["structured_reduce_max_tokens"]
    rows = _rows(run_dir)
    if expect is not None and len(rows) != expect:
        problems.append(f"{len(rows)} examples, expected {expect}")
    spans: dict[str, list | None] = {}
    for example_id, row in rows.items():
        tag = f"{example_id}:"
        if row.get("error"):
            problems.append(f"{tag} error {row['error'][:160]}")
            continue
        m = row.get("metrics") or {}
        if m.get("document_coverage_fraction") != 1.0:
            problems.append(f"{tag} coverage {m.get('document_coverage_fraction')}")
        if m.get("map_calls") != m.get("group_count"):
            problems.append(f"{tag} map_calls {m.get('map_calls')} != groups {m.get('group_count')}")
        local = m.get("reader_input_tokens", [])
        served = m.get("reader_served_prompt_tokens", [])
        if len(local) != m.get("group_count"):
            problems.append(f"{tag} {len(local)} reader counts for {m.get('group_count')} groups")
        if local and max(local) > budget:
            problems.append(f"{tag} reader input {max(local)} > budget {budget}")
        if local != served:
            bad = sum(1 for a, b in zip(local, served) if a != b)
            problems.append(f"{tag} {bad} reader calls priced differently from the server (local vs served)")
        if m.get("sub_payload_truncated_calls"):
            problems.append(f"{tag} {m['sub_payload_truncated_calls']} truncated sub-calls")
        reducer = m.get("reducer_input_tokens")
        if reducer is not None and reducer + reduce_cap > context:
            problems.append(f"{tag} reducer {reducer} + {reduce_cap} > {context}")
        path = run_dir / "transcripts" / f"{example_id}.json"
        text = path.read_text() if path.exists() else ""
        if not text:
            problems.append(f"{tag} no transcript")
        for notice in NOTICES:
            if notice in text:
                problems.append(f"{tag} transcript contains {notice}")
        spans[example_id] = _spans(run_dir, example_id)
    return problems, spans


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--expect", type=int, default=None, help="examples each run must hold")
    args = ap.parse_args()

    by_subset: dict[str, dict[str, tuple[Path, dict]]] = {}
    failures = 0
    for config_path in sorted(args.results.glob("*/config.yaml")):
        config = yaml.safe_load(config_path.read_text()) or {}
        if config.get("structured_order"):
            by_subset.setdefault(config["data_dir"], {})[config["structured_order"]] = (config_path.parent, config)
    if not by_subset:
        sys.exit(f"no ordered runs under {args.results}")
    for subset, arms in sorted(by_subset.items()):
        spans_by_arm = {}
        for arm, (run_dir, config) in sorted(arms.items()):
            problems, spans = check_run(run_dir, config, args.expect)
            spans_by_arm[arm] = spans
            status = "OK" if not problems else f"FAIL ({len(problems)})"
            print(f"{subset:14s} {arm:11s} {status}")
            for p in problems[:20]:
                print(f"    {p}")
            failures += len(problems)
        if set(spans_by_arm) != {"source", "bm25-graph"}:
            print(f"{subset:14s} MISSING an arm: have {sorted(spans_by_arm)}")
            failures += 1
            continue
        source, graph = spans_by_arm["source"], spans_by_arm["bm25-graph"]
        if set(source) != set(graph):
            print(f"{subset:14s} arms ran different questions")
            failures += 1
        for example_id in set(source) & set(graph):
            if source[example_id] is not None and graph[example_id] is not None:
                # Bisection can split a chunk differently per arm; compare what was covered.
                if _merged(source[example_id]) != _merged(graph[example_id]):
                    print(f"{subset:14s} {example_id}: arms read different source text")
                    failures += 1
    print("GATE PASSED" if not failures else f"GATE FAILED: {failures} problem(s)")
    sys.exit(1 if failures else 0)


def _merged(spans: list) -> list:
    out: list = []
    for start, end in spans:
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


if __name__ == "__main__":
    main()
