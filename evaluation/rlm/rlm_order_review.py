# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Label the cause of every REVIEW row of an ordered-chunk stage, one line each.

    python -m evaluation.rlm.rlm_order_review <results dir>

REVIEW rows are the gate's (rlm_order_gate): an ending a harness guard forced, or a row
that passed no chunk to llm_query. The labels are what an acceptance review samples
from. No scores are printed.
  refusal loop   3 or more over-budget reads refused, and no chunk ever read
  case trap      a capitalised term tested against `c.text.lower()`, which never matches
  ungrounded     the last FINAL literal the grounding guard rejected
  abstained      FINAL_NONE without a chunk read
  <end reason>   anything else, shown with the last cell the root ran
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .rlm_order_gate import ANSWERED_ENDINGS

CASE_TRAP = re.compile(r"""(["'])([^"'\n]*[A-Z][^"'\n]*)\1\s+in\s+\w+\.text\.lower\(\)""")
LITERAL = re.compile(r"""FINAL\((["'])(.*?)\1\)""")


def needs_review(row: dict) -> bool:
    """The gate's rule: an ending a guard forced, or no chunk passed to llm_query."""
    reads = (row.get("metrics") or {}).get("chunk_payload_calls")
    return row.get("end_reason") not in ANSWERED_ENDINGS or not reads


def cause(row: dict, transcript: dict) -> str:
    """Why a REVIEW row is one, read from its transcript."""
    m = row.get("metrics") or {}
    steps = [s for s in transcript.get("transcript", []) if isinstance(s, dict)]
    code = "\n".join(s.get("code") or "" for s in steps)
    end = row.get("end_reason")
    trap = CASE_TRAP.search(code)
    if m.get("chunk_budget_refusals", 0) >= 3 and not m.get("chunk_payload_calls"):
        return f"refusal loop: {m['chunk_budget_refusals']} refused, {m.get('chunks_per_call_advertised')} fit per call"
    if trap:
        return f"case trap: {trap.group(0)[:60]}"
    if end == "ungrounded_final":
        literals = LITERAL.findall(code)
        return f"ungrounded: {literals[-1][1][:90] if literals else '?'}"
    if end == "abstained":
        return "abstained: without a chunk read"
    last = next((s.get("code") for s in reversed(steps) if s.get("code")), "") or ""
    return f"{end}: last cell {last.strip()[:90]!r}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    args = ap.parse_args()
    causes: Counter = Counter()
    for run_dir in sorted(p for p in args.results.iterdir() if (p / "checkpoint.jsonl").exists()):
        subset, arm = run_dir.name.split("__")[1], run_dir.name.split("order-")[-1].split("__")[0]
        rows: dict[str, dict] = {}
        for line in (run_dir / "checkpoint.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row  # a retried row's last record stands, as in the gate
        for example_id, row in rows.items():
            if not needs_review(row):
                continue
            transcript = json.loads((run_dir / "transcripts" / f"{example_id}.json").read_text())
            label = cause(row, transcript)
            causes[label.split(":")[0]] += 1
            row_number = example_id.rsplit(":", 1)[-1]
            print(f"{subset:16s} {arm:10s} {row_number:>3s} {str(row.get('end_reason')):18s} {label}")
    print("causes:", dict(causes))


if __name__ == "__main__":
    main()
