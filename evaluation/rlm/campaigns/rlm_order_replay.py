# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay revision 6's grounding rule on a finished revision-5 ordered-chunk campaign.

    python -m evaluation.rlm.campaigns.rlm_order_replay <results dir> <dataset>

Revision 6 changes no prompt text, only which FINAL literals are accepted
(composed_answer_grounded). So up to the first literal revision 5 rejected, a revision-6
run IS the recorded run, and if revision 6 accepts that literal the row ends there with
it as the prediction; if not, revision 6 rejects it too and the recorded run goes on. For
every row this walks the literals revision 5 rejected -- every FINAL literal except the
one a final_called row ended on, and every ungrounded prose FINAL -- and rescores the row
with the first one revision 6 accepts.

Transcripts keep the display-truncated observation, not the full output the harness
grounded against, so this can only UNDER-count the rows revision 6 would accept.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from evaluation.rlm.analysis.paired import load_rows, per_row_scores
from evaluation.rlm.harness.rlm import TEXT_FINAL_RE, _static_string, composed_answer_grounded


def final_literal(code: str) -> str | None:
    """The static string a cell passes to FINAL(...), if it passes one."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "FINAL" and node.args:
            return _static_string(node.args[0])
    return None


def first_accepted(transcript: dict, end_reason: str, answer_prefix: str) -> tuple[int, str | None]:
    """(how many literals revision 5 rejected, the first of them revision 6 accepts)."""
    steps = [s for s in transcript["transcript"] if isinstance(s, dict) and not s.get("forced")]
    code_steps = [k for k, s in enumerate(steps) if s.get("code")]
    accepted = code_steps[-1] if end_reason == "final_called" and code_steps else None
    task_text = f"{transcript.get('question', '')}\n{answer_prefix}"
    seen, rejected = "", 0
    for k, s in enumerate(steps):
        literal: str | None = None
        if s.get("code"):
            seen += "\n" + str(s.get("observation", ""))
            if k != accepted:
                literal = final_literal(s["code"])
        elif s.get("observation") == "[ungrounded prose FINAL rejected]":
            match = TEXT_FINAL_RE.search(s.get("reply") or "")
            literal = match.group(1).strip() if match else None
        if literal:
            rejected += 1
            if composed_answer_grounded(literal, seen, task_text):
                return rejected, literal
    return rejected, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("dataset")
    args = ap.parse_args()
    means: dict[tuple, tuple[float, float]] = {}
    total_rows = total_rejected = total_accepted = 0
    for run_dir in sorted(p for p in args.results.iterdir() if (p / "checkpoint.jsonl").exists()):
        frame, config = load_rows(run_dir)
        accepted: dict[str, str] = {}
        rejected_rows = 0
        for _, row in frame.iterrows():
            transcript = json.loads((run_dir / "transcripts" / f"{row['id']}.json").read_text())
            prefix = row.get("answer_prefix")
            rejected, literal = first_accepted(
                transcript, str(row.get("end_reason") or ""), prefix if isinstance(prefix, str) else ""
            )
            rejected_rows += rejected > 0
            if literal is not None:
                accepted[str(row["id"])] = literal
        _, before = per_row_scores(frame, args.dataset)
        replayed = frame.copy()
        for example_id, literal in accepted.items():
            replayed.loc[replayed["id"] == example_id, "predicted_answer"] = literal
        _, after = per_row_scores(replayed, args.dataset)
        subset, arm = config.get("data_dir"), config.get("chunk_order")
        n = len(before)
        mean_before = sum(before.values()) / n if n else 0.0
        mean_after = sum(after.values()) / n if n else 0.0
        means[(subset, arm)] = (mean_before, mean_after)
        print(
            f"{subset:16s} {arm:10s} n={n:3d} rows with a rejected literal={rejected_rows:2d} "
            f"accepted under r6={len(accepted):2d}  mean {mean_before:.3f} -> {mean_after:.3f} "
            f"({mean_after - mean_before:+.3f})"
        )
        for example_id, literal in accepted.items():
            old = frame.loc[frame["id"] == example_id, "predicted_answer"].iloc[0]
            print(
                f"    {example_id}: {before.get(example_id)} -> {after.get(example_id)}"
                f"  r6 {literal[:70]!r}  r5 {str(old)[:50]!r}"
            )
        total_rows, total_rejected = total_rows + n, total_rejected + rejected_rows
        total_accepted += len(accepted)
    print(f"TOTAL rows {total_rows}, with a rejected literal {total_rejected}, accepted under r6 {total_accepted}")
    for subset in sorted({s for s, _ in means}):
        query, graph = means.get((subset, "bm25-query")), means.get((subset, "bm25-graph"))
        if query and graph:
            print(f"{subset:16s} query - graph: r5 {query[0] - graph[0]:+.3f}, r6 replay {query[1] - graph[1]:+.3f}")


if __name__ == "__main__":
    main()
