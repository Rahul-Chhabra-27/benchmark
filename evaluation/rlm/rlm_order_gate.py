# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the complete ordered-chunk campaign before launch or reporting.

Use --pilot --expect 3 for the pilot; its IDs must be the first three comparator
rows. A failed gate preserves all rows: it blocks acceptance, not analysis. A row that a
harness guard ended, or that never read a chunk, is behaviour: it is listed under REVIEW,
and a run WARNS past REVIEW_WARN_SHARE of such rows and FAILS past REVIEW_FAIL_SHARE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from ..benchmarks.registry import RULER_32K_TASKS
from .rlm import CHUNK_ORDER_REVISION

ARMS = ("bm25-graph", "bm25-query")
LOFT_SUBSETS = ("nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k")
NOTICES = ("[SUB-CALL LIMIT", "[TIME LIMIT", "[SUB-MODEL ERROR]")
# The endings in which the ROOT committed to an answer or an abstention.
ANSWERED_ENDINGS = ("final_called", "abstained", "final_in_prose", "final_var_in_prose")
# Rows for REVIEW: those a harness guard ended (ungrounded_final, max_steps, run_timeout,
# gave_up_no_code, the repetition, error-loop and no-output breakers) and those that never
# passed a chunk to llm_query. One is behaviour -- a 4B root on a hard question, under the
# guards base RLM ran too -- and scores as it stands. A run WARNS past the first share and
# FAILS past the second: at that rate it measures the harness or an unused view, not the
# order (revision 1: 2 of 5 query rows never read; revision 2: 3 of 15 graph rows looped).
# Each share always tolerates one row.
REVIEW_WARN_SHARE = 0.10
REVIEW_FAIL_SHARE = 0.25


def _rows(run_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    path = run_dir / "checkpoint.jsonl"
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)  # A partial/corrupt checkpoint cannot pass.
            rows[str(record["id"])] = record
    return rows


def _transcript(run_dir: Path, example_id: str) -> dict:
    return json.loads((run_dir / "transcripts" / f"{example_id}.json").read_text())


def _kv_read_problems(m: dict, calls: int, config: dict) -> list[str]:
    """A B x F read reaches an in-process KVzip model, not a server, so there is no server
    count to match. Check instead that the press did what the cell names: every read at
    most N tokens, pressed by exactly 1 - 1/F, and keeping at most B."""
    budget = config.get("chunk_call_tokens")
    kept = config.get("chunk_kv_budget_tokens")
    factor = config.get("compression_factor")
    local = m.get("reader_input_tokens")
    if not isinstance(local, list) or len(local) != calls or any(type(t) is not int or t <= 0 for t in local):
        return [f"{calls} chunk reads but missing or invalid local token counts"]
    problems = []
    if isinstance(budget, int) and any(t > budget for t in local):
        problems.append("an over-budget chunk read reached the sub model")
    if not calls:
        return problems
    kv = m.get("sub_kv") or {}
    if not kv.get("calls"):
        return problems + ["chunk reads left no KV stats: the sub model never ran"]
    if isinstance(budget, int) and kv.get("max_context_tokens", 0) > budget:
        problems.append(f"a read held {kv['max_context_tokens']} context tokens, over N={budget}")
    # One token of slack: the pipeline floors n x (1 - ratio).
    if isinstance(kept, int) and kv.get("max_retained_context_tokens", 0) > kept + 1:
        problems.append(f"a read kept {kv['max_retained_context_tokens']} tokens, over B={kept}")
    if not factor:
        problems.append("a kvzip run without a compression_factor")
    elif factor == 1:
        if kv.get("pressed_calls"):
            problems.append("the F=1 control pressed a read")
    else:
        ratio = 1.0 - 1.0 / factor
        if kv.get("pressed_calls") != kv["calls"]:
            problems.append(f"{kv.get('pressed_calls')} of {kv['calls']} sub-calls pressed, not every one")
        for key in ("min_compression_ratio", "max_compression_ratio"):
            if abs(kv.get(key, -1.0) - ratio) > 1e-6:
                problems.append(f"{key} {kv.get(key)} is not 1 - 1/F = {ratio:.6f}")
    return problems


def check_run(
    run_dir: Path, config: dict, expect: int | None, revision: int = CHUNK_ORDER_REVISION
) -> tuple[list[str], dict]:
    problems: list[str] = []
    budget = config.get("chunk_call_tokens")
    if not isinstance(budget, int) or budget <= 0:
        problems.append("missing or invalid chunk token budget")
    if config.get("chunk_order_revision") != revision:
        problems.append(f"expected chunk_order_revision {revision}")
    try:
        rows = _rows(run_dir)
    except (ValueError, KeyError) as exc:
        rows = {}
        problems.append(f"invalid checkpoint: {exc}")
    if expect is not None and len(rows) != expect:
        problems.append(f"{len(rows)} examples, expected {expect}")
    behaviour: Counter = Counter()
    views: set = set()
    documents: set = set()
    review: dict[str, str] = {}
    for example_id, row in rows.items():
        tag = f"{example_id}:"
        if row.get("error"):
            problems.append(f"{tag} error {str(row['error'])[:160]}")
        if "finished" not in row or not row.get("end_reason"):
            problems.append(f"{tag} missing finished flag or end_reason")
        elif not row["finished"] or row["end_reason"] not in ANSWERED_ENDINGS:
            review[example_id] = row["end_reason"]
        m = row.get("metrics") or {}
        if not m.get("chunks_total") or not m.get("chunk_view_sha256") or not m.get("document_sha256"):
            problems.append(f"{tag} missing chunk view or document provenance")
        calls = m.get("chunk_payload_calls", 0)
        if not calls:
            review[example_id] = f"{review[example_id]}+no_reads" if example_id in review else "no_reads"
        local = m.get("reader_input_tokens")
        served = m.get("reader_served_prompt_tokens")
        if config.get("sub_backend") == "kvzip":
            problems.extend(f"{tag} {p}" for p in _kv_read_problems(m, calls, config))
        elif not isinstance(local, list) or not isinstance(served, list) or len(local) != calls or len(served) != calls:
            problems.append(f"{tag} {calls} chunk reads but missing or inconsistent token counts")
        else:
            if any(type(t) is not int or t <= 0 for t in local + served):
                problems.append(f"{tag} missing or invalid local/server token count")
            if local != served:
                problems.append(f"{tag} chunk reads priced differently from the server: {local} vs {served}")
            if isinstance(budget, int) and any(isinstance(t, int) and t > budget for t in local + served):
                problems.append(f"{tag} an over-budget chunk read reached the server")
        try:
            transcript = _transcript(run_dir, example_id)
            if not transcript.get("transcript"):
                raise ValueError("empty transcript")
            if transcript.get("end_reason") != row.get("end_reason") or transcript.get("pred") != row.get("pred"):
                raise ValueError("transcript disagrees with checkpoint")
            text = json.dumps(transcript)
            for notice in NOTICES:
                if notice in text:
                    problems.append(f"{tag} transcript contains {notice}")
        except (OSError, ValueError) as exc:
            problems.append(f"{tag} missing or invalid transcript: {exc}")
        behaviour[f"end_reason={row.get('end_reason')}"] += 1
        behaviour["abstentions"] += row.get("end_reason") == "abstained"
        behaviour["refused over-budget reads"] += m.get("chunk_budget_refusals", 0)
        behaviour["root overflow evictions"] += m.get("overflow_evictions", 0)
        behaviour["premature abstentions rejected"] += m.get("premature_abstentions", 0)
        behaviour["prose Final Answer cells"] += m.get("prose_answer_cells", 0)
        behaviour["empty chunk reads refused"] += m.get("empty_chunk_reads", 0)
        views.add(m.get("chunk_view_sha256"))
        documents.add(m.get("document_sha256"))
    listed = [f"{example_id}:{why}" for example_id, why in sorted(review.items())]
    warnings: list[str] = []
    warn_at, fail_at = (max(1, int(share * len(rows))) for share in (REVIEW_WARN_SHARE, REVIEW_FAIL_SHARE))
    if len(review) > fail_at:
        problems.append(f"{len(review)} of {len(rows)} rows need review, over the {fail_at} a run may have: {listed}")
    elif len(review) > warn_at:
        warnings.append(f"{len(review)} of {len(rows)} rows need review, over the {warn_at} expected")
    behaviour["guard endings"] = sum(1 for why in review.values() if why != "no_reads")
    behaviour["rows without a chunk read"] = sum(1 for why in review.values() if why.endswith("no_reads"))
    info = dict(behaviour=behaviour, views=views, documents=documents, ids=set(rows), rows=rows)
    info.update(review=listed, warnings=warnings)
    return problems, info


def _configs(root: Path, key: str) -> dict[tuple, tuple[Path, dict]]:
    found = {}
    for path in sorted(root.glob("*/config.yaml")):
        config = yaml.safe_load(path.read_text()) or {}
        if config.get(key):
            slot = (config["dataset"], config["data_dir"], config[key])
            if slot in found:
                raise ValueError(f"duplicate run for {slot}")
            found[slot] = (path.parent, config)
    return found


def check_comparator(run_dir: Path, rows: dict, comparator: Path, pilot: bool, expect: int) -> list[str]:
    """Match exact IDs for full runs, the comparator's first N test IDs for pilots."""
    problems = []
    s_rows = _rows(comparator)
    ordered_ids = sorted(s_rows, key=lambda value: int(value.rsplit(":", 1)[1]))
    expected_ids = set(ordered_ids[:expect] if pilot else ordered_ids)
    if len(expected_ids) != expect or set(rows) != expected_ids:
        problems.append(f"IDs differ from {'first pilot rows of' if pilot else 'full'} structured {comparator.name}")
    for example_id in set(rows) & expected_ids:
        ours, theirs = rows[example_id], s_rows[example_id]
        doc = (ours.get("metrics") or {}).get("document_sha256")
        if not doc or doc != (theirs.get("metrics") or {}).get("organization_sha256"):
            problems.append(f"{example_id}: document differs from structured comparator")
        a, b = _transcript(run_dir, example_id), _transcript(comparator, example_id)
        if a.get("question") != b.get("question") or a.get("answers") != b.get("answers"):
            problems.append(f"{example_id}: question or answers differ from structured comparator")
    return problems


def check_graph_artifact(rows: dict, artifacts: list[dict]) -> list[str]:
    problems = []
    for example_id, row in rows.items():
        m = row.get("metrics") or {}
        candidates = [
            a
            for a in artifacts
            if a.get("document_sha256") == m.get("document_sha256") and a.get("config", {}).get("order") == "bm25-graph"
        ]
        if len(candidates) != 1:
            problems.append(f"{example_id}: expected one structured graph artifact, found {len(candidates)}")
            continue
        a = candidates[0]
        spans, order = a["spans"], a["order"]
        if sorted(order) != list(range(len(spans))):
            problems.append(f"{example_id}: invalid graph artifact permutation")
            continue
        digest = hashlib.sha256(json.dumps([spans[i] for i in order]).encode()).hexdigest()
        if digest != m.get("chunk_view_sha256") or len(spans) != m.get("chunks_total"):
            problems.append(f"{example_id}: graph view differs from structured artifact order")
        if m.get("organization_cache_hit") is not True:
            problems.append(f"{example_id}: graph organization was not a cache hit")
    return problems


def check_campaign(
    results: Path,
    dataset: str,
    expect: int,
    structured: Path | None,
    pilot: bool,
    organization: Path | None = None,
    revision: int = CHUNK_ORDER_REVISION,
    require_comparator: bool = True,
) -> list[str]:
    runs = _configs(results, "chunk_order")
    subsets = LOFT_SUBSETS if dataset == "loft" else RULER_32K_TASKS
    expected = {(dataset, subset, arm) for subset in subsets for arm in ARMS}
    problems = [f"missing run: {key}" for key in sorted(expected - set(runs))]
    problems += [f"unexpected run: {key}" for key in sorted(set(runs) - expected)]
    comparators = _configs(structured, "structured_order") if structured else {}
    if dataset == "loft" and structured is None:
        if require_comparator:
            problems.append("LOFT requires --structured-results for row and organization validation")
        else:
            # A reproduction without the structured campaign's results: say what goes unproven.
            print(
                "WARN: --no-comparator: rows, questions and the graph order are NOT checked against "
                "the structured graph-order campaign"
            )
    info = {}
    for key, (run_dir, config) in sorted(runs.items()):
        failures, info[key] = check_run(run_dir, config, expect, revision)
        if not (run_dir / "metrics.json").exists():
            failures.append("run has no completed metrics.json")
        if config.get("mode") != "rlm" or (dataset == "loft" and config.get("split") != "test"):
            failures.append("expected agentic RLM and LOFT test split")
        print(f"{key}: {len(info[key]['ids'])} rows; {dict(info[key]['behaviour'])}")
        if info[key]["review"]:
            print(f"REVIEW {key}: {info[key]['review']}")
        for warning in info[key]["warnings"]:
            print(f"WARN {key}: {warning}")
        problems.extend(f"{key}: {p}" for p in failures)
    for subset in subsets:
        gkey, qkey = (dataset, subset, "bm25-graph"), (dataset, subset, "bm25-query")
        if gkey not in info or qkey not in info:
            continue
        graph, query = info[gkey], info[qkey]
        if graph["ids"] != query["ids"]:
            problems.append(f"{subset}: arms ran different IDs")
        if dataset == "loft" and len(graph["views"]) != 1:
            problems.append(f"{subset}: graph used multiple orders on one document")
        for example_id in graph["ids"] & query["ids"]:
            gm, qm = (info[k]["rows"][example_id].get("metrics") or {} for k in (gkey, qkey))
            if (gm.get("document_sha256"), gm.get("chunks_total")) != (
                qm.get("document_sha256"),
                qm.get("chunks_total"),
            ):
                problems.append(f"{subset}/{example_id}: arms received different documents or chunk counts")
        if dataset != "loft" or structured is None:
            continue
        for s_arm in ("source", "bm25-graph"):
            skey = (dataset, subset, s_arm)
            if skey not in comparators:
                problems.append(f"missing structured comparator: {skey}")
                continue
            s_dir, s_config = comparators[skey]
            for key in (gkey, qkey):
                problems.extend(check_comparator(runs[key][0], info[key]["rows"], s_dir, pilot, expect))
            if s_arm == "bm25-graph":
                # The structured campaign's own cache: where its config recorded it, unless it
                # has since been archived elsewhere (--organization-dir).
                org = organization or s_config.get("organization_dir")
                artifacts = [json.loads(p.read_text()) for p in Path(org).glob("*.json")] if org else []
                problems.extend(check_graph_artifact(graph["rows"], artifacts))
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--expect", type=int, default=55)
    ap.add_argument("--dataset", choices=("loft", "ruler128k"), default="loft")
    ap.add_argument("--pilot", action="store_true", help="compare to the first --expect structured rows")
    ap.add_argument("--structured-results", type=Path)
    ap.add_argument(
        "--organization-dir",
        type=Path,
        help="the structured campaign's graph artifacts, if moved since (default: the organization_dir it recorded)",
    )
    ap.add_argument(
        "--revision",
        type=int,
        default=CHUNK_ORDER_REVISION,
        help=f"the chunk_order_revision every run must carry (default {CHUNK_ORDER_REVISION}; LOFT ran 5)",
    )
    ap.add_argument(
        "--no-comparator",
        action="store_true",
        help="LOFT without the structured campaign's results (e.g. off infolab): skip the checks against it",
    )
    args = ap.parse_args()
    if args.expect <= 0:
        ap.error("--expect must be positive")
    try:
        problems = check_campaign(
            args.results,
            args.dataset,
            args.expect,
            args.structured_results,
            args.pilot,
            args.organization_dir,
            args.revision,
            require_comparator=not args.no_comparator,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems = [f"invalid or incomplete artifacts: {exc}"]
    for problem in problems:
        print(f"FAIL: {problem}")
    print("GATE PASSED" if not problems else f"GATE FAILED: {len(problems)} problem(s)")
    sys.exit(bool(problems))


if __name__ == "__main__":
    main()
