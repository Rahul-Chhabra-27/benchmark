# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report the ordered-chunk agentic campaign, one subset at a time.

    python -m evaluation.rlm.rlm_order_report --results <campaign>/results --out <campaign>/report \
        [--structured-results /mnt/nas/gautammahale/archive/graphorder/results] \
        [--guards /mnt/nas/gautammahale/benchmark/evaluation/results/loft128k_guards]

For each subset: its own metric (subspan_em; coverage for qampari and quest; string_match
for RULER -- never pooled), the paired difference bm25-query - bm25-graph, and each agentic
arm against each structured arm of the graph-order campaign on the same rows: the same
chunks in the same order, read by the REPL loop instead of a fixed map->reduce. Cost:
input/output tokens per question, root peak context, steps, sub-calls, latency.
Behaviour: chunk reads, how deep into the order the root went, refusals, abstentions.
The loose gold diagnostics say where a gold string sat in each view and whether the root
read that chunk -- a string match, not proof the chunk carried the evidence.

Base RLM and vanilla from loft128k_guards are CONTEXT ONLY: 110-row runs over dev+test
with other prompts. They are aligned to the same test rows by position (test row i is
guards position i+10), the alignment is checked against the gold answers row by row,
and they are rescored per row -- never paired, never subtracted.

--tuned-rows N: the first N test rows of every subset informed prompt revisions, so every
paired comparison is repeated on the remaining rows alone. The harness-interventions table
counts what the harness refused or rejected, and the rows a harness guard ended.

Also writes campaign.json: code revision, document hashes, question ids, every config.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ..compare import headline_score
from .paired import compare, load_rows, per_row_scores
from .rlm_order_gate import ANSWERED_ENDINGS

ARMS = ("bm25-query", "bm25-graph")
STRUCTURED_ARMS = ("source", "bm25-graph")
GUARDS_OFFSET = 10  # LOFT's dev rows precede test in a 110-row run


def _configs(root: Path | None, key: str) -> dict[tuple, dict[str, Path]]:
    """{(dataset, data_dir): {config[key]: run_dir}} for every run that sets `key`."""
    found: dict[tuple, dict[str, Path]] = {}
    if root is None or not root.exists():
        return found
    for config_path in sorted(root.glob("*/config.yaml")):
        config = yaml.safe_load(config_path.read_text()) or {}
        arm = config.get(key)
        if not arm:
            continue
        slot = found.setdefault((config.get("dataset"), config.get("data_dir")), {})
        if arm in slot:
            raise SystemExit(f"two {arm} runs for {config.get('data_dir')}: {slot[arm]} and {config_path.parent}")
        slot[arm] = config_path.parent
    return found


def _metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _headline(run_dir: Path, dataset: str, task: str):
    scores = {k: v for k, v in _metrics(run_dir).items() if k != "runtime"}
    return headline_score(scores, dataset, task)


def _git_revision(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _guards_id(example_id: str) -> str | None:
    """Our test-split id -> the id of the same row in a 110-row dev+test run."""
    match = re.match(r"^(.*?)(\d+)$", example_id)
    return f"{match.group(1)}{int(match.group(2)) + GUARDS_OFFSET}" if match else None


def guards_reference(guards: Path | None, dataset: str, subset: str, model_slug: str, ours: Path) -> dict:
    """Base RLM and vanilla on OUR rows, rescored per row. Context only; {} when unaligned."""
    if guards is None or dataset != "loft":
        return {}
    frame, _ = load_rows(ours)
    gold = {str(r["id"]): list(r["answers"]) for _, r in frame.iterrows()}
    out: dict = {}
    for mode in ("rlm", "vanilla"):
        run_dir = guards / f"loft__{subset}__{model_slug}__{mode}"
        if not (run_dir / "checkpoint.jsonl").exists():
            continue
        g_frame, _ = load_rows(run_dir)
        g_gold = {str(r["id"]): list(r["answers"]) for _, r in g_frame.iterrows()}
        mapping = {ours_id: _guards_id(ours_id) for ours_id in gold}
        if any(g_gold.get(g_id) != gold[o_id] for o_id, g_id in mapping.items()):
            # The position rule failed somewhere: better no reference than a wrong one.
            out[mode] = dict(score=None, note="rows did not align on gold answers")
            continue
        metric, scores = per_row_scores(g_frame[g_frame["id"].astype(str).isin(set(mapping.values()))], "loft")
        values = [scores[g_id] for g_id in mapping.values() if g_id in scores]
        out[mode] = dict(metric=metric, score=sum(values) / len(values) if values else None, n=len(values))
    return out


def _arm_row(dataset: str, subset: str, arm: str, run_dir: Path) -> dict:
    metrics = _metrics(run_dir)
    runtime = metrics.get("runtime", {})
    name, value = _headline(run_dir, dataset, subset)
    return {
        "dataset": dataset,
        "subset": subset,
        "arm": arm,
        "metric": name,
        "score": value,
        "n": runtime.get("scored"),
        "errors": runtime.get("errors"),
        "avg_prompt_tokens": runtime.get("average_prompt_tokens"),
        "avg_completion_tokens": runtime.get("average_completion_tokens"),
        "avg_peak_root_context": runtime.get("average_peak_context_tokens"),
        "avg_steps": runtime.get("average_steps"),
        "avg_sub_calls": runtime.get("average_sub_calls"),
        "avg_latency_s": runtime.get("average_latency_s"),
        "abstained": runtime.get("abstained"),
        "avg_chunk_reads": runtime.get("average_chunk_payload_calls"),
        "chunk_read_example_fraction": runtime.get("chunk_read_example_fraction"),
        "avg_chunks_read": runtime.get("average_chunks_read"),
        "avg_deepest_position": runtime.get("average_deepest_position_read"),
        "avg_chunks_total": runtime.get("average_chunks_total"),
        "chunk_budget_refusals": runtime.get("chunk_budget_refusals"),
        "reader_token_count_mismatches": runtime.get("reader_token_count_mismatches"),
        "coverage_fraction": runtime.get("document_coverage_fraction"),
        "gold_present_fraction": runtime.get("gold_present_fraction"),
        "median_gold_first_position": runtime.get("median_gold_first_position"),
        "gold_in_top5_fraction": runtime.get("gold_in_top5_fraction"),
        "gold_in_read_chunks_fraction": runtime.get("gold_in_read_chunks_fraction"),
        "premature_abstentions_rejected": runtime.get("premature_abstentions_rejected"),
        "prose_answer_cells": runtime.get("prose_answer_cells"),
        "empty_chunk_reads": runtime.get("empty_chunk_reads"),
        "run": run_dir.name,
    }


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pair(a: Path, b: Path, boot: int, seed: int, only_ids: set[str] | None = None) -> dict | None:
    try:
        return compare(a, b, resamples=boot, seed=seed, only_ids=only_ids)
    except SystemExit as exc:  # different rows or metrics: say so, never subtract
        return {"refused": str(exc)}


def held_out_ids(ids, tuned_rows: int) -> set[str]:
    """The rows no revision was tuned on: every id whose trailing row number is >= tuned_rows."""
    held = set()
    for example_id in ids:
        match = re.search(r"(\d+)$", str(example_id))
        if match and int(match.group(1)) >= tuned_rows:
            held.add(str(example_id))
    return held


def _pair_line(subset: str, label: str, p: dict) -> str:
    return (
        f"| {subset} | {label} | {p['metric']} | {p['n']} | {p['mean_b']:.3f} | {p['mean_a']:.3f} | "
        f"{p['delta']:+.3f} | [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}] | "
        f"{p['wins']}/{p['losses']}/{p['ties']} | {p['sign_p']:.3g} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--structured-results", type=Path, default=None, help="the graph-order campaign's results")
    ap.add_argument("--guards", type=Path, default=None, help="loft128k_guards, for context-only reference rows")
    ap.add_argument("--repo", type=Path, default=Path("."), help="checkout the campaign ran, for the revision")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tuned-rows",
        type=int,
        default=0,
        help="the first N test rows of each subset informed prompt revisions; also pair on the rest",
    )
    args = ap.parse_args()

    runs = _configs(args.results, "chunk_order")
    structured = _configs(args.structured_results, "structured_order")
    arm_rows: list[dict] = []
    paired_rows: list[dict] = []
    provenance: dict[str, Any] = {"code_revision": _git_revision(args.repo), "subsets": {}}
    headline = [
        "| subset | metric | n | query | graph | Δ query−graph | 95% CI | W/L/T | sign p "
        "| structured source | structured graph | base RLM (ctx) | vanilla (ctx) |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    versus = [
        "| subset | agentic | structured | agentic score | structured score | Δ | 95% CI | W/L/T | sign p |",
        "|---|---|---|---:|---:|---:|---|---|---:|",
    ]
    held_lines = [
        "| subset | comparison (B − A) | metric | n | B | A | Δ | 95% CI | W/L/T | sign p |",
        "|---|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for (dataset, subset), arms in sorted(runs.items()):
        entry: dict = {"runs": {arm: str(path) for arm, path in arms.items()}, "configs": {}, "question_ids": {}}
        for arm in ARMS:
            if arm not in arms:
                continue
            arm_rows.append(_arm_row(dataset, subset, arm, arms[arm]))
            frame, config = load_rows(arms[arm])
            if {"end_reason", "finished"} <= set(frame.columns):
                guarded = ~frame["finished"].astype(bool) | ~frame["end_reason"].isin(ANSWERED_ENDINGS)
                arm_rows[-1]["guard_endings"] = int(guarded.sum())
            entry["configs"][arm] = config
            entry["question_ids"][arm] = sorted(str(i) for i in frame["id"])
            docs = {(m or {}).get("document_sha256") for m in frame.get("metrics", []) if isinstance(m, dict)}
            entry.setdefault("document_sha256", sorted(d for d in docs if d))
        pair = _pair(arms["bm25-graph"], arms["bm25-query"], args.boot, args.seed) if set(ARMS) <= set(arms) else None
        s_arms = structured.get((dataset, subset), {})
        s_scores = {s: _headline(p, dataset, subset)[1] for s, p in s_arms.items()}
        model_slug = str(arms[ARMS[0] if ARMS[0] in arms else ARMS[1]].name).split("__")[2]
        refs = guards_reference(args.guards, dataset, subset, model_slug, next(iter(arms.values())))
        entry["guards_reference"] = refs
        if pair and "refused" not in pair:
            pair.update(dataset=dataset, subset=subset, comparison="bm25-query - bm25-graph")
            paired_rows.append(pair)
            headline.append(
                f"| {subset} | {pair['metric']} | {pair['n']} | {pair['mean_b']:.3f} | {pair['mean_a']:.3f} | "
                f"{pair['delta']:+.3f} | [{pair['ci_lo']:+.3f}, {pair['ci_hi']:+.3f}] | "
                f"{pair['wins']}/{pair['losses']}/{pair['ties']} | {pair['sign_p']:.3g} | "
                f"{_fmt(s_scores.get('source'))} | {_fmt(s_scores.get('bm25-graph'))} | "
                f"{_fmt(refs.get('rlm', {}).get('score'))} | {_fmt(refs.get('vanilla', {}).get('score'))} |"
            )
        elif pair:
            headline.append(f"| {subset} | — | — | — | — | refused: {pair['refused'][:80]} | | | | | | | |")
        held = held_out_ids(entry["question_ids"].get(ARMS[0], []), args.tuned_rows) if args.tuned_rows else set()
        if held and set(ARMS) <= set(arms):
            kept = _pair(arms["bm25-graph"], arms["bm25-query"], args.boot, args.seed, only_ids=held)
            if kept and "refused" not in kept:
                label = f"rows >= {args.tuned_rows}"
                kept.update(dataset=dataset, subset=subset, comparison=f"bm25-query - bm25-graph, {label}")
                paired_rows.append(kept)
                held_lines.append(_pair_line(subset, "bm25-query − bm25-graph", kept))
        for agentic in ARMS:
            for s_arm in STRUCTURED_ARMS:
                if agentic not in arms or s_arm not in s_arms:
                    continue
                result = _pair(s_arms[s_arm], arms[agentic], args.boot, args.seed)
                if not result or "refused" in result:
                    continue
                result.update(dataset=dataset, subset=subset, comparison=f"agentic {agentic} - structured {s_arm}")
                paired_rows.append(result)
                versus.append(
                    f"| {subset} | {agentic} | {s_arm} | {result['mean_b']:.3f} | {result['mean_a']:.3f} | "
                    f"{result['delta']:+.3f} | [{result['ci_lo']:+.3f}, {result['ci_hi']:+.3f}] | "
                    f"{result['wins']}/{result['losses']}/{result['ties']} | {result['sign_p']:.3g} |"
                )
                if held:
                    kept = _pair(s_arms[s_arm], arms[agentic], args.boot, args.seed, only_ids=held)
                    if kept and "refused" not in kept:
                        comparison = f"agentic {agentic} - structured {s_arm}, rows >= {args.tuned_rows}"
                        kept.update(dataset=dataset, subset=subset, comparison=comparison)
                        paired_rows.append(kept)
                        held_lines.append(_pair_line(subset, f"{agentic} − structured {s_arm}", kept))
        provenance["subsets"][f"{dataset}/{subset}"] = entry

    lines = ["# Agentic RLM over ordered chunks — Qwen3-4B-Instruct-2507", ""]
    lines += ["## Headline, per subset (paired: bm25-query − bm25-graph)", "", *headline, ""]
    lines += [
        "One example at n=55 is 0.018; an interval narrower than that is row-sampling noise only. "
        "Base RLM and vanilla are loft128k_guards rows aligned to these questions: context, not a control "
        "(different prompts; no list-shape note on qampari/quest).",
        "",
    ]
    if len(versus) > 2:
        lines += ["## Agentic vs structured, same rows, same chunks and order", "", *versus, ""]
    if len(held_lines) > 2:
        lines += [
            f"## Held out from tuning: rows {args.tuned_rows} and up of each subset",
            "",
            f"Rows 0–{args.tuned_rows - 1} of every subset informed the prompt revisions, so these pairs leave "
            "them out. B is the first arm named, A the second.",
            "",
            *held_lines,
            "",
        ]
    lines += [
        "## Cost and behaviour, per arm",
        "",
        "| subset | arm | score | in tok/q | out tok/q | peak root ctx | steps | sub-calls | latency s "
        "| chunk reads/q | read any | chunks read | deepest pos | refusals | abstained | errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in arm_rows:
        lines.append(
            f"| {r['subset']} | {r['arm']} | {_fmt(r['score'])} | {_fmt(r['avg_prompt_tokens'], 0)} | "
            f"{_fmt(r['avg_completion_tokens'], 0)} | {_fmt(r['avg_peak_root_context'], 0)} | "
            f"{_fmt(r['avg_steps'], 1)} | {_fmt(r['avg_sub_calls'], 1)} | {_fmt(r['avg_latency_s'], 1)} | "
            f"{_fmt(r['avg_chunk_reads'], 2)} | {_fmt(r['chunk_read_example_fraction'], 2)} | "
            f"{_fmt(r['avg_chunks_read'], 1)} | {_fmt(r['avg_deepest_position'], 1)} | "
            f"{_fmt(r['chunk_budget_refusals'])} | {_fmt(r['abstained'])} | {_fmt(r['errors'])} |"
        )
    lines += [
        "",
        "## Harness interventions, per arm",
        "",
        "| subset | arm | guard endings | abstentions rejected | prose Final Answer cells "
        "| empty reads refused | over-budget reads refused |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in arm_rows:
        lines.append(
            f"| {r['subset']} | {r['arm']} | {_fmt(r.get('guard_endings'))} | "
            f"{_fmt(r['premature_abstentions_rejected'])} | {_fmt(r['prose_answer_cells'])} | "
            f"{_fmt(r['empty_chunk_reads'])} | {_fmt(r['chunk_budget_refusals'])} |"
        )
    lines += [
        "",
        "## Where the gold sat (loose string match — not evidence sufficiency)",
        "",
        "| subset | arm | chunks | gold present | median gold position | gold in top 5 | gold chunk read |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in arm_rows:
        lines.append(
            f"| {r['subset']} | {r['arm']} | {_fmt(r['avg_chunks_total'], 0)} | {_fmt(r['gold_present_fraction'])} | "
            f"{_fmt(r['median_gold_first_position'], 1)} | {_fmt(r['gold_in_top5_fraction'])} | "
            f"{_fmt(r['gold_in_read_chunks_fraction'])} |"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    for name, rows in (("report.csv", arm_rows), ("paired.csv", paired_rows)):
        if rows:
            with open(args.out / name, "w", newline="") as handle:
                fields = sorted({key for row in rows for key in row})
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    provenance["paired"] = paired_rows
    (args.out / "campaign.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
