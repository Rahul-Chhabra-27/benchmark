# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report the BM25-graph vs source-order campaign, one subset at a time.

    python -m evaluation.rlm.graph_order_report --results <campaign>/results \
        --out <campaign> [--historical /mnt/nas/gautammahale/archive/qagnostic/results]

For each subset: the subset's own metric (subspan_em, or coverage for qampari and
quest -- never pooled), the paired difference graph - source with a bootstrap CI,
input/output tokens, latency, map calls, group sizes, reducer exposure, errors and
preprocessing CPU. Historical exhaustive runs are listed as CONTEXT ONLY: their
windows are 8000/1600 and they read one window per call, so they differ from both
arms in more than order.

Also writes campaign.json: code revision, document hashes, exact question ids,
and every run's config -- what is needed to reproduce or audit the table.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ..compare import headline_score
from .paired import compare, load_rows

ARMS = ("source", "bm25-graph")


def _runs(results: Path) -> dict[str, dict[str, Path]]:
    """{subset: {arm: run_dir}} for every ordered run under `results`."""
    found: dict[str, dict[str, Path]] = {}
    for config_path in sorted(results.glob("*/config.yaml")):
        config = yaml.safe_load(config_path.read_text()) or {}
        arm = config.get("structured_order")
        if arm not in ARMS:
            continue
        subset = config.get("data_dir")
        if arm in found.setdefault(subset, {}):
            raise SystemExit(f"two {arm} runs for {subset}: {found[subset][arm]} and {config_path.parent}")
        found[subset][arm] = config_path.parent
    return found


def _metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _headline(metrics: dict, task: str):
    scores = {k: v for k, v in metrics.items() if k != "runtime"}
    return headline_score(scores, "loft", task)


def _historical(root: Path | None, subset: str) -> list[dict]:
    """Exhaustive (`sel-all`) runs of the query-agnostic campaign, as context."""
    if root is None or not root.exists():
        return []
    rows = []
    for run_dir in sorted(root.glob(f"loft__{subset}__*__structured__*__sel-all__*")):
        metrics = _metrics(run_dir)
        if not metrics:
            continue
        name, value = _headline(metrics, subset)
        runtime = metrics.get("runtime", {})
        rows.append(dict(run=run_dir.name, metric=name, score=value, n=runtime.get("scored"),
                         tokens=runtime.get("average_tokens"), latency_s=runtime.get("average_latency_s"),
                         map_calls=runtime.get("average_map_calls")))
    return rows


def _git_revision(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _arm_row(subset: str, arm: str, run_dir: Path) -> dict:
    runtime = _metrics(run_dir).get("runtime", {})
    name, value = _headline(_metrics(run_dir), subset)
    return {
        "subset": subset,
        "arm": arm,
        "metric": name,
        "score": value,
        "n": runtime.get("scored"),
        "errors": runtime.get("errors"),
        "avg_prompt_tokens": runtime.get("average_prompt_tokens"),
        "avg_completion_tokens": runtime.get("average_completion_tokens"),
        "avg_latency_s": runtime.get("average_latency_s"),
        "avg_map_calls": runtime.get("average_map_calls"),
        "avg_group_count": runtime.get("average_group_count"),
        "avg_chunks_per_group": runtime.get("average_chunks_per_group"),
        "avg_reader_input_tokens": runtime.get("average_reader_input_tokens"),
        "max_reader_input_tokens": runtime.get("max_reader_input_tokens"),
        "reader_token_count_mismatches": runtime.get("reader_token_count_mismatches"),
        "avg_reducer_input_tokens": runtime.get("average_reducer_input_tokens"),
        "max_reducer_input_tokens": runtime.get("max_reducer_input_tokens"),
        "avg_retained_findings": runtime.get("average_retained_finding_count"),
        "avg_reducer_finding_chars": runtime.get("average_reducer_finding_chars"),
        "coverage_fraction": runtime.get("document_coverage_fraction"),
        # One build per corpus, shared by every query on it: report the total and
        # the per-query share, never charge the full build to each question.
        "organization_build_cpu_s": runtime.get("organization_build_cpu_seconds"),
        "organization_cpu_s_per_query": (
            runtime["organization_build_cpu_seconds"] / runtime["scored"]
            if runtime.get("scored") and runtime.get("organization_build_cpu_seconds") is not None
            else None
        ),
        "avg_packing_cpu_s": runtime.get("average_packing_cpu_seconds"),
        "run": run_dir.name,
    }


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--historical", type=Path, default=None, help="query-agnostic results dir holding sel-all runs")
    ap.add_argument("--repo", type=Path, default=Path("."), help="checkout the campaign ran, for the revision")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = _runs(args.results)
    arm_rows: list[dict] = []
    paired_rows: list[dict] = []
    provenance: dict[str, Any] = {"code_revision": _git_revision(args.repo), "subsets": {}}
    lines = ["# BM25-graph vs source order — LOFT-128k, Qwen3-4B-Instruct-2507", ""]
    for subset in sorted(runs):
        arms = runs[subset]
        for arm in ARMS:
            if arm in arms:
                arm_rows.append(_arm_row(subset, arm, arms[arm]))
        entry: dict = {"runs": {arm: str(path) for arm, path in arms.items()}}
        for arm, path in arms.items():
            frame, config = load_rows(path)
            entry.setdefault("configs", {})[arm] = config
            entry.setdefault("question_ids", {})[arm] = sorted(str(i) for i in frame["id"])
            hashes = {(m or {}).get("organization_sha256") for m in frame.get("metrics", []) if isinstance(m, dict)}
            entry.setdefault("document_sha256", {})[arm] = sorted(h for h in hashes if h)
        if set(ARMS) <= set(arms):
            result = compare(arms["source"], arms["bm25-graph"], resamples=args.boot, seed=args.seed)
            result["subset"] = subset
            paired_rows.append(result)
        provenance["subsets"][subset] = entry

    lines += ["## Paired difference (bm25-graph − source), per subset", "",
              "| subset | metric | n | source | graph | Δ | 95% CI | W/L/T | sign p | errors s/g |",
              "|---|---|---:|---:|---:|---:|---|---|---:|---|"]
    for r in paired_rows:
        lines.append(
            f"| {r['subset']} | {r['metric']} | {r['n']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | "
            f"{r['delta']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
            f"{r['wins']}/{r['losses']}/{r['ties']} | {r['sign_p']:.3g} | {r['errors_a']}/{r['errors_b']} |"
        )
    lines += ["", "One example at n=55 is 0.018; an interval narrower than that is row-sampling noise only.", "",
              "## Cost and exposure, per arm", "",
              "| subset | arm | score | in tok/q | out tok/q | latency s | groups | chunks/group | reader in (avg/max) "
              "| reducer in (avg/max) | findings | coverage | org CPU s (total / per q) | pack CPU s | errors |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|"]
    for r in arm_rows:
        lines.append(
            f"| {r['subset']} | {r['arm']} | {_fmt(r['score'])} | {_fmt(r['avg_prompt_tokens'], 0)} | "
            f"{_fmt(r['avg_completion_tokens'], 0)} | {_fmt(r['avg_latency_s'], 1)} | "
            f"{_fmt(r['avg_group_count'], 1)} | "
            f"{_fmt(r['avg_chunks_per_group'], 1)} | {_fmt(r['avg_reader_input_tokens'], 0)} / "
            f"{_fmt(r['max_reader_input_tokens'])} | {_fmt(r['avg_reducer_input_tokens'], 0)} / "
            f"{_fmt(r['max_reducer_input_tokens'])} | {_fmt(r['avg_retained_findings'], 1)} | "
            f"{_fmt(r['coverage_fraction'])} | {_fmt(r['organization_build_cpu_s'], 2)} / "
            f"{_fmt(r['organization_cpu_s_per_query'], 4)} | {_fmt(r['avg_packing_cpu_s'], 2)} | {_fmt(r['errors'])} |"
        )
    historical = {s: _historical(args.historical, s) for s in sorted(runs)}
    if any(historical.values()):
        lines += ["", "## Historical exhaustive reference (context only — different geometry)", "",
                  "8000-char windows with 1600 overlap, one window per call, finding cap ≤ 2000. Not a control "
                  "for either arm.", "", "| subset | metric | score | n | tokens/q | latency s | map calls |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for subset, rows in historical.items():
            for h in rows:
                lines.append(f"| {subset} | {h['metric']} | {_fmt(h['score'])} | {_fmt(h['n'])} | "
                             f"{_fmt(h['tokens'], 0)} | {_fmt(h['latency_s'], 1)} | {_fmt(h['map_calls'], 1)} |")
        provenance["historical"] = historical

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    if arm_rows:
        with open(args.out / "report.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(arm_rows[0]))
            writer.writeheader()
            writer.writerows(arm_rows)
    if paired_rows:
        with open(args.out / "paired.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)
    provenance["paired"] = paired_rows
    (args.out / "campaign.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
