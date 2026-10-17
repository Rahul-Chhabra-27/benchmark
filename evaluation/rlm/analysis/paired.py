# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two arms, the same rows, and an interval around the difference between them.

The campaign's rule of thumb -- "one example is 0.018, so deltas under ~0.02 are
noise" -- is the right instinct with no number attached. Every paired count quoted
in the results docs ("paired 7-3", "15-3, p=0.0075") was worked out by hand, and
nothing in the repo computes one. This does.

WHAT MAKES THE COMPARISON PAIRED. `--ids-file` and `--limit` both guarantee the same
rows in the same order, so two arms differ on a row for a reason and not because they
saw different rows. This tool INTERSECTS on example id and refuses a mismatch rather
than averaging over a union -- averaging is what throws away the pairing that makes a
55-row comparison worth anything.

WHY IT RESCORES INSTEAD OF READING metrics.json. `metrics.json` holds run-level means.
A paired test needs the per-row score, so each row is re-scored through the SAME
canonical scorer the run used -- `score_prediction_frame` -> the LOFT scorer -- one
row at a time. There is exactly one definition of the headline number in this repo and
this is not a second one.

THE TRAP THIS FILE EXISTS TO AVOID. `predictions.csv` is written with
`frame.to_csv`, so the `answers` LIST round-trips as the string "['a', 'b']".
`calculate_metrics` wraps a non-list in `[value]`, so the gold becomes a single answer
that is a Python repr and matches nothing -- every row scores 0.000 and a delta of
0.000 gets reported with total confidence. `checkpoint.jsonl` is real JSON and is read
in preference; the CSV path is parsed back with `ast.literal_eval`.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from evaluation.benchmarks.results import score_prediction_frame
from evaluation.compare import headline_score

# One example at n=55 is 0.018 and the campaign measured run-to-run drift of about
# that much under four concurrent lanes at temperature 0.0 (vLLM batching numerics).
# A confidence interval narrower than one example is describing sampling noise only.


def noise_floor(n: int) -> float:
    return 1.0 / n if n else 0.0


def _delist(value):
    """A list column as it survives a CSV round-trip, back to a list."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str):
        return [] if pd.isna(value) else [value]
    text = value.strip()
    if text.startswith(("[", "(")):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [value]
        return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    return [value]


def load_rows(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """One run's per-example rows and its config.

    `checkpoint.jsonl` first: it is real JSON, so `answers` arrives as a list and the
    CSV round-trip trap cannot fire. It is also pre-dedup -- an errored example is
    RETRIED on resume and the retry APPENDS -- so the same `drop_duplicates` the CSV
    path applies has to be repeated here or a stale failure is counted twice.
    """
    config = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    checkpoint = run_dir / "checkpoint.jsonl"
    if checkpoint.exists():
        records = []
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # torn trailing line; the example was rerun and re-appended
        frame = pd.DataFrame(records).rename(columns={"pred": "predicted_answer"})
    else:
        frame = pd.read_csv(run_dir / "predictions.csv")
        for column in ("answers", "answer"):
            if column in frame:
                frame[column] = frame[column].map(_delist)
    if "id" in frame:
        frame = frame.drop_duplicates("id", keep="last").reset_index(drop=True)
    return frame, config


def per_row_scores(frame: pd.DataFrame, dataset: str) -> tuple[Optional[str], dict[str, float]]:
    """Each row's canonical score, keyed by example id, plus the metric's name.

    An errored example is a MISSING measurement, not a wrong answer -- the same
    distinction `write_run_artifacts` makes -- so it is dropped rather than scored 0.
    """
    scored = frame[frame["error"].isna()] if "error" in frame else frame
    metric_name = None
    scores: dict[str, float] = {}
    for _, row in scored.iterrows():
        one = pd.DataFrame([row])
        metrics = score_prediction_frame(dataset, one)
        name, value = headline_score(metrics, dataset, str(row.get("task") or ""))
        if value is None:
            continue
        metric_name = metric_name or name
        scores[str(row["id"])] = float(value)
    return metric_name, scores


def bootstrap_ci(diffs: list[float], resamples: int, seed: int, level: float = 0.95):
    """A percentile interval over resamples OF THE PAIRED DIFFERENCES.

    Resampling the two arms independently would discard the pairing, which is the
    whole reason the arms were run on the same rows.
    """
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(resamples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((1 - level) / 2 * resamples)]
    hi = means[min(resamples - 1, int((1 + level) / 2 * resamples))]
    return (lo, hi)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p over the non-ties. No scipy: this is 12 lines."""
    n = wins + losses
    if n == 0:
        return 1.0
    from math import comb

    observed = min(wins, losses)
    tail = sum(comb(n, i) for i in range(observed + 1))
    return min(1.0, 2.0 * tail / (2**n))


def compare(
    a_dir: Path,
    b_dir: Path,
    resamples: int = 10000,
    seed: int = 0,
    allow_subset: bool = False,
    only_ids: Optional[set[str]] = None,
) -> dict:
    a_frame, a_config = load_rows(a_dir)
    b_frame, b_config = load_rows(b_dir)
    dataset = a_config.get("dataset")
    if dataset != b_config.get("dataset"):
        raise SystemExit(f"different datasets ({dataset} vs {b_config.get('dataset')}); these are not comparable")

    a_metric, a_scores = per_row_scores(a_frame, dataset)
    b_metric, b_scores = per_row_scores(b_frame, dataset)
    if only_ids is not None:
        # A deliberate slice -- e.g. the rows no prompt revision was tuned on -- taken
        # before the same-rows check below, which still applies within it.
        a_scores = {i: s for i, s in a_scores.items() if i in only_ids}
        b_scores = {i: s for i, s in b_scores.items() if i in only_ids}
    if a_metric and b_metric and a_metric != b_metric:
        # subspan_em against coverage is not a difference, it is two different questions.
        raise SystemExit(f"arms report different metrics ({a_metric} vs {b_metric}); refusing to subtract them")

    shared = sorted(set(a_scores) & set(b_scores))
    only_a, only_b = sorted(set(a_scores) - set(b_scores)), sorted(set(b_scores) - set(a_scores))
    if (only_a or only_b) and not allow_subset:
        raise SystemExit(
            f"the arms do not describe the same rows: {len(only_a)} only in A, {len(only_b)} only in B "
            f"(e.g. {(only_a or only_b)[:3]}). Averaging over the union would throw away the pairing. "
            "Pass --allow-subset to compare the intersection deliberately"
        )

    diffs = [b_scores[i] - a_scores[i] for i in shared]
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    lo, hi = bootstrap_ci(diffs, resamples, seed)
    mean_a = sum(a_scores[i] for i in shared) / len(shared) if shared else 0.0
    mean_b = sum(b_scores[i] for i in shared) / len(shared) if shared else 0.0
    return {
        "a": a_dir.name,
        "b": b_dir.name,
        "task": a_config.get("data_dir"),
        "metric": a_metric or b_metric,
        "n": len(shared),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "errors_a": int(a_frame["error"].notna().sum()) if "error" in a_frame else 0,
        "errors_b": int(b_frame["error"].notna().sum()) if "error" in b_frame else 0,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": mean_b - mean_a,
        "ci_lo": lo,
        "ci_hi": hi,
        "wins": wins,
        "losses": losses,
        "ties": len(shared) - wins - losses,
        "sign_p": sign_test(wins, losses),
        "noise_floor": noise_floor(len(shared)),
    }


def render(result: dict) -> str:
    half_width = (result["ci_hi"] - result["ci_lo"]) / 2
    floor = result["noise_floor"]
    lines = [
        f"{result['task']}   A={result['a']}",
        f"{' ' * len(str(result['task']))}   B={result['b']}   metric={result['metric']}",
        f"  paired rows   {result['n']:5d}   (A only {result['only_a']}, B only {result['only_b']}, "
        f"errors dropped {result['errors_a']}/{result['errors_b']})",
        f"  mean A        {result['mean_a']:.3f}",
        f"  mean B        {result['mean_b']:.3f}",
        f"  delta (B-A)  {result['delta']:+.3f}   95% CI [{result['ci_lo']:+.3f}, {result['ci_hi']:+.3f}]",
        f"  win/loss/tie  {result['wins']} / {result['losses']} / {result['ties']}"
        f"      sign test p = {result['sign_p']:.3g}",
    ]
    # An interval narrower than one example is describing row-sampling noise alone --
    # it cannot see the run-to-run drift the campaign measured at about that size.
    verdict = "BELOW the floor -- do not report this as a difference" if half_width < floor else "above the floor"
    lines.append(
        f"  noise floor  +/-{floor:.3f} (1 example at n={result['n']}); " f"CI half-width {half_width:.3f} {verdict}"
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="the reference arm's run directory")
    ap.add_argument("--b", required=True, help="the arm being tested; the delta is B - A")
    ap.add_argument("--boot", type=int, default=10000, help="paired bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed, so the interval is reproducible")
    ap.add_argument(
        "--allow-subset",
        action="store_true",
        help="compare the intersection when the arms ran different rows. Off by default: a silently "
        "smaller comparison is a different experiment",
    )
    ap.add_argument("--csv", default=None, help="also append the row to this CSV")
    args = ap.parse_args()

    result = compare(Path(args.a), Path(args.b), args.boot, args.seed, args.allow_subset)
    print(render(result))
    if args.csv:
        path = Path(args.csv)
        frame = pd.DataFrame([result])
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)


if __name__ == "__main__":
    main()
