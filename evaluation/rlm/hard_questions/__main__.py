# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m evaluation.rlm.hard_questions --results ROOT [ROOT ...] --out DIR

Finds the questions vanilla solves but RLM + ordering does not, and those only a
compressed RLM solves, across every finished run under the given roots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .arms import discover
from .find import classify, score_matrix, summarize, write


def _markdown(frame) -> str:
    header = "| " + " | ".join(str(c) for c in [frame.index.name or "", *frame.columns]) + " |"
    rule = "|" + "---|" * (len(frame.columns) + 1)
    rows = ["| " + " | ".join(str(v) for v in [index, *row]) + " |" for index, row in frame.iterrows()]
    return "\n".join([header, rule, *rows])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evaluation.rlm.hard_questions", description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, nargs="+", required=True, help="roots holding finished runs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--solved", type=float, default=0.5, help="official score at which a question counts as solved")
    ap.add_argument("--arms", nargs="*", default=None, help="only these arm labels (default: every arm found)")
    args = ap.parse_args(argv)

    arms = discover(args.results)
    if args.arms is not None:
        arms = {label: arm for label, arm in arms.items() if label in args.arms}
    print("arms found:")
    for label, arm in sorted(arms.items(), key=lambda item: (item[1].kind, item[0])):
        print(f"  {arm.kind:<11} {label}  ({len(arm.run_dirs)} task dirs)")
    table = classify(score_matrix(arms), arms, args.solved)
    counts = summarize(table)
    means = (100 * table.drop(columns=["category"]).groupby(level="task").mean()).round(2)
    written = write(table, args.out)
    report = (
        f"# Hard questions (solved = official score >= {args.solved:g})\n\n"
        "Questions per category. `validation_set` = vanilla_only + compression_rescues.\n\n"
        f"{_markdown(counts)}\n\nMean official score per arm on the compared questions:\n\n{_markdown(means)}\n"
    )
    (args.out / "summary.md").write_text(report)
    print("\n" + report)
    print(f"{len(written)} ids files in {args.out / 'ids'}; run one as a validation set with")
    print(
        "  python -m evaluation.rlm.minimal_rlm run <config>.yaml "
        f"--set data.ids_file={args.out / 'ids'}/<task>__validation_set.txt"
    )


if __name__ == "__main__":
    main()
