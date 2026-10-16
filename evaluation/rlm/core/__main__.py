# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m evaluation.rlm.core {check,run} CONFIG.yaml [--set key=value ...]

    check   validate the config and print what it will do (no model is loaded)
    run     run it; re-running resumes

--set overrides a key for this invocation, e.g. --set data.limit=3 --set name=smoke.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .config import ExperimentConfig, config_to_dict, load_config
from .prompts import system_prompt


def _overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        out[key] = yaml.safe_load(value)  # so 3 is an int and null is None
    return out


def describe(config: ExperimentConfig) -> str:
    c = config.compression
    reads = (
        f"in-process kvpress, B={c.budget_tokens:,} tokens ({c.budget_mb:g} MB) x F={c.factor}"
        if c.in_process
        else "plain reads over http (no kvpress)"
    )
    return "\n".join(
        [
            f"experiment   {config.name}",
            f"data         {config.data.dataset} {config.data.task_dirs()} split={config.data.split} "
            f"limit={config.data.limit}",
            f"root         {config.models.root} @ {config.models.base_url}",
            f"ordering     {config.ordering}",
            f"reads        {reads}; one read <= {config.read_tokens:,} tokens",
            f"limits       {config.rlm.max_steps} turns, {config.rlm.max_sub_calls} reads, "
            f"{config.rlm.run_timeout_s:g}s per question",
            f"output       {Path(config.output_dir) / config.name}/<task>/",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evaluation.rlm.core", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["check", "run"])
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--show-prompt", action="store_true", help="check: also print the root prompt")
    args = ap.parse_args(argv)
    config = load_config(args.config, **_overrides(args.set))
    print(describe(config))
    if args.command == "check":
        if args.show_prompt:
            print("\n--- root prompt (with a placeholder question) ---")
            print(
                system_prompt(
                    question="<question>",
                    context_chars=500_000,
                    read_tokens=config.read_tokens,
                    max_steps=config.rlm.max_steps,
                    output_chars=config.rlm.output_chars,
                    ordering=config.ordering,
                    n_chunks=250,
                    chunk_chars=2000,
                    answer_format="Final Answer:",
                )
            )
        print("\n--- resolved config ---")
        print(json.dumps(config_to_dict(config), indent=2))
        return
    from .run import run_experiment

    results = run_experiment(config)
    print(f"\n=== {config.name}")
    for task, metrics in results.items():
        score = {k: round(v, 4) for k, v in metrics.items() if k in ("f1", "coverage", "subspan_em", "em")}
        print(f"  {task:<16} {score}  runtime={metrics['runtime']}")


if __name__ == "__main__":
    main()
