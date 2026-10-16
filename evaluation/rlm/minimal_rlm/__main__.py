# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m evaluation.rlm.minimal_rlm {check,run} CONFIG.yaml [--set key=value ...]

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
from .loop import RLM
from .run import make_limits


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
    if c.in_process:
        chosen = (
            f"F chosen per read from {list(c.factors)} (default {c.factor})" if len(c.factors) > 1 else f"F={c.factor}"
        )
        reads = f"in-process kvpress, B={c.budget_tokens:,} tokens ({c.budget_mb:g} MB), {chosen}"
        if c.offload:
            reads += ", offloading on"
    else:
        reads = "plain reads over http (no kvpress)"
    shown = ", shown to the root" if config.cost.show_root else ", recorded only"
    costs = f"gpu {config.cost.gpu_token:g}/token, offloaded {config.cost.offloaded_token:g}/token{shown}"
    return "\n".join(
        [
            f"experiment   {config.name}",
            f"data         {config.data.dataset} {config.data.task_dirs()} split={config.data.split} "
            f"limit={config.data.limit}",
            f"root         {config.models.root} @ {config.models.base_url}",
            f"ordering     {config.ordering}",
            f"reads        {reads}; one read <= {config.largest_read_tokens:,} tokens",
            f"cost         {costs}",
            f"limits       {config.rlm.max_steps} turns, {config.rlm.max_sub_calls} reads, "
            f"{config.rlm.run_timeout_s:g}s per question",
            f"output       {Path(config.output_dir) / config.name}/<task>/",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evaluation.rlm.minimal_rlm", description=__doc__.split("\n")[0])
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
            rlm = RLM(None, None, make_limits(config), config.ordering)  # type: ignore[arg-type]
            print(rlm.system_prompt("<question>", 500_000, 250, 2000, "Final Answer:"))
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
