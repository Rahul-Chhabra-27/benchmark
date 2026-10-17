# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m evaluation.rlm.configurable_rlm {list,check,run} [NAME] [--set key=value ...]

    list         the experiments in experiments.yaml
    check NAME   what one experiment will do, and its root prompt (no model is loaded)
    run NAME     run it; re-running resumes

--set overrides a key for this invocation, e.g. --set data.limit=3 --set name=smoke.
--file reads another experiments file instead of the one next to this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .config import EXPERIMENTS_FILE, ExperimentConfig, config_to_dict, load_experiment, read_experiments
from .rlm import RLM
from .run import make_features, make_limits, make_reader


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
    tuned = config.loop == "tuned"
    loop = "tuned: our original rlm.py, via harness/run_benchmark.py" if tuned else "base: the minimal RLM"
    shown = ", shown to the root" if config.cost.show_root else ", recorded only"
    costs = f"gpu {config.cost.gpu_token:g}/token, offloaded {config.cost.offloaded_token:g}/token{shown}"
    return "\n".join(
        [
            f"experiment   {config.name}",
            f"data         {config.data.dataset} {config.data.task_dirs()} split={config.data.split} "
            f"limit={config.data.limit}",
            f"root         {config.models.root} @ {config.models.base_url}",
            f"ordering     {config.ordering}",
            f"loop         {loop}",
            f"reads        {reads}" + ("" if tuned else f"; one read <= {config.largest_read_tokens:,} tokens"),
            *([] if tuned else [f"cost         {costs}"]),
            f"limits       {config.rlm.max_steps} turns, {config.rlm.max_sub_calls} reads, "
            f"{config.rlm.run_timeout_s:g}s per question",
            f"output       {Path(config.output_dir) / config.name}/<task>/",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evaluation.rlm.configurable_rlm", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["list", "check", "run"])
    ap.add_argument("name", nargs="?", help="an experiment in the experiments file")
    ap.add_argument("--file", type=Path, default=EXPERIMENTS_FILE, help="experiments file (default: experiments.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    if args.command == "list":
        for name in read_experiments(args.file)[1]:
            config = load_experiment(name, args.file)
            c = config.compression
            factors = "x" + "/".join(str(f) for f in c.factors)
            compression = "none" if c.budget_mb is None else f"{factors} at {c.budget_mb:g} MB"
            extras = "".join(
                f"  {k}" for k, on in (("offload", c.offload), ("cost shown", config.cost.show_root)) if on
            )
            tuned = "  loop=tuned" if config.loop == "tuned" else ""
            print(f"  {name:<18} ordering={config.ordering:<11} compression={compression}{extras}{tuned}")
        return
    if not args.name:
        ap.error(f"{args.command} needs an experiment name; see `list`")
    config = load_experiment(args.name, args.file, _overrides(args.set))
    print(describe(config))
    if args.command == "check" and config.loop == "tuned":
        from .tuned import command

        print("\n--- runs our original rlm.py, one command per task ---")
        for task in config.data.task_dirs():
            print(" ".join(command(config, task)))
        return
    if args.command == "check":
        rlm = RLM(None, make_reader(config, load_model=False), make_limits(config), make_features(config))
        print("\n--- root prompt (with a placeholder question) ---")
        print(rlm.system_prompt("<question>", 500_000, "Final Answer:"))
        print("\n--- resolved settings ---")
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
