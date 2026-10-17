# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`loop: tuned`: run our original, fully tuned RLM (harness/rlm.py) exactly as it is.

The base RLM (rlm.py in this folder) is the RLM idea with a short prompt. Our tuned RLM
-- the long prompt, worked examples, search coaching, FINAL_NONE, the grounding check,
the repetition breaker, labelled chunk reads, the KVzip path -- is not re-implemented
here. An experiment with `loop: tuned` is handed to the runner that produced every
earlier number, harness/run_benchmark.py --mode rlm, with its settings translated to
that runner's flags (the same flags scripts/rlm/rlm_order_lane.sh passes). Same code,
same arguments: the same behaviour, and results comparable with the earlier campaigns.

What carries over from experiments.yaml:

    data        dataset, tasks, split, limit or ids_file
    models      root, sub, base_url; thinking: false -> --no-think (unset: the server's default)
    ordering    bm25-query | bm25-graph -> --chunk-order (+ the bm25-graph cache)
    compression budget_mb + factor -> KVzip sub-calls: --sub-backend kvzip, --press,
                --memory-budget, --compression-factor; with ordering a fixed B x F cell,
                without it rlm.py's own slice planner (--max-subcall-chars auto)
    rlm         max_steps, max_sub_calls, exec_timeout_s, run_timeout_s

Everything else is rlm.py's own default. Factor choice, offloading and cost do not exist
in rlm.py, so a tuned experiment that asks for them is refused (config.validate).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import yaml

from .config import ExperimentConfig

RUNNER = "evaluation.rlm.harness.run_benchmark"


def results_dir(config: ExperimentConfig) -> Path:
    return Path(config.output_dir) / config.name


# run_benchmark's own defaults: a setting equal to one of these is left off the command,
# as the original launchers did, so a tuned run's command reads like theirs.
RUNNER_DEFAULTS = {"max_sub_calls": 40, "exec_timeout_s": 60.0, "run_timeout_s": 900.0}


def min_free_gib(budget_mb: float, factor: int) -> int:
    """GPU memory a B x F read needs free, the formula rlm_order_lane.sh uses for --sub-min-free-gib.

    kvpress keeps the full uncompressed KV of a read (N = B x F tokens at 147,456 bytes each,
    x2 headroom) on top of ~9 GiB of model weights and activations.
    """
    tokens = int(int(budget_mb * 1e6 / 147456) * factor)
    return math.ceil(9 + tokens * 147456 * 2 / 2**30)


def command(config: ExperimentConfig, task: str, python: str = sys.executable) -> list[str]:
    """The run_benchmark command for one task, flag for flag as rlm_order_lane.sh builds it."""
    data, models, rlm, c = config.data, config.models, config.rlm, config.compression
    cmd = [python, "-m", RUNNER, "--dataset", data.dataset, "--data-dir", task]
    if data.split:
        cmd += ["--split", data.split]
    cmd += ["--ids-file", data.ids_file] if data.ids_file else (["--limit", str(data.limit)] if data.limit else [])
    cmd += ["--mode", "rlm"]
    if config.ordering != "none":
        organization = config.organization_dir or str(Path(config.output_dir) / "organization")
        cmd += ["--chunk-order", config.ordering, "--organization-dir", organization]
    cmd += ["--max-steps", str(rlm.max_steps)]
    cmd += ["--root-model", models.root, "--sub-model", models.sub_model, "--base-url", models.base_url]
    if c.in_process:
        cmd += [
            "--sub-backend", "kvzip",
            "--press", "kvzip" if c.factor > 1 else "no_press",
            "--memory-budget", f"{c.budget_mb:g}", "--memory-budget-unit", "MB",
            "--compression-factor", str(c.factor),
            "--sub-min-free-gib", str(min_free_gib(c.budget_mb, c.factor)),
        ]  # fmt: skip
        if config.ordering == "none":
            # Without ordered chunks rlm.py sizes each slice from B and F itself (its
            # --max-subcall-chars auto planner); with them it reads fixed B x F cells.
            cmd += ["--max-subcall-chars", "auto"]
    # Only what differs from run_benchmark's defaults (the launchers passed none of these).
    for key, flag in (("max_sub_calls", "--max-sub-calls"), ("exec_timeout_s", "--exec-timeout"),
                      ("run_timeout_s", "--run-timeout")):  # fmt: skip
        if getattr(rlm, key) != RUNNER_DEFAULTS[key]:
            cmd += [flag, f"{getattr(rlm, key):g}"]
    if models.thinking is False:  # unset = the server's default, as the launchers left it
        cmd.append("--no-think")
    return cmd + ["--out", str(results_dir(config))]


def task_metrics(config: ExperimentConfig, task: str) -> dict:
    """The metrics.json run_benchmark wrote for this task (its run directory names itself)."""
    for config_path in sorted(results_dir(config).rglob("config.yaml")):
        written = yaml.safe_load(config_path.read_text()) or {}
        metrics = config_path.parent / "metrics.json"
        if written.get("data_dir") == task and written.get("mode") == "rlm" and metrics.exists():
            return json.loads(metrics.read_text())
    return {}


def run_tuned(
    config: ExperimentConfig,
    log: Callable[[str], None] = print,
    execute: Optional[Callable[[list[str]], int]] = None,
) -> dict[str, dict]:
    """Run every task through our original RLM; `execute` is injectable for tests."""
    execute = execute or (lambda cmd: subprocess.call(cmd))
    results = {}
    for task in config.data.task_dirs():
        cmd = command(config, task)
        log(f"=== {config.name}: {task} (loop=tuned: harness/rlm.py via run_benchmark)\n    {' '.join(cmd)}")
        code = execute(cmd)
        if code != 0:
            raise SystemExit(f"run_benchmark exited with {code} on {task}")
        results[task] = task_metrics(config, task)
    return results
