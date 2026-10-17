# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A clean base RLM with features on top, each switched on in experiments.yaml.

    rlm.py            the base RLM: prompt, REPL, loop, plain reads -- imports no feature
    ordering.py       feature: the `chunks` list (bm25-query / bm25-graph)
    tuning.py         feature: our prompt tuning and guards, from evaluation/rlm/rlm.py
    compression.py    feature: KV budget, factor per read, offloading, cost (a reader)
    config.py         the settings, read from experiments.yaml
    run.py            settings -> every question -> result files

    python -m evaluation.rlm.configurable_rlm list
    python -m evaluation.rlm.configurable_rlm run order_graph

README.md in this folder has the commands and experiments.
"""

from .config import ExperimentConfig, config_from_dict, load_experiment
from .rlm import RLM, Limits, Result

__all__ = ["ExperimentConfig", "Limits", "RLM", "Result", "config_from_dict", "load_experiment"]
