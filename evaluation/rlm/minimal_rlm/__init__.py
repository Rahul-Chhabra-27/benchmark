# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A minimal, config-driven RLM. Three files and one experiments file:

    config.py         the settings, read from experiments.yaml
    rlm.py            the RLM itself: prompts, ordering, readers, REPL, loop, per-call records
    run.py            settings -> every question -> result files
    experiments.yaml  every experiment: shared defaults, then what each one changes

    python -m evaluation.rlm.minimal_rlm list
    python -m evaluation.rlm.minimal_rlm run order_graph
"""

from .config import ExperimentConfig, config_from_dict, load_experiment
from .rlm import RLM, Limits, Result

__all__ = ["ExperimentConfig", "Limits", "RLM", "Result", "config_from_dict", "load_experiment"]
