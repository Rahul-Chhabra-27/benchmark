# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A clean, config-driven RLM. See README.md in this folder.

    config.py    the experiment file: data, models, ordering, compression, limits
    prompts.py   every word the models see: a short base prompt + one block per option
    ordering.py  none | bm25-query | bm25-graph -> the `chunks` list
    readers.py   how llm_query reads: plain http, or kvpress at factor F
    repl.py      the persistent Python REPL the root writes into
    loop.py      the RLM loop itself
    stats.py     one record per sub-call
    run.py       config -> per-question loop -> result files

    python -m evaluation.rlm.core run evaluation/rlm/configs/graph.yaml
"""

from .config import ExperimentConfig, config_from_dict, load_config
from .loop import RLM, Limits, Result

__all__ = ["ExperimentConfig", "Limits", "RLM", "Result", "config_from_dict", "load_config"]
