# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find finished runs and say what each one is, from its config.yaml alone.

An arm is one configuration (possibly spread over one directory per task):

    kind        what it is                                   example label
    vanilla     the whole document in the model's window     vanilla
    rlm         RLM, no ordering, no compression             rlm
    ordered     RLM over ordered chunks, no compression      rlm+bm25-graph
    compressed  RLM whose reads are KV-compressed (F > 1)    rlm+bm25-graph 1024MB x2
    other       anything else (structured, skylight, ...)    structured

Both the older harness (run_benchmark.py) and minimal_rlm write a config.yaml
next to their checkpoint.jsonl; both are understood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import yaml

KINDS = ("vanilla", "rlm", "ordered", "compressed", "other")


@dataclass
class Arm:
    label: str
    kind: str
    dataset: str
    run_dirs: list[Path] = field(default_factory=list)


def describe(config: dict) -> Optional[tuple[str, str, str]]:
    """(label, kind, dataset) for a run's config, or None if it is not an RLM-family run."""
    if "ordering" in config and isinstance(config.get("compression"), dict):  # minimal_rlm
        factor = int(config["compression"].get("factor") or 1)
        ordering = config.get("ordering") or "none"
        kind = "compressed" if factor > 1 else "ordered" if ordering != "none" else "rlm"
        return f"minimal_rlm:{config['name']}", kind, config["data"]["dataset"]
    if config.get("backend") != "rlm":
        return None
    mode, dataset = config.get("mode"), config.get("dataset")
    if mode == "vanilla":
        return "vanilla", "vanilla", dataset
    order = config.get("chunk_order")
    factor = config.get("compression_factor")
    budget = config.get("sub_kv_memory_budget")
    label = str(mode) + (f"+{order}" if order else "")
    if config.get("sub_backend") == "kvzip" and factor is not None and budget is not None:
        label += f" {float(budget):g}{config.get('sub_kv_memory_budget_unit') or 'MB'} x{float(factor):g}"
    if mode not in ("rlm", "clean-rlm"):
        return label, "other", dataset
    kind = "compressed" if (factor or 1) > 1 else "ordered" if order else "rlm"
    return label, kind, dataset


def discover(roots: Iterable[Path]) -> dict[str, Arm]:
    """Every arm under `roots`, keyed by label. One label may span several task directories."""
    arms: dict[str, Arm] = {}
    for root in roots:
        for config_path in sorted(Path(root).rglob("config.yaml")):
            if not (config_path.parent / "checkpoint.jsonl").exists():
                continue
            described = describe(yaml.safe_load(config_path.read_text()) or {})
            if described is None:
                continue
            label, kind, dataset = described
            arm = arms.setdefault(label, Arm(label, kind, dataset))
            arm.run_dirs.append(config_path.parent)
    return arms
