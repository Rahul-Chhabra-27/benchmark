# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One experiment = one YAML file. Every option the RLM has is a key here.

    name: hotpotqa_graph_x2
    data:
      dataset: loft
      tasks: [hotpotqa]          # LOFT: _128k is added
      split: test
      limit: 55
    models:
      root: Qwen/Qwen3-4B-Instruct-2507
      base_url: http://localhost:8002/v1
    ordering: bm25-graph         # none | bm25-query | bm25-graph
    compression:
      factor: 2                  # 1 = no kvpress | 2 | 4 | 8
      budget_mb: 1024            # B: KV one read may keep; reads carry B x F tokens
    rlm:
      max_steps: 30

Unknown keys are refused, so a typo cannot silently run the default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

ORDERINGS = ("none", "bm25-query", "bm25-graph")
FACTORS = (1, 2, 4, 8)
# Qwen3-4B-Instruct-2507: 36 layers x 8 KV heads x 128 dims x 2 (K, V) x 2 bytes.
KV_BYTES_PER_TOKEN = 147_456
BYTES_PER_MB = 1000**2  # decimal, as kvpress.pipeline.MEMORY_UNIT_TO_BYTES


@dataclass(frozen=True)
class DataConfig:
    tasks: tuple[str, ...]
    dataset: str = "loft"
    split: Optional[str] = "test"
    limit: Optional[int] = None
    loft_length: str = "128k"
    # Only these question ids (one per line), e.g. a hard_questions validation set.
    # Ids of other tasks in the same file are ignored, so one file can serve every task.
    ids_file: Optional[str] = None

    def ids(self) -> Optional[set[str]]:
        if self.ids_file is None:
            return None
        lines = Path(self.ids_file).read_text().splitlines()
        return {line.strip() for line in lines if line.strip() and not line.startswith("#")}

    def task_dirs(self) -> list[str]:
        """LOFT subsets may be named bare (`hotpotqa`); the loader wants `hotpotqa_128k`."""
        if self.dataset != "loft":
            return list(self.tasks)
        return [t if "_" in t else f"{t}_{self.loft_length}" for t in self.tasks]


@dataclass(frozen=True)
class ModelConfig:
    root: str = "Qwen/Qwen3-4B-Instruct-2507"
    sub: Optional[str] = None  # defaults to the root model
    base_url: str = "http://localhost:8002/v1"
    thinking: bool = False  # Qwen3 thinking mode for root and http sub-calls

    @property
    def sub_model(self) -> str:
        return self.sub or self.root


@dataclass(frozen=True)
class CompressionConfig:
    factor: int = 1  # 1 = plain reads, no kvpress
    budget_mb: Optional[float] = None
    kv_bytes_per_token: int = KV_BYTES_PER_TOKEN

    @property
    def budget_tokens(self) -> Optional[int]:
        if self.budget_mb is None:
            return None
        return int(self.budget_mb * BYTES_PER_MB // self.kv_bytes_per_token)

    @property
    def in_process(self) -> bool:
        """Reads go through the in-process kvpress model (else through the http server).

        Any budget means B x F semantics, and F = 1 then runs the same model with no
        press -- the no_press control every compressed cell is compared against.
        """
        return self.budget_mb is not None


@dataclass(frozen=True)
class RLMConfig:
    max_steps: int = 30
    max_sub_calls: int = 40
    exec_timeout_s: float = 60.0  # one code cell, excluding sub-call time
    run_timeout_s: float = 900.0  # one question, wall clock
    output_chars: int = 4000  # REPL output shown to the root per turn
    read_tokens: int = 8192  # max tokens one llm_query reads, when no budget is set


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: DataConfig
    models: ModelConfig = field(default_factory=ModelConfig)
    ordering: str = "none"
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    rlm: RLMConfig = field(default_factory=RLMConfig)
    output_dir: str = "evaluation/results/rlm/minimal_rlm"
    organization_dir: Optional[str] = None  # bm25-graph cache; default <output_dir>/organization

    @property
    def read_tokens(self) -> int:
        """The most tokens one llm_query may read: N = B x F, or rlm.read_tokens."""
        budget = self.compression.budget_tokens
        return budget * self.compression.factor if budget is not None else self.rlm.read_tokens

    def validate(self) -> "ExperimentConfig":
        if not self.data.tasks:
            raise ValueError("data.tasks is empty")
        if self.data.ids_file is not None and self.data.limit is not None:
            raise ValueError("data.ids_file and data.limit both choose which questions run; set only one")
        if self.ordering not in ORDERINGS:
            raise ValueError(f"ordering must be one of {ORDERINGS}, got {self.ordering!r}")
        if self.compression.factor not in FACTORS:
            raise ValueError(f"compression.factor must be one of {FACTORS}, got {self.compression.factor}")
        if self.compression.factor > 1 and self.compression.budget_mb is None:
            raise ValueError("compression.factor > 1 needs compression.budget_mb (B), so a read is B x F tokens")
        if self.compression.budget_mb is not None and self.compression.budget_tokens < 1:
            raise ValueError(f"compression.budget_mb={self.compression.budget_mb} holds no KV token")
        for name in ("max_steps", "max_sub_calls", "output_chars", "read_tokens"):
            if getattr(self.rlm, name) < 1:
                raise ValueError(f"rlm.{name} must be positive")
        return self


def _build(cls, raw: Any, where: str):
    """A dataclass from a mapping, refusing keys the dataclass does not have."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping, got {type(raw).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {unknown}; allowed: {sorted(known)}")
    return cls(**raw)


def config_from_dict(raw: dict) -> ExperimentConfig:
    raw = dict(raw or {})
    if "name" not in raw or "data" not in raw:
        raise ValueError("a config needs at least `name` and `data`")
    data = dict(raw.pop("data") or {})
    tasks = data.get("tasks")
    data["tasks"] = tuple([tasks] if isinstance(tasks, str) else tasks or ())
    sections = {
        "data": _build(DataConfig, data, "data"),
        "models": _build(ModelConfig, raw.pop("models", None), "models"),
        "compression": _build(CompressionConfig, raw.pop("compression", None), "compression"),
        "rlm": _build(RLMConfig, raw.pop("rlm", None), "rlm"),
    }
    if raw.get("ordering") is None:  # YAML reads a bare `none` as the string, but `null` as None
        raw["ordering"] = "none"
    return _build(ExperimentConfig, {**raw, **sections}, "the config").validate()


def _merge(base: dict, top: dict) -> dict:
    """`top` over `base`, section by section."""
    out = dict(base)
    for key, value in top.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def read_yaml(path: str | Path, _seen: tuple = ()) -> dict:
    """A config file, with `extends: other.yaml` (relative to it) merged underneath."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"config extends itself in a loop: {' -> '.join(str(p) for p in (*_seen, path))}")
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _merge(read_yaml(path.parent / parent, (*_seen, path)), raw)


def load_config(path: str | Path, **overrides: Any) -> ExperimentConfig:
    """Read a YAML experiment. `overrides` replace top-level or `section.key` values."""
    raw = read_yaml(path)
    for key, value in overrides.items():
        section, _, leaf = key.partition(".")
        if leaf:
            raw.setdefault(section, {})[leaf] = value
        else:
            raw[key] = value
    return config_from_dict(raw)


def config_to_dict(config: ExperimentConfig) -> dict:
    out = asdict(config)
    out["data"]["tasks"] = list(config.data.tasks)
    out["read_tokens"] = config.read_tokens
    out["budget_tokens"] = config.compression.budget_tokens
    return out
