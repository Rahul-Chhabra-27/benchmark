# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The settings of an experiment, and how they are read from experiments.yaml.

experiments.yaml (next to this file) holds every experiment: a `defaults` section that
applies to all of them, and an `experiments` section where each named experiment lists
only what it changes. Every option the RLM has is a key here: `data`, `models`,
`ordering` (none | bm25-query | bm25-graph), `compression` (factor, budget_mb, choices,
offload), `cost`, `rlm` (limits). Unknown keys are refused, so a typo cannot silently
run the default.
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
    factor: int = 1  # 1 = plain reads, no kvpress; with `choices`, the default factor
    budget_mb: Optional[float] = None
    kv_bytes_per_token: int = KV_BYTES_PER_TOKEN
    # Factors the root may choose per read with llm_query(..., factor=F). Empty: only `factor`.
    choices: tuple[int, ...] = ()
    # Allow a read larger than B x F: the KV beyond B is counted as offloaded to CPU (and
    # costed) instead of the read being refused. Accounting only -- kvpress keeps the KV on
    # the GPU -- so a read is still capped, at max_read_tokens (default B x the largest factor:
    # as wide as the widest compressed read, so e.g. x4 + 2x offloaded reads as far as x8).
    offload: bool = False
    max_read_tokens: Optional[int] = None

    @property
    def factors(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.choices))) if self.choices else (self.factor,)

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
class CostConfig:
    """What a read costs, so a run reports accuracy AND cost (and the root can minimise it)."""

    gpu_token: float = 0.0  # one KV token kept in GPU memory
    offloaded_token: float = 100.0  # one KV token fetched back from CPU
    show_root: bool = False  # tell the root each read's cost and the running total


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
    cost: CostConfig = field(default_factory=CostConfig)
    rlm: RLMConfig = field(default_factory=RLMConfig)
    output_dir: str = "evaluation/results/rlm/minimal_rlm"
    organization_dir: Optional[str] = None  # bm25-graph cache; default <output_dir>/organization

    def read_limit(self, factor: int) -> int:
        """The most tokens one read at `factor` may carry: B x F, the offload cap, or rlm.read_tokens."""
        c = self.compression
        if c.budget_tokens is None:
            return self.rlm.read_tokens
        if c.offload:
            return c.max_read_tokens or c.budget_tokens * max(c.factors)
        return c.budget_tokens * factor

    @property
    def read_tokens(self) -> int:
        """The read limit at the default factor (what the root is told first)."""
        return self.read_limit(self.compression.factor)

    @property
    def largest_read_tokens(self) -> int:
        """The read limit at the largest allowed factor: what the sub model must hold."""
        return max(self.read_limit(f) for f in self.compression.factors)

    def validate(self) -> "ExperimentConfig":
        if not self.data.tasks:
            raise ValueError("data.tasks is empty")
        if self.data.ids_file is not None and self.data.limit is not None:
            raise ValueError("data.ids_file and data.limit both choose which questions run; set only one")
        if self.ordering not in ORDERINGS:
            raise ValueError(f"ordering must be one of {ORDERINGS}, got {self.ordering!r}")
        c = self.compression
        bad = [f for f in (c.factor, *c.factors) if f not in FACTORS]
        if bad:
            raise ValueError(f"compression factors must be in {FACTORS}, got {bad}")
        if c.factor not in c.factors:
            raise ValueError(f"compression.factor {c.factor} must be one of compression.choices {list(c.factors)}")
        if max(c.factors) > 1 and c.budget_mb is None:
            raise ValueError("a compression factor > 1 needs compression.budget_mb (B), so a read is B x F tokens")
        if c.offload and c.budget_mb is None:
            raise ValueError("compression.offload needs compression.budget_mb: offloaded is KV beyond B")
        if self.cost.show_root and c.budget_mb is None:
            raise ValueError("cost.show_root needs compression.budget_mb: costs are about KV kept and offloaded")
        if c.max_read_tokens is not None and c.max_read_tokens < 1:
            raise ValueError("compression.max_read_tokens must be positive")
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


def _tuple_choices(section: Any) -> Any:
    if isinstance(section, dict) and section.get("choices") is not None:
        return {**section, "choices": tuple(int(f) for f in section["choices"])}
    return section


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
        "compression": _build(CompressionConfig, _tuple_choices(raw.pop("compression", None)), "compression"),
        "cost": _build(CostConfig, raw.pop("cost", None), "cost"),
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


EXPERIMENTS_FILE = Path(__file__).with_name("experiments.yaml")


def read_experiments(path: str | Path = EXPERIMENTS_FILE) -> tuple[dict, dict]:
    """(defaults, experiments) from an experiments file."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = sorted(set(raw) - {"defaults", "experiments"})
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s) {unknown}; use `defaults` and `experiments`")
    return raw.get("defaults") or {}, raw.get("experiments") or {}


def load_experiment(
    experiment: str, path: str | Path = EXPERIMENTS_FILE, overrides: Optional[dict] = None
) -> ExperimentConfig:
    """One named experiment: its keys over `defaults`, then `overrides` ({"key" or "section.key": value})."""
    defaults, experiments = read_experiments(path)
    if experiment not in experiments:
        raise ValueError(f"no experiment {experiment!r} in {path}; choose from {sorted(experiments)}")
    raw = _merge(defaults, {"name": experiment, **(experiments[experiment] or {})})
    for key, value in (overrides or {}).items():
        section, _, leaf = key.partition(".")
        if leaf:
            raw.setdefault(section, {})[leaf] = value
        else:
            raw[key] = value
    return config_from_dict(raw)


def config_to_dict(config: ExperimentConfig) -> dict:
    out = asdict(config)
    out["data"]["tasks"] = list(config.data.tasks)
    out["compression"]["choices"] = list(config.compression.choices)  # YAML cannot hold a tuple
    out["read_tokens"] = config.read_tokens
    out["budget_tokens"] = config.compression.budget_tokens
    return out
