# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The compression add-on: reads that are KV-compressed under a memory budget.

Plugged into the base RLM (rlm.py) as its Reader, only when an experiment sets
compression.budget_mb. The base RLM never imports this file; everything about
compression lives here:

    budget      B, the KV one read may keep on the GPU; a read at factor F carries up to B x F tokens
    factors     F = 1 is no press (the control); F > 1 presses each read by 1 - 1/F (KVzip).
                With `choices`, the root picks F per read: llm_query(question, text, factor=4)
    offloading  with `offload`, a read may exceed B x F: KV beyond B is offloaded to CPU.
                offload_mode: count only counts it (kvpress keeps the KV on the GPU);
                offload_mode: real runs such a read with an offloaded KV cache (each layer
                on the CPU until it computes) and records its real peak GPU memory
    cost        GPU tokens x gpu_token + offloaded tokens x offloaded_token, per read; with
                cost.show_root the root sees each read's cost and the running total

One in-process kvpress model serves every factor (its press is set before each read), so
choosing a factor needs no extra servers and one copy of the weights.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .config import ExperimentConfig
from .rlm import SUB_PROMPT, Read, read_limit_help, summarize_sizes

FACTOR_NOTE = (
    "\nMemory: one read keeps at most {budget:,} tokens of KV memory (B). llm_query(question, text, factor=F)"
    "\nwith F in {factors} reads up to B x F tokens, compressed F times: F=1 is exact, a higher F reads"
    "\nwider but keeps less detail. Without factor= a read uses F={default}.\n"
)
OFFLOAD_NOTE = (
    "\nA read may be larger than B x F (up to {cap:,} tokens): the memory beyond B is offloaded to CPU"
    "\nand fetched back when the read is answered.\n"
)
COST_NOTE = (
    "\nCost: every read reports what it cost: {gpu:g} per memory token kept on the GPU, {offloaded:g} per"
    "\ntoken offloaded to CPU. Answer correctly at the lowest total cost.\n"
)
COST_LINE = (
    "[cost: read {read:,} tokens at x{factor} -> {gpu:,} on GPU, {offloaded:,} offloaded; {cost:g} (total {total:g})]"
)


@dataclass(frozen=True)
class Placement:
    """Where one read's KV lives: kept = read / F; up to B on the GPU, the rest offloaded."""

    kept: int
    gpu: int
    offloaded: int

    @staticmethod
    def of(read_tokens: int, factor: int, budget_tokens: int) -> "Placement":
        kept = math.ceil(read_tokens / factor)
        return Placement(kept, min(kept, budget_tokens), max(0, kept - budget_tokens))

    def cost(self, gpu_token: float, offloaded_token: float) -> float:
        return self.gpu * gpu_token + self.offloaded * offloaded_token


class CompressedReader:
    """A Reader whose reads go through one kvpress model, compressed by the chosen factor."""

    def __init__(
        self,
        sub: Any,
        *,
        budget_tokens: int,
        factors: tuple[int, ...] = (1,),
        default_factor: int = 1,
        read_limits: Optional[Mapping[int, int]] = None,
        offload: bool = False,
        real_offload: bool = False,
        gpu_token_cost: float = 0.0,
        offloaded_token_cost: float = 100.0,
        show_cost: bool = False,
    ):
        self.sub = sub  # a KVzipSubClient (or None, to describe the reader without a model)
        self.press = getattr(sub, "press", None)  # the KVzipPress, or None when nothing compresses
        self.budget_tokens = budget_tokens
        self.factors = tuple(factors)
        self.default_factor = default_factor
        self.read_limits = dict(read_limits or {f: budget_tokens * f for f in self.factors})
        self.offload = offload
        self.real_offload = real_offload
        self.gpu_token_cost = gpu_token_cost
        self.offloaded_token_cost = offloaded_token_cost
        self.show_cost = show_cost
        self.total_cost = 0.0
        self.read_help = read_limit_help(self.read_limits[default_factor])
        notes = ""
        if len(self.factors) > 1:
            notes += FACTOR_NOTE.format(budget=budget_tokens, factors=list(self.factors), default=default_factor)
        if offload:
            notes += OFFLOAD_NOTE.format(cap=self.read_limits[max(self.factors)])
        if show_cost:
            notes += COST_NOTE.format(gpu=gpu_token_cost, offloaded=offloaded_token_cost)
        self.notes = notes

    @classmethod
    def from_config(cls, config: ExperimentConfig, sub: Any = None) -> "CompressedReader":
        c, cost = config.compression, config.cost
        return cls(
            sub,
            budget_tokens=c.budget_tokens,
            factors=c.factors,
            default_factor=c.factor,
            read_limits={f: config.read_limit(f) for f in c.factors},
            offload=c.offload,
            real_offload=c.offload_mode == "real",
            gpu_token_cost=cost.gpu_token,
            offloaded_token_cost=cost.offloaded_token,
            show_cost=cost.show_root,
        )

    def begin(self) -> None:
        self.total_cost = 0.0

    def _use(self, factor: int) -> None:
        """Press the next read by exactly 1 - 1/F (no press at all for F = 1)."""
        if factor == 1:
            self.sub.press = None
        else:
            if self.press is None:
                raise ValueError(f"factor {factor} needs a press; the reader was built without one")
            self.press.compression_ratio = 1 - 1 / factor
            self.sub.press = self.press
            self.sub.press_min_tokens = 1  # every read at F > 1 is pressed, so F is exact
        self.sub.compression_factor = float(factor)

    def read(self, question: str, text: str, factor: Any = None, **options: Any) -> Read:
        if options:
            return Read("", refused=f"[REFUSED] llm_query takes no {sorted(options)}; only factor=F.")
        factor = self.default_factor if factor is None else factor
        if factor not in self.factors:
            return Read("", refused=f"[REFUSED] factor={factor!r} is not available; use one of {list(self.factors)}.")
        tokens = int(self.sub.token_len(text))
        limit = self.read_limits[factor]
        details: dict = {
            "read_tokens": tokens,
            "limit_tokens": limit,
            "factor": factor,
            "budget_tokens": self.budget_tokens,
        }
        if tokens > limit:
            refusal = (
                f"[REFUSED] that text is {tokens:,} tokens; one llm_query at factor {factor} reads at most "
                f"{limit:,}. Split it into smaller pieces."
            )
            return Read("", details, refused=refusal)
        placement = Placement.of(tokens, factor, self.budget_tokens)
        cost = placement.cost(self.gpu_token_cost, self.offloaded_token_cost)
        self.total_cost += cost
        details.update(
            kept_tokens=placement.kept,
            gpu_tokens=placement.gpu,
            offloaded_tokens=placement.offloaded,
            cost=cost,
            # Would this read NOT have fit in B uncompressed? False means compression bought nothing.
            compression_needed=(tokens > self.budget_tokens) if factor > 1 else None,
        )
        self._use(factor)
        offloaded = self.real_offload and placement.offloaded > 0
        started = time.monotonic()
        answer = self.sub.chat_split(
            question=SUB_PROMPT.format(question=question), context=text, **({"offload": True} if offloaded else {})
        )
        details["offload_mode"] = ("real" if offloaded else "count") if placement.offloaded else None
        peak = getattr(self.sub, "last_gpu_peak_bytes", None)
        if peak is not None:
            details["gpu_peak_mb"] = round(peak / 1e6, 1)
        note = ""
        if self.show_cost:
            note = COST_LINE.format(
                read=tokens,
                factor=factor,
                gpu=placement.gpu,
                offloaded=placement.offloaded,
                cost=cost,
                total=self.total_cost,
            )
        return Read(str(answer or ""), details, note=note, seconds=time.monotonic() - started)

    def summarize(self, details: list[dict]) -> dict:
        """One question's reads: sizes, factors used, KV offloaded, cost, unneeded compression."""
        return {
            **summarize_sizes(details),
            "kept_tokens_max": max((d.get("kept_tokens", 0) for d in details), default=0),
            "offloaded_tokens": sum(d.get("offloaded_tokens", 0) for d in details),
            "offloaded_calls": sum(1 for d in details if d.get("offloaded_tokens")),
            "cost": sum(d.get("cost", 0.0) for d in details),
            "factors": dict(sorted(Counter(f"x{d['factor']}" for d in details if "factor" in d).items())),
            "compressed_unneeded": sum(1 for d in details if d.get("compression_needed") is False),
        }


def load_kv_model(config: ExperimentConfig) -> Any:
    """The in-process kvpress model, built for the largest factor and the widest read."""
    from evaluation.rlm.shared.kvzip_backend import KVzipSubClient  # torch and kvpress, only when needed

    largest = max(config.compression.factors)
    return KVzipSubClient(
        model=config.models.sub_model,
        press_name="kvzip" if largest > 1 else "no_press",
        memory_budget=config.compression.budget_mb,
        memory_budget_unit="MB",
        max_context_tokens=config.largest_read_tokens,
        compression_factor=largest,
    )
