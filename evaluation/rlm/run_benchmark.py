# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run vanilla vs RLM on a long-context task. Resumable JSONL checkpointing.

Usage:
  python -m evaluation.rlm.run_benchmark --dataset niah --mode rlm --limit 1 --debug
  python -m evaluation.rlm.run_benchmark --dataset ruler32k --data-dir niah_single_1 --limit 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import yaml

from evaluation import compare
from evaluation.benchmarks.registry import LOFT_TASKS, LONGBENCH_128K_SUBSETS, RULER_32K_TASKS, SCORER_REGISTRY
from evaluation.benchmarks.results import score_prediction_frame
from evaluation.results_layout import RLM_RESULTS_DIR

from . import chunkset, retrieval, window_notes
from .chunkset import ChunkSet, ChunkSetError, file_digest
from .client import LLMClient
from .datasets import available_datasets, canonical_dataset_name, load_examples
from .organization import CONFIG as ORGANIZATION_CONFIG
from .organization import ORDERS, ModelTokens, ordered_hits, organize, pack
from .retrieval import Hit
from .rlm import (
    CHUNK_ORDER_REVISION,
    CHUNK_ORDERS,
    CHUNK_READS_BEFORE_ABSTAIN,
    DEFAULT_CHUNK_CALL_TOKENS,
    DEFAULT_REDUCE_FINDING_CHARS,
    RLM,
    SEARCH_ABLATABLE,
    STRUCTURED_ANSWER_SHAPES,
    STRUCTURED_MAP_GRANULARITIES,
    STRUCTURED_MAP_QUESTIONS,
    SUB_SYSTEM_PROMPT,
    MemoryBudget,
    Scratchpad,
    bare_query,
    vanilla_answer,
)
from .sizing import (
    DEFAULT_MIN_TOKENS,
    DEFAULT_RESERVE_TOKENS,
    FIXED_CHUNK_CHAR_OVERSHOOT,
    compression_ratio_from_factor,
    fixed_chunk_char_cap,
)

# The historical hand-picked chunk size. Named because it is compared against in
# build_run_dir_components as well as being the argparse default -- as two separate
# literals, changing one would silently shift every run directory name.
DEFAULT_MAX_SUBCALL_CHARS = 32000

# Same reason as above: these are both argparse defaults and compared against in
# build_run_dir_components, so they must be one literal each.
DEFAULT_SEARCH_WINDOW = retrieval.DEFAULT_CHUNK_CHARS
DEFAULT_SEARCH_OVERLAP = retrieval.DEFAULT_OVERLAP

# Chunks per example for the structured arm. Matches retrieval's DEFAULT_K so a
# `bm25` structured run and a --search-k 5 agentic run read the same windows, and
# the only difference between them is the pipeline.
DEFAULT_CHUNKS_K = retrieval.DEFAULT_K

# DEFAULT_REDUCE_FINDING_CHARS is imported from .rlm rather than restated here for
# the same reason: it is the argparse default AND is compared against in
# build_run_dir_components, and a second literal could drift from the cap the reduce
# step actually applies.


# The bracketed shape LOFT's multi-value scorer parses, as a worked example. Only
# ever shown for a task that IS scored that way -- see answer_list_example.
LOFT_LIST_ANSWER_EXAMPLE = '["first answer", "second answer", "third answer"]'


def answer_list_example(dataset: Optional[str], task: Optional[str], mode: str, ordered_rlm: bool = False) -> str:
    """The list example this run's reduce step should show, or "" for none.

    Narrow on purpose. LOFT ships one bare `Final Answer: ` prefix for every subset,
    but only qampari and quest are scored through the list parser, and an arm that is
    never shown the shape it is parsed with is measured as wrong when it was right --
    3 of 3 correct answers reporting coverage 0.000, measured. Single-value subsets
    are untouched, so every nq/hotpotqa run keeps its exact prompt and its numbers.

    The ordered-chunk agentic arms (`ordered_rlm`) are shown it too, in their FINAL
    note: they are paired against structured arms that were. Every other agentic run
    keeps the prompt it was scored with.
    """
    if dataset != "loft" or not task or not (mode == "structured" or (mode == "rlm" and ordered_rlm)):
        return ""
    return LOFT_LIST_ANSWER_EXAMPLE if compare.loft_score_key(task) == "coverage" else ""


def _subcall_chars(value: str) -> Any:
    """Parse --max-subcall-chars: a positive int, or the string 'auto'."""
    if value.strip().lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer or 'auto', got {value!r}")
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be positive, got {parsed}")
    return parsed


# Knobs that change what the numbers MEAN, but that the run-dir name deliberately
# does not carry: a slug with all of them in it would be unreadable, and adding
# them would rename every existing run directory. They are enforced on RESUME
# instead, against the config.yaml the first run wrote -- so two different
# experiments can still never merge into one checkpoint.jsonl, which is the
# property build_run_dir_components exists to protect.
RESUME_CRITICAL_KEYS = (
    "limit",
    "split",
    # Same reason as `limit` and `split`: they pick WHICH rows are evaluated, and
    # neither reaches the run-directory name. A `--sample-fraction 0.5` run and a
    # full run of the same cell otherwise land in one directory, share one
    # checkpoint.jsonl, and merge two different samples into one score.
    "sample_fraction",
    "sample_seed",
    "root_model",
    # Not in the slug: adding it there would rename every run directory that has
    # ever been written, orphaning their checkpoints silently. Erroring on resume
    # is both safer and louder.
    "sub_model",
    "max_steps",
    "exec_timeout",
    "run_timeout",
    "max_sub_calls",
    "min_sub_calls_before_abstain",
    "max_unproductive_steps",
    "max_error_steps",
    "vanilla_char_limit",
    "vanilla_max_prompt_tokens",
    "sub_max_context_tokens",
    "press_min_tokens",
    # In the slug too, but keep it here as well: the slug only distinguishes
    # on/off-by-value, and resuming a floor-2048 run into a floor-16384 checkpoint
    # would merge two different read regimes.
    "min_subcall_chars",
    # In the slug too (search_k is), but listed here for the same reason
    # min_subcall_chars is: the slug distinguishes depth, not window geometry, and
    # resuming a 2000-char-window run into a 4000-char-window checkpoint would merge
    # two different retrievers. --sample-fraction is the cautionary precedent: it
    # picked which rows were evaluated, reached neither the name nor this list, and
    # two different samples merged into one score.
    "search_k",
    "search_window_chars",
    "search_overlap_chars",
    "max_search_calls",
    "search_ablate",
    "sub_max_tokens",
    "data_dir",
    # The structured arm. `chunks_file_sha256` rather than the path: two runs
    # pointed at the same filename with different contents would otherwise resume
    # into one checkpoint and merge two different selections into one score. Same
    # for `ids_sha256` -- an id set picks WHICH rows are evaluated, exactly the
    # property --sample-fraction is listed above for.
    "chunks_source",
    "chunks_k",
    "chunks_file_sha256",
    # WHICH selector wrote that file. `chunks_file_sha256` already refuses a resume
    # across two different files, so this is belt and braces there -- it earns its
    # place by making config.yaml self-describing, so the sweep analysis can key on
    # the selector without reopening every chunk file to find out.
    "chunks_tag",
    "notes_file_sha256",
    "notes_select",
    "notes_k",
    "ids_sha256",
    # The three structured axes. In the slug too (only when non-default, so existing
    # directories keep their names), but listed here for the same reason `search_k`
    # is: the slug says which components were withheld, and a checkpoint written
    # before these flags existed carries no value at all for them.
    "structured_map_question",
    "structured_map_granularity",
    "structured_answer_shape",
    # The geometry axes. In the slug too (only when non-default), but listed here for
    # the same reason search_window_chars is: resuming a 400-char-exposure run into a
    # 4000-char one would merge two different read regimes under one score.
    "structured_map_chars",
    "reduce_finding_chars",
    "structured_hops",
    "answer_list_format",
    "chunks_query",
    # The ordered whole-document arms: resuming one group budget or organization
    # revision into another would merge two different reads into one score.
    "structured_order",
    "structured_group_input_tokens",
    "structured_context_tokens",
    "structured_reader_max_tokens",
    "structured_reduce_max_tokens",
    "organization_config",
    # The ordered-chunk agentic arms: the order, the code revision that renders it, and
    # the per-call read budget each change what the root is shown or may send.
    "chunk_order",
    "chunk_order_revision",
    "chunk_call_tokens",
    # Dicts: comparing them whole covers their nested knobs too
    # (keep_recent_turns inside memory_budget, max_notes_tokens inside
    # scratchpad), which have no top-level key of their own.
    "memory_budget",
    "scratchpad",
    "target_compression_ratio",
    "compression_factor",
    "fixed_chunk",
    "no_think",
)


def resolve_compression_target(args: argparse.Namespace) -> float | None:
    """Validate sizing flags and return the ratio used by the existing planner."""
    auto_chunk = args.max_subcall_chars == "auto"
    factor = getattr(args, "compression_factor", None)
    ratio = getattr(args, "target_compression_ratio", None)
    fixed_chunk = bool(getattr(args, "fixed_chunk", False))

    if factor is not None and ratio is not None:
        raise ValueError("--compression-factor and --target-compression-ratio are mutually exclusive")
    if factor is not None:
        ratio = compression_ratio_from_factor(factor)
    if fixed_chunk and (args.sub_backend != "kvzip" or not auto_chunk):
        raise ValueError("--fixed-chunk requires --sub-backend kvzip and --max-subcall-chars auto")
    if auto_chunk and args.sub_backend != "kvzip":
        raise ValueError("--max-subcall-chars auto requires --sub-backend kvzip (the http path has no KV budget)")
    if auto_chunk and ratio is None:
        raise ValueError("--max-subcall-chars auto requires --compression-factor or --target-compression-ratio")
    if ratio is not None:
        if not auto_chunk:
            raise ValueError("compression factor/target ratio only applies with --max-subcall-chars auto")
        if not 0.0 <= ratio < 1.0:
            raise ValueError("--target-compression-ratio must be in [0.0, 1.0)")
    if fixed_chunk:
        if getattr(args, "min_subcall_chars", 0):
            raise ValueError("--fixed-chunk derives its own floor; do not also pass --min-subcall-chars")
    return ratio


def resume_conflicts(run_dir: Path, config: dict) -> list[str]:
    """How this run's config differs from the one that wrote the checkpoint."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return []
    try:
        prior = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        # An unreadable config must not silently DISABLE the guard -- that is
        # the exact failure mode (merging two experiments) it exists to prevent.
        raise SystemExit(
            f"{config_path} is corrupt ({exc}); cannot verify this run matches the "
            "checkpoint beside it. Fix or remove the run directory."
        )
    conflicts = [
        f"{key}: checkpoint was written with {prior.get(key)!r}, this run has {config.get(key)!r}"
        for key in RESUME_CRITICAL_KEYS
        if key in prior and prior.get(key) != config.get(key)
    ]
    # Zero-overlap retrieval previously indexed only the first window. Missing
    # revisions must conflict here, unlike newly added optional settings above.
    if "search_index_revision" in config and prior.get("search_index_revision") != config["search_index_revision"]:
        conflicts.append(
            "search_index_revision: zero-overlap retrieval changed; use a fresh results directory "
            "instead of resuming a checkpoint from a different index revision"
        )
    # Same shape, same reason: revision 1 read one window per map call and merely
    # offered the answer format. A checkpoint written before either fix describes a
    # different pipeline, and its absence of the key is exactly the case to catch.
    if "structured_revision" in config and prior.get("structured_revision") != config["structured_revision"]:
        conflicts.append(
            "structured_revision: the map step now reads one CONTIGUOUS PASSAGE per call rather "
            "than one window, and the reduce step requires the answer shape rather than offering "
            "it; use a fresh results directory instead of resuming a revision-1 checkpoint"
        )
    return conflicts


def _prior_resolved_chars(run_dir: Path) -> int | None:
    """The chunk size a previous run of this directory settled on, if recorded."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(
            f"{config_path} is corrupt ({exc}); cannot verify this run's auto-sized "
            "chunk matches the checkpoint beside it. Fix or remove the run directory."
        )
    prior = config.get("max_subcall_chars_resolved")
    return int(prior) if prior else None


def normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def is_correct(pred: str | None, answers: list[str]) -> bool:
    if pred is None:
        return False
    p = normalize(pred)
    return any(normalize(a) in p for a in answers)


def load_done(path: Path) -> set:
    """IDs already completed. Records that died with a harness/API exception
    (e.g. a transient server outage) don't count -- re-running retries them."""
    if not path.exists():
        return set()
    done = set()
    for line in open(path):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A hard kill can tear the last append mid-line. The rest of the
            # file is intact; treat the torn record as not-done so it reruns.
            print(f"!! skipping a torn checkpoint line in {path} (interrupted write); its example will rerun")
            continue
        if not record.get("error"):
            done.add(record["id"])
    return done


def write_run_artifacts(
    checkpoint_path: Path,
    run_dir: Path,
    dataset_name: str,
    config: dict,
) -> dict:
    """Materialize the common predictions/config/metrics result contract.

    The headline score comes from the shared canonical scorer, so it is the same
    number a KVPress run of this dataset reports and the two are directly
    comparable. Everything RLM-specific lands under ``runtime``.
    """
    records = []
    for line in checkpoint_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Same torn-trailing-line tolerance as load_done: the interrupted
            # example was rerun and re-appended, so nothing is lost by skipping.
            continue
    frame = pd.DataFrame(records).rename(columns={"pred": "predicted_answer"})
    # An errored example is RETRIED on resume (load_done ignores it) and the retry
    # APPENDS a record -- the failed one stays in the checkpoint, because a JSONL
    # written for crash-safety is never rewritten in place. Without this de-dup the
    # stale failure is counted a second time: `errors` and `examples` both inflate
    # and predictions.csv carries the id twice. `errors` is the column a reader
    # checks before believing a score, so it has to mean what it says.
    if "id" in frame:
        frame = frame.drop_duplicates("id", keep="last").reset_index(drop=True)
    frame.to_csv(run_dir / "predictions.csv", index=False)

    # An example that died with an API/harness exception is a MISSING measurement,
    # not a wrong answer. score_prediction_frame fills a null prediction with "",
    # which every scorer then marks 0 -- so a server outage would show up as the RLM
    # arm being less accurate than KVPress, which never makes network calls and so
    # can never take this hit. Score only the examples that actually produced an
    # answer, and report the shortfall separately.
    errored = frame["error"].notna() if "error" in frame else pd.Series(False, index=frame.index)
    scored_frame = frame[~errored]

    metrics: dict
    if dataset_name in SCORER_REGISTRY:
        metrics = score_prediction_frame(dataset_name, scored_frame) if len(scored_frame) else {}
    else:
        count = len(scored_frame)
        metrics = {"accuracy": float(scored_frame["correct"].sum() / count) if count else 0.0}

    count = len(frame)
    scored = len(scored_frame)
    runtime = {
        "examples": count,
        "scored": scored,
        "errors": int(errored.sum()),
        # `_loose` in the name: this is a substring containment check, systematically
        # more generous than the canonical scorer that produced the headline metric
        # sitting beside it in this same file. It is a progress indicator for a
        # running sweep, never a result.
        "progress_match_loose": float(scored_frame["correct"].sum() / scored) if scored else 0.0,
        "average_tokens": float(scored_frame["tokens"].mean()) if scored else 0.0,
        "average_latency_s": float(scored_frame["latency_s"].mean()) if scored else 0.0,
        "unfinished": int((~scored_frame["finished"].astype(bool)).sum()) if scored else 0,
    }

    # Abstentions are a deliberate ending (FINAL_NONE), not a failure: the model
    # searched and reported finding nothing. They score 0 like any wrong answer, so
    # without this column an abstaining arm and a hallucinating one look identical.
    if scored and "end_reason" in scored_frame:
        runtime["abstained"] = int((scored_frame["end_reason"] == "abstained").sum())

    # Peak simultaneous root context is RLM's analogue of KVPress's
    # `average_retained_context_tokens` -- both say "how much context did the model
    # hold at once", which is the axis a memory sweep varies.
    if scored and "metrics" in scored_frame:
        peaks = [
            m.get("peak_context_tokens")
            for m in scored_frame["metrics"]
            if isinstance(m, dict) and m.get("peak_context_tokens")
        ]
        if peaks:
            runtime["average_peak_context_tokens"] = float(sum(peaks) / len(peaks))

    # The vanilla arm's ceiling: if the gold never survived truncation, its score is
    # a property of the char/token limit, not of the model. KVPress compresses
    # rather than truncates, so this column is what makes the gap interpretable.
    if scored and "truncated" in scored_frame:
        truncated = scored_frame["truncated"].fillna(False).astype(bool)
        runtime["truncated_fraction"] = float(truncated.sum() / scored)
        if "context_chars_used" in scored_frame and "context_chars" in scored_frame:
            retained = scored_frame["context_chars_used"] / scored_frame["context_chars"]
            runtime["average_context_chars_retained"] = float(retained.mean())

    # How big the root's slices actually were, on EVERY backend. This is the number
    # the chunk-size question turns on, and until it was recorded the only way to
    # answer it was to re-parse saved transcripts: sub_call_tokens folds in the
    # question and the decoded answer, so it can only bound the slice from above.
    # Reported next to subcall_chars_advertised -- the ratio between them is how far
    # the root is from using the budget it was offered.
    if scored and "metrics" in scored_frame:
        payloads = [
            (m.get("sub_payload_chars", 0), m.get("sub_calls", 0))
            for m in scored_frame["metrics"]
            if isinstance(m, dict) and m.get("sub_calls")
        ]
        total_calls = sum(calls for _, calls in payloads)
        if total_calls:
            runtime["average_sub_payload_chars"] = float(sum(chars for chars, _ in payloads) / total_calls)
            runtime["max_sub_payload_chars"] = max(
                (m.get("sub_payload_chars_max", 0) for m in scored_frame["metrics"] if isinstance(m, dict)),
                default=0,
            )
            expanded = sum(m.get("sub_slices_expanded", 0) for m in scored_frame["metrics"] if isinstance(m, dict))
            runtime["sub_slice_expanded_fraction"] = float(expanded / total_calls)

        rlm_metrics = [m for m in scored_frame["metrics"] if isinstance(m, dict)]
        runtime["sub_slice_unlocatable_calls"] = int(sum(m.get("sub_slice_unlocatable_calls", 0) for m in rlm_metrics))
        coverage = [
            m.get("document_coverage_fraction") for m in rlm_metrics if m.get("document_coverage_fraction") is not None
        ]
        if coverage:
            runtime["document_coverage_fraction"] = float(sum(coverage) / len(coverage))
        # The effort thesis had no run-level aggregate at all: the grid granted 50
        # steps and up to 64 sub-calls while the median example used 2 and 1.
        for column in ("steps", "sub_calls"):
            if column in scored_frame:
                runtime[f"average_{column}"] = float(scored_frame[column].mean())
        # Search arm only, so a non-search run's runtime block is unchanged.
        searches = [m.get("search_calls") for m in rlm_metrics if m.get("search_calls") is not None]
        if searches:
            runtime["search_calls"] = int(sum(searches))
            runtime["average_search_calls"] = float(sum(searches) / len(searches))
            runtime["search_hits_returned"] = int(sum(m.get("search_hits_returned", 0) for m in rlm_metrics))
            # The number that says whether the tool was actually USED, as opposed to
            # merely offered: hits returned but never handed to llm_query.
            runtime["search_hits_read"] = int(sum(m.get("search_hits_read", 0) for m in rlm_metrics))
            runtime["search_k_clamped_calls"] = int(sum(m.get("search_k_clamped_calls", 0) for m in rlm_metrics))
        if "gold_in_retrieved" in scored_frame:
            # Loose answer-string presence diagnostic, not a score ceiling:
            # the root can read elsewhere and a match need not supply evidence.
            gold = scored_frame["gold_in_retrieved"].dropna()
            if len(gold):
                runtime["gold_in_retrieved_fraction"] = float(gold.astype(bool).sum() / len(gold))
        # The same diagnostic split per window, so a k-sweep can say whether a cell
        # gained score by adding SIGNAL or lost it by adding DISTRACTORS. Both rise
        # together with k, and one number cannot separate them.
        for column in ("gold_windows_in_chunks", "nongold_windows_in_chunks"):
            if column in scored_frame:
                values = scored_frame[column].dropna()
                if len(values):
                    runtime[f"average_{column}"] = float(values.mean())

    # Sub-side KV retention: the direct analogue of KVPress's
    # `average_retained_context_tokens`, but measured over llm_query sub-calls --
    # "how much of each context slice did the sub model effectively attend to".
    if scored and "metrics" in scored_frame:
        sub_kv = [
            m["sub_kv"]
            for m in scored_frame["metrics"]
            if isinstance(m, dict) and isinstance(m.get("sub_kv"), dict) and m["sub_kv"]
        ]
        if sub_kv:
            n_kv = len(sub_kv)
            total_calls = sum(s.get("calls", 0) for s in sub_kv)
            runtime["average_sub_context_tokens"] = float(
                sum(s.get("average_context_tokens", 0.0) for s in sub_kv) / n_kv
            )
            runtime["average_sub_retained_context_tokens"] = float(
                sum(s.get("average_retained_context_tokens", 0.0) for s in sub_kv) / n_kv
            )
            runtime["average_sub_compression_ratio"] = float(
                sum(s.get("average_compression_ratio", 0.0) for s in sub_kv) / n_kv
            )
            average_ratio = runtime["average_sub_compression_ratio"]
            runtime["realized_compression_factor"] = float(1.0 / (1.0 - average_ratio)) if average_ratio < 1.0 else None
            runtime["sub_context_tokens_on_target_fraction"] = (
                float(sum(s.get("context_tokens_on_target_calls", 0) for s in sub_kv) / total_calls)
                if total_calls
                else 0.0
            )
            # How often the root actually used the two-arg form. Near 0 means the
            # arm degenerated to dense one-arg calls and is NOT measuring the press.
            runtime["sub_split_call_fraction"] = (
                float(sum(s.get("split_calls", 0) for s in sub_kv) / total_calls) if total_calls else 0.0
            )
            # How often a sub-call actually went through KVzip vs. being skipped
            # (too small to clear press_min_tokens, or already fit the budget).
            # This is the direct answer to "what fraction of sub-calls actually
            # got compressed" -- distinct from sub_split_call_fraction, which
            # only says whether the two-arg form was used, not whether pruning
            # ran on top of it.
            runtime["sub_pressed_call_fraction"] = (
                float(sum(s.get("pressed_calls", 0) for s in sub_kv) / total_calls) if total_calls else 0.0
            )
            # The cost axis compared against KVPress is peak context held at once.
            # Counting only the ROOT flatters this arm: while the root holds ~2k
            # tokens, the sub model is holding a whole slice of KV on the GPU, and
            # KVzip masks rather than frees it. Report the larger of the two as the
            # honest peak, keeping both components visible above.
            root_peak = runtime.get("average_peak_context_tokens")
            sub_peak = runtime.get("average_sub_context_tokens")
            if root_peak is not None and sub_peak is not None:
                runtime["average_peak_context_tokens_root"] = root_peak
                runtime["average_peak_context_tokens"] = float(max(root_peak, sub_peak))

    # The size we ASKED for, next to average_sub_compression_ratio which is what we
    # GOT. They differ whenever the root under-fills the chunk -- the expected
    # failure mode of auto sizing, and invisible without both numbers side by side.
    # Outside the kvzip guard: the advertised cap governs the http arms too, and the
    # gap between it and average_sub_payload_chars is the finding, not a kvzip detail.
    if config.get("mode") in ("rlm", "structured"):
        runtime["subcall_chars_advertised"] = config.get("max_subcall_chars")
        runtime["subcall_chars_floor"] = config.get("min_subcall_chars") or 0
    if scored and (config.get("mode") == "structured" or config.get("chunks_source")):
        # `chunks_provided` is what the selector offered and `map_calls` what was
        # actually read; they diverge when the per-example deadline or the sub-call
        # cap cuts the map short, which is the difference between a thin selection
        # and a truncated run. `chunks_with_findings` near 0 is this arm's version
        # of sub_pressed_call_fraction near 0: the pipeline ran and learned nothing.
        # map_calls is now passages, not windows: chunks_provided minus map_spans is
        # how many adjacent windows were coalesced, which is the whole point of the
        # clustering and invisible without both numbers.
        # The agentic provided-chunk arm has no map step, so only the size of the
        # set it was handed is meaningful there.
        keys = ("chunks_provided", "map_spans", "map_calls", "chunks_with_findings")
        if config.get("mode") != "structured":
            keys = ("chunks_provided",)
        for key in keys:
            values = [(row or {}).get(key) for row in scored_frame.get("metrics", [])]
            values = [v for v in values if v is not None]
            runtime[f"average_{key}"] = float(sum(values) / len(values)) if values else 0.0
        if config.get("structured_order"):
            runtime.update(ordered_runtime(scored_frame))
    if scored and config.get("chunk_order"):
        runtime.update(chunk_order_runtime(scored_frame))
    if config.get("sub_backend") == "kvzip":
        runtime["subcall_sizing_mode"] = config.get("subcall_sizing_mode")
        runtime["subcall_target_compression_ratio"] = config.get("target_compression_ratio")
        runtime["subcall_sizing_binding"] = (config.get("subcall_sizing") or {}).get("binding")

    metrics["runtime"] = runtime
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return metrics


DEFAULT_GROUP_INPUT_TOKENS = 8192
DEFAULT_STRUCTURED_CONTEXT_TOKENS = 65536
DEFAULT_READER_MAX_TOKENS = 512
DEFAULT_REDUCE_MAX_TOKENS = 1024


def ordered_runtime(scored_frame: pd.DataFrame) -> dict:
    """Run-level aggregates for the ordered whole-document arms.

    Group sizes, reader and reducer exposure, and preprocessing cost -- the numbers
    that say whether two arms with equal CAPS actually spent equal tokens.
    """
    rows = [m for m in scored_frame.get("metrics", []) if isinstance(m, dict)]
    out: dict = {}

    def mean(values):
        values = [v for v in values if v is not None]
        return float(sum(values) / len(values)) if values else 0.0

    out["average_group_count"] = mean([m.get("group_count") for m in rows])
    out["average_chunks_per_group"] = mean([c for m in rows for c in m.get("group_chunk_counts", [])])
    out.update(reader_token_runtime(rows))
    out["average_reducer_input_tokens"] = mean([m.get("reducer_input_tokens") for m in rows])
    out["max_reducer_input_tokens"] = int(max((m.get("reducer_input_tokens") or 0) for m in rows)) if rows else 0
    out["average_retained_finding_count"] = mean([m.get("retained_finding_count") for m in rows])
    out["average_reducer_finding_chars"] = mean([m.get("reducer_finding_chars") for m in rows])
    # One build per document; its cost is shared by every query on that document.
    builds = {m.get("organization_sha256"): m.get("organization_build_cpu_seconds") for m in rows}
    out["organization_build_cpu_seconds"] = float(sum(v for v in builds.values() if v is not None))
    out["organization_documents"] = len(builds)
    out["average_packing_cpu_seconds"] = mean([m.get("packing_cpu_seconds") for m in rows])
    for column in ("prompt_tokens", "completion_tokens"):
        if column in scored_frame:
            out[f"average_{column}"] = float(scored_frame[column].mean())
    return out


def reader_token_runtime(rows: list[dict]) -> dict:
    """Reader exposure and pricing, for every arm that reads through OrderedPayload.

    `reader_token_count_mismatches` compares the local chat-template count with the
    server's own prompt_tokens, call by call: zero is the proof that a budget was
    priced on what vLLM actually received.
    """
    reader = [t for m in rows for t in m.get("reader_input_tokens", [])]
    served = [t for m in rows for t in m.get("reader_served_prompt_tokens", [])]
    pairs = [
        pair for m in rows for pair in zip(m.get("reader_input_tokens", []), m.get("reader_served_prompt_tokens", []))
    ]
    mismatches = sum(1 for local, server in pairs if server is not None and local != server)
    return {
        "average_reader_input_tokens": float(sum(reader) / len(reader)) if reader else 0.0,
        "max_reader_input_tokens": int(max(reader)) if reader else 0,
        "reader_token_count_mismatches": int(mismatches),
        "reader_served_counts_missing": int(sum(1 for t in served if t is None)),
    }


def chunk_order_query(question: str, dataset: Optional[str]) -> str:
    """What BM25 ranks the chunks against for --chunk-order bm25-query: the question alone.

    LOFT wraps its query in a page of citation instructions, which `bare_query` strips
    (the same reason `chunks_query` in main exists). RULER puts its instructions around
    a final `Question:` line instead -- and "Answer the question based on the given
    documents" shares "documents" with every document header in a qa haystack -- so for
    RULER the query is the text after the last `Question:`.
    """
    query = bare_query(question)
    if (dataset or "").startswith("ruler"):
        marker = query.rfind("Question:")
        if marker >= 0:
            query = query[marker + len("Question:") :].strip()
    return query


def build_chunk_view(
    document: str,
    question: str,
    order: str,
    dataset: Optional[str],
    organization_dir: Optional[str],
    organizations: dict,
) -> tuple[list[Hit], dict]:
    """EVERY chunk of the document, in the order an ordered-chunk arm hands the root.

    Both orders cut the same 2000-character, zero-overlap grid (organization.CONFIG).
    `bm25-graph` is the question-free organization -- the structured campaign's own
    artifact, cached by document hash in `organizations` and on disk -- and
    `bm25-query` ranks those same chunks by BM25 against the question. So the two views
    are permutations of one partition, and the arms differ in order and nothing else
    they are handed.
    """
    started = time.process_time()
    size, overlap = ORGANIZATION_CONFIG["chunk_chars"], ORGANIZATION_CONFIG["overlap"]
    digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
    stats: dict = {"chunk_order": order, "document_sha256": digest}
    if order == "bm25-graph":
        if digest not in organizations:
            artifact, _, cache_hit = organize(document, "bm25-graph", organization_dir)
            organizations[digest] = (artifact, cache_hit)
        artifact, cache_hit = organizations[digest]
        view = ordered_hits(document, artifact)
        stats.update(
            organization_sha256=digest,
            organization_cache_hit=cache_hit,
            organization_build_cpu_seconds=artifact["build_cpu_seconds"],
            organization_edges=len(artifact["edges"]),
            organization_forest_edges=len(artifact["forest"]),
        )
    elif order == "bm25-query":
        index = retrieval.get_index(document, size, overlap)
        grid = [(start, min(start + size, len(document))) for start in range(0, len(document), size)]
        if index.spans != grid:
            raise ValueError("the BM25 index and the organization grid disagree on chunk spans")
        query = chunk_order_query(question, dataset)
        view = index.rank_all(query)
        stats.update(chunk_order_query=query, chunks_scored=sum(1 for hit in view if hit.score > 0))
    else:
        raise ValueError(f"unknown chunk order {order!r}; choose from {list(CHUNK_ORDERS)}")
    stats.update(
        chunks_total=len(view),
        chunk_view_sha256=hashlib.sha256(json.dumps([[hit.start, hit.end] for hit in view]).encode()).hexdigest(),
        chunk_view_cpu_seconds=time.process_time() - started,
    )
    return view, stats


def chunk_diagnostics(view: Sequence[Hit], ranks_read: Sequence[int], answers: list[str]) -> dict:
    """Where the gold sat in the view, and whether the root read it. LOOSE and post hoc.

    A gold answer STRING inside a chunk -- exactly as loose as gold_in_retrieved
    (c448c97): a match does not establish the chunk carries sufficient evidence. None
    of this is ever shown to the model.
    """
    gold = [hit.rank for hit in view if is_correct(hit.text, answers)]
    read = set(ranks_read)
    return {
        "gold_first_position": gold[0] if gold else None,
        "gold_chunks": len(gold),
        "gold_in_read_chunks": bool(read & set(gold)),
        "chunks_read": len(read),
        "deepest_position_read": max(read) if read else 0,
    }


def chunk_order_runtime(scored_frame: pd.DataFrame) -> dict:
    """Run-level aggregates for the ordered-chunk agentic arms.

    What the root was offered, how much of it it read and how deep into the order it
    went, where the gold sat (loosely), and the same reader pricing check the
    structured ordered arms report.
    """
    rows = [m for m in scored_frame.get("metrics", []) if isinstance(m, dict)]

    def mean(values):
        values = [v for v in values if v is not None]
        return float(sum(values) / len(values)) if values else 0.0

    out: dict = reader_token_runtime(rows)
    out["average_chunks_total"] = mean([m.get("chunks_total") for m in rows])
    out["average_chunks_per_call_advertised"] = mean([m.get("chunks_per_call_advertised") for m in rows])
    out["average_chunk_payload_calls"] = mean([m.get("chunk_payload_calls") for m in rows])
    # The number that says whether the chunks were USED: an example that never passed
    # one to llm_query answered from string ops or from nothing.
    out["chunk_read_example_fraction"] = (
        float(sum(1 for m in rows if m.get("chunk_payload_calls")) / len(rows)) if rows else 0.0
    )
    out["chunk_budget_refusals"] = int(sum(m.get("chunk_budget_refusals", 0) for m in rows))
    # Harness rejections, totalled so a campaign shows how often each one fired.
    out["premature_abstentions_rejected"] = int(sum(m.get("premature_abstentions", 0) for m in rows))
    out["prose_answer_cells"] = int(sum(m.get("prose_answer_cells", 0) for m in rows))
    out["empty_chunk_reads"] = int(sum(m.get("empty_chunk_reads", 0) for m in rows))
    for column in ("chunks_read", "deepest_position_read"):
        if column in scored_frame:
            out[f"average_{column}"] = mean(list(scored_frame[column].dropna()))
    total = len(scored_frame)
    if "gold_first_position" in scored_frame and total:
        first = scored_frame["gold_first_position"].dropna()
        out["gold_present_fraction"] = float(len(first) / total)
        out["median_gold_first_position"] = float(first.median()) if len(first) else None
        out["gold_in_top5_fraction"] = float((first <= 5).sum() / total)
    if "gold_in_read_chunks" in scored_frame:
        read = scored_frame["gold_in_read_chunks"].dropna()
        out["gold_in_read_chunks_fraction"] = float(read.astype(bool).sum() / len(read)) if len(read) else 0.0
    # One build per document; its cost is shared by every query on that document.
    builds = {
        m.get("organization_sha256"): m.get("organization_build_cpu_seconds")
        for m in rows
        if m.get("organization_sha256")
    }
    out["organization_build_cpu_seconds"] = float(sum(v for v in builds.values() if v is not None))
    out["organization_documents"] = len(builds)
    out["average_chunk_view_cpu_seconds"] = mean([m.get("chunk_view_cpu_seconds") for m in rows])
    for column in ("prompt_tokens", "completion_tokens"):
        if column in scored_frame:
            out[f"average_{column}"] = float(scored_frame[column].mean())
    return out


def build_run_dir_components(args: argparse.Namespace, mode: str, scratchpad: Scratchpad | None) -> list[str]:
    """Name the run directory after everything that changes results.

    Two configurations that differ in any result-affecting knob must land in
    different directories, or they share a checkpoint.jsonl and silently merge.
    (compare.py reads config.yaml, not this name, so nothing else depends on the
    format.)
    """
    dataset_name = canonical_dataset_name(args.dataset or args.legacy_dataset)
    model_slug = args.root_model.replace("/", "_")
    task_slug = (args.data_dir or "default").replace("/", "_")
    components = [dataset_name, task_slug, model_slug, mode]
    # Only when it is not the whole subset: `all` is what every existing run used,
    # so naming it would rename every existing directory for no new information.
    if getattr(args, "split", "all") != "all":
        components.append(f"split-{args.split}")
    if mode == "rlm" and args.max_context_tokens is not None:
        components.append(f"ctx{args.max_context_tokens}")
    if mode == "rlm" and scratchpad is not None:
        components.append("scratchpad")
    if mode == "rlm" and args.sub_backend == "kvzip":
        # Backend-prefixed ("kvzip-kvzip1GB", not a bare "kvzip1GB"): kvpress's
        # own KVzipPress (masking, not real eviction) produces different
        # results than the standalone-KVzip backend it replaced, so resuming
        # into an old checkpoint from that backend would silently merge two
        # experiments.
        components.append(f"{args.sub_backend}-{args.press}{args.memory_budget:g}{args.memory_budget_unit}")
    # Auto-sized runs must NOT stamp the resolved char count into the name: the
    # resolution reads live GPU free memory, so a resume could resolve slightly
    # differently, land in a fresh directory, and silently start over instead of
    # hitting the _prior_resolved_chars guard. Only hand-picked sizes are named.
    sizing_mode = getattr(args, "subcall_sizing_mode", "fixed")
    if mode == "rlm" and sizing_mode == "fixed" and args.max_subcall_chars != DEFAULT_MAX_SUBCALL_CHARS:
        components.append(f"sub{args.max_subcall_chars}")
    # A floor changes both the prompt and the slices, so a floored run must never
    # share a checkpoint with the unfloored run of the same subset. Only stamped
    # when non-zero, so every existing run directory keeps its name.
    if mode == "rlm" and getattr(args, "min_subcall_chars", 0) and not getattr(args, "fixed_chunk", False):
        components.append(f"min{args.min_subcall_chars}")
    # After the `sub` component, so every hand-sized slug stays byte-identical to
    # what it was before auto-sizing existed. This marker is the ONLY thing
    # separating an auto run that happens to resolve to exactly the default size
    # from the hand-sized default -- neither carries a `sub<N>` component.
    if mode == "rlm" and sizing_mode == "auto":
        factor = getattr(args, "compression_factor", None)
        components.append(f"autosubx{factor:g}" if factor is not None else f"autosub{args.target_compression_ratio:g}")
    if mode == "rlm" and getattr(args, "fixed_chunk", False):
        components.append("fixed")
    # Last, and only when retrieval is on, so every directory written before this
    # arm existed keeps its exact name and its checkpoint stays reachable.
    if mode == "rlm" and getattr(args, "search_k", 0):
        components.append(f"bm25k{args.search_k}")
        if (args.search_window, args.search_overlap) != (DEFAULT_SEARCH_WINDOW, DEFAULT_SEARCH_OVERLAP):
            components.append(f"win{args.search_window}o{args.search_overlap}")
        # Only when ablating, so a full-treatment run keeps the name it already has.
        for part in ("locate", "example", "gate"):
            if part in getattr(args, "search_ablate", ()):
                components.append(f"no-{part}")
    # Last of all, and only for the structured arm, so every directory written
    # before this mode existed keeps its exact name and its checkpoint stays
    # reachable. `mode` already reads "structured"; this says which chunks.
    if mode == "structured" and getattr(args, "structured_order", None):
        # Every chunk is read, so there is no source and no k to name -- only the
        # order and, when not the default, the group budget.
        components.append(f"order-{args.structured_order}")
        if args.structured_group_input_tokens != DEFAULT_GROUP_INPUT_TOKENS:
            components.append(f"grp{args.structured_group_input_tokens}")
        if getattr(args, "reduce_finding_chars", DEFAULT_REDUCE_FINDING_CHARS) != DEFAULT_REDUCE_FINDING_CHARS:
            components.append(f"find{args.reduce_finding_chars}")
        if getattr(args, "structured_map_question", "task") != "task":
            components.append(f"mapq-{args.structured_map_question}")
        if answer_list_example(canonical_dataset_name(args.dataset or args.legacy_dataset), task_slug, mode):
            if getattr(args, "answer_list_format", "auto") != "off":
                components.append("listfmt")
    elif mode == "structured":
        components.append(f"chunks-{args.chunks_source}{args.chunks_k}")
        # WHICH selector chose them. Stamped only when a tag exists, so every chunk
        # file written before the query-agnostic selectors -- the oracle sets and the
        # two hand-made ones -- names the directory it always named. Without this,
        # oracle / lead / stride / random / exhaustive at the same k are all
        # `chunks-file5` and the second arm to run dies on a resume conflict.
        if getattr(args, "chunks_tag", None):
            components.append(f"sel-{args.chunks_tag}")
        # The query-free index arm. `notes-all` and `notes-bm25k<K>` are different
        # claims -- one is query-free selection, the other is query-aware lookup over
        # a query-agnostic index -- so they must never share a directory.
        if getattr(args, "chunks_source", None) == "notes":
            select = getattr(args, "notes_select", "all")
            components.append(f"notes-{select}" if select == "all" else f"notes-bm25k{args.notes_k}")
        # The window geometry, on the same non-default rule. WITHOUT this the whole
        # geometry axis collapses: --mode structured rejects --search-k, so the
        # `mode == "rlm" and args.search_k` gate above can never fire here and every
        # window cell of a sweep would name the same directory. The first cell would
        # run and every later one would die on resume_conflicts -- loud, but the
        # sweep would be impossible to run at all.
        if (args.search_window, args.search_overlap) != (DEFAULT_SEARCH_WINDOW, DEFAULT_SEARCH_OVERLAP):
            components.append(f"win{args.search_window}o{args.search_overlap}")
        # The revision-2 axes, stamped only when NON-default, so every directory
        # written by the revision-2 campaign keeps its exact name and its checkpoint
        # stays reachable -- the same rule --search-ablate follows above.
        if getattr(args, "structured_map_question", "task") != "task":
            components.append(f"mapq-{args.structured_map_question}")
        if getattr(args, "structured_map_granularity", "passage") != "passage":
            components.append(f"mapg-{args.structured_map_granularity}")
        if getattr(args, "structured_answer_shape", "require") != "require":
            components.append(f"shape-{args.structured_answer_shape}")
        # How much of each passage the map step SEES, and how much of each finding
        # survives into the reduce step. Both bound the read independently of the
        # window width, which is the whole point of having them: a window axis that
        # a downstream cap silently truncates measures the cap, not the window.
        if getattr(args, "structured_map_chars", 0):
            components.append(f"mapchars{args.structured_map_chars}")
        if getattr(args, "reduce_finding_chars", DEFAULT_REDUCE_FINDING_CHARS) != DEFAULT_REDUCE_FINDING_CHARS:
            components.append(f"find{args.reduce_finding_chars}")
        if getattr(args, "structured_hops", 1) != 1:
            components.append(f"hops{args.structured_hops}")
        # Stamped when the list shape IS shown -- the inverse of the usual
        # non-default rule, and deliberately. The change here is the new behaviour,
        # so it is the new behaviour that must claim a new directory; every qampari
        # and quest run already on disk keeps its name and its checkpoint.
        # Stamped when the query is the corrected one, for the same reason `listfmt`
        # is: the change is the new behaviour, so the new behaviour claims the new
        # directory and the four arms already on the NAS keep their checkpoints.
        if getattr(args, "chunks_query", "task") == "bare" and args.chunks_source == "bm25":
            components.append("qbare")
        if answer_list_example(canonical_dataset_name(args.dataset or args.legacy_dataset), task_slug, mode):
            if getattr(args, "answer_list_format", "auto") != "off":
                components.append("listfmt")
    # The agentic root handed EVERY chunk in a fixed order. Only when set, so every
    # agentic directory written before it keeps its exact name.
    if mode == "rlm" and getattr(args, "chunk_order", None):
        components.append(f"order-{args.chunk_order}")
        if getattr(args, "chunk_call_tokens", DEFAULT_CHUNK_CALL_TOKENS) != DEFAULT_CHUNK_CALL_TOKENS:
            components.append(f"ctok{args.chunk_call_tokens}")
        if answer_list_example(dataset_name, task_slug, mode, ordered_rlm=True):
            if getattr(args, "answer_list_format", "auto") != "off":
                components.append("listfmt")
    # The agentic arm reading a chunk set chosen outside the run. It still carries
    # `bm25k<N>` from the block above, because search() is how the chunks arrive and
    # k still bounds them; this component is what says the windows are not BM25's.
    if mode == "rlm" and getattr(args, "chunks_source", None):
        components.append(f"chunks-{args.chunks_source}{args.chunks_k}")
        if getattr(args, "chunks_tag", None):
            components.append(f"sel-{args.chunks_tag}")
    # Last of all, and for EVERY mode: --ids-file picks WHICH rows run and reaches
    # nothing else in the name, so without this an id-restricted run resumes into the
    # full run's checkpoint and merges two experiments. Kept last so every structured
    # directory written before the agentic arm existed keeps its exact name.
    if getattr(args, "ids_sha256", None):
        components.append(f"ids-{args.ids_sha256[:8]}")
    return components


def main() -> None:
    ap = argparse.ArgumentParser()
    dataset_selection = ap.add_mutually_exclusive_group(required=True)
    dataset_selection.add_argument("--dataset", choices=available_datasets())
    dataset_selection.add_argument(
        "--task",
        dest="legacy_dataset",
        choices=available_datasets(),
        help="deprecated alias for --dataset",
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help="shared Hugging Face config/subset, e.g. niah_single_1 for ruler32k",
    )
    ap.add_argument("--mode", default="both", choices=["vanilla", "rlm", "structured", "both"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--split",
        default="all",
        choices=["all", "dev", "test"],
        help="restrict to one split where the dataset has them (LOFT ships dev then "
        "test). Datasets with a single split ignore this. Use --split dev for smoke "
        "runs and --split test for real ones: with 'all', a small --limit silently "
        "evaluates the 10 dev rows only, which lands in the results tree looking "
        "like a real run that scored 0.000",
    )
    ap.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="evaluate a reproducible random subset of the (post-split) examples, e.g. "
        "0.5 for half -- unlike --limit, this shuffles with --sample-seed before "
        "selecting, instead of just taking the first N rows",
    )
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="random seed for --sample-fraction; ignored without it",
    )
    ap.add_argument(
        "--root-model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="served model id for vanilla baseline AND the RLM root",
    )
    ap.add_argument(
        "--sub-model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="served model id for recursive sub-calls (cheap is fine)",
    )
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible server, e.g. vLLM")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--vanilla-char-limit", type=int, default=400_000)
    ap.add_argument(
        "--vanilla-max-prompt-tokens",
        type=int,
        default=None,
        help="hard TOKEN ceiling for the vanilla prompt; the char limit alone "
        "overflows the served window on densely-tokenising subsets (RULER cwe / "
        "niah_multikey_3), which the server rejects with a 400 and which then "
        "scores 0.0 from a harness error rather than from the model. Set to "
        "(max-model-len - max-tokens - margin), e.g. 34000 for a 40960 window",
    )
    # --- RLM memory-budget knobs (no budget unless --max-context-tokens is set) ---
    # Eviction-only: out-of-budget turns are simply dropped (no notes/summarization).
    ap.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="cap the RLM root's context window (tokens); unset = unbounded (legacy)",
    )
    ap.add_argument(
        "--keep-recent-turns",
        type=int,
        default=3,
        help="recent (assistant,observation) pairs to keep verbatim under budget",
    )
    # --- RLM scratchpad knobs (off unless --scratchpad; orthogonal to the budget) ---
    ap.add_argument(
        "--scratchpad",
        action="store_true",
        help="give the RLM root a note() tool; notes are re-shown every turn and survive budget eviction",
    )
    ap.add_argument(
        "--max-notes-tokens",
        type=int,
        default=1024,
        help="cap on the scratchpad block injected into context",
    )
    # --- Runaway guards. Each is independent; pass 0 to disable just that one. ---
    ap.add_argument(
        "--exec-timeout",
        type=float,
        default=60.0,
        help="wall-clock seconds of PURE-PYTHON execution a single RLM code block may run before "
        "being aborted (guards against model-generated infinite loops); time spent inside "
        "llm_query sub-calls is excluded. 0 disables",
    )
    ap.add_argument(
        "--run-timeout",
        type=float,
        default=900.0,
        help="wall-clock seconds for ONE example before the RLM gives up (end_reason=run_timeout). "
        "exec-timeout does not bound llm_query time, so without this a single pathological "
        "example can stall a job for hours. 0 disables",
    )
    ap.add_argument(
        "--max-sub-calls",
        type=int,
        default=40,
        help="cap on llm_query calls per example; further calls return a notice instead of "
        "hitting the API. 0 disables",
    )
    ap.add_argument(
        "--min-sub-calls-before-abstain",
        type=int,
        default=1,
        help="llm_query calls an abstention must be backed by; FINAL_NONE below this is "
        "rejected with a nudge (at most twice, sharing the FINAL nudge budget) telling the "
        "root to search distinctive keywords rather than the whole question. Guards the "
        "cheapest way out of a hard example: claiming absence without having read anything. "
        "0 disables",
    )
    ap.add_argument(
        "--max-unproductive-steps",
        type=int,
        default=3,
        help="consecutive code cells printing NOTHING before the run is broken out of with a "
        "forced grounded answer (end_reason=unproductive_broken); nudged one step earlier. "
        "The repetition breaker only catches identical code, which a model rambling in "
        "comments slips past while learning nothing. 0 disables",
    )
    ap.add_argument(
        "--max-error-steps",
        type=int,
        default=3,
        help="consecutive code cells raising an exception before the run is broken out of "
        "(end_reason=error_loop_broken); nudged one step earlier. Neither other breaker sees "
        "this: an exception counts as output, and broken code that changes between attempts "
        "is never an exact repeat. 0 disables",
    )
    # --- Sub-call backend (arm 4: KV-compressed sub-calls; see kvzip_backend.py) ---
    # `http` keeps today's behavior: sub-calls go to the same OpenAI-compatible
    # server as the root. `kvzip` loads the sub model IN-PROCESS through
    # kvpress's own KVzipPress + KVPressTextGenerationPipeline -- the same
    # press and memory-budget mechanics used by every LOFT/RULER/synthetic-kv
    # benchmark this session. Note: kvpress's KVzipPress masks evicted KV
    # rather than freeing it, so this does NOT reduce actual GPU memory usage
    # the way a real-eviction backend would -- it buys methodological
    # consistency with the rest of the benchmarking, not compute savings.
    # "nim" is a deprecated alias for "http": existing run-dir slugs and
    # config.yaml resume guards were written with it, so it must keep parsing.
    ap.add_argument("--sub-backend", default="http", choices=["http", "kvzip", "nim"])
    ap.add_argument(
        "--press",
        default="kvzip",
        choices=["kvzip", "no_press"],
        help="for --sub-backend kvzip; no_press = same load/generate path, no pruning (press control)",
    )
    ap.add_argument(
        "--memory-budget",
        type=float,
        default=1.0,
        help="KV memory budget per sub-call (kvzip backend), converted to a "
        "compression_ratio the same way LOFT-32k/128k's matrix runs are "
        "(matrix_constants.py's EXTENDED_KV_BUDGETS: 0.256, 0.512, 1, 2, 4 with "
        "--memory-budget-unit GB)",
    )
    ap.add_argument(
        "--memory-budget-unit",
        default="GB",
        choices=["MB", "GB", "tokens"],
        help="unit for --memory-budget (kvzip backend)",
    )
    ap.add_argument(
        "--max-subcall-chars",
        type=_subcall_chars,
        default=DEFAULT_MAX_SUBCALL_CHARS,
        help="char cap per llm_query prompt/context slice (was constructor-only); the split "
        "prompt advertises this cap to the root, so raising it is what makes the "
        "'compression enables bigger reads' arm real. Pass 'auto' to derive it from "
        "--memory-budget and --target-compression-ratio instead of picking it by hand "
        "(requires --sub-backend kvzip)",
    )
    ap.add_argument(
        "--min-subcall-chars",
        type=int,
        default=0,
        help="FLOOR on the context slice sent to the sub model (0 = off, the behaviour of "
        "every arm so far). --max-subcall-chars is a ceiling the root may use and mostly "
        "does not: across the finished 128k and 1m campaigns it sent a mean of 1023 chars "
        "against a 32000 cap. This makes the size a rule -- the split prompt switches to "
        "its large-read variants and any shorter slice is widened in the source document. "
        "Required for the KV-compression arm to measure anything: a slice under "
        "--press-min-tokens skips the press entirely",
    )
    # --- BM25 retrieval over the document (arm 5) ---
    # The root's only locate primitive was `str.find`, which returns the FIRST
    # literal occurrence. Across the 879 finished LOFT-1m transcripts 80.5% of
    # examples never surfaced the gold answer into any REPL output at all, against
    # 5.2% that saw it and still answered wrong -- a recall failure sitting upstream
    # of every knob the fixed-chunk grid sweeps.
    ap.add_argument(
        "--search-k",
        type=int,
        default=0,
        help="give the root a BM25 `search(query)` over the whole document, returning this "
        "many ranked windows (0 = off, which is every arm run so far and leaves the REPL "
        "namespace unchanged). This is the grid's retrieval-depth axis: the model may ask "
        "for fewer windows but never more, so a worked example cannot override the cell",
    )
    ap.add_argument(
        "--search-window",
        type=int,
        default=DEFAULT_SEARCH_WINDOW,
        help="character width of each indexed window (--search-k)",
    )
    ap.add_argument(
        "--search-overlap",
        type=int,
        default=DEFAULT_SEARCH_OVERLAP,
        help="character overlap between adjacent indexed windows, so an answer straddling a "
        "boundary still matches one of them (--search-k)",
    )
    ap.add_argument(
        "--search-ablate",
        default="",
        help="comma-separated parts of the search treatment to WITHHOLD, so the bundle can be "
        "taken apart: locate (the prose telling the root to search), example (the worked "
        "session demonstrating it), gate (an abstention must be backed by a search). Empty = "
        "the full treatment. Turning retrieval on changes four things at once, and a combined "
        "result cannot say which did the work; --search-k 0 withholds the tool itself",
    )
    ap.add_argument(
        "--max-search-calls",
        type=int,
        default=200,
        help="runaway guard on search() calls per example, in the shape of --max-sub-calls. "
        "Searching is pure Python over a prebuilt index, so this should never bind",
    )
    # --- Structured (map -> reduce) arm over PROVIDED chunks ---
    # The search arm let the root FIND things; this hands it the chunks and takes
    # the REPL away, so "who picked the text" and "was the loop worth it" become two
    # axes instead of one bundled treatment.
    ap.add_argument(
        "--chunks-source",
        default=None,
        choices=["bm25", "file", "notes"],
        help="where --mode structured gets its chunks: `bm25` selects the top --chunks-k windows "
        "in-harness (the cheap-selector control), `notes` reads a query-free index built by "
        "window_notes.py, `file` reads them from --chunks-file, written "
        "offline by a larger model. Required by --mode structured and rejected by every other mode",
    )
    ap.add_argument(
        "--chunks-file",
        default=None,
        help="JSONL of pre-selected chunks, one record per example id (see chunkset.py). The file's "
        "CONTENT hash is stamped into config.yaml, so editing it invalidates a resume",
    )
    ap.add_argument(
        "--chunks-tag",
        default=None,
        help="short name for WHICH selector wrote --chunks-file, stamped into the run directory as "
        "`sel-<tag>`. Normally left unset: the chunk file names itself in its own `selector_tag` "
        "field and this only overrides that. Without a tag every file-sourced arm at the same k -- "
        "oracle, lead, stride, random, exhaustive -- names the SAME directory and the second one "
        "dies on a resume conflict",
    )
    ap.add_argument(
        "--notes-file",
        default=None,
        help="a window_notes.py index: one note per window, written by a pass over the corpus that "
        "never saw the question. Read with --chunks-source notes",
    )
    ap.add_argument(
        "--notes-select",
        choices=["all", "bm25"],
        default="all",
        help="which notes reach the reduce step. `all` is query-free end to end until the reduce; "
        "`bm25` ranks the NOTES on the bare question, which is query-aware LOOKUP over a "
        "query-agnostic INDEX and must be labelled as such",
    )
    ap.add_argument(
        "--notes-k",
        type=int,
        default=DEFAULT_CHUNKS_K,
        help="notes kept by --notes-select bm25",
    )
    ap.add_argument(
        "--chunks-k",
        type=int,
        default=DEFAULT_CHUNKS_K,
        help="chunks per example for --chunks-source bm25, and the size the `file` source is held "
        "to: a provided chunk set may not send more than --chunks-k x --search-window characters. "
        "Without that cap the provided-chunk arm simply reads more than the arm it is compared "
        "against, and the result measures payload size instead of selection quality",
    )
    ap.add_argument(
        "--ids-file",
        default=None,
        help="evaluate only the example ids in this file (one per line; blank lines and #comments "
        "ignored), so several arms can be scored on exactly the same rows. Applied BEFORE --limit, "
        "which it is therefore mutually exclusive with",
    )
    # The three components of the revision-2 structured treatment, one flag each, in
    # the shape of --search-ablate. Every default reproduces revision 2 exactly, so a
    # run that passes none of them is byte-identical to the arm already on the NAS.
    ap.add_argument(
        "--structured-map-question",
        default="task",
        choices=list(STRUCTURED_MAP_QUESTIONS),
        help="what the map step asks about each passage: `task` interpolates the benchmark's "
        "task string verbatim (LOFT's is a page of citation instructions, which derails a 4B "
        "sub-model reading one window), `query` strips it to the query inside. The reduce step "
        "always sees the full task -- it carries the output contract the scorer matches on",
    )
    ap.add_argument(
        "--structured-map-granularity",
        default="passage",
        choices=list(STRUCTURED_MAP_GRANULARITIES),
        help="how many windows go into one map call: `passage` merges touching windows into "
        "contiguous runs, `window` reads each provided window alone, `all` reads every window "
        "in ONE call -- which is the payload the agentic root assembles, and the difference "
        "that separates `splitting the text` from `the pipeline`",
    )
    ap.add_argument(
        "--structured-answer-shape",
        default="require",
        choices=list(STRUCTURED_ANSWER_SHAPES),
        help="whether the reduce step REQUIRES the benchmark's answer shape (a bare value on a "
        "final line) or merely OFFERS it as a format. `offer` is revision 1's wording, restored "
        "so the shape change can be scored apart from the clustering change it shipped with",
    )
    ap.add_argument(
        "--structured-map-chars",
        type=int,
        default=0,
        help="characters of each passage the map step actually SEES (0 = all of it, which is "
        "every arm run so far). --search-window sets the retrieval unit AND the read size "
        "together; this separates them, so a window-axis result can be told apart from an "
        "exposure effect",
    )
    ap.add_argument(
        "--reduce-finding-chars",
        type=int,
        default=DEFAULT_REDUCE_FINDING_CHARS,
        help="characters of each map finding pasted into the reduce prompt. Left at 2000 a wide "
        "window is truncated back at the reduce step, so the top of a window sweep would measure "
        "this cap rather than the window; raise it with --search-window",
    )
    ap.add_argument(
        "--chunks-query",
        default="task",
        choices=["task", "bare"],
        help="what BM25 is asked for --chunks-source bm25. `bare` strips the benchmark's "
        "instruction wrapper down to the query inside it; `task` sends the whole thing and "
        "is the default ONLY because it is what every structured run on disk did. On LOFT "
        "the wrapper costs recall@2 nearly everything -- nq 0.00 against 0.95 -- so pass "
        "`bare` for any new measurement and read a `task` run as handicapped retrieval",
    )
    ap.add_argument(
        "--answer-list-format",
        default="auto",
        choices=["auto", "off"],
        help="`auto` shows the structured reduce step the BRACKETED list shape LOFT's "
        "multi-value scorer (qampari, quest) actually parses; every other task is "
        "untouched. `off` is the old behaviour, in which a reply naming all five gold "
        "answers as prose extracted as one string and scored 0",
    )
    ap.add_argument(
        "--structured-hops",
        type=int,
        default=1,
        choices=[1, 2],
        help="retrieval rounds. 1 = the shipped arm. 2 adds a round whose query is built from "
        "what the FIRST round's map step found -- not from the retrieved text, which was measured "
        "on this corpus to make recall worse. There is deliberately no 3",
    )
    # --- Ordered whole-document reading (organization.py) ---
    # Every chunk of the document, in a question-free order, packed into reader
    # groups under one token budget. Replaces --chunks-source: nothing is selected.
    ap.add_argument(
        "--structured-order",
        default=None,
        choices=list(ORDERS),
        help="read EVERY 2000-char zero-overlap chunk in this question-free order, packed into "
        "reader groups of --structured-group-input-tokens: `source` (document order, the control) "
        "or `bm25-graph` (chunk-to-chunk BM25 graph, spanning forest, depth-first). Requires "
        "--mode structured and --max-sub-calls 0, and replaces --chunks-source",
    )
    ap.add_argument(
        "--structured-group-input-tokens",
        type=int,
        default=DEFAULT_GROUP_INPUT_TOKENS,
        help="reader-group budget in model tokens, counted with the served model's chat template on "
        "the COMPLETE reader message (instructions, question, source labels)",
    )
    ap.add_argument(
        "--structured-context-tokens",
        type=int,
        default=DEFAULT_STRUCTURED_CONTEXT_TOKENS,
        help="served context limit; every reader and reducer request is checked against it "
        "(input + output cap) BEFORE it is sent",
    )
    ap.add_argument(
        "--structured-reader-max-tokens",
        type=int,
        default=DEFAULT_READER_MAX_TOKENS,
        help="output cap per reader call under --structured-order",
    )
    ap.add_argument(
        "--structured-reduce-max-tokens",
        type=int,
        default=DEFAULT_REDUCE_MAX_TOKENS,
        help="output cap for the reduce call under --structured-order",
    )
    ap.add_argument(
        "--organization-dir",
        default=None,
        help="cache + archive of organization artifacts (spans, order, graph edges, forest), keyed "
        "by document hash and configuration. Defaults to <out>/organization",
    )
    # --- The AGENTIC root over every chunk, in a fixed order (organization.py) ---
    # --structured-order reads every chunk through a fixed map -> reduce, so the root
    # never chooses what to read. These hand the same chunks to the REPL loop as a
    # `chunks` list, in an order the root is told about.
    ap.add_argument(
        "--chunk-order",
        default=None,
        choices=list(CHUNK_ORDERS),
        help="hand the AGENTIC root every 2000-char zero-overlap chunk as a `chunks` list, in this order, "
        "and tell it what the order is: `bm25-graph` (the question-free organization --structured-order "
        "reads) or `bm25-query` (every chunk ranked by BM25 against the bare question). Requires --mode "
        "rlm; there is no search() and no --chunks-source beside it",
    )
    ap.add_argument(
        "--chunk-call-tokens",
        type=int,
        default=DEFAULT_CHUNK_CALL_TOKENS,
        help="most tokens one llm_query over chunks may carry, counted with the served model's chat "
        "template on the complete reader message. A longer list is refused, never truncated. The "
        "default is one structured reader group",
    )
    compression_target = ap.add_mutually_exclusive_group()
    compression_target.add_argument(
        "--target-compression-ratio",
        type=float,
        default=None,
        help="with --max-subcall-chars auto: the compression ratio the chunk is sized to HIT "
        "when the root fills it (chunk = token_budget / (1 - ratio)). This only sets the size "
        "advertised to the root -- the press still derives each call's actual ratio from the "
        "slice the root really sent, reported as runtime.average_sub_compression_ratio",
    )
    compression_target.add_argument(
        "--compression-factor",
        type=float,
        default=None,
        help="with --max-subcall-chars auto: human-readable chunk/retained-token factor; "
        "converted to ratio = 1 - 1/factor. A factor of 1 is uncompressed",
    )
    ap.add_argument(
        "--fixed-chunk",
        action="store_true",
        help="make the auto-sized chunk exact: derive an oversized character floor, reserve "
        "question room in the character cap, and token-truncate every sub-call to the "
        "budget-derived chunk. Fails if the model window or GPU cap would shrink the cell",
    )
    ap.add_argument(
        "--subcall-reserve-tokens",
        type=int,
        default=DEFAULT_RESERVE_TOKENS,
        help="tokens held back from the sub model's window when auto-sizing, to cover the "
        "question and the decoded answer (the context cap alone budgets for neither)",
    )
    ap.add_argument(
        "--subcall-min-tokens",
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help="smallest chunk auto-sizing may resolve to before it refuses the run. The 1024 "
        "default dates from the era of a FIXED 1024-token press floor, below which pressing "
        "was skipped and the chunk was pointless. press_min_tokens is now derived from the "
        "memory budget, so a sub-1024 budget presses normally and the old floor only blocks "
        "otherwise-valid small-budget grid cells. Lower it to run them",
    )
    ap.add_argument(
        "--sub-max-tokens",
        type=int,
        default=512,
        help="max_new_tokens per kvzip sub-call; HF greedy decode is ~30-40 tok/s, so the "
        "HTTP-path default of 4096 is a wall-clock hazard in-process",
    )
    ap.add_argument(
        "--sub-device",
        default=None,
        help="GPU for the in-process sub model, e.g. cuda:1 (default: auto-pick the GPU with "
        "the most free memory; either way the choice is preflight-checked against "
        "--sub-min-free-gib before the model loads)",
    )
    ap.add_argument(
        "--sub-min-free-gib",
        type=float,
        default=14.0,
        help="minimum free GPU memory (GiB) required by the preflight check before loading the "
        "sub model (weights + KV headroom; ~14 covers Qwen3-4B bf16 at 34k-token sub-calls)",
    )
    ap.add_argument(
        "--sub-max-context-tokens",
        type=int,
        default=34000,
        help="token-level truncation of the sub-call context (a 131072-char slice of dense "
        "text can exceed 32k tokens)",
    )
    ap.add_argument(
        "--press-min-tokens",
        type=int,
        default=None,
        help="one-arg llm_query prompts below this many tokens skip the press; "
        "defaults to the memory budget's own token capacity (kvzip_backend.py "
        "derives it) rather than a fixed constant",
    )
    ap.add_argument("--out", default=RLM_RESULTS_DIR)
    ap.add_argument("--debug", action="store_true", help="print every RLM step (model reply, code, REPL output) live")
    ap.add_argument(
        "--no-think",
        action="store_true",
        help="disable Qwen3 thinking mode (chat_template_kwargs.enable_thinking=False)",
    )
    args = ap.parse_args()
    if args.sub_backend == "nim":  # deprecated alias; the slug never carried it
        args.sub_backend = "http"

    requested_dataset = args.dataset or args.legacy_dataset
    dataset_name = canonical_dataset_name(requested_dataset)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if dataset_name == "ruler32k" and args.data_dir not in RULER_32K_TASKS:
        ap.error("--dataset ruler32k requires --data-dir with one of: " + ", ".join(RULER_32K_TASKS))
    if dataset_name == "ruler128k" and args.data_dir not in RULER_32K_TASKS:
        ap.error("--dataset ruler128k requires --data-dir with one of: " + ", ".join(RULER_32K_TASKS))
    # Without this, a typo'd or omitted subset surfaces as a HF 404 deep inside
    # _load_loft after the clients are already built.
    if dataset_name == "loft" and args.data_dir not in LOFT_TASKS:
        ap.error("--dataset loft requires --data-dir with one of: " + ", ".join(LOFT_TASKS))
    if dataset_name == "longbench128k" and args.data_dir not in LONGBENCH_128K_SUBSETS:
        ap.error("--dataset longbench128k requires --data-dir with one of: " + ", ".join(LONGBENCH_128K_SUBSETS))

    # --- auto chunk sizing: validate the combination before anything expensive ---
    auto_chunk = args.max_subcall_chars == "auto"
    args.subcall_sizing_mode = "auto" if auto_chunk else "fixed"
    try:
        args.target_compression_ratio = resolve_compression_target(args)
    except ValueError as exc:
        ap.error(str(exc))
    if args.min_subcall_chars < 0:
        ap.error("--min-subcall-chars must be >= 0 (0 disables the floor)")
    if args.search_k < 0:
        ap.error("--search-k must be >= 0 (0 disables retrieval)")
    args.search_ablate = tuple(x.strip() for x in str(args.search_ablate).split(",") if x.strip())
    unknown_ablate = set(args.search_ablate) - SEARCH_ABLATABLE
    if unknown_ablate:
        ap.error(f"--search-ablate: unknown {sorted(unknown_ablate)}; choose from {sorted(SEARCH_ABLATABLE)}")
    if args.search_ablate and not args.search_k:
        ap.error("--search-ablate needs --search-k > 0; without the tool there is nothing to ablate")
    if args.search_k and not 0 <= args.search_overlap < args.search_window:
        ap.error(
            f"--search-overlap ({args.search_overlap}) must satisfy 0 <= overlap < "
            f"--search-window ({args.search_window})"
        )
    if args.chunks_k < 1:
        ap.error("--chunks-k must be >= 1")
    ordered_defaults = (
        args.structured_group_input_tokens == DEFAULT_GROUP_INPUT_TOKENS
        and args.structured_context_tokens == DEFAULT_STRUCTURED_CONTEXT_TOKENS
        and args.structured_reader_max_tokens == DEFAULT_READER_MAX_TOKENS
        and args.structured_reduce_max_tokens == DEFAULT_REDUCE_MAX_TOKENS
        # --chunk-order reads the same organization, so it may name the cache too.
        and (args.organization_dir is None or args.chunk_order is not None)
    )
    if args.structured_order is None and not ordered_defaults:
        ap.error("the ordered-group flags only apply with --structured-order")
    if args.chunk_order is None and args.chunk_call_tokens != DEFAULT_CHUNK_CALL_TOKENS:
        ap.error("--chunk-call-tokens only applies with --chunk-order")
    if args.chunk_order is not None:
        # Each refusal is a way the arm would stop being "the agentic root, handed every
        # chunk in one named order": a second locate tool, another chunk source, or a
        # read the budget cannot price.
        if args.mode != "rlm":
            ap.error("--chunk-order hands the AGENTIC root its chunks; it needs --mode rlm")
        if args.search_k:
            ap.error("--chunk-order gives the root every chunk already ordered; it has no search() (drop --search-k)")
        if args.chunks_source is not None:
            ap.error("--chunk-order hands the root every chunk; it replaces --chunks-source")
        if args.min_subcall_chars:
            ap.error("--chunk-order reads whole chunks; --min-subcall-chars does not apply")
        if args.sub_backend != "http":
            ap.error("--chunk-order needs --sub-backend http (a chunk read is priced on a plain chat request)")
        if args.chunk_call_tokens <= 0:
            ap.error("--chunk-call-tokens must be positive")
        if args.chunk_call_tokens + LLMClient.max_tokens > args.structured_context_tokens:
            ap.error("--chunk-call-tokens plus the sub-call output cap exceeds --structured-context-tokens")
    if args.structured_order is not None:
        # Every refusal below is a way the arm would silently read less than the whole
        # document, or read it differently from the arm it is compared against.
        if args.mode != "structured":
            ap.error("--structured-order needs --mode structured")
        if args.chunks_source is not None:
            ap.error("--structured-order reads every chunk; it replaces --chunks-source")
        if args.sub_backend != "http":
            ap.error("--structured-order needs --sub-backend http (the budget is priced on a plain chat request)")
        if args.max_sub_calls:
            ap.error("--structured-order needs --max-sub-calls 0: a capped map would drop groups")
        if args.structured_hops != 1 or args.structured_map_chars or args.structured_answer_shape != "require":
            ap.error("--structured-order is one retrieval-free pass over whole chunks with the required answer shape")
        if args.structured_map_granularity != "passage" or args.search_k:
            ap.error("--structured-order builds its own groups; drop --structured-map-granularity / --search-k")
        if (
            min(
                args.structured_group_input_tokens, args.structured_reader_max_tokens, args.structured_reduce_max_tokens
            )
            <= 0
        ):
            ap.error("ordered-group token budgets must be positive")
        if args.structured_group_input_tokens + args.structured_reader_max_tokens > args.structured_context_tokens:
            ap.error("--structured-group-input-tokens plus the reader output cap exceeds --structured-context-tokens")
    elif args.mode == "structured":
        if args.chunks_source is None:
            ap.error("--mode structured requires --chunks-source {bm25,file}")
        if args.search_k:
            # search() is a REPL primitive and the structured arm has no REPL, so the
            # flag would change nothing while still stamping `bm25k5` into the run
            # directory -- naming an agentic retrieval depth the run never had.
            ap.error("--mode structured has no REPL for search() to live in; use --chunks-k, not --search-k")
    elif args.mode == "rlm" and args.chunks_source is not None:
        # A FROZEN chunk set read by the AGENTIC root: search() returns the same
        # windows for every query, so what varies between two such runs is only which
        # windows were chosen. `bm25` is not a duplicate of --search-k here -- an
        # adaptive search and a frozen top-k behave very differently in a loop whose
        # main move is to rephrase and search again (measured: 1.5 search calls per
        # example adaptive, 17 frozen) -- and it is the only fair control for a
        # `file` arm, because both sides then pay that cost equally. It also reads
        # exactly the windows --mode structured --chunks-source bm25 reads, so
        # file-vs-bm25 means the same thing on both pipelines.
        if not args.search_k:
            ap.error(
                "--mode rlm --chunks-source file delivers the chunks through search(), which is "
                "only bound with --search-k > 0. Use the same --search-k as the arm you compare against"
            )
    elif args.chunks_source is not None:
        ap.error(f"--chunks-source is only meaningful with --mode structured or --mode rlm, " f"not --mode {args.mode}")
    if args.mode != "structured" and (
        args.structured_map_question != "task"
        or args.structured_map_granularity != "passage"
        or args.structured_answer_shape != "require"
        or args.structured_map_chars
        or args.reduce_finding_chars != DEFAULT_REDUCE_FINDING_CHARS
        or args.structured_hops != 1
    ):
        # They change the map and reduce steps, which only exist in the structured
        # pipeline. Silently ignoring one would stamp an axis into config.yaml that
        # the run never had.
        ap.error(f"the --structured-* axes only apply to --mode structured, not --mode {args.mode}")
    if args.chunks_source == "file" and not args.chunks_file:
        ap.error("--chunks-source file requires --chunks-file")
    if args.chunks_source == "bm25" and args.chunks_file:
        ap.error("--chunks-file is only read by --chunks-source file; bm25 selects in-harness")
    if args.chunks_source == "notes" and not args.notes_file:
        ap.error("--chunks-source notes requires --notes-file")
    if args.notes_file and args.chunks_source != "notes":
        ap.error("--notes-file is only read by --chunks-source notes")
    if args.chunks_source == "notes" and args.mode != "structured":
        ap.error("--chunks-source notes replaces the structured map step; it needs --mode structured")
    if args.chunks_tag and args.chunks_source != "file":
        ap.error("--chunks-tag names the selector behind --chunks-file; it is only read by --chunks-source file")
    if args.ids_file and args.limit is not None:
        # --ids-file is applied BEFORE --limit, so combining them means "the first N
        # of the named rows" -- which reads like a cap but silently changes which
        # experiment ran. Make the caller pick one.
        ap.error("--ids-file and --limit both choose which rows run; pass only one")

    # A retrieval payload is k windows wide, and _cap_subcall truncates whatever
    # exceeds --max-subcall-chars. At the 2000-char default window that caps the
    # REAL depth near k=16, so a k=20 and a k=50 cell would both quietly read ~16
    # windows and the top of the axis would be several cells measuring the same
    # thing. Refuse up front, the way --fixed-chunk refuses a cell the GPU would
    # shrink, rather than producing a plausible number for a depth never run.
    if args.search_k and not auto_chunk:
        needed = args.search_k * args.search_window
        room = args.max_subcall_chars - args.max_subcall_chars // 4
        if needed > room:
            ap.error(
                f"--search-k {args.search_k} x --search-window {args.search_window} = {needed} chars "
                f"of payload, but --max-subcall-chars {args.max_subcall_chars} leaves only {room} for "
                f"it (the question reserves a quarter). The run would silently read about "
                f"{room // args.search_window} windows instead of {args.search_k}. Raise "
                f"--max-subcall-chars to at least {needed * 4 // 3 + 1}."
            )
    # The same refusal for the structured arm, which reaches none of the above:
    # --mode structured rejects --search-k, so args.search_k is 0 and the guard never
    # fires there. At --structured-map-granularity all a k=10 x 8000-char cell hands
    # _cap_subcall 80,000 characters and reads about 24,000 of them, reporting a depth
    # it never ran. Other granularities send one cluster per call, so the binding size
    # is one window -- but a window wider than the cap is the same silent shrink.
    if args.mode == "structured" and not auto_chunk:
        whole_payload = args.structured_map_granularity == "all"
        per_call = args.chunks_k * args.search_window if whole_payload else args.search_window
        room = args.max_subcall_chars - args.max_subcall_chars // 4
        if per_call > room:
            unit = "k x window" if whole_payload else "one window"
            ap.error(
                f"a map call would carry {per_call} chars ({unit} at --structured-map-granularity "
                f"{args.structured_map_granularity}), but --max-subcall-chars {args.max_subcall_chars} "
                f"leaves only {room} for it (the question reserves a quarter). The run would silently "
                f"read about {room} chars per call. Raise --max-subcall-chars to at least "
                f"{per_call * 4 // 3 + 1}."
            )
    # A floor above the ceiling is silently self-cancelling: _expand_subcall widens
    # to the floor and _cap_subcall immediately truncates back to the cap, so the
    # run looks configured for big reads and produces the size it always did.
    if not auto_chunk and args.min_subcall_chars and args.min_subcall_chars > args.max_subcall_chars:
        ap.error(
            f"--min-subcall-chars ({args.min_subcall_chars}) exceeds --max-subcall-chars "
            f"({args.max_subcall_chars}); the floor would be truncated away by the cap"
        )

    # Hash the SORTED SET, not the file: two id files listing the same rows in a
    # different order, or with a duplicate, describe the same experiment and must
    # resume each other.
    example_ids = None
    args.ids_sha256 = None
    if args.ids_file:
        lines = Path(args.ids_file).read_text().splitlines()
        wanted = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        if not wanted:
            ap.error(f"--ids-file {args.ids_file} lists no ids")
        example_ids = set(wanted)
        args.ids_sha256 = hashlib.sha256("\n".join(sorted(example_ids)).encode("utf-8")).hexdigest()

    chunk_set = None
    args.chunks_file_sha256 = None
    if args.chunks_source == "file":
        # Loaded and validated before a single example runs, for the same reason
        # the data pack is checked before the grid loads a model: a malformed chunk
        # file discovered on example 40 has already spent the run.
        try:
            chunk_set = ChunkSet.from_file(
                args.chunks_file,
                chunk_chars=args.search_window,
                overlap=args.search_overlap,
                max_payload_chars=args.chunks_k * args.search_window,
            )
        except ChunkSetError as exc:
            ap.error(str(exc))
        args.chunks_file_sha256 = file_digest(Path(args.chunks_file))
    args.notes_file_sha256 = file_digest(Path(args.notes_file)) if args.notes_file else None

    # WHICH selector wrote the file, as a run-directory component. The file names
    # itself; --chunks-tag only overrides that, and disagreeing with the file is an
    # error rather than a silent rename -- naming an arm something other than what it
    # is, is the exact failure this mechanism exists to prevent.
    #
    # Stamped only when a tag EXISTS, which is what keeps the invariant every other
    # stamp here keeps: no chunk file written before the query-agnostic selectors
    # carries `selector_tag`, so every oracle and hand-made arm already on disk names
    # the directory it always named and its checkpoint stays reachable.
    if chunk_set is not None:
        try:
            file_tag = chunk_set.tag
        except ChunkSetError as exc:
            ap.error(str(exc))
        if args.chunks_tag and file_tag and args.chunks_tag != file_tag:
            ap.error(
                f"--chunks-tag {args.chunks_tag!r} disagrees with the chunk file's own selector_tag "
                f"{file_tag!r}. The run directory would name the arm something other than what wrote it"
            )
        args.chunks_tag = args.chunks_tag or file_tag

        # A map call reads one CLUSTER, and `_cluster_hits` merges windows that touch
        # or overlap. The retrieval grid overlaps by construction, so a selection that
        # tiles the document collapses into ONE passage spanning the whole corpus:
        # _cap_subcall truncates it to the cap and the run reports a full sweep having
        # read the first cap-worth of characters. The guard above cannot see this --
        # it prices one WINDOW for every granularity but `all`.
        if args.mode == "structured" and args.structured_map_granularity == "passage" and not auto_chunk:
            widest = chunk_set.max_cluster_chars()
            room = args.max_subcall_chars - args.max_subcall_chars // 4
            if widest > room:
                ap.error(
                    f"this chunk set's windows merge into a single {widest}-char passage at "
                    f"--structured-map-granularity passage, but --max-subcall-chars "
                    f"{args.max_subcall_chars} leaves only {room} for one call. The run would read "
                    f"the first {room} chars in ONE map call and report a sweep of "
                    f"{chunk_set.max_windows} windows. Pass --structured-map-granularity window."
                )

        # Past --max-sub-calls, llm_query returns a NOTICE STRING instead of calling
        # the model -- and that string does not start with the "nothing relevant"
        # sentinel, so the findings filter counts every one of them as a finding and
        # pastes it into the reduce prompt as evidence. A truncated sweep therefore
        # reports MORE findings than a healthy one and passes every gate that looks
        # for a starved cell. Refuse it instead.
        if args.mode == "structured" and args.max_sub_calls:
            wanted = chunk_set.max_windows
            if wanted > args.max_sub_calls:
                ap.error(
                    f"this chunk set selects up to {wanted} windows per example but --max-sub-calls "
                    f"is {args.max_sub_calls}. Calls past the budget return a notice string that the "
                    f"findings filter counts as a FINDING, so the run would read {args.max_sub_calls} "
                    f"windows and report {wanted} of them as read. Pass --max-sub-calls 0 to disable "
                    f"the budget, or raise it above {wanted}."
                )

    # The notes index cannot be verified before the first example, because the check
    # is against the DOCUMENT and the loader yields it. It is loaded on the first row
    # and refuses there -- one example spent rather than the whole run, which is the
    # same bargain the chunk-set load makes with the file it can read up front.
    notes_state: dict = {"notes": None, "index": None}

    def notes_for(example: dict) -> dict:
        if notes_state["notes"] is None:
            notes = window_notes.load_notes(
                args.notes_file, example["context"], args.search_window, args.search_overlap
            )
            notes_state["notes"] = notes
            if args.notes_select == "bm25":
                spans = chunkset.window_spans(len(example["context"]), args.search_window, args.search_overlap)
                notes_state["index"] = window_notes.NotesIndex(notes, spans)
        return notes_state["notes"]

    def chunks_for(example: dict) -> list:
        """The chunks this example is read through: the structured arm's map step, or
        what `search()` returns to an agentic root handed a chunk set.

        Both sources return `Hit`s on the same window grid, so the two differ in
        WHICH windows were chosen and in nothing else -- which is the whole point
        of having a cheap-selector control at all.
        """
        if args.chunks_source == "notes":
            document = example["context"]
            notes = notes_for(example)
            spans = chunkset.window_spans(len(document), args.search_window, args.search_overlap)
            if args.notes_select == "bm25":
                # Query-aware LOOKUP over a query-agnostic INDEX. The query is the bare
                # question for the same reason retrieval uses it: LOFT's wrapper is a
                # page of boilerplate present in every note.
                chosen = sorted(notes_state["index"].search(bare_query(example["question"]), args.notes_k))
            else:
                chosen = sorted(notes)
            return [
                Hit(rank=rank, start=spans[w][0], end=spans[w][1], score=0.0, text=document[spans[w][0] : spans[w][1]])
                for rank, w in enumerate(chosen, start=1)
                if w < len(spans)
            ]
        if chunk_set is not None:
            return chunk_set.hits_for(example["id"], example["context"])
        index = retrieval.get_index(example["context"], args.search_window, args.search_overlap)
        return index.search(chunks_query(example["question"]), args.chunks_k)

    def chunks_query(question: str) -> str:
        """What BM25 is actually asked. LOFT's `task` is not a question.

        `bare_query` was written for the MAP step, where LOFT's page of citation and
        formatting instructions derails a 4B reading one passage. The retriever was
        handed the same wrapper and nobody applied the same reasoning -- so the query
        BM25 scored on was dominated by "documents", "answer", "print", "title",
        "format", "list", terms that occur in every window of the corpus.

        Measured offline over 55 rows per subset, recall of the gold answer string:

            task           r@2          r@5          r@10
            nq         0.00 -> 0.95  0.65 -> 0.96  0.80 -> 0.96
            hotpotqa   0.05 -> 0.71  0.64 -> 0.89  0.82 -> 0.91
            musique    0.11 -> 0.38  0.29 -> 0.58  0.53 -> 0.78
            qampari    0.00 -> 0.53* 0.53 -> 0.98  0.95 -> 1.00

        At k=2 it is total: the top two windows never contained the answer on nq or
        qampari. That is not a property of BM25 at depth 2, it is the boilerplate
        outranking the question.

        It also confounds the campaign's headline. The agentic `--search-k` arm lets
        the MODEL write its query and gets a clean one; the structured arm did not,
        so "structure loses to the BM25 agentic arm" was partly measuring which of
        them was allowed to ask a sensible question.

        `bare_query` is conservative -- a task with no `query:` line comes back
        untouched -- so benchmarks whose task IS the question are unaffected.
        """
        return bare_query(question) if args.chunks_query == "bare" else question

    def retrieve_for(example: dict):
        """BM25 over this example's window grid, for `--structured-hops 2`.

        Always BM25, even under `--chunks-source file`: a chunk file names round 1
        only, and a selector that never saw the first round's findings cannot have
        chosen round 2. Making the follow-up the pipeline's own keeps `file` and
        `bm25` differing in exactly the windows they start from.
        """
        index = retrieval.get_index(example["context"], args.search_window, args.search_overlap)
        return lambda query, k: index.search(query, k)

    organizations: dict = {}

    def groups_for(document: str, task: str):
        """Organize the document (question-free, cached), then pack it for this question.

        Organization sees only the document. The QUESTION enters at packing, and only
        as a length: the reader message carries it, so the budget must count it.
        """
        digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
        if digest not in organizations:
            artifact, _, cache_hit = organize(document, args.structured_order, args.organization_dir)
            organizations[digest] = (artifact, cache_hit)
        artifact, cache_hit = organizations[digest]
        started = time.process_time()
        groups = pack(
            document,
            ordered_hits(document, artifact),
            rlm.structured_question(task),
            SUB_SYSTEM_PROMPT,
            ordered_counter,
            args.structured_group_input_tokens,
        )
        stats = {
            "organization_sha256": digest,
            "organization_cache_hit": cache_hit,
            "organization_build_cpu_seconds": artifact["build_cpu_seconds"],
            "organization_chunks": len(artifact["spans"]),
            "organization_edges": len(artifact["edges"]),
            "organization_forest_edges": len(artifact["forest"]),
            "packing_cpu_seconds": time.process_time() - started,
            "split_chunk_pieces": sum(len(g.hits) for g in groups) - len(artifact["spans"]),
        }
        return groups, stats

    if (args.structured_order or args.chunk_order == "bm25-graph") and args.organization_dir is None:
        args.organization_dir = str(Path(args.out) / "organization")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["vanilla", "rlm"] if args.mode == "both" else [args.mode]

    root = LLMClient(model=args.root_model, base_url=args.base_url)
    sub: Any
    if args.sub_backend == "kvzip":
        if modes == ["vanilla"]:
            ap.error("--sub-backend kvzip only affects RLM sub-calls; use --mode rlm (or both)")
        # Imported lazily: the http path must stay importable in a venv without
        # torch or kvpress's deps.
        from .kvzip_backend import KVzipSubClient

        sub = KVzipSubClient(
            model=args.sub_model,
            press_name=args.press,
            memory_budget=args.memory_budget,
            memory_budget_unit=args.memory_budget_unit,
            device=args.sub_device,
            max_new_tokens=args.sub_max_tokens,
            max_context_tokens=args.sub_max_context_tokens,
            press_min_tokens=args.press_min_tokens,
            min_free_gib=args.sub_min_free_gib,
        )
    else:
        sub = LLMClient(model=args.sub_model, base_url=args.base_url)
    if args.no_think:
        eb = {"chat_template_kwargs": {"enable_thinking": False}}
        root.extra_body = sub.extra_body = eb

    # Resolve the advertised chunk size ONCE, before the RLM is built: it is
    # rendered into the root's system prompt, so it must be identical for every
    # example of the run. Needs a real document for the chars-per-token
    # calibration, and examples only load inside the mode loop below -- so pull
    # one here. After this block args.max_subcall_chars is always an int.
    subcall_sizing = None
    if auto_chunk:
        sample = next(iter(load_examples(requested_dataset, args.data_dir, 1, args.split)))
        subcall_sizing = sub.plan_subcall_chunk(
            document=sample["context"],
            target_compression_ratio=args.target_compression_ratio,
            cli_max_context_tokens=args.sub_max_context_tokens,
            reserve_tokens=args.subcall_reserve_tokens,
            min_tokens=args.subcall_min_tokens,
            char_overshoot=(FIXED_CHUNK_CHAR_OVERSHOOT if args.fixed_chunk else 1.0),
            require_budget_binding=args.fixed_chunk,
            # In fixed mode this value is the OUTPUT of planning (N), not an
            # input cap. Applying the legacy 34K ceiling here would silently
            # collapse every larger grid cell before N could be installed.
            apply_cli_context_cap=not args.fixed_chunk,
        )
        if args.fixed_chunk:
            args.min_subcall_chars = subcall_sizing.chars
            args.max_subcall_chars = fixed_chunk_char_cap(args.min_subcall_chars)
            args.sub_max_context_tokens = subcall_sizing.tokens
            sub.max_context_tokens = subcall_sizing.tokens
        else:
            args.max_subcall_chars = subcall_sizing.chars
        # A target ratio is only actually delivered when the KV BUDGET is what
        # limits the chunk. If the sub model's window or --sub-max-context-tokens
        # binds first, the run silently measures a different ratio than the one
        # asked for, and only runtime.subcall_sizing_binding records it.
        if not args.fixed_chunk and subcall_sizing.binding != "budget":
            print(
                f"!! --target-compression-ratio {args.target_compression_ratio:g} is NOT what this run "
                f"will deliver: the chunk size is limited by {subcall_sizing.binding!r} "
                f"({subcall_sizing.tokens} tokens), not by the KV budget. Raise "
                "--sub-max-context-tokens (and check the sub model's window) to hit the "
                "requested ratio."
            )
        # Same self-cancelling pair as the fixed path, but only checkable here: on
        # the auto path the cap is not a number until the sizing has run.
        if args.min_subcall_chars > args.max_subcall_chars:
            raise SystemExit(
                f"--min-subcall-chars ({args.min_subcall_chars}) exceeds the auto-sized chunk "
                f"({args.max_subcall_chars} chars, bound by {subcall_sizing.binding!r}); the floor "
                "would be truncated away by the cap. Lower the floor or raise the budget."
            )

    ordered_counter = None
    if args.structured_order:
        # One served model answers both roles; the caps are what differ.
        sub.max_tokens = args.structured_reader_max_tokens
        root.max_tokens = args.structured_reduce_max_tokens
        ordered_counter = ModelTokens(args.sub_model)
    if args.chunk_order:
        # The agentic caps stay the base RLM's; only the chunk reads are priced exactly.
        ordered_counter = ModelTokens(args.sub_model)

    budget = None
    if args.max_context_tokens is not None:
        budget = MemoryBudget(
            max_context_tokens=args.max_context_tokens,
            keep_recent_turns=args.keep_recent_turns,
        )
    scratchpad = Scratchpad(max_notes_tokens=args.max_notes_tokens) if args.scratchpad else None
    rlm = RLM(
        root_client=root,
        sub_client=sub,
        max_steps=args.max_steps,
        budget=budget,
        scratchpad=scratchpad,
        # `or None` is what makes each guard independently switchable off from the
        # CLI: argparse cannot pass None for a float/int flag, so 0 is the off
        # switch and is translated here.
        exec_timeout=args.exec_timeout or None,
        run_timeout=args.run_timeout or None,
        max_sub_calls=args.max_sub_calls or None,
        max_subcall_chars=args.max_subcall_chars,
        min_subcall_chars=args.min_subcall_chars,
        # In-process sub-calls can take minutes each; without an in-call check a
        # single code cell looping over slices sails past the per-step deadline.
        subcall_deadline_check=(args.sub_backend == "kvzip"),
        min_sub_calls_before_abstain=args.min_sub_calls_before_abstain,
        max_unproductive_steps=args.max_unproductive_steps,
        max_error_steps=args.max_error_steps,
        search_k=args.search_k,
        search_window_chars=args.search_window,
        search_overlap_chars=args.search_overlap,
        max_search_calls=args.max_search_calls,
        search_ablate=args.search_ablate,
        structured_map_question=args.structured_map_question,
        structured_map_granularity=args.structured_map_granularity,
        structured_answer_shape=args.structured_answer_shape,
        structured_map_chars=args.structured_map_chars,
        reduce_finding_chars=args.reduce_finding_chars,
        structured_hops=args.structured_hops,
        ordered_token_counter=ordered_counter,
        structured_group_input_tokens=args.structured_group_input_tokens,
        structured_context_tokens=args.structured_context_tokens,
        structured_answer_list=(
            ""
            if args.answer_list_format == "off"
            else answer_list_example(
                canonical_dataset_name(args.dataset or args.legacy_dataset),
                args.data_dir,
                args.mode,
                ordered_rlm=bool(args.chunk_order),
            )
        ),
        chunk_order=args.chunk_order,
        chunk_call_tokens=args.chunk_call_tokens,
    )

    for mode in modes:
        run_dir = out_dir / "__".join(build_run_dir_components(args, mode, scratchpad))
        run_dir.mkdir(parents=True, exist_ok=True)
        res_path = run_dir / "checkpoint.jsonl"
        tdir = run_dir / "transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        done = load_done(res_path)
        run_config = build_run_config(
            args,
            mode,
            dataset_name,
            budget,
            scratchpad,
            subcall_sizing,
            # Resolved on the constructed client: --press-min-tokens defaults to
            # None and kvzip_backend derives it from the memory budget, so the
            # effective value only exists on `sub`, not on `args`.
            getattr(sub, "press_min_tokens", None),
            ",".join(chunk_set.selectors) if chunk_set is not None else None,
        )
        if done:
            # The run-dir name cannot carry every result-affecting knob, so a resume
            # is checked against the config the first run wrote. Without this, a
            # second run with a different --limit (or --max-steps, or sub model)
            # appends to the same checkpoint and the two silently merge into one set
            # of numbers -- exactly what the directory naming exists to prevent.
            conflicts = resume_conflicts(run_dir, run_config)
            if conflicts:
                ap.error(
                    f"{run_dir} holds {len(done)} examples from a different configuration:\n  "
                    + "\n  ".join(conflicts)
                    + "\nResuming would merge two experiments. Re-run with the original "
                    "settings, or point --out somewhere else."
                )
        if auto_chunk and done:
            # Auto sizing reads live GPU free memory, which a co-tenant process
            # changes -- so a resumed auto run can resolve to a different size and
            # merge two differently-sized experiments into one checkpoint. The
            # run-dir name cannot catch this (it deliberately carries only the
            # ratio, so auto runs remain resumable at all), so compare explicitly.
            prior = _prior_resolved_chars(run_dir)
            if prior and abs(args.max_subcall_chars - prior) > 0.02 * prior:
                ap.error(
                    f"{run_dir} holds {len(done)} examples sized at {prior} chars, but this run "
                    f"resolved to {args.max_subcall_chars}. Resuming would merge two different "
                    "configurations. Free the GPU and retry, or pass "
                    f"--max-subcall-chars {prior} to reproduce the original size."
                )
        # Written BEFORE the examples run, not only at the end: a run interrupted
        # part-way through would otherwise leave a checkpoint with no config beside
        # it, and the resume checks would have nothing to compare against. It must
        # come AFTER both of those checks, which read the PREVIOUS run's config --
        # writing it any earlier makes each check compare this run against itself.
        (run_dir / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))
        print(f"== {dataset_name}/{args.data_dir or 'default'} / {mode} " f"-> {run_dir} ({len(done)} already done)")

        n, correct = 0, 0
        with open(res_path, "a") as fout:
            for ex in load_examples(
                requested_dataset,
                args.data_dir,
                args.limit,
                args.split,
                sample_fraction=args.sample_fraction,
                sample_seed=args.sample_seed,
                ids=example_ids,
            ):
                if ex["id"] in done:
                    continue
                t0 = time.time()
                tok0 = root.usage.total_tokens + sub.usage.total_tokens
                prompt0 = root.usage.prompt_tokens + sub.usage.prompt_tokens
                completion0 = root.usage.completion_tokens + sub.usage.completion_tokens
                sub0 = sub.usage.calls
                record = {
                    "id": ex["id"],
                    "dataset": dataset_name,
                    "task": ex.get("task") or args.data_dir or dataset_name,
                    "mode": mode,
                    "answers": ex["answers"],
                    "answer": ex.get("scoring", {}).get("answer", ex["answers"]),
                    # LOFT's scorer reads this column to know which cue to look for
                    # and falls back to a hardcoded "Final Answer: " when it is
                    # absent. It was being appended to the question below but never
                    # recorded, so the scorer silently used the wrong cue.
                    "answer_prefix": ex.get("answer_prefix") or "",
                }
                record.update(ex.get("scoring", {}))
                question = ex["question"]
                # The benchmark's output contract goes to BOTH arms, as a format
                # instruction rather than stacked onto the question a second time:
                # the RLM renders it into its system prompt and vanilla appends it
                # to the document prompt. Scoring is string-matching, so an arm
                # asked for a different answer SHAPE is measured as less accurate.
                answer_format = ex.get("answer_prefix") or None
                vstats: dict = {}
                chunk_view = None
                try:
                    if mode == "vanilla":
                        pred = vanilla_answer(
                            root,
                            ex["context"],
                            question,
                            char_limit=args.vanilla_char_limit,
                            answer_format=answer_format,
                            max_prompt_tokens=args.vanilla_max_prompt_tokens,
                            stats=vstats,
                        )
                        record.update(pred=pred, steps=1, finished=True, end_reason="")
                    else:
                        if mode == "structured" and args.structured_order:
                            groups, organization_stats = groups_for(ex["context"], question)
                            r = rlm.run_structured(
                                ex["context"], question, [], answer_format=answer_format, ordered_groups=groups
                            )
                            r.metrics.update(organization_stats)
                        elif mode == "structured":
                            r = rlm.run_structured(
                                ex["context"],
                                question,
                                chunks_for(ex),
                                answer_format=answer_format,
                                retrieve=retrieve_for(ex) if args.structured_hops > 1 else None,
                                window_notes=notes_for(ex) if args.chunks_source == "notes" else None,
                            )
                        elif args.chunk_order:
                            chunk_view, view_stats = build_chunk_view(
                                ex["context"],
                                question,
                                args.chunk_order,
                                dataset_name,
                                args.organization_dir,
                                organizations,
                            )
                            r = rlm.run(ex["context"], question, answer_format=answer_format, ordered_chunks=chunk_view)
                            r.metrics.update(view_stats)
                        else:
                            r = rlm.run(
                                ex["context"],
                                question,
                                answer_format=answer_format,
                                chunks=chunks_for(ex) if args.chunks_source else None,
                            )
                        if hasattr(sub, "pop_example_stats"):
                            r.metrics["sub_kv"] = sub.pop_example_stats() or None
                        record.update(
                            pred=r.answer,
                            steps=r.steps,
                            finished=r.finished,
                            end_reason=r.end_reason,
                            metrics=r.metrics,
                        )
                        # Always save the full transcript for post-mortems:
                        with open(tdir / f"{ex['id']}.json", "w") as tf:
                            json.dump(
                                {
                                    "question": question,
                                    "answers": ex["answers"],
                                    "pred": r.answer,
                                    "end_reason": r.end_reason,
                                    "metrics": r.metrics,
                                    "transcript": r.transcript,
                                },
                                tf,
                                indent=2,
                            )
                        if args.debug:
                            for t in r.transcript:
                                print(f"\n--- step {t['step']} ---")
                                print("MODEL REPLY:\n" + (t.get("reply") or "")[:2000])
                                print("EXECUTED CODE:\n" + (t.get("code") or "<none>"))
                                print("REPL OUTPUT:\n" + t["observation"][:2000])
                            print(f"\nEND: reason={r.end_reason} pred={r.answer!r}\n")
                except Exception as e:
                    record.update(
                        pred=None, error=f"{type(e).__name__}: {e}", steps=0, finished=False, end_reason="exception"
                    )
                # Was the gold even inside the prompt we sent? Without this a
                # truncated vanilla arm's score is indistinguishable from a model
                # failure -- and unlike KVPress, which compresses the full context,
                # this arm can simply not have seen the answer. Merged outside the
                # try: vanilla_answer fills the dict before re-raising, and the
                # errored record is exactly the one a triage pass needs it on.
                record.update(vstats)
                # Check for answer strings in returned windows after the example;
                # gold answers never reach the model. This loose diagnostic does
                # not establish whether retrieval supplied sufficient evidence.
                spans = (
                    (record.get("metrics") or {}).pop("search_retrieved_spans", None)
                    if mode in ("rlm", "structured")
                    else None
                )
                if spans:
                    retrieved = "".join(ex["context"][start:end] for start, end in spans)
                    record["gold_in_retrieved"] = is_correct(retrieved, ex["answers"])
                    # The same loose test, applied one window at a time. Raising k adds
                    # SIGNAL and DISTRACTORS at once, so an undecomposed k-curve cannot
                    # say which drives its turnover; these two counts are what let the
                    # sweep separate them afterwards. Loose in exactly the way
                    # gold_in_retrieved is (c448c97): a matching string does not
                    # establish the window carries sufficient evidence.
                    gold_windows = sum(1 for start, end in spans if is_correct(ex["context"][start:end], ex["answers"]))
                    record["gold_windows_in_chunks"] = gold_windows
                    record["nongold_windows_in_chunks"] = len(spans) - gold_windows
                # Where the gold sat in the ordered view and whether the root read it --
                # the same loose string test, applied after the example ends.
                if chunk_view is not None and record.get("metrics"):
                    ranks_read = record["metrics"].get("chunk_ranks_read") or []
                    record.update(chunk_diagnostics(chunk_view, ranks_read, ex["answers"]))
                record["correct"] = is_correct(record.get("pred"), ex["answers"])
                record["latency_s"] = round(time.time() - t0, 2)
                record["tokens"] = root.usage.total_tokens + sub.usage.total_tokens - tok0
                # The same server-counted usage, split into input and output.
                record["prompt_tokens"] = root.usage.prompt_tokens + sub.usage.prompt_tokens - prompt0
                record["completion_tokens"] = root.usage.completion_tokens + sub.usage.completion_tokens - completion0
                record["sub_calls"] = sub.usage.calls - sub0
                record["ctx_chars"] = len(ex["context"])
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                n += 1
                correct += record["correct"]
                print(
                    f"  [{ex['id']}] correct={record['correct']} "
                    f"steps={record['steps']} sub_calls={record['sub_calls']} "
                    f"end={record.get('end_reason', '')} tokens={record['tokens']} "
                    f"t={record['latency_s']}s"
                )
        if n:
            print(f"== {mode}: {correct}/{n} correct ({100*correct/n:.1f}%)")
        metrics = write_run_artifacts(
            res_path,
            run_dir,
            dataset_name,
            run_config,
        )
        print(f"== shared artifacts written to {run_dir}; metrics={metrics}")
    print(
        f"Total API calls: root={root.usage.calls}, sub={sub.usage.calls}; "
        f"tokens: {root.usage.total_tokens + sub.usage.total_tokens}"
    )


def build_run_config(
    args: argparse.Namespace,
    mode: str,
    dataset_name: str,
    budget: MemoryBudget | None,
    scratchpad: Scratchpad | None,
    subcall_sizing: Any,
    resolved_press_min_tokens: int | None = None,
    chunk_selector: str | None = None,
) -> dict:
    """The run's full settings, as written to config.yaml.

    Built before the examples run rather than after, because it is what a resume
    is checked against (see resume_conflicts).
    """
    return {
        "backend": "rlm",
        "dataset": dataset_name,
        "data_dir": args.data_dir,
        # `model` duplicates root_model under the key KVPress's config.yaml
        # uses, so a comparison joins the two backends on (dataset, data_dir,
        # model) without special-casing either one.
        "model": args.root_model,
        "mode": mode,
        "root_model": args.root_model,
        "sub_model": args.sub_model,
        "base_url": args.base_url,
        "limit": args.limit,
        "split": args.split,
        "sample_fraction": args.sample_fraction,
        # Only meaningful alongside a fraction -- recorded unconditionally, two
        # unsampled runs that merely passed different --sample-seed values would
        # refuse to resume each other over a knob neither of them used.
        "sample_seed": args.sample_seed if args.sample_fraction is not None else None,
        "max_steps": args.max_steps,
        "vanilla_char_limit": args.vanilla_char_limit,
        "vanilla_max_prompt_tokens": args.vanilla_max_prompt_tokens,
        "memory_budget": asdict(budget) if budget else None,
        "scratchpad": asdict(scratchpad) if scratchpad else None,
        "exec_timeout": args.exec_timeout or None,
        "run_timeout": args.run_timeout or None,
        "max_sub_calls": args.max_sub_calls or None,
        "min_sub_calls_before_abstain": args.min_sub_calls_before_abstain,
        "max_unproductive_steps": args.max_unproductive_steps,
        "max_error_steps": args.max_error_steps,
        "no_think": bool(args.no_think),
        "sub_backend": args.sub_backend,
        "press": args.press if args.sub_backend == "kvzip" else None,
        # sub_kv_-prefixed: "memory_budget" above is the ROOT's own eviction
        # budget (MemoryBudget), an unrelated concept -- this is the KV
        # compression budget applied per llm_query sub-call.
        "sub_kv_memory_budget": args.memory_budget if args.sub_backend == "kvzip" else None,
        "sub_kv_memory_budget_unit": args.memory_budget_unit if args.sub_backend == "kvzip" else None,
        "max_subcall_chars": args.max_subcall_chars,
        "min_subcall_chars": args.min_subcall_chars,
        # How that number was arrived at. `subcall_sizing` carries every cap
        # considered and which one bound, so a surprising size can be explained
        # without re-running; `max_subcall_chars_resolved` is what the resume
        # check compares against.
        "subcall_sizing_mode": args.subcall_sizing_mode,
        "target_compression_ratio": args.target_compression_ratio,
        "compression_factor": args.compression_factor,
        "fixed_chunk": bool(args.fixed_chunk),
        "max_subcall_chars_resolved": args.max_subcall_chars,
        "subcall_sizing": asdict(subcall_sizing) if subcall_sizing else None,
        "sub_max_tokens": args.sub_max_tokens if args.sub_backend == "kvzip" else None,
        "sub_max_context_tokens": args.sub_max_context_tokens if args.sub_backend == "kvzip" else None,
        # The client's resolved value, not args.press_min_tokens: with the flag
        # left at its None default, kvzip_backend derives this from the memory
        # budget, so args would serialise None while the run used a real floor.
        "press_min_tokens": resolved_press_min_tokens if args.sub_backend == "kvzip" else None,
        "search_k": args.search_k,
        "search_window_chars": args.search_window,
        "search_overlap_chars": args.search_overlap,
        "max_search_calls": args.max_search_calls,
        "search_ablate": ",".join(args.search_ablate),
        # The structured arm. `selector` is provenance, not a knob: it names what
        # chose the chunks, which is the one thing about this arm that cannot be
        # reconstructed from the flags.
        "chunks_source": args.chunks_source,
        "chunks_k": args.chunks_k if args.chunks_source else None,
        "chunks_file": args.chunks_file,
        "chunks_file_sha256": getattr(args, "chunks_file_sha256", None),
        "chunks_tag": getattr(args, "chunks_tag", None),
        "notes_file": getattr(args, "notes_file", None),
        "notes_file_sha256": getattr(args, "notes_file_sha256", None),
        "notes_select": getattr(args, "notes_select", None) if args.chunks_source == "notes" else None,
        "notes_k": getattr(args, "notes_k", None) if args.chunks_source == "notes" else None,
        # The three components of the revision-2 treatment, recorded whatever their
        # value: `structured_revision` names the CODE generation, and these name the
        # configuration it ran in. A run that leaves them alone is revision 2 exactly.
        **(
            {
                "structured_map_question": args.structured_map_question,
                "structured_map_granularity": args.structured_map_granularity,
                "structured_answer_shape": args.structured_answer_shape,
                # getattr, like chunks_file_sha256 above: build_run_config is called
                # from tests with hand-built namespaces, and a knob added later must
                # not make an older caller raise instead of recording its default.
                "structured_map_chars": getattr(args, "structured_map_chars", 0),
                "reduce_finding_chars": getattr(args, "reduce_finding_chars", DEFAULT_REDUCE_FINDING_CHARS),
                "structured_hops": getattr(args, "structured_hops", 1),
                "answer_list_format": getattr(args, "answer_list_format", "auto"),
                "chunks_query": getattr(args, "chunks_query", "task"),
            }
            if mode == "structured"
            else {}
        ),
        # The ordered whole-document arms. Only present when the order is set, so
        # every config.yaml written before them is unchanged.
        **(
            {
                "structured_order": args.structured_order,
                "structured_group_input_tokens": args.structured_group_input_tokens,
                "structured_context_tokens": args.structured_context_tokens,
                "structured_reader_max_tokens": args.structured_reader_max_tokens,
                "structured_reduce_max_tokens": args.structured_reduce_max_tokens,
                "organization_config": dict(ORGANIZATION_CONFIG),
                "organization_dir": str(getattr(args, "organization_dir", None)),
            }
            if mode == "structured" and getattr(args, "structured_order", None)
            else {}
        ),
        # The ordered-chunk agentic arms. Only present when the order is set, so every
        # agentic config.yaml written before them is unchanged.
        **(
            {
                "chunk_order": args.chunk_order,
                "chunk_order_revision": CHUNK_ORDER_REVISION,
                "chunk_reads_before_abstain": CHUNK_READS_BEFORE_ABSTAIN,
                "chunk_call_tokens": getattr(args, "chunk_call_tokens", DEFAULT_CHUNK_CALL_TOKENS),
                "chunk_order_query": "bare" if args.chunk_order == "bm25-query" else None,
                "structured_context_tokens": getattr(
                    args, "structured_context_tokens", DEFAULT_STRUCTURED_CONTEXT_TOKENS
                ),
                "organization_config": dict(ORGANIZATION_CONFIG),
                "organization_dir": str(getattr(args, "organization_dir", None)),
                "answer_list_format": getattr(args, "answer_list_format", "auto"),
            }
            if mode == "rlm" and getattr(args, "chunk_order", None)
            else {}
        ),
        "chunks_selector": chunk_selector,
        "ids_file": args.ids_file,
        "ids_sha256": getattr(args, "ids_sha256", None),
        # Only zero-overlap RLM retrieval changed; preserve other runs' configs.
        **({"search_index_revision": 2} if mode == "rlm" and args.search_k > 0 and args.search_overlap == 0 else {}),
        **({"structured_revision": 2} if mode == "structured" else {}),
        # Which semantics produced document_coverage_fraction. Before this, spans
        # were derived AFTER the dense fold set chunk=None, so coverage was
        # structurally 0.0 on every --sub-backend http run; it is now resolved
        # before the fold. A coverage number must not be compared across that line.
        "coverage_attribution": "spans_v2",
    }


if __name__ == "__main__":
    main()
