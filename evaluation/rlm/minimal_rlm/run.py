# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one experiment config over its tasks and write the standard result files.

For each task, under <output_dir>/<name>/<task>/:

    config.yaml         the resolved config (a later run must match it to resume)
    checkpoint.jsonl    one line per answered question, appended as it finishes
    transcripts/        the root's turns per question
    predictions.csv     one row per question
    metrics.json        the official score, plus what the run did (steps, endings, reads)

Re-running the same config resumes: questions already answered are skipped.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pandas as pd
import yaml

from evaluation.benchmarks.registry import SCORER_REGISTRY
from evaluation.benchmarks.results import score_prediction_frame

from ..client import LLMClient
from ..datasets import load_examples
from .config import ExperimentConfig, config_to_dict
from .loop import RLM, Limits, Result
from .ordering import build_chunks, chunk_chars
from .readers import Reader, make_reader
from .stats import summarize

LIST_TASKS = ("qampari", "quest")  # LOFT tasks scored by coverage over a list


def make_root(config: ExperimentConfig) -> LLMClient:
    root = LLMClient(model=config.models.root, base_url=config.models.base_url)
    if not config.models.thinking:
        root.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    return root


def make_limits(config: ExperimentConfig) -> Limits:
    """Everything the loop enforces and records, from a config."""
    rlm = config.rlm
    return Limits(
        max_steps=rlm.max_steps,
        max_sub_calls=rlm.max_sub_calls,
        exec_timeout_s=rlm.exec_timeout_s,
        run_timeout_s=rlm.run_timeout_s,
        output_chars=rlm.output_chars,
        read_tokens=config.read_tokens,
        budget_tokens=config.compression.budget_tokens,
        factors=config.compression.factors,
        default_factor=config.compression.factor,
        read_limits={f: config.read_limit(f) for f in config.compression.factors},
        offload=config.compression.offload,
        gpu_token_cost=config.cost.gpu_token,
        offloaded_token_cost=config.cost.offloaded_token,
        show_cost=config.cost.show_root,
    )


def make_rlm(config: ExperimentConfig, root: Any, reader: Reader) -> RLM:
    return RLM(root, reader, make_limits(config), config.ordering)


def loose_match(pred: Optional[str], answers: Iterable[str]) -> bool:
    """Progress indicator only: does any gold answer appear in the prediction?"""
    text = " ".join(str(pred or "").lower().split())
    return any(" ".join(str(a).lower().split()) in text for a in answers if str(a).strip())


def _answered(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # torn last line of a killed run
            continue
        if not record.get("error"):
            done.add(record["id"])
    return done


def _claim(run_dir: Path, config: ExperimentConfig) -> None:
    """Write config.yaml, or refuse if the directory holds a different experiment."""
    path = run_dir / "config.yaml"
    resolved = config_to_dict(config)
    if path.exists():
        prior = yaml.safe_load(path.read_text())
        if prior != resolved:
            changed = sorted(k for k in set(prior) | set(resolved) if prior.get(k) != resolved.get(k))
            raise SystemExit(
                f"{run_dir} was written by a different config (differs in {changed}). "
                "Give this experiment a new `name` instead of resuming into it."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(resolved, sort_keys=False))


def record_for(example: dict, task: str, result: Result) -> dict:
    record = {
        "id": example["id"],
        "task": task,
        "answers": example["answers"],
        "answer": example.get("scoring", {}).get("answer", example["answers"]),
        "answer_prefix": example.get("answer_prefix") or "",
        **example.get("scoring", {}),
        "pred": result.answer,
        "steps": result.steps,
        "end_reason": result.end_reason,
        "finished": result.end_reason == "final",
        "latency_s": round(result.seconds, 3),
        "correct": loose_match(result.answer, example["answers"]),
        "error": None,
        "reads": summarize(result.calls),
        "calls": [c.to_dict() for c in result.calls],
    }
    return record


def task_metrics(dataset: str, records: list[dict]) -> dict:
    frame = pd.DataFrame(records).drop_duplicates("id", keep="last")
    scored = frame[frame["error"].isna()]
    metrics: dict = {}
    if dataset in SCORER_REGISTRY and len(scored):
        metrics = dict(score_prediction_frame(dataset, scored.rename(columns={"pred": "predicted_answer"})))
    reads = [r for r in scored.get("reads", []) if isinstance(r, dict)]
    metrics["runtime"] = {
        "examples": len(frame),
        "scored": len(scored),
        "errors": int(frame["error"].notna().sum()),
        "progress_match_loose": float(scored["correct"].mean()) if len(scored) else 0.0,
        "steps_mean": float(scored["steps"].mean()) if len(scored) else 0.0,
        "end_reasons": dict(Counter(scored["end_reason"])),
        "sub_calls_mean": sum(r["sub_calls"] for r in reads) / len(reads) if reads else 0.0,
        "refused_calls": sum(r["refused_calls"] for r in reads),
        "read_fill_mean": (sum(r.get("read_fill_mean", 0.0) for r in reads) / len(reads) if reads else 0.0),
        # Cost and the decisions behind it (see stats.summarize): what a question cost on
        # average, how much KV went to CPU, which factors were used, and how often a read
        # was compressed although it would have fit uncompressed.
        "cost_mean": sum(r.get("cost", 0.0) for r in reads) / len(reads) if reads else 0.0,
        "offloaded_tokens_mean": sum(r.get("offloaded_tokens", 0) for r in reads) / len(reads) if reads else 0.0,
        "factors": dict(sum((Counter(r.get("factors", {})) for r in reads), Counter())),
        "compressed_unneeded": sum(r.get("compressed_unneeded", 0) for r in reads),
        "contributing_calls": sum(r.get("contributing_calls", 0) for r in reads),
    }
    return metrics


def run_task(
    config: ExperimentConfig,
    task: str,
    rlm: RLM,
    examples: Iterable[dict],
    log: Callable[[str], None] = print,
) -> dict:
    run_dir = Path(config.output_dir) / config.name / task
    _claim(run_dir, config)
    checkpoint = run_dir / "checkpoint.jsonl"
    transcripts = run_dir / "transcripts"
    transcripts.mkdir(exist_ok=True)
    done = _answered(checkpoint)
    organization_dir = config.organization_dir or str(Path(config.output_dir) / "organization")
    dataset = config.data.dataset
    list_answer = dataset == "loft" and task.startswith(LIST_TASKS)
    for example in examples:
        if example["id"] in done:
            continue
        started = time.monotonic()
        try:
            chunks = build_chunks(example["context"], example["question"], config.ordering, organization_dir)
            result = rlm.run(
                example["context"],
                example["question"],
                chunks=chunks,
                answer_format=example.get("answer_prefix") or None,
                list_answer=list_answer,
                chunk_chars=chunk_chars(),
            )
            record = record_for(example, task, result)
            (transcripts / f"{example['id'].replace(':', '_')}.json").write_text(
                json.dumps(result.transcript, indent=1)
            )
        except Exception as exc:  # noqa: BLE001 - recorded and retried on the next run
            record = {"id": example["id"], "task": task, "error": f"{type(exc).__name__}: {exc}"}
        with checkpoint.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        log(
            f"  [{example['id']}] {record.get('end_reason', 'ERROR')} steps={record.get('steps')} "
            f"reads={record.get('reads', {}).get('sub_calls')} match={record.get('correct')} "
            f"{time.monotonic() - started:.0f}s"
        )

    records = [json.loads(line) for line in checkpoint.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(records).drop_duplicates("id", keep="last")
    columns = [c for c in ("id", "task", "pred", "answers", "steps", "end_reason", "latency_s", "error") if c in frame]
    frame[columns].rename(columns={"pred": "predicted_answer"}).to_csv(run_dir / "predictions.csv", index=False)
    metrics = task_metrics(dataset, records)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def run_experiment(
    config: ExperimentConfig,
    root: Any = None,
    reader: Optional[Reader] = None,
    load: Callable[..., Iterable[dict]] = load_examples,
    log: Callable[[str], None] = print,
) -> dict[str, dict]:
    """Run every task of a config. `root`, `reader` and `load` are injectable for tests."""
    rlm = make_rlm(config, root or make_root(config), reader or make_reader(config))
    results = {}
    wanted = config.data.ids()
    for task in config.data.task_dirs():
        log(f"=== {config.name}: {task} (ordering={config.ordering}, F={config.compression.factor})")
        examples = load(config.data.dataset, task, config.data.limit, config.data.split)
        if wanted is not None:
            examples = [e for e in examples if e["id"] in wanted]
            if not examples:
                log(f"    no id of {task} is in {config.data.ids_file}; skipped")
                continue
        results[task] = run_task(config, task, rlm, examples, log)
    return results
