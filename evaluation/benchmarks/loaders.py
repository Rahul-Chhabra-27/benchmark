# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Centralized Hugging Face loading and normalization for evaluation datasets."""

import logging
from typing import Any, Mapping, Optional

import pandas as pd
from datasets import load_dataset


logger = logging.getLogger(__name__)


def _require_task(dataset_name: str, task: Optional[str]) -> str:
    if task is None:
        raise ValueError(f"{dataset_name} requires a task name in data_dir")
    return task


def _load_loft(task: str) -> pd.DataFrame:
    parts = task.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid LOFT subset {task!r}; expected dataset_length")

    length = parts[-1]
    dataset_name = "_".join(parts[:-1])
    dataset_id = f"f20180301/loft-rag-{dataset_name}-{length}"
    dataset_dict = load_dataset(dataset_id)

    split_frames = []
    for split_name in ("dev", "test"):
        if split_name in dataset_dict:
            split_df = dataset_dict[split_name].to_pandas()
            split_df["split"] = split_name
            split_frames.append(split_df)
    if not split_frames:
        raise ValueError(f"No dev or test split found for {task} ({dataset_id})")

    df = pd.concat(split_frames, ignore_index=True)
    df["task"] = task

    required_columns = [
        "context",
        "question",
        "answers",
        "task",
        "answer_prefix",
        "max_new_tokens",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"  ✓ Loaded {len(df)} LOFT samples from {task}")
    return df


def _load_synthetic_kv(
    dataset_name: str,
    dataset_id: str,
    task: str,
    metadata_override: bool,
) -> pd.DataFrame:
    expected_task = "32k" if dataset_name == "synthetic_kv_32k" else "64k"
    if task != expected_task:
        raise ValueError(
            f"Unknown synthetic-KV configuration {task!r}; expected {expected_task!r}"
        )
    if metadata_override:
        raise ValueError(
            "Synthetic-KV metadata override is disabled: use the Hugging Face context"
        )

    logger.info("Loading native Synthetic-KV context from %s", dataset_id)
    dataset = load_dataset(dataset_id, split="test")
    expanded_rows: list[dict[str, Any]] = []
    for compact_row in dataset:
        context = compact_row["context"]
        questions = compact_row["questions"]
        answers = compact_row["answers"]
        if len(questions) != len(answers):
            raise ValueError(
                f"Mismatched questions/answers for {compact_row['context_id']}: "
                f"{len(questions)} != {len(answers)}"
            )

        max_new_tokens = int(compact_row.get("max_new_tokens", 32))
        context_length = int(compact_row.get("context_tokens", 65536))
        for question, answer in zip(questions, answers):
            expanded_rows.append(
                {
                    "context_id": compact_row["context_id"],
                    "context": context,
                    "question": question,
                    "answer": [answer],
                    "task": f"synthetic_kv_{expected_task}",
                    "answer_prefix": str(compact_row.get("answer_prefix", "")),
                    "max_new_tokens": max_new_tokens,
                    "context_length": context_length,
                }
            )
    if not expanded_rows:
        raise ValueError("The synthetic-KV dataset contains no questions")

    print(
        f"  ✓ Expanded {len(dataset)} Synthetic-KV context(s) into "
        f"{len(expanded_rows)} questions || context tokens = "
        f"{expanded_rows[0]['context_length']}"
    )
    return pd.DataFrame(expanded_rows)


def load_benchmark_dataset(
    dataset_name: str,
    task: Optional[str],
    dataset_registry: Mapping[str, str],
    synthetic_metadata_override: bool = False,
) -> pd.DataFrame:
    """Load and normalize one evaluation task into a pandas DataFrame."""
    if dataset_name not in dataset_registry:
        raise ValueError(f"Unknown evaluation dataset: {dataset_name!r}")
    dataset_id = dataset_registry[dataset_name]

    if dataset_name == "loft":
        df = _load_loft(_require_task(dataset_name, task))
    elif dataset_name in {"synthetic_kv", "synthetic_kv_32k"}:
        df = _load_synthetic_kv(
            dataset_name,
            dataset_id,
            _require_task(dataset_name, task),
            synthetic_metadata_override,
        )
    else:
        raise ValueError(
            f"No specialized loader for {dataset_name!r}; use the standard KVPress path"
        )

    return df
