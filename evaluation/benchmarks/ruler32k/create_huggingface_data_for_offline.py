# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset

from evaluation.benchmarks.registry import DATASET_REGISTRY, RULER_32K_TASKS

huggingface_dataset_id = DATASET_REGISTRY["ruler32k"]

for subset in RULER_32K_TASKS:
    ds = load_dataset(
        huggingface_dataset_id,
        data_dir=subset,
        split="test",
    )
    print(f"cached: {subset} ({len(ds)} samples)")

print(
    "Done. Cache folders are named after the config (e.g. 'cwe', 'fwe', ...) "
    "which is exactly what the shared benchmark loader expects to find."
)
