# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for centralized evaluation dataset loading."""

import unittest
from unittest.mock import patch

import pandas as pd

from benchmarks import loaders


class FakeDataset(list):
    """Minimal iterable dataset needed by the Synthetic-KV loader."""


class FakeSplit:
    def __init__(self, frame):
        self.frame = frame

    def to_pandas(self):
        return self.frame.copy()


class LoaderTest(unittest.TestCase):
    def test_loft_rejects_missing_original_required_columns(self):
        incomplete_frame = pd.DataFrame(
            {
                "context": ["context"],
                "question": ["question"],
                "answer_prefix": ["Answer: "],
                "max_new_tokens": [32],
            }
        )
        dataset = {"test": FakeSplit(incomplete_frame)}

        with patch.object(loaders, "load_dataset", return_value=dataset):
            with self.assertRaisesRegex(ValueError, "answers"):
                loaders._load_loft("nq_32k")

    def test_synthetic_kv_expands_compact_context(self):
        dataset = FakeDataset(
            [
                {
                    "context_id": "context-1",
                    "context": "Records:\n[K_AAAAAAAAAAAA: V_BBBBBBBBBBBB]",
                    "questions": ["Value for K_AAAAAAAAAAAA?"],
                    "answers": ["V_BBBBBBBBBBBB"],
                    "answer_prefix": "Answer: ",
                    "max_new_tokens": 32,
                    "context_tokens": 31967,
                }
            ]
        )
        with patch.object(loaders, "load_dataset", return_value=dataset):
            df = loaders.load_benchmark_dataset(
                dataset_name="synthetic_kv_32k",
                task="32k",
                dataset_registry={"synthetic_kv_32k": "example/32k"},
            )

        self.assertEqual(len(df), 1)
        self.assertTrue(df.loc[0, "context"].startswith("Records:"))
        self.assertIn("K_AAAAAAAAAAAA", df.loc[0, "context"])
        self.assertIn("V_BBBBBBBBBBBB", df.loc[0, "context"])
        self.assertEqual(df.loc[0, "question"], "Value for K_AAAAAAAAAAAA?")
        self.assertEqual(df.loc[0, "answer"], ["V_BBBBBBBBBBBB"])
        self.assertEqual(df.loc[0, "context_length"], 31967)

if __name__ == "__main__":
    unittest.main()
