# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared result scoring boundary."""

import unittest
from unittest.mock import patch

import pandas as pd
from benchmarks import results


class ResultScoringTest(unittest.TestCase):
    def test_score_prediction_frame_is_backend_neutral_and_non_mutating(self):
        frame = pd.DataFrame(
            {
                "predicted_answer": [None],
                "answer": [["needle"]],
                "task": ["niah_single_1"],
            }
        )

        def scorer(scoring_frame):
            self.assertEqual(scoring_frame.loc[0, "predicted_answer"], "")
            return 25.0

        with patch.dict(results.SCORER_REGISTRY, {"ruler32k": scorer}, clear=True):
            metrics = results.score_prediction_frame("ruler32k", frame)

        self.assertEqual(metrics, {"score": 25.0})
        self.assertIsNone(frame.loc[0, "predicted_answer"])

    def test_score_prediction_frame_requires_registered_dataset(self):
        frame = pd.DataFrame({"predicted_answer": ["answer"]})
        with patch.dict(results.SCORER_REGISTRY, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "No scorer"):
                results.score_prediction_frame("missing", frame)


if __name__ == "__main__":
    unittest.main()
