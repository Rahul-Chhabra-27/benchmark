# SPDX-License-Identifier: Apache-2.0
"""Tests for Synthetic-KV exact and prefix-tolerant string matching."""

import unittest

import pandas as pd

from calculate_metrics import calculate_metrics


class SyntheticKvMetricsTest(unittest.TestCase):
    def score(self, prediction: str, reference: str) -> dict:
        frame = pd.DataFrame(
            {
                "task": ["synthetic_kv_32k"],
                "predicted_answer": [prediction],
                "answer": [[reference]],
            }
        )
        return calculate_metrics(frame)["synthetic_kv_32k"]

    def test_bare_value_is_exact_and_string_match(self) -> None:
        result = self.score("ABCDEF123456", "ABCDEF123456")
        self.assertEqual(result["exact_match"], 100.0)
        self.assertEqual(result["string_match"], 100.0)

    def test_prediction_v_prefix_is_only_string_match(self) -> None:
        result = self.score("V_ABCDEF123456", "ABCDEF123456")
        self.assertEqual(result["exact_match"], 0.0)
        self.assertEqual(result["string_match"], 100.0)

    def test_reference_v_prefix_is_only_string_match(self) -> None:
        result = self.score("ABCDEF123456", "V_ABCDEF123456")
        self.assertEqual(result["exact_match"], 0.0)
        self.assertEqual(result["string_match"], 100.0)

    def test_verbose_prefixed_value_is_string_match(self) -> None:
        result = self.score("Answer: V_ABCDEF123456.", "ABCDEF123456")
        self.assertEqual(result["exact_match"], 0.0)
        self.assertEqual(result["string_match"], 100.0)

    def test_wrong_value_does_not_match(self) -> None:
        result = self.score("V_000000000000", "ABCDEF123456")
        self.assertEqual(result["exact_match"], 0.0)
        self.assertEqual(result["string_match"], 0.0)


if __name__ == "__main__":
    unittest.main()
