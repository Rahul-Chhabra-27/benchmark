# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""run_benchmark plumbing for --structured-order: names, config, refusals, aggregates."""

import contextlib
import io
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

from evaluation.rlm.harness import run_benchmark as rb
from tests.evaluation.test_rlm_run_benchmark import _args

ORDERED = dict(
    mode="structured",
    structured_order="bm25-graph",
    structured_group_input_tokens=8192,
    structured_context_tokens=65536,
    structured_reader_max_tokens=512,
    structured_reduce_max_tokens=1024,
    organization_dir="/tmp/org",
    max_sub_calls=0,
)


class RunDirTest(unittest.TestCase):
    def test_the_order_names_the_directory_instead_of_a_chunk_source(self):
        components = rb.build_run_dir_components(_args(**ORDERED), "structured", None)
        self.assertEqual(components[-1], "order-bm25-graph")
        self.assertFalse(any(c.startswith("chunks-") for c in components))

    def test_the_two_arms_never_share_a_directory(self):
        graph = rb.build_run_dir_components(_args(**ORDERED), "structured", None)
        source = rb.build_run_dir_components(_args(**{**ORDERED, "structured_order": "source"}), "structured", None)
        self.assertNotEqual(graph, source)

    def test_non_default_group_budget_is_stamped(self):
        components = rb.build_run_dir_components(
            _args(**{**ORDERED, "structured_group_input_tokens": 4096}), "structured", None
        )
        self.assertIn("grp4096", components)

    def test_multi_value_subsets_keep_the_list_shape_stamp(self):
        components = rb.build_run_dir_components(_args(**ORDERED, data_dir="qampari_128k"), "structured", None)
        self.assertEqual(components[-2:], ["order-bm25-graph", "listfmt"])

    def test_legacy_structured_names_are_unchanged(self):
        legacy = rb.build_run_dir_components(_args(mode="structured", chunks_source="bm25"), "structured", None)
        self.assertEqual(legacy[-1], "chunks-bm255")


class ConfigTest(unittest.TestCase):
    def test_ordered_config_records_the_order_budgets_and_frozen_organization(self):
        config = rb.build_run_config(_args(**ORDERED), "structured", "loft", None, None, None)
        self.assertEqual(config["structured_order"], "bm25-graph")
        self.assertEqual(config["structured_group_input_tokens"], 8192)
        self.assertEqual(config["structured_reader_max_tokens"], 512)
        self.assertEqual(config["structured_reduce_max_tokens"], 1024)
        self.assertEqual(config["organization_config"]["revision"], 1)

    def test_legacy_config_has_no_ordered_keys(self):
        legacy = _args(mode="structured", chunks_source="bm25")
        config = rb.build_run_config(legacy, "structured", "loft", None, None, None)
        self.assertNotIn("structured_order", config)


class RuntimeTest(unittest.TestCase):
    def test_aggregates_group_exposure_and_token_mismatches(self):
        frame = pd.DataFrame(
            {
                "prompt_tokens": [100, 300],
                "completion_tokens": [10, 30],
                "metrics": [
                    dict(
                        group_count=2,
                        group_chunk_counts=[3, 1],
                        reader_input_tokens=[50, 20],
                        reader_served_prompt_tokens=[50, 20],
                        reducer_input_tokens=40,
                        retained_finding_count=2,
                        reducer_finding_chars=80,
                        organization_sha256="d",
                        organization_build_cpu_seconds=1.5,
                        packing_cpu_seconds=0.1,
                    ),
                    dict(
                        group_count=4,
                        group_chunk_counts=[1, 1, 1, 1],
                        reader_input_tokens=[60, 10, 10, 10],
                        reader_served_prompt_tokens=[61, 10, 10, None],
                        reducer_input_tokens=90,
                        retained_finding_count=1,
                        reducer_finding_chars=20,
                        organization_sha256="d",
                        organization_build_cpu_seconds=1.5,
                        packing_cpu_seconds=0.3,
                    ),
                ],
            }
        )
        out = rb.ordered_runtime(frame)
        self.assertEqual(out["average_group_count"], 3.0)
        self.assertEqual(out["max_reader_input_tokens"], 60)
        self.assertEqual(out["reader_token_count_mismatches"], 1)
        self.assertEqual(out["reader_served_counts_missing"], 1)
        self.assertEqual(out["max_reducer_input_tokens"], 90)
        # One document, built once: its cost is counted once, not per query.
        self.assertEqual(out["organization_build_cpu_seconds"], 1.5)
        self.assertEqual(out["average_prompt_tokens"], 200.0)


class RefusalTest(unittest.TestCase):
    BASE = [
        "run_benchmark",
        "--dataset",
        "loft",
        "--data-dir",
        "nq_128k",
        "--mode",
        "structured",
        "--structured-order",
        "source",
        "--max-sub-calls",
        "0",
    ]

    def refused(self, argv):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(sys, "argv", [*argv, "--out", tmp]):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                    rb.main()
        self.assertEqual(caught.exception.code, 2)
        return stderr.getvalue()

    def test_order_replaces_chunk_sources(self):
        self.assertIn("replaces --chunks-source", self.refused([*self.BASE, "--chunks-source", "bm25"]))

    def test_a_capped_map_is_refused(self):
        argv = [a for a in self.BASE if a not in ("--max-sub-calls", "0")]
        self.assertIn("--max-sub-calls 0", self.refused(argv))

    def test_order_needs_the_structured_mode(self):
        argv = [*self.BASE]
        argv[argv.index("structured")] = "rlm"
        self.assertIn("needs --mode structured", self.refused(argv))

    def test_second_hop_and_map_truncation_are_refused(self):
        self.refused([*self.BASE, "--structured-hops", "2"])
        self.refused([*self.BASE, "--structured-map-chars", "500"])

    def test_group_plus_output_must_fit_the_context(self):
        self.assertIn("exceeds", self.refused([*self.BASE, "--structured-group-input-tokens", "65500"]))

    def test_ordered_flags_without_an_order_are_refused(self):
        argv = [
            "run_benchmark",
            "--dataset",
            "loft",
            "--data-dir",
            "nq_128k",
            "--mode",
            "structured",
            "--chunks-source",
            "bm25",
            "--structured-group-input-tokens",
            "4096",
        ]
        self.assertIn("only apply with --structured-order", self.refused(argv))


if __name__ == "__main__":
    unittest.main()
