# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The paired comparison, and the CSV round-trip that would silently zero it.

Pure: no client, no torch, no GPU.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from evaluation.rlm.paired import _delist, bootstrap_ci, compare, load_rows, per_row_scores, sign_test


def write_run(root: Path, name: str, rows, dataset="loft", task="nq_128k", as_csv=False) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(f"dataset: {dataset}\ndata_dir: {task}\nmode: structured\n")
    records = [
        {
            "id": row_id,
            "task": task,
            "answers": answers,
            "answer_prefix": "Final Answer: ",
            ("predicted_answer" if as_csv else "pred"): prediction,
        }
        for row_id, answers, prediction in rows
    ]
    if as_csv:
        pd.DataFrame(records).to_csv(run / "predictions.csv", index=False)
    else:
        (run / "checkpoint.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return run


class DelistTest(unittest.TestCase):
    """`frame.to_csv` writes a list as its repr, and one gold that is a repr matches nothing."""

    def test_a_list_survives_a_csv_round_trip(self):
        self.assertEqual(_delist("['Adam Smith', 'A. Smith']"), ["Adam Smith", "A. Smith"])

    def test_an_answer_containing_a_comma_is_not_split_on_it(self):
        self.assertEqual(_delist("['Smith, Adam']"), ["Smith, Adam"])

    def test_a_bare_string_becomes_a_one_answer_list(self):
        self.assertEqual(_delist("Adam Smith"), ["Adam Smith"])

    def test_an_already_parsed_list_is_left_alone(self):
        self.assertEqual(_delist(["a", "b"]), ["a", "b"])

    def test_a_string_that_is_not_a_literal_is_kept_verbatim(self):
        self.assertEqual(_delist("[unclosed"), ["[unclosed"])


class PerRowScoreTest(unittest.TestCase):
    ROWS = [
        ("loft:nq_128k:0", ["whale"], "Final Answer: whale"),
        ("loft:nq_128k:1", ["shark"], "Final Answer: otter"),
    ]

    def test_the_csv_path_scores_the_same_as_the_checkpoint_path(self):
        # The whole point of _delist: without it the CSV path reports a different,
        # plausible number rather than an obvious failure.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_frame, _ = load_rows(write_run(root, "as-json", self.ROWS))
            csv_frame, _ = load_rows(write_run(root, "as-csv", self.ROWS, as_csv=True))
            self.assertEqual(per_row_scores(json_frame, "loft")[1], per_row_scores(csv_frame, "loft")[1])

    def test_an_errored_row_is_dropped_rather_than_scored_zero(self):
        # A server outage is a missing measurement, not a wrong answer.
        frame = pd.DataFrame(
            [
                {"id": "a", "task": "nq_128k", "answers": ["whale"],
                 "predicted_answer": "Final Answer: whale", "error": None},
                {"id": "b", "task": "nq_128k", "answers": ["shark"],
                 "predicted_answer": None, "error": "boom"},
            ]
        )
        _, scores = per_row_scores(frame, "loft")
        self.assertEqual(list(scores), ["a"])

    def test_a_retried_row_is_counted_once(self):
        # An errored example is retried on resume and the retry APPENDS.
        with TemporaryDirectory() as tmp:
            run = write_run(Path(tmp), "r", [("loft:nq_128k:0", ["whale"], "Final Answer: otter")])
            with open(run / "checkpoint.jsonl", "a") as fh:
                fh.write(json.dumps({"id": "loft:nq_128k:0", "task": "nq_128k", "answers": ["whale"],
                                     "answer_prefix": "Final Answer: ", "pred": "Final Answer: whale"}) + "\n")
            frame, _ = load_rows(run)
            _, scores = per_row_scores(frame, "loft")
            self.assertEqual(scores, {"loft:nq_128k:0": 1.0}, "the retry, not the stale failure")


class SignTestTest(unittest.TestCase):
    def test_a_hand_computable_case(self):
        # 3 wins, 1 loss: two-sided p = 2 * (C(4,0) + C(4,1)) / 2^4 = 10/16.
        self.assertAlmostEqual(sign_test(3, 1), 0.625)

    def test_all_ties_is_not_evidence_of_anything(self):
        self.assertEqual(sign_test(0, 0), 1.0)

    def test_a_clean_sweep_is_the_smallest_p_available(self):
        self.assertAlmostEqual(sign_test(5, 0), 2 / 32)


class BootstrapTest(unittest.TestCase):
    def test_no_difference_gives_an_interval_containing_zero(self):
        lo, hi = bootstrap_ci([0.0] * 20, resamples=500, seed=0)
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)

    def test_it_is_reproducible_from_the_seed(self):
        diffs = [0.4, -0.1, 0.2, 0.9, -0.3] * 6
        self.assertEqual(bootstrap_ci(diffs, 400, seed=7), bootstrap_ci(diffs, 400, seed=7))


class CompareTest(unittest.TestCase):
    ROWS_A = [(f"loft:nq_128k:{i}", ["whale"], "Final Answer: otter") for i in range(4)]

    def test_identical_arms_report_no_difference(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = compare(write_run(root, "a", self.ROWS_A), write_run(root, "b", self.ROWS_A), resamples=200)
        self.assertEqual(result["delta"], 0.0)
        self.assertEqual((result["wins"], result["losses"]), (0, 0))
        self.assertLessEqual(result["ci_lo"], 0.0)
        self.assertGreaterEqual(result["ci_hi"], 0.0)

    def test_three_wins_and_one_loss_gives_the_hand_computed_p(self):
        rows_b = [
            ("loft:nq_128k:0", ["whale"], "Final Answer: whale"),
            ("loft:nq_128k:1", ["whale"], "Final Answer: whale"),
            ("loft:nq_128k:2", ["whale"], "Final Answer: whale"),
            ("loft:nq_128k:3", ["whale"], "Final Answer: otter"),
        ]
        rows_a = [
            ("loft:nq_128k:0", ["whale"], "Final Answer: otter"),
            ("loft:nq_128k:1", ["whale"], "Final Answer: otter"),
            ("loft:nq_128k:2", ["whale"], "Final Answer: otter"),
            ("loft:nq_128k:3", ["whale"], "Final Answer: whale"),
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = compare(write_run(root, "a", rows_a), write_run(root, "b", rows_b), resamples=200)
        self.assertEqual((result["wins"], result["losses"], result["ties"]), (3, 1, 0))
        self.assertAlmostEqual(result["sign_p"], 0.625)

    def test_arms_that_ran_different_rows_are_refused(self):
        # Averaging over the union throws away the pairing --ids-file exists to buy.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = write_run(root, "a", self.ROWS_A)
            b = write_run(root, "b", self.ROWS_A[:2] + [("loft:nq_128k:9", ["whale"], "Final Answer: whale")])
            with self.assertRaises(SystemExit) as caught:
                compare(a, b, resamples=50)
        self.assertIn("do not describe the same rows", str(caught.exception))

    def test_the_intersection_can_be_compared_deliberately(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = write_run(root, "a", self.ROWS_A)
            b = write_run(root, "b", self.ROWS_A[:2])
            result = compare(a, b, resamples=50, allow_subset=True)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["only_a"], 2)

    def test_a_slice_of_ids_is_paired_on_its_own(self):
        # The rows no prompt revision was tuned on, compared by themselves.
        rows_b = [(f"loft:nq_128k:{i}", ["whale"], "Final Answer: whale") for i in range(4)]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = write_run(root, "a", self.ROWS_A)
            b = write_run(root, "b", rows_b)
            result = compare(a, b, resamples=50, only_ids={"loft:nq_128k:2", "loft:nq_128k:3"})
        self.assertEqual((result["n"], result["wins"], result["only_a"]), (2, 2, 0))
        self.assertEqual(result["delta"], 1.0)

    def test_subspan_em_is_never_subtracted_from_coverage(self):
        # qampari scores on coverage and nq on subspan_em; a delta across that line
        # is two different questions, not a difference.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = write_run(root, "a", self.ROWS_A, task="nq_128k")
            rows_b = [(f"loft:nq_128k:{i}", ["whale"], "Final Answer: whale") for i in range(4)]
            b = write_run(root, "b", rows_b, task="qampari_128k")
            with self.assertRaises(SystemExit) as caught:
                compare(a, b, resamples=50)
        self.assertIn("different metrics", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
