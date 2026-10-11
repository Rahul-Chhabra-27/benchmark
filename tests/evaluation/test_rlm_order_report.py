# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ordered-chunk report's slice of the rows no prompt revision was tuned on."""

from evaluation.rlm.rlm_order_report import held_out_ids, noise_note


def test_the_noise_note_speaks_the_metrics_scale():
    assert "moves a mean by 0.018;" in noise_note(55, percent_scale=False)
    assert "moves a mean by 1.8 points of a 0–100 score;" in noise_note(55, percent_scale=True)


def test_rows_below_the_tuned_count_are_left_out():
    ids = [f"loft:nq_128k:{i}" for i in range(6)]
    assert held_out_ids(ids, 3) == {"loft:nq_128k:3", "loft:nq_128k:4", "loft:nq_128k:5"}


def test_the_row_number_is_the_trailing_one_and_zero_keeps_everything():
    ids = ["loft:hotpotqa_128k:2", "loft:hotpotqa_128k:12", "loft:hotpotqa_128k:30"]
    assert held_out_ids(ids, 3) == {"loft:hotpotqa_128k:12", "loft:hotpotqa_128k:30"}
    assert held_out_ids(ids, 0) == set(ids)
