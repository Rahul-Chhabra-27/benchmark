# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rlm_order_review: which rows an acceptance review samples, and why each one is there."""

from evaluation.rlm.rlm_order_review import cause, needs_review


def row(end="final_called", reads=1, **metrics):
    return dict(end_reason=end, metrics=dict(chunk_payload_calls=reads, **metrics))


def transcript(*cells):
    return dict(transcript=[dict(step=i, code=cell, observation="") for i, cell in enumerate(cells)])


def test_the_gate_rule_picks_the_rows():
    assert not needs_review(row())
    assert not needs_review(row(end="abstained", reads=3))
    assert needs_review(row(reads=0))
    assert needs_review(row(end="max_steps"))


def test_a_refusal_loop_is_named_with_the_budget():
    r = row(end="error_loop_broken", reads=0, chunk_budget_refusals=3, chunks_per_call_advertised=5)
    assert cause(r, transcript("llm_query(q, chunks[i-2:i+6])")) == "refusal loop: 3 refused, 5 fit per call"


def test_a_capitalised_filter_against_lowered_text_is_the_case_trap():
    t = transcript('hits = [i for i, c in enumerate(chunks) if "Duchy" in c.text.lower()]')
    assert cause(row(end="abstained", reads=0), t) == 'case trap: "Duchy" in c.text.lower()'


def test_an_ungrounded_ending_shows_the_last_rejected_literal():
    t = transcript('FINAL("1 variables: A")', 'FINAL("2 variables: A and B")')
    assert cause(row(end="ungrounded_final"), t) == "ungrounded: 2 variables: A and B"


def test_anything_else_shows_the_last_cell():
    label = cause(row(end="max_steps"), transcript("x = 1", "print(hits[43])"))
    assert label == "max_steps: last cell 'print(hits[43])'"
