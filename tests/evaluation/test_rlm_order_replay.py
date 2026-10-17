# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rlm_order_replay: which FINAL literals revision 5 rejected, and which revision 6 accepts."""

from evaluation.rlm.campaigns.rlm_order_replay import final_literal, first_accepted

READ = 'a = llm_query("q", chunks[:2])\nprint(a)'


def step(code, observation="[no output]"):
    return dict(step=1, reply=f"```python\n{code}\n```", code=code, observation=observation)


def transcript(*steps):
    return dict(question="Which cello sonatas?", transcript=list(steps))


def test_the_static_string_passed_to_final_is_found():
    assert final_literal('x = 1\nFINAL("Op. " + "1")') == "Op. 1"
    assert final_literal("FINAL(ans)") is None
    assert final_literal("not python at all") is None


def test_a_rejected_composed_literal_is_accepted_under_revision_6():
    t = transcript(
        step(READ, "The cello sonata is Op. 1111."),
        step('b = llm_query("q", chunks[2:4])\nprint(b)', "The violin sonata is Op. 2222."),
        step('FINAL("Op. 1111 and Op. 2222")'),
        step("FINAL(a)"),
    )
    assert first_accepted(t, "final_called", "") == (1, "Op. 1111 and Op. 2222")


def test_a_guess_stays_rejected_and_the_final_a_row_ended_on_is_not_counted():
    t = transcript(
        step(READ, "The cello sonata is Op. 1111."),
        step('FINAL("Op. 3333")'),
        step('FINAL("The cello sonata is Op. 1111.")'),
    )
    assert first_accepted(t, "final_called", "") == (1, None)


def test_the_last_literal_of_an_ungrounded_ending_was_rejected_too():
    t = transcript(step(READ, "Op. 1111, then Op. 2222."), step('FINAL("1111 and 2222")'))
    assert first_accepted(t, "ungrounded_final", "") == (1, "1111 and 2222")


def test_a_rejected_prose_final_is_replayed():
    t = transcript(
        step(READ, "The cello sonata is Op. 1111."),
        dict(step=2, reply='FINAL("Op. 1111")', code=None, observation="[ungrounded prose FINAL rejected]"),
    )
    assert first_accepted(t, "ungrounded_final", "") == (1, "Op. 1111")
