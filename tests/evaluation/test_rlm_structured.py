# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The structured (map -> reduce) arm.

Pure: fake clients, no torch, no GPU. The properties under test are the ones that
make this arm COMPARABLE to the agentic ones -- one sub-call per chunk, spans
attributed to the document, the same sub_* counters -- plus the endings that must
stay distinguishable from one another.
"""

import unittest

from evaluation.rlm.retrieval import Hit
from evaluation.rlm.rlm import RLM, STRUCTURED_NOTHING

DOCUMENT = "".join(f"passage {i} says thing {i}. " for i in range(400))


def hits(*windows):
    return [Hit(rank=r, start=s, end=e, score=1.0, text=DOCUMENT[s:e]) for r, (s, e) in enumerate(windows, start=1)]


class _Client:
    """Records what it was asked; answers from a list, then repeats the last."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.asked: list = []
        self.calls = 0
        self.usage = None

    def chat(self, messages, **kw):
        self.asked.append(messages[-1]["content"])
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def build(sub_replies, root_replies=("Final Answer: whale",), **kw):
    root = _Client(root_replies)
    sub = _Client(sub_replies)
    return root, sub, RLM(root_client=root, sub_client=sub, **kw)


class MapTest(unittest.TestCase):
    def test_one_sub_call_per_chunk_and_each_carries_its_own_text(self):
        chunks = hits((0, 50), (100, 150), (200, 250))
        root, sub, rlm = build(["fact A", "fact B", "fact C"])

        result = rlm.run_structured(DOCUMENT, "what?", chunks)

        self.assertEqual(sub.calls, 3)
        self.assertEqual(result.metrics["map_calls"], 3)
        self.assertEqual(result.metrics["sub_calls"], 3)
        self.assertEqual(result.metrics["chunks_provided"], 3)
        for chunk, asked in zip(chunks, sub.asked):
            self.assertIn(chunk.text, asked)

    def test_the_question_is_put_to_every_chunk(self):
        root, sub, rlm = build(["x"])
        rlm.run_structured(DOCUMENT, "who invented the kazoo?", hits((0, 50), (60, 110)))
        for asked in sub.asked:
            self.assertIn("who invented the kazoo?", asked)

    def test_chunks_are_attributed_to_real_document_spans(self):
        # Coverage is what separates a payload the harness can vouch for from one
        # the model assembled. 100 of 10000 chars read == 0.01.
        root, sub, rlm = build(["fact"])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 50), (100, 150)))
        self.assertAlmostEqual(result.metrics["document_coverage_fraction"], 100 / len(DOCUMENT))

    def test_provided_spans_are_recorded_where_retrieved_spans_go(self):
        # run_benchmark's gold-in-retrieved diagnostic reads this key; recording it
        # here is what lets the structured arm reuse that machinery unchanged.
        root, sub, rlm = build(["fact"])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 50), (100, 150)))
        self.assertEqual(result.metrics["search_retrieved_spans"], [(0, 50), (100, 150)])

    def test_max_sub_calls_bounds_the_map(self):
        root, sub, rlm = build(["fact"], max_sub_calls=2)
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 50), (100, 150), (200, 250), (300, 350)))
        # Past the cap llm_query returns a notice instead of hitting the server.
        self.assertEqual(sub.calls, 2)
        self.assertEqual(result.metrics["map_calls"], 4)


class ReduceTest(unittest.TestCase):
    def test_the_reduce_sees_every_finding(self):
        root, sub, rlm = build(["the whale sank it", "captain was Ahab"])
        rlm.run_structured(DOCUMENT, "q", hits((0, 50), (100, 150)))
        self.assertEqual(root.calls, 1)
        self.assertIn("the whale sank it", root.asked[0])
        self.assertIn("captain was Ahab", root.asked[0])

    def test_chunks_reporting_nothing_are_kept_out_of_the_reduce(self):
        root, sub, rlm = build([STRUCTURED_NOTHING, "real finding"])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 50), (100, 150)))
        self.assertEqual(result.metrics["chunks_with_findings"], 1)
        self.assertNotIn(STRUCTURED_NOTHING, root.asked[0])
        self.assertIn("real finding", root.asked[0])

    def test_the_nothing_sentinel_is_matched_loosely(self):
        # A 4B model returns the sentinel with punctuation attached often enough
        # that equality would paste "NOTHING RELEVANT." in as evidence.
        root, sub, rlm = build(["nothing relevant.", "  NOTHING RELEVANT - sorry"])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 50), (100, 150)))
        self.assertEqual(result.metrics["chunks_with_findings"], 0)
        self.assertEqual(result.end_reason, "abstained")

    def test_the_answer_format_reaches_the_reduce_prompt(self):
        root, sub, rlm = build(["fact"])
        rlm.run_structured(DOCUMENT, "q", hits((0, 50)), answer_format="Final Answer: ['x']")
        self.assertIn("Final Answer: ['x']", root.asked[0])

    def test_the_reduce_reply_is_the_prediction(self):
        root, sub, rlm = build(["fact"], root_replies=["Final Answer: the whale"])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 50)))
        self.assertEqual(result.answer, "Final Answer: the whale")
        self.assertEqual(result.end_reason, "structured_complete")
        self.assertTrue(result.finished)

    def test_root_context_is_measured_for_the_cost_axis(self):
        root, sub, rlm = build(["fact"])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 50)))
        self.assertGreater(result.metrics["peak_context_tokens"], 0)
        self.assertGreater(result.metrics["root_prompt_tokens"], 0)


class EndingsTest(unittest.TestCase):
    def test_no_chunks_is_distinct_from_an_abstention(self):
        # An empty chunk file is a harness failure, not the model finding nothing.
        root, sub, rlm = build(["fact"])
        result = rlm.run_structured(DOCUMENT, "q", [])
        self.assertEqual(result.end_reason, "structured_no_chunks")
        self.assertFalse(result.finished)
        self.assertIsNone(result.answer)
        self.assertEqual(sub.calls, 0)
        self.assertEqual(root.calls, 0)

    def test_every_chunk_silent_is_an_abstention_and_costs_no_reduce_call(self):
        root, sub, rlm = build([STRUCTURED_NOTHING])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 50), (100, 150)))
        self.assertEqual(result.end_reason, "abstained")
        self.assertIsNone(result.answer)
        self.assertEqual(root.calls, 0)
        self.assertIn("abstain_reason", result.metrics)

    def test_the_abstention_gate_that_governs_the_repl_does_not_apply_here(self):
        # min_sub_calls_before_abstain guards a root claiming absence without
        # reading. The map step read every chunk it was given, so the gate has
        # nothing to catch and must not block a legitimate ending.
        root, sub, rlm = build([STRUCTURED_NOTHING], min_sub_calls_before_abstain=99)
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 50)))
        self.assertEqual(result.end_reason, "abstained")

    def test_the_transcript_keeps_the_shape_post_mortems_expect(self):
        root, sub, rlm = build(["fact"])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 50), (100, 150)))
        self.assertEqual(len(result.transcript), 3)  # two maps + one reduce
        for entry in result.transcript:
            self.assertEqual({"step", "reply", "code", "observation"}, set(entry))
        self.assertEqual(result.steps, 3)


class GroundingTest(unittest.TestCase):
    def test_an_answer_absent_from_every_finding_is_flagged_not_dropped(self):
        # The REPL's guard rejects and nudges; there is no loop to nudge into here,
        # and the root is shown nothing but real chunk text -- so this is recorded.
        root, sub, rlm = build(["the captain was Ahab"], root_replies=["Nantucket"])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 50)))
        self.assertEqual(result.answer, "Nantucket")
        self.assertTrue(result.metrics["reduce_answer_not_quoted"])

    def test_an_answer_quoted_from_a_finding_is_not_flagged(self):
        root, sub, rlm = build(["the captain was Ahab"], root_replies=["Ahab"])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 50)))
        self.assertNotIn("reduce_answer_not_quoted", result.metrics)


if __name__ == "__main__":
    unittest.main()
