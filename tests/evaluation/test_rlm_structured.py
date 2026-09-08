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
from evaluation.rlm.rlm import RLM, STRUCTURED_NOTHING, _cluster_hits, _map_groups, bare_query

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
    def test_one_sub_call_per_passage_and_each_carries_its_own_text(self):
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


class ClusteringTest(unittest.TestCase):
    """Retrieval windows overlap by 400 chars, and a document straddling a boundary
    is deliberately selected as an adjacent PAIR. One call per window would re-read
    the shared text and show each call half a document."""

    def test_overlapping_windows_become_one_passage(self):
        clusters = _cluster_hits(hits((0, 2000), (1600, 3600)))
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 2)

    def test_touching_windows_become_one_passage(self):
        self.assertEqual(len(_cluster_hits(hits((0, 2000), (2000, 4000)))), 1)

    def test_a_gap_keeps_passages_apart(self):
        self.assertEqual(len(_cluster_hits(hits((0, 2000), (2001, 4000)))), 2)

    def test_clusters_come_out_in_document_order_whatever_the_rank_order(self):
        # BM25 and the selector both return best-first, which is not document order.
        clusters = _cluster_hits(hits((8000, 10000), (0, 2000), (4000, 6000)))
        self.assertEqual([c[0].start for c in clusters], [0, 4000, 8000])

    def test_an_adjacent_pair_costs_one_map_call_not_two(self):
        root, sub, rlm = build(["fact"])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 2000), (1600, 3600)))
        self.assertEqual(sub.calls, 1)
        self.assertEqual(result.metrics["map_calls"], 1)
        self.assertEqual(result.metrics["map_spans"], 1)
        # chunks_provided still counts what the selector OFFERED, so the difference
        # between the two is how many windows were coalesced.
        self.assertEqual(result.metrics["chunks_provided"], 2)

    def test_the_merged_passage_is_sent_contiguously(self):
        # The defect this fixes: a document split across two calls, each seeing half.
        root, sub, rlm = build(["fact"])
        rlm.run_structured(DOCUMENT, "q", hits((0, 2000), (1600, 3600)))
        self.assertIn(DOCUMENT[0:3600], sub.asked[0])

    def test_the_shared_overlap_is_not_repeated(self):
        root, sub, rlm = build(["fact"])
        rlm.run_structured(DOCUMENT, "q", hits((0, 2000), (1600, 3600)))
        overlap = DOCUMENT[1600:2000]
        self.assertEqual(sub.asked[0].count(overlap), 1)

    def test_distant_windows_still_get_a_call_each(self):
        root, sub, rlm = build(["fact"])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 2000), (6000, 8000)))
        self.assertEqual(sub.calls, 2)
        self.assertEqual(result.metrics["map_spans"], 2)

    def test_coverage_counts_a_merged_passage_once(self):
        root, sub, rlm = build(["fact"])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 2000), (1600, 3600)))
        self.assertAlmostEqual(result.metrics["document_coverage_fraction"], 3600 / len(DOCUMENT))


class AnswerShapeTest(unittest.TestCase):
    """The agentic arms answer through FINAL(x) -- a bare value by protocol. A reduce
    step is a chat completion, and 'be concise' is satisfied by a sentence."""

    def test_the_reduce_requires_the_shape_rather_than_offering_it(self):
        root, sub, rlm = build(["fact"])
        rlm.run_structured(DOCUMENT, "q", hits((0, 50)), answer_format="Final Answer: ")
        prompt = root.asked[0]
        self.assertIn("Final Answer:", prompt)
        self.assertIn("no sentence built around it", prompt)
        self.assertIn("no restatement of the question", prompt)

    def test_without_an_answer_format_it_still_asks_for_the_bare_answer(self):
        root, sub, rlm = build(["fact"])
        rlm.run_structured(DOCUMENT, "q", hits((0, 50)))
        self.assertIn("answer itself and nothing else", root.asked[0])

    def test_the_reduce_announces_document_order(self):
        root, sub, rlm = build(["fact"])
        rlm.run_structured(DOCUMENT, "q", hits((0, 50)))
        self.assertIn("in document order", root.asked[0])


LOFT_TASK = (
    "====== Now let's start! ======\n"
    "Based on the documents above, can you answer the following query? Print out the "
    "TITLE and ID of the documents you use to answer. Then format the answers into a list.\n"
    "query: when did season 4 of glee come out\n"
)


class MapQuestionTest(unittest.TestCase):
    """LOFT's `task` is a page of citation instructions with the query on the last
    line. Interpolated per-window it derails a 4B sub-model; at the reduce step it
    carries the output contract the scorer matches on."""

    def test_the_loft_wrapper_is_stripped_to_its_query(self):
        self.assertEqual(bare_query(LOFT_TASK), "when did season 4 of glee come out")

    def test_a_task_that_is_already_a_question_is_untouched(self):
        self.assertEqual(bare_query("who invented the kazoo?"), "who invented the kazoo?")

    def test_text_with_no_query_line_is_untouched(self):
        task = "Read the documents.\nThen answer.\nWhat is the capital of Peru?"
        self.assertEqual(bare_query(task), task)

    def test_the_last_query_line_wins(self):
        # A document quoted inside the task can contain the word too; the wrapper's
        # own query is always last.
        self.assertEqual(bare_query("query: a decoy\nquery: the real one"), "the real one")

    def test_the_map_step_asks_only_the_query_when_told_to(self):
        root, sub, rlm = build(["fact"], structured_map_question="query")
        rlm.run_structured(DOCUMENT, LOFT_TASK, hits((0, 50)), answer_format="Final Answer: ")
        self.assertIn("when did season 4 of glee come out", sub.asked[0])
        self.assertNotIn("Print out the TITLE", sub.asked[0])

    def test_the_reduce_step_still_sees_the_whole_task(self):
        root, sub, rlm = build(["fact"], structured_map_question="query")
        rlm.run_structured(DOCUMENT, LOFT_TASK, hits((0, 50)), answer_format="Final Answer: ")
        self.assertIn("Print out the TITLE", root.asked[0])

    def test_the_default_interpolates_the_task_verbatim(self):
        root, sub, rlm = build(["fact"])
        rlm.run_structured(DOCUMENT, LOFT_TASK, hits((0, 50)))
        self.assertIn("Print out the TITLE", sub.asked[0])


class MapGranularityTest(unittest.TestCase):
    """How many windows go into one map call, as an axis rather than a property of
    the pipeline: `all` is the payload the agentic root assembles."""

    def test_window_reads_each_provided_window_alone(self):
        root, sub, rlm = build(["fact"], structured_map_granularity="window")
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 2000), (1600, 3600)))
        self.assertEqual(sub.calls, 2)
        self.assertEqual(result.metrics["map_spans"], 2)

    def test_all_reads_every_window_in_one_call(self):
        root, sub, rlm = build(["fact"], structured_map_granularity="all")
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 2000), (6000, 8000)))
        self.assertEqual(sub.calls, 1)
        self.assertEqual(result.metrics["map_spans"], 1)
        self.assertEqual(result.metrics["chunks_provided"], 2)
        # Both windows reach the one call, which is what makes it arm B's payload.
        self.assertIn(DOCUMENT[0:2000], sub.asked[0])
        self.assertIn(DOCUMENT[6000:8000], sub.asked[0])

    def test_all_still_folds_the_shared_overlap_once(self):
        root, sub, rlm = build(["fact"], structured_map_granularity="all")
        rlm.run_structured(DOCUMENT, "q", hits((0, 2000), (1600, 3600)))
        self.assertEqual(sub.asked[0].count(DOCUMENT[1600:2000]), 1)

    def test_every_granularity_emits_document_order(self):
        # Ordering is held constant across the axis on purpose: revision 1 emitted
        # rank order, and restoring that here would put a third variable inside a
        # two-way ablation.
        ranked = hits((8000, 10000), (0, 2000), (4000, 6000))
        for granularity in ("passage", "window", "all"):
            groups = _map_groups(ranked, granularity)
            starts = [hit.start for group in groups for hit in group]
            self.assertEqual(starts, sorted(starts), granularity)

    def test_an_unknown_granularity_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            build(["fact"], structured_map_granularity="paragraph")

    def test_the_default_is_the_shipped_passage_clustering(self):
        root, sub, rlm = build(["fact"])
        result = rlm.run_structured(DOCUMENT, "q", hits((0, 2000), (1600, 3600)))
        self.assertEqual(result.metrics["map_spans"], 1)


class AnswerShapeOfferTest(unittest.TestCase):
    """Revision 1's wording, restored whole, so the shape change can be scored apart
    from the clustering change it shipped with."""

    def test_offer_restores_the_revision_one_format_note(self):
        root, sub, rlm = build(["fact"], structured_answer_shape="offer")
        rlm.run_structured(DOCUMENT, "q", hits((0, 50)), answer_format="Final Answer: ")
        prompt = root.asked[0]
        self.assertIn("Format your answer as: Final Answer:", prompt)
        self.assertIn("Be concise.", prompt)
        self.assertNotIn("no sentence built around it", prompt)

    def test_offer_without_an_answer_format_adds_no_note_at_all(self):
        root, sub, rlm = build(["fact"], structured_answer_shape="offer")
        rlm.run_structured(DOCUMENT, "q", hits((0, 50)))
        self.assertNotIn("answer itself and nothing else", root.asked[0])
        self.assertIn("Be concise.", root.asked[0])

    def test_require_is_the_default_and_is_unchanged(self):
        root, sub, rlm = build(["fact"])
        rlm.run_structured(DOCUMENT, "q", hits((0, 50)), answer_format="Final Answer: ")
        self.assertIn("no sentence built around it", root.asked[0])
        self.assertNotIn("Be concise.", root.asked[0])

    def test_an_unknown_shape_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            build(["fact"], structured_answer_shape="demand")
