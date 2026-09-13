# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Question-free organization, lossless packing, and the ordered structured read.

Pure: a character-count token counter and fake clients, no torch, no GPU. One test
uses the real Qwen tokenizer and skips when it is not cached.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.rlm.organization import (
    CONFIG,
    ModelTokens,
    OrderedPayload,
    graph_order,
    ordered_hits,
    organize,
    pack,
    reader_messages,
)
from evaluation.rlm.rlm import RLM, SUB_SYSTEM_PROMPT

SIZE = CONFIG["chunk_chars"]
REPO = Path(__file__).resolve().parents[2]


def chunk(words):
    """Exactly SIZE characters of the given words, repeated."""
    text = (" ".join(words) + " ") * (SIZE // len(" ".join(words)) + 2)
    return text[:SIZE]


TOPIC_A = ["zebra", "giraffe", "savanna", "lion"]
TOPIC_B = ["piano", "violin", "cello", "sonata"]
TOPIC_C = ["glacier", "fjord", "iceberg", "tundra"]
# A B A C B -- graph order must pull the two A's and the two B's together.
DOCUMENT = "".join(chunk(t) for t in (TOPIC_A, TOPIC_B, TOPIC_A, TOPIC_C, TOPIC_B))


class CharCounter:
    """Deterministic stand-in for the chat-template count: ~4 chars per token."""

    def __init__(self):
        self.calls = 0

    def count_messages(self, messages):
        self.calls += 1
        return sum(len(m["content"]) for m in messages) // 4 + 8


class GraphOrderTest(unittest.TestCase):
    def test_self_scores_are_removed_before_normalizing_and_directions_are_averaged(self):
        # The diagonal (100) would set every row's scale if it were not dropped.
        rows = [[100, 10, 0], [2, 100, 1], [0, 0, 100]]
        order, edges, forest = graph_order(rows)
        # row0 -> [0, 1, 0]; row1 -> [1, 0, 0.5]; row2 has no positive off-diagonal.
        self.assertEqual(edges, [(0, 1, 1.0), (1, 2, 0.25)])
        self.assertEqual(forest, edges)
        self.assertEqual(order, [0, 1, 2])

    def test_depth_first_takes_the_strongest_edge_first(self):
        rows = [[0, 1, 2], [1, 0, 0], [2, 0, 0]]  # w01 = 0.75, w02 = 1.0
        self.assertEqual(graph_order(rows)[0], [0, 2, 1])

    def test_ties_resolve_by_source_index(self):
        rows = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
        order, edges, _ = graph_order(rows, neighbors=1)
        self.assertEqual(order, [0, 1, 2])
        # Node 0 keeps only its top-1 edge, and the tie goes to the lower index.
        self.assertIn((0, 1, 1.0), edges)

    def test_disconnected_components_each_start_at_their_earliest_chunk(self):
        rows = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
        order, _, forest = graph_order(rows)
        self.assertEqual(order, [0, 2, 1, 3])
        self.assertEqual(len(forest), 2)

    def test_isolated_chunks_are_kept(self):
        rows = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
        self.assertEqual(graph_order(rows)[0], [0, 1, 2])

    def test_neighbor_cap_is_a_union_over_nodes(self):
        # Node 0's best neighbour is 1, but node 2's best is 0: the union keeps both.
        rows = [[0, 5, 1], [5, 0, 0], [1, 0, 0]]
        _, edges, _ = graph_order(rows, neighbors=1)
        self.assertEqual({(i, j) for i, j, _ in edges}, {(0, 1), (0, 2)})


class OrganizeTest(unittest.TestCase):
    def test_graph_groups_related_chunks_and_keeps_every_chunk_once(self):
        artifact, _, hit = organize(DOCUMENT, "bm25-graph")
        self.assertFalse(hit)
        self.assertEqual(artifact["order"], [0, 2, 1, 4, 3])
        self.assertEqual(sorted(artifact["order"]), list(range(5)))
        self.assertEqual(artifact["spans"], [[i * SIZE, (i + 1) * SIZE] for i in range(5)])

    def test_source_order_is_the_identity_with_no_graph(self):
        artifact, _, _ = organize(DOCUMENT, "source")
        self.assertEqual(artifact["order"], list(range(5)))
        self.assertEqual(artifact["edges"], [])

    def test_empty_lexical_content_is_an_isolated_chunk(self):
        document = DOCUMENT[:SIZE] + chunk(["the", "and", "of"]) + DOCUMENT[:SIZE]
        artifact, _, _ = organize(document, "bm25-graph")
        self.assertEqual(artifact["order"], [0, 2, 1])
        self.assertNotIn(1, {n for i, j, _ in artifact["edges"] for n in (i, j)})

    def test_empty_document(self):
        artifact, _, _ = organize("", "bm25-graph")
        self.assertEqual((artifact["spans"], artifact["order"]), ([], []))

    def test_last_chunk_may_be_short(self):
        artifact, _, _ = organize(DOCUMENT + "zebra tail", "bm25-graph")
        self.assertEqual(artifact["spans"][-1], [5 * SIZE, 5 * SIZE + 10])

    def test_unknown_order_is_refused(self):
        with self.assertRaises(ValueError):
            organize(DOCUMENT, "llm")

    def test_cache_round_trip_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, _, hit1 = organize(DOCUMENT, "bm25-graph", tmp)
            second, _, hit2 = organize(DOCUMENT, "bm25-graph", tmp)
            self.assertEqual((hit1, hit2), (False, True))
            self.assertEqual(first, second)
            (path,) = Path(tmp).glob("*.json")
            tampered = json.loads(path.read_text())
            tampered["config"]["neighbors"] = 99
            path.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                organize(DOCUMENT, "bm25-graph", tmp)

    def test_order_does_not_depend_on_the_hash_seed(self):
        # BM25 sums floats over query terms; iterating a set of strings varies with
        # PYTHONHASHSEED, which could flip near-ties between processes.
        script = (
            "import json; from evaluation.rlm.organization import organize; "
            "from tests.evaluation.test_rlm_organization import DOCUMENT; "
            "a = organize(DOCUMENT, 'bm25-graph')[0]; print(json.dumps([a['order'], a['edges']]))"
        )
        outputs = set()
        for seed in ("1", "2", "3"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(REPO))
            outputs.add(subprocess.run([sys.executable, "-c", script], env=env, cwd=REPO,
                                       capture_output=True, text=True, check=True).stdout)
        self.assertEqual(len(outputs), 1)

    def test_ordered_hits_follow_the_artifact_and_carry_source_text(self):
        artifact, _, _ = organize(DOCUMENT, "bm25-graph")
        hits = ordered_hits(DOCUMENT, artifact)
        self.assertEqual([h.start // SIZE for h in hits], artifact["order"])
        for h in hits:
            self.assertEqual(h.text, DOCUMENT[h.start:h.end])
        with self.assertRaises(ValueError):
            ordered_hits(DOCUMENT + "x", artifact)


class PackTest(unittest.TestCase):
    def setUp(self):
        artifact, _, _ = organize(DOCUMENT, "bm25-graph")
        self.hits = ordered_hits(DOCUMENT, artifact)
        self.counter = CharCounter()

    def _pack(self, budget, question="what?"):
        return pack(DOCUMENT, self.hits, question, SUB_SYSTEM_PROMPT, self.counter, budget)

    def size(self, group):
        """The complete reader message's count, exactly as pack() priced it."""
        return self.counter.count_messages(reader_messages(SUB_SYSTEM_PROMPT, "what?", group, DOCUMENT))

    def test_every_chunk_once_in_order_and_within_budget(self):
        budget = 1300  # fits two chunks per group under the char counter
        groups = self._pack(budget)
        flat = [h for g in groups for h in g.hits]
        self.assertEqual(flat, self.hits)
        self.assertEqual(sum(h.end - h.start for h in flat), len(DOCUMENT))
        for g in groups:
            self.assertLessEqual(self.size(g), budget)
        self.assertEqual([len(g.hits) for g in groups], [2, 2, 1])

    def test_distant_chunks_keep_graph_order_and_separate_labels(self):
        groups = self._pack(1300)
        text = groups[0].render(DOCUMENT)
        self.assertLess(text.index(f"[source chars 0:{SIZE}]"), text.index(f"[source chars {2 * SIZE}:{3 * SIZE}]"))
        self.assertNotIn(f"[source chars 0:{3 * SIZE}]", text)

    def test_the_question_counts_against_the_budget(self):
        # ~500 tokens of question: two whole chunks no longer fit beside it.
        self.assertEqual(len(self._pack(1300, question="q " * 1000)), 5)
        self.assertEqual(len(self._pack(1300)), 3)

    def test_oversized_chunks_are_split_without_losing_text(self):
        groups = self._pack(400)  # under one whole chunk
        flat = [h for g in groups for h in g.hits]
        self.assertGreater(len(flat), len(self.hits))
        # Pieces stay contiguous and in the graph order of their parent chunk.
        self.assertEqual("".join(h.text for h in flat), "".join(h.text for h in self.hits))
        for h in flat:
            self.assertEqual(h.text, DOCUMENT[h.start:h.end])
        for g in groups:
            self.assertLessEqual(self.size(g), 400)

    def test_a_budget_the_instructions_alone_exceed_is_refused(self):
        with self.assertRaises(ValueError):
            self._pack(10)
        with self.assertRaises(ValueError):
            self._pack(0)

    def test_payload_render_refuses_text_that_is_not_the_source(self):
        bad = self.hits[0].__class__(1, 0, 10, 0.0, "not the doc")
        with self.assertRaises(ValueError):
            OrderedPayload((bad,)).render(DOCUMENT)


class _Usage:
    def __init__(self):
        self.last_prompt_tokens = None


class _Client:
    """Records full message lists; reports the prompt size a server would."""

    def __init__(self, replies, counter=None, max_tokens=512):
        self.replies = list(replies)
        self.messages: list = []
        self.counter = counter
        self.max_tokens = max_tokens
        self.usage = _Usage()

    def chat(self, messages, **kw):
        self.messages.append(messages)
        if self.counter is not None:
            self.usage.last_prompt_tokens = self.counter.count_messages(messages)
        return self.replies[min(len(self.messages) - 1, len(self.replies) - 1)]


class OrderedStructuredTest(unittest.TestCase):
    TASK = "Which animals live on the savanna?"

    def build(self, sub_replies, budget=1300, **kw):
        counter = CharCounter()
        sub = _Client(sub_replies, counter)
        root = _Client(["Final Answer: zebra"], max_tokens=1024)
        rlm = RLM(
            root_client=root,
            sub_client=sub,
            ordered_token_counter=counter,
            structured_group_input_tokens=kw.pop("reader_budget", budget),
            max_sub_calls=kw.pop("max_sub_calls", None),
            run_timeout=None,
            **kw,
        )
        artifact, _, _ = organize(DOCUMENT, "bm25-graph")
        groups = pack(DOCUMENT, ordered_hits(DOCUMENT, artifact), rlm.structured_question(self.TASK),
                      SUB_SYSTEM_PROMPT, counter, budget)
        return rlm, sub, root, groups

    def test_reader_payloads_and_reduce_order_follow_the_groups(self):
        rlm, sub, root, groups = self.build(["fact one", "fact two", "fact three"])
        result = rlm.run_structured(DOCUMENT, self.TASK, [], ordered_groups=groups)
        question = rlm.structured_question(self.TASK)
        self.assertEqual(sub.messages, [reader_messages(SUB_SYSTEM_PROMPT, question, g, DOCUMENT) for g in groups])
        reduce = root.messages[0][-1]["content"]
        self.assertIn("in reader-group order", reduce)
        self.assertLess(reduce.index("fact one"), reduce.index("fact two"))
        self.assertLess(reduce.index("fact two"), reduce.index("fact three"))
        m = result.metrics
        self.assertEqual(m["document_coverage_fraction"], 1.0)
        self.assertEqual((m["map_calls"], m["group_count"]), (3, 3))
        self.assertEqual(m["group_chunk_counts"], [2, 2, 1])
        self.assertEqual(m["reader_input_tokens"], m["reader_served_prompt_tokens"])
        self.assertEqual(m["retained_finding_count"], 3)
        self.assertEqual(result.end_reason, "structured_complete")
        self.assertEqual(result.transcript[0]["source_spans"], [(0, SIZE), (2 * SIZE, 3 * SIZE)])

    def test_legacy_structured_runs_still_say_document_order(self):
        rlm, sub, root, groups = self.build(["fact"])
        rlm.run_structured(DOCUMENT, self.TASK, list(groups[0].hits))
        self.assertIn("in document order", root.messages[0][-1]["content"])

    def test_a_gap_in_coverage_is_an_error(self):
        rlm, _, _, groups = self.build(["fact"])
        with self.assertRaises(ValueError):
            rlm.run_structured(DOCUMENT, self.TASK, [], ordered_groups=groups[:-1])

    def test_a_group_over_the_reader_budget_is_an_error(self):
        rlm, sub, _, groups = self.build(["fact"], reader_budget=1000)
        with self.assertRaises(ValueError):
            rlm.run_structured(DOCUMENT, self.TASK, [], ordered_groups=groups)
        self.assertEqual(sub.messages, [])  # refused before sending

    def test_a_failed_reader_is_an_error(self):
        rlm, _, _, groups = self.build([""])
        with self.assertRaises(ValueError):
            rlm.run_structured(DOCUMENT, self.TASK, [], ordered_groups=groups)

    def test_a_map_call_limit_is_an_error_not_a_finding(self):
        rlm, _, _, groups = self.build(["fact"], max_sub_calls=1)
        with self.assertRaises(ValueError):
            rlm.run_structured(DOCUMENT, self.TASK, [], ordered_groups=groups)

    def test_reducer_overflow_is_an_error(self):
        rlm, _, root, groups = self.build(["fact"])
        root.max_tokens = rlm.structured_context_tokens
        with self.assertRaises(ValueError):
            rlm.run_structured(DOCUMENT, self.TASK, [], ordered_groups=groups)
        self.assertEqual(root.messages, [])

    def test_ordered_groups_refuse_retrieval_and_notes(self):
        rlm, _, _, groups = self.build(["fact"])
        with self.assertRaises(ValueError):
            rlm.run_structured(DOCUMENT, self.TASK, [], ordered_groups=groups, retrieve=lambda q, k: [])


def _cached_tokenizer():
    try:
        return ModelTokens("Qwen/Qwen3-4B-Instruct-2507")
    except Exception:
        return None


class RealTokenizerTest(unittest.TestCase):
    def test_counts_the_full_chat_request(self):
        counter = _cached_tokenizer()
        if counter is None:
            self.skipTest("Qwen3-4B-Instruct-2507 tokenizer not cached")
        short = counter.count_messages([dict(role="system", content="s"), dict(role="user", content="hi")])
        longer = counter.count_messages([dict(role="system", content="s"), dict(role="user", content="hi " * 100)])
        self.assertGreater(short, 2)  # template tokens are counted, not just content
        self.assertGreaterEqual(longer - short, 99)


if __name__ == "__main__":
    unittest.main()
