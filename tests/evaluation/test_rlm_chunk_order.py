# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ordered-chunk agentic arms (--chunk-order): the views, the REPL, the prompt, the CLI.

Pure: a character-count token counter and fake clients, no torch, no GPU.
"""

import contextlib
import io
import sys
import tempfile
from unittest import mock

import pandas as pd
import pytest

from evaluation.benchmarks.loaders import _load_ruler128k, split_ruler_input
from evaluation.benchmarks.loft.calculate_metrics import extract_prediction
from evaluation.benchmarks.registry import DATASET_REGISTRY, SCORER_REGISTRY
from evaluation.benchmarks.ruler.create_huggingface_dataset import ANSWER_PATTERNS, QUESTION_PATTERNS
from evaluation.rlm import retrieval
from evaluation.rlm import run_benchmark as rb
from evaluation.rlm.organization import CONFIG, OrderedPayload, ordered_hits, organize, reader_messages
from evaluation.rlm.retrieval import Hit
from evaluation.rlm.rlm import (
    CHUNK_ORDER_NOTES,
    CHUNK_READS_BEFORE_ABSTAIN,
    CHUNK_WORKED_SESSIONS,
    HIT_SEPARATOR,
    PREMATURE_ABSTAIN_NUDGE,
    PREMATURE_ABSTAIN_NUDGE_CHUNK_READS,
    PREMATURE_ABSTAIN_NUDGE_CHUNKS,
    PROSE_ANSWER_CELL_ERROR,
    RLM,
    SUB_SYSTEM_PROMPT,
    UNGROUNDED_FINAL_CHUNKS,
    composed_answer_grounded,
)
from tests.evaluation.test_rlm_run_benchmark import _args

SIZE = CONFIG["chunk_chars"]


def chunk(words):
    """Exactly SIZE characters of the given words, repeated."""
    text = (" ".join(words) + " ") * (SIZE // len(" ".join(words)) + 2)
    return text[:SIZE]


TOPIC_A = ["zebra", "giraffe", "savanna", "lion"]
TOPIC_B = ["piano", "violin", "cello", "sonata"]
TOPIC_C = ["glacier", "fjord", "iceberg", "tundra"]
# A B A C B: graph order pulls the two A's and the two B's together, and a question
# about the cello ranks the two B chunks first.
DOCUMENT = "".join(chunk(t) for t in (TOPIC_A, TOPIC_B, TOPIC_A, TOPIC_C, TOPIC_B))


class CharCounter:
    """Deterministic stand-in for the chat-template count: ~4 chars per token."""

    def count_messages(self, messages):
        return sum(len(m["content"]) for m in messages) // 4 + 8


class FakeClient:
    """A plain chat backend -- no `chat_split`, like the real LLMClient."""

    def __init__(self, reply="sub answer"):
        self.reply = reply
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        return self.reply


class ScriptedRoot(FakeClient):
    """A root that replies with the given code cells, one per turn."""

    def __init__(self, cells):
        super().__init__()
        self.cells = list(cells)

    def chat(self, messages, **kw):
        self.calls.append(messages)
        return f"```python\n{self.cells.pop(0)}\n```"


READ_THEN_ANSWER = ['ans = llm_query("Which cello sonata?", chunks[:2])\nprint(ans)', "FINAL(ans)"]


@pytest.fixture(autouse=True)
def _clean_cache():
    retrieval.reset_cache()
    yield
    retrieval.reset_cache()


def query_view(question="cello sonata"):
    return retrieval.ChunkIndex(DOCUMENT, chunk_chars=SIZE, overlap=0).rank_all(question)


def graph_view():
    artifact, _, _ = organize(DOCUMENT, "bm25-graph")
    return ordered_hits(DOCUMENT, artifact)


def base_metrics():
    return {
        "sub_calls": 0,
        "sub_call_tokens": 0,
        "sub_cache_hits": 0,
        "sub_fit_failures": 0,
        "sub_slice_unlocatable_calls": 0,
        "document_coverage_fraction": 0.0,
    }


def build(order="bm25-query", view=None, sub=None, **kw):
    """An ordered-chunk RLM and its REPL, without running the loop."""
    rlm = RLM(
        root_client=FakeClient(),
        sub_client=sub or FakeClient(),
        chunk_order=order,
        ordered_token_counter=CharCounter(),
        **kw,
    )
    metrics = base_metrics()
    env = rlm._make_env(DOCUMENT, metrics, {}, [], ordered_chunks=view if view is not None else query_view())
    return rlm, env, metrics


def run_once(order, view, cells=READ_THEN_ANSWER, **kw):
    """One full agentic run; returns the result, the system prompt and the root."""
    root = ScriptedRoot(cells)
    rlm = RLM(
        root_client=root,
        sub_client=FakeClient(),
        chunk_order=order,
        ordered_token_counter=CharCounter(),
        **kw,
    )
    result = rlm.run(DOCUMENT, "Which cello sonata?", ordered_chunks=view)
    return result, root.calls[0][0]["content"], root


class TestRankAll:
    def test_every_chunk_exactly_once_ranked_in_list_order(self):
        index = retrieval.ChunkIndex(DOCUMENT, chunk_chars=SIZE, overlap=0)
        view = index.rank_all("cello sonata")
        assert sorted((h.start, h.end) for h in view) == index.spans
        assert [h.rank for h in view] == list(range(1, index.n_chunks + 1))
        assert all(h.text == DOCUMENT[h.start : h.end] for h in view)

    def test_scored_chunks_come_first_and_the_rest_keep_source_order(self):
        view = query_view("cello sonata")
        # The two B chunks tie on score; the tie goes to the earlier chunk.
        assert [h.start // SIZE for h in view] == [1, 4, 0, 2, 3]
        assert all(h.score > 0 for h in view[:2]) and all(h.score == 0.0 for h in view[2:])

    def test_its_scored_prefix_is_exactly_search(self):
        index = retrieval.ChunkIndex(DOCUMENT, chunk_chars=SIZE, overlap=0)
        ranked = [h for h in index.rank_all("cello zebra") if h.score > 0]
        assert ranked == index.search("cello zebra", k=index.n_chunks)

    def test_a_query_sharing_no_term_leaves_source_order(self):
        assert [h.start // SIZE for h in query_view("xylophone")] == [0, 1, 2, 3, 4]


class TestViews:
    def test_the_graph_view_is_the_organization_and_its_cache(self, tmp_path):
        view, stats = rb.build_chunk_view(DOCUMENT, "q", "bm25-graph", "loft", str(tmp_path), {})
        assert view == graph_view()
        assert stats["organization_cache_hit"] is False
        again, stats = rb.build_chunk_view(DOCUMENT, "another q", "bm25-graph", "loft", str(tmp_path), {})
        assert again == view and stats["organization_cache_hit"] is True

    def test_both_orders_are_permutations_of_one_partition(self, tmp_path):
        graph, _ = rb.build_chunk_view(DOCUMENT, "cello", "bm25-graph", "loft", str(tmp_path), {})
        query, stats = rb.build_chunk_view(DOCUMENT, "cello", "bm25-query", "loft", None, {})
        assert sorted((h.start, h.end) for h in graph) == sorted((h.start, h.end) for h in query)
        assert stats["chunks_total"] == 5 and stats["chunks_scored"] == 2

    def test_the_query_view_follows_the_question_and_the_graph_view_does_not(self, tmp_path):
        views = {
            (order, q): rb.build_chunk_view(DOCUMENT, q, order, "loft", str(tmp_path), {})[1]["chunk_view_sha256"]
            for order in ("bm25-graph", "bm25-query")
            for q in ("cello", "glacier")
        }
        assert views[("bm25-graph", "cello")] == views[("bm25-graph", "glacier")]
        assert views[("bm25-query", "cello")] != views[("bm25-query", "glacier")]

    def test_the_query_view_ranks_on_the_bare_question(self):
        task = "You will be given a list of documents. Print the TITLE and ID of each.\nquery: which cello sonata"
        _, stats = rb.build_chunk_view(DOCUMENT, task, "bm25-query", "loft", None, {})
        assert stats["chunk_order_query"] == "which cello sonata"

    def test_ruler_queries_drop_the_instructions_around_the_question(self):
        question = (
            "Answer the question based on the given documents. Only give me the answer and do not output any "
            "other words.\n\nQuestion: Were Scott Derrickson and Ed Wood of the same nationality? "
        )
        expected = "Were Scott Derrickson and Ed Wood of the same nationality?"
        assert rb.chunk_order_query(question, "ruler128k") == expected
        # LOFT is never cut at a `Question:` its query might contain.
        assert rb.chunk_order_query("query: Question: what", "loft") == "Question: what"


class TestRepl:
    def test_chunks_are_bound_and_search_is_not(self):
        _, env, _ = build()
        assert env["chunks"] == query_view()
        assert "search" not in env

    def test_legacy_runs_bind_no_chunks(self):
        rlm = RLM(root_client=FakeClient(), sub_client=FakeClient())
        assert "chunks" not in rlm._make_env(DOCUMENT, base_metrics(), {}, [])

    def test_a_read_keeps_the_order_it_was_given_and_labels_every_chunk(self):
        sub = FakeClient()
        _, env, metrics = build(sub=sub)
        picked = env["chunks"][2::-1]
        env["llm_query"]("Q?", picked)
        expected = reader_messages(SUB_SYSTEM_PROMPT, "Q?", OrderedPayload(tuple(picked)), DOCUMENT)
        assert sub.calls == [expected]
        assert metrics["chunk_payload_calls"] == 1
        assert metrics["chunk_ranks_read"] == [3, 2, 1]
        assert metrics["reader_input_tokens"] == [CharCounter().count_messages(expected)]
        assert metrics["reader_served_prompt_tokens"] == [None]

    def test_one_chunk_on_its_own_is_a_read(self):
        sub = FakeClient()
        _, env, metrics = build(sub=sub)
        env["llm_query"]("Q?", env["chunks"][0])
        assert metrics["chunk_payload_calls"] == 1 and len(sub.calls) == 1

    def test_a_generator_of_chunks_is_read_once_in_order(self):
        sub = FakeClient()
        _, env, metrics = build(sub=sub)
        env["llm_query"]("Q?", (c for c in env["chunks"][:2]))
        assert metrics["chunk_ranks_read"] == [1, 2]

    def test_coverage_counts_what_was_read(self):
        _, env, metrics = build()
        env["llm_query"]("Q?", env["chunks"][:2])
        assert metrics["document_coverage_fraction"] == pytest.approx(2 * SIZE / len(DOCUMENT))

    def test_hits_that_are_not_this_views_chunks_keep_the_merge_path(self):
        sub = FakeClient()
        _, env, metrics = build(sub=sub)
        foreign = [Hit(1, 10, 110, 0.0, DOCUMENT[10:110]), Hit(2, 5000, 5100, 0.0, DOCUMENT[5000:5100])]
        env["llm_query"]("Q?", foreign)
        assert HIT_SEPARATOR in sub.calls[0][1]["content"]
        assert metrics.get("chunk_payload_calls", 0) == 0

    def test_a_repeated_read_is_answered_from_cache_without_a_lonely_local_count(self):
        sub = FakeClient()
        _, env, metrics = build(sub=sub)
        env["llm_query"]("Q?", env["chunks"][:2])
        env["llm_query"]("Q?", env["chunks"][:2])
        assert len(sub.calls) == 1
        assert len(metrics["reader_input_tokens"]) == len(metrics["reader_served_prompt_tokens"]) == 1

    def test_an_over_budget_read_is_refused_before_anything_is_sent(self):
        sub = FakeClient()
        # A chunk is ~520 tokens to CharCounter, so a 1200-token call fits two of them.
        _, env, metrics = build(sub=sub, chunk_call_tokens=1200)
        with pytest.raises(ValueError, match=r"only the first 2 fit"):
            env["llm_query"]("Q?", env["chunks"][:3])
        assert sub.calls == []
        assert metrics["chunk_budget_refusals"] == 1
        assert metrics.get("reader_input_tokens", []) == []

    def test_a_single_chunk_over_budget_asks_for_a_shorter_question(self):
        _, env, _ = build(chunk_call_tokens=100)
        with pytest.raises(ValueError, match="Shorten the question"):
            env["llm_query"]("Q?", env["chunks"][0])

    def test_an_empty_list_of_chunks_is_refused_and_names_the_case_trap(self):
        sub = FakeClient()
        _, env, metrics = build(sub=sub)
        # The pilot's failure: a capitalised term against .lower() text matches nothing.
        with pytest.raises(ValueError, match="EMPTY.*lower case"):
            env["llm_query"]("Q?", [c for c in env["chunks"] if "Cello" in c.text.lower()])
        with pytest.raises(ValueError, match="EMPTY"):
            env["llm_query"]("Q?", (c for c in env["chunks"] if False))
        assert sub.calls == [] and metrics["empty_chunk_reads"] == 2
        assert metrics.get("chunk_payload_calls", 0) == 0


class TestRun:
    def test_a_run_reports_what_was_offered_and_what_was_read(self):
        result, _, _ = run_once("bm25-query", query_view())
        m = result.metrics
        assert result.end_reason == "final_called" and result.answer == "sub answer"
        assert m["chunks_total"] == 5 and m["chunk_payload_calls"] == 1 and m["chunk_ranks_read"] == [1, 2]
        assert len(m["reader_input_tokens"]) == len(m["reader_served_prompt_tokens"]) == 1
        assert m["chunks_per_call_advertised"] >= 1

    def test_an_order_without_a_view_is_refused(self):
        rlm = RLM(root_client=FakeClient(), chunk_order="bm25-query", ordered_token_counter=CharCounter())
        with pytest.raises(ValueError, match="come together"):
            rlm.run(DOCUMENT, "q")

    def test_a_view_without_an_order_is_refused(self):
        with pytest.raises(ValueError, match="come together"):
            RLM(root_client=FakeClient()).run(DOCUMENT, "q", ordered_chunks=query_view())

    def test_a_view_that_is_not_the_whole_document_is_refused(self):
        partial = [Hit(i, h.start, h.end, h.score, h.text) for i, h in enumerate(query_view()[:4], start=1)]
        rlm = RLM(root_client=FakeClient(), chunk_order="bm25-query", ordered_token_counter=CharCounter())
        with pytest.raises(ValueError, match="exactly once"):
            rlm.run(DOCUMENT, "q", ordered_chunks=partial)

    def test_a_view_without_a_tokenizer_is_refused(self):
        rlm = RLM(root_client=FakeClient(), chunk_order="bm25-query")
        with pytest.raises(ValueError, match="tokenizer"):
            rlm.run(DOCUMENT, "q", ordered_chunks=query_view())

    def test_search_and_unknown_orders_are_refused_but_a_split_sub_is_a_bxf_cell(self):
        with pytest.raises(ValueError, match="no search"):
            RLM(root_client=FakeClient(), chunk_order="bm25-query", search_k=5)
        # A split-capable sub (KVzipSubClient) is how a B x F cell reads its chunks.
        assert RLM(root_client=FakeClient(), sub_client=SplitClient(), chunk_order="bm25-query").chunk_order
        with pytest.raises(ValueError, match="unknown chunk_order"):
            RLM(root_client=FakeClient(), chunk_order="random")

    def test_a_premature_abstention_is_nudged_toward_the_chunks(self):
        cells = ['FINAL_NONE("absent")', *READ_THEN_ANSWER]
        _, _, root = run_once("bm25-query", query_view(), cells=cells)
        nudges = [m["content"] for m in root.calls[1] if m["role"] == "user"]
        assert PREMATURE_ABSTAIN_NUDGE_CHUNKS in nudges and PREMATURE_ABSTAIN_NUDGE not in nudges

    def test_one_read_is_too_little_to_abstain_and_the_nudge_says_how_much_was_read(self):
        cells = [
            'a = llm_query("Which cello sonata?", chunks[:2])\nprint(a)',
            'FINAL_NONE("absent")',
            'b = llm_query("Which violin?", chunks[2:4])\nprint(b)',
            'FINAL_NONE("absent")',
            'c = llm_query("Which piano?", chunks[4:])\nprint(c)',
            'FINAL_NONE("absent")',
        ]
        assert CHUNK_READS_BEFORE_ABSTAIN == 3
        result, _, root = run_once("bm25-query", query_view(), cells=cells)
        nudges = [m["content"] for m in root.calls[-1] if m["role"] == "user"]
        assert PREMATURE_ABSTAIN_NUDGE_CHUNK_READS.format(read=2, total=5) in nudges
        assert PREMATURE_ABSTAIN_NUDGE_CHUNK_READS.format(read=4, total=5) in nudges
        assert result.end_reason == "abstained" and result.metrics["premature_abstentions"] == 2
        assert result.metrics["chunk_payload_calls"] == 3

    def test_legacy_runs_still_abstain_after_one_sub_call(self):
        cells = ['ans = llm_query("Which cello sonata? " + context[:100])\nprint(ans)', 'FINAL_NONE("absent")']
        result = RLM(root_client=ScriptedRoot(cells), sub_client=FakeClient()).run(DOCUMENT, "Which cello sonata?")
        assert result.end_reason == "abstained" and "premature_abstentions" not in result.metrics

    def test_a_final_answer_line_is_told_to_call_final(self):
        cells = [READ_THEN_ANSWER[0], "Final Answer: sub answer", "FINAL(ans)"]
        result, _, root = run_once("bm25-query", query_view(), cells=cells)
        assert result.end_reason == "final_called" and result.answer == "sub answer"
        assert result.metrics["prose_answer_cells"] == 1
        assert any(PROSE_ANSWER_CELL_ERROR in m["content"] for m in root.calls[2] if m["role"] == "user")

    def test_an_ungrounded_literal_is_rejected_without_the_find_recipe(self):
        cells = [READ_THEN_ANSWER[0], 'FINAL("made up answer")', "FINAL(ans)"]
        result, _, root = run_once("bm25-query", query_view(), cells=cells)
        rejections = [m["content"] for m in root.calls[2] if m["role"] == "user"]
        assert UNGROUNDED_FINAL_CHUNKS.format(val="made up answer") in rejections
        assert not any("context.find" in r for r in rejections)
        assert result.end_reason == "final_called" and result.answer == "sub answer"

    def test_the_final_answer_diagnosis_is_for_ordered_runs_only(self):
        rlm, env, _ = build()
        prose = (PROSE_ANSWER_CELL_ERROR, PROSE_ANSWER_CELL_ERROR)
        assert rlm._exec("x = 1\nFinal Answer: 2000", env, 4000) == prose
        # `answer: x` is an annotation, valid Python: it runs as written.
        assert "x" in rlm._exec('answer: str = "x"\nprint(answer)', env, 4000)[0]
        legacy = RLM(root_client=FakeClient(), sub_client=FakeClient())
        obs, _ = legacy._exec("Final Answer: 2000", legacy._make_env(DOCUMENT, base_metrics(), {}, []), 4000)
        assert obs.startswith("[SYNTAX ERROR] Your code block was written on a single line")


class Replies(FakeClient):
    """A sub-model that answers each read with the next reply in turn."""

    def __init__(self, *replies):
        super().__init__()
        self.replies = list(replies)

    def chat(self, messages, **kw):
        self.calls.append(messages)
        return self.replies.pop(0)


def run_composed(cells, replies, answer_format=None):
    """One ordered-chunk run whose reads return `replies`; returns the result and the root."""
    root = ScriptedRoot(cells)
    rlm = RLM(
        root_client=root, sub_client=Replies(*replies), chunk_order="bm25-query", ordered_token_counter=CharCounter()
    )
    return rlm.run(DOCUMENT, "Which cello sonatas?", answer_format=answer_format, ordered_chunks=query_view()), root


READ_ONE = 'a = llm_query("Which cello sonata?", chunks[:2])\nprint(a)'


class TestComposedAnswers:
    """Revision 6: FINAL accepts a literal composed from several reads, and nothing looser."""

    SEEN = "Reader: the first value is 3914404.\nReader: dusty-president -> 4968669.\nTotal: 2,548 items."

    def test_values_read_on_different_calls_ground_their_combination(self):
        assert composed_answer_grounded("3914404 and 4968669", self.SEEN, "Which values?")
        assert not composed_answer_grounded("3914404 and 1937991", self.SEEN, "Which values?")

    def test_digit_group_commas_do_not_matter(self):
        assert composed_answer_grounded("2548", self.SEEN, "How many items?")
        assert composed_answer_grounded("25,480", "the count was 25480", "How many?")

    def test_a_uuid_cannot_be_completed_past_what_was_printed(self):
        seen = "The uuid is c44540a1-cace-4a6"
        assert composed_answer_grounded("c44540a1-cace-4a6", seen, "Which uuid?")
        assert not composed_answer_grounded("c44540a1-cace-4a67-ad45-88ad7478a40e", seen, "Which uuid?")

    def test_task_words_alone_and_negations_ground_nothing(self):
        task = "What is the dusty-president value?"
        assert not composed_answer_grounded("the dusty-president value", self.SEEN, task)
        assert not composed_answer_grounded("3914404, the rest are none", self.SEEN + " none", "Which values?")

    def test_an_answer_composed_from_several_reads_is_accepted(self):
        cells = [
            READ_ONE,
            'b = llm_query("Which violin sonata?", chunks[2:4])\nprint(b)',
            'FINAL("Op. 1111 and Op. 2222")',
        ]
        result, _ = run_composed(cells, ["The cello sonata is Op. 1111.", "The violin sonata is Op. 2222."])
        assert result.end_reason == "final_called" and result.answer == "Op. 1111 and Op. 2222"

    def test_the_answer_template_filled_with_a_read_value_is_accepted(self):
        cells = [READ_ONE, 'FINAL("The sonata for cello is Op. 1111.")', "FINAL(a)"]
        result, _ = run_composed(cells, ["It is Op. 1111."], answer_format="The sonata for cello is")
        assert result.end_reason == "final_called" and result.answer == "The sonata for cello is Op. 1111."
        # Without the template, its words are the root's own, and no read printed them.
        result, _ = run_composed(cells, ["It is Op. 1111."])
        assert result.answer == "It is Op. 1111."

    def test_a_value_no_read_returned_or_an_added_negation_is_still_rejected(self):
        for literal, reply in (
            ("Op. 1111 and Op. 3333", "The cello sonata is Op. 1111."),
            ("Op. 1111, others not found", "The cello sonata is Op. 1111. Others are not found."),
        ):
            result, root = run_composed([READ_ONE, f'FINAL("{literal}")', "FINAL(a)"], [reply])
            rejections = [m["content"] for m in root.calls[2] if m["role"] == "user"]
            assert UNGROUNDED_FINAL_CHUNKS.format(val=literal) in rejections
            assert result.end_reason == "final_called" and result.answer == reply

    def test_legacy_runs_keep_the_word_for_word_test(self):
        cells = [
            'a = llm_query("Which cello sonata? " + context[:100])\nprint(a)',
            'b = llm_query("Which violin sonata? " + context[100:200])\nprint(b)',
            'FINAL("Op. 1111 and Op. 2222")',
            "FINAL(a)",
        ]
        replies = Replies("The cello sonata is Op. 1111.", "The violin sonata is Op. 2222.")
        result = RLM(root_client=ScriptedRoot(cells), sub_client=replies).run(DOCUMENT, "Which cello sonatas?")
        assert result.end_reason == "final_called" and result.answer == "The cello sonata is Op. 1111."


class TestPrompt:
    @pytest.mark.parametrize("order", ["bm25-query", "bm25-graph"])
    def test_each_order_is_described_and_demonstrated(self, order):
        view = query_view() if order == "bm25-query" else graph_view()
        _, prompt, _ = run_once(order, view)
        assert CHUNK_ORDER_NOTES[order] in prompt and CHUNK_WORKED_SESSIONS[order] in prompt
        # The help wraps across lines; compare with the whitespace folded.
        assert "5 consecutive pieces of 2000 characters" in " ".join(prompt.split())
        # The demonstration is what the root copies: neither arm may show it .find.
        assert ".find(" not in prompt

    def test_the_orders_differ_in_their_description_and_session_only(self):
        views = {"bm25-query": query_view(), "bm25-graph": graph_view()}
        prompts = {order: run_once(order, view)[1] for order, view in views.items()}
        stripped = {
            order: prompt.replace(CHUNK_ORDER_NOTES[order], "").replace(CHUNK_WORKED_SESSIONS[order], "")
            for order, prompt in prompts.items()
        }
        assert stripped["bm25-query"] == stripped["bm25-graph"]

    def test_the_list_shape_is_shown_only_when_asked(self):
        _, with_list, _ = run_once("bm25-query", query_view(), structured_answer_list='["a", "b"]')
        _, without, _ = run_once("bm25-query", query_view())
        assert "The answer is a LIST of separate items: pass FINAL(...) a bracketed" in with_list
        assert 'for example ["a", "b"]' in with_list
        assert "The answer is a LIST of separate items" not in without

    def test_legacy_agentic_prompts_ignore_the_list_example(self):
        root = ScriptedRoot(["FINAL(context[:5])"])
        RLM(root_client=root, sub_client=FakeClient(), structured_answer_list='["a"]').run(DOCUMENT, "q")
        assert "The answer is a LIST of separate items" not in root.calls[0][0]["content"]
        assert "`chunks`" not in root.calls[0][0]["content"]

    def test_a_list_passed_to_final_parses_the_way_the_multi_value_scorer_reads_it(self):
        assert extract_prediction(str(["Rival Schools", "Quicksand"])) == ["Rival Schools", "Quicksand"]
        assert extract_prediction('["a", "b"]') == ["a", "b"]

    @pytest.mark.parametrize("order", ["bm25-query", "bm25-graph"])
    def test_both_orders_are_told_to_write_filter_terms_in_lower_case(self, order):
        _, prompt, _ = run_once(order, query_view() if order == "bm25-query" else graph_view())
        assert '"steven spielberg" can match it, "Steven Spielberg" never will' in " ".join(prompt.split())


class SplitClient(FakeClient):
    """A split-capable sub, like KVzipSubClient: `context` is the side the press compresses."""

    def __init__(self, reply="sub answer"):
        super().__init__(reply)
        self.splits = []

    def chat_split(self, question, context, system=None, **kw):
        self.splits.append(dict(question=question, context=context, system=system))
        return self.reply


class TestKVzipReads:
    """A B x F cell: the same ordered read, served by a KV-press sub through chat_split."""

    def test_a_chunk_read_rides_the_compressed_context_side(self):
        sub = SplitClient()
        _, env, metrics = build(sub=sub)
        picked = env["chunks"][:2]
        env["llm_query"]("Which cello sonata?", picked)
        rendered = OrderedPayload(tuple(picked)).render(DOCUMENT)
        assert sub.splits == [dict(question="Which cello sonata?", context=rendered, system=SUB_SYSTEM_PROMPT)]
        assert sub.calls == []
        assert metrics["chunk_payload_calls"] == 1
        assert metrics["reader_served_prompt_tokens"] == [None]
        assert metrics["document_coverage_fraction"] == pytest.approx(2 * SIZE / len(DOCUMENT))

    def test_a_read_over_n_is_refused_and_nothing_is_sent(self):
        sub = SplitClient()
        _, env, metrics = build(sub=sub, chunk_call_tokens=600)
        with pytest.raises(ValueError, match="refused"):
            env["llm_query"]("Q?", env["chunks"][:3])
        assert sub.splits == [] and metrics["chunk_budget_refusals"] == 1


ORDERED_RLM = dict(mode="rlm", chunk_order="bm25-query", chunk_call_tokens=8192, split="test")


class TestRunDirAndConfig:
    def test_the_order_names_the_directory(self):
        components = rb.build_run_dir_components(_args(**ORDERED_RLM), "rlm", None)
        assert components[-2:] == ["split-test", "order-bm25-query"]

    def test_the_two_orders_never_share_a_directory(self):
        graph = rb.build_run_dir_components(_args(**{**ORDERED_RLM, "chunk_order": "bm25-graph"}), "rlm", None)
        assert graph != rb.build_run_dir_components(_args(**ORDERED_RLM), "rlm", None)

    def test_multi_value_subsets_carry_the_list_shape_stamp(self):
        components = rb.build_run_dir_components(_args(**ORDERED_RLM, data_dir="qampari_128k"), "rlm", None)
        assert components[-2:] == ["order-bm25-query", "listfmt"]

    def test_a_non_default_budget_is_stamped(self):
        components = rb.build_run_dir_components(_args(**{**ORDERED_RLM, "chunk_call_tokens": 4096}), "rlm", None)
        assert components[-1] == "ctok4096"

    def test_legacy_agentic_names_are_unchanged(self):
        assert rb.build_run_dir_components(_args(mode="rlm"), "rlm", None) == [
            "loft",
            "nq_128k",
            "Qwen_Qwen3-4B-Instruct-2507",
            "rlm",
        ]
        assert rb.build_run_dir_components(_args(mode="rlm", data_dir="qampari_128k"), "rlm", None)[-1] == "rlm"

    def test_config_records_the_order_its_revision_and_budget(self):
        config = rb.build_run_config(_args(**ORDERED_RLM), "rlm", "loft", None, None, None)
        assert config["chunk_order"] == "bm25-query" and config["chunk_order_revision"] == rb.CHUNK_ORDER_REVISION
        assert config["chunk_call_tokens"] == 8192 and config["chunk_order_query"] == "bare"
        assert config["organization_config"]["chunk_chars"] == 2000

    BXF = dict(
        sub_backend="kvzip",
        memory_budget=512.0,
        memory_budget_unit="MB",
        compression_factor=4.0,
        chunk_call_tokens=13888,
        chunk_kv_budget_tokens=3472,
        sub_max_tokens=512,
        sub_max_context_tokens=13888,
    )

    def test_a_bxf_cell_names_its_budget_and_factor(self):
        components = rb.build_run_dir_components(_args(**{**ORDERED_RLM, **self.BXF}), "rlm", None)
        assert components[components.index("rlm") + 1 :] == [
            "split-test",
            "kvzip-kvzip512MB",
            "cf4",
            "order-bm25-query",
            "ctok13888",
        ]

    def test_a_bxf_cell_records_what_a_read_keeps(self):
        config = rb.build_run_config(_args(**{**ORDERED_RLM, **self.BXF}), "rlm", "loft", None, None, None)
        assert config["chunk_read_backend"] == "kvzip" and config["chunk_kv_budget_tokens"] == 3472
        assert config["chunk_call_tokens"] == 13888 and config["compression_factor"] == 4.0

    def test_an_http_ordered_config_is_unchanged(self):
        config = rb.build_run_config(_args(**ORDERED_RLM), "rlm", "loft", None, None, None)
        assert "chunk_read_backend" not in config and "chunk_kv_budget_tokens" not in config

    def test_legacy_agentic_config_has_no_chunk_keys(self):
        assert "chunk_order" not in rb.build_run_config(_args(mode="rlm"), "rlm", "loft", None, None, None)

    def test_the_order_keys_guard_a_resume(self):
        assert {"chunk_order", "chunk_order_revision", "chunk_call_tokens"} <= set(rb.RESUME_CRITICAL_KEYS)

    def test_the_list_example_reaches_ordered_agentic_runs_only(self):
        assert rb.answer_list_example("loft", "qampari_128k", "rlm", ordered_rlm=True) == rb.LOFT_LIST_ANSWER_EXAMPLE
        assert rb.answer_list_example("loft", "qampari_128k", "rlm") == ""
        assert rb.answer_list_example("loft", "nq_128k", "rlm", ordered_rlm=True) == ""
        assert rb.answer_list_example("loft", "quest_128k", "structured") == rb.LOFT_LIST_ANSWER_EXAMPLE


class TestRefusals:
    BASE = [
        "run_benchmark",
        "--dataset",
        "loft",
        "--data-dir",
        "nq_128k",
        "--mode",
        "rlm",
        "--chunk-order",
        "bm25-query",
    ]

    def refused(self, argv):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(sys, "argv", [*argv, "--out", tmp]):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit) as caught:
                    rb.main()
        assert caught.value.code == 2
        return stderr.getvalue()

    def test_the_order_needs_the_agentic_mode(self):
        argv = [*self.BASE]
        argv[argv.index("rlm")] = "structured"
        assert "needs --mode rlm" in self.refused(argv)

    def test_skylight_does_not_take_a_precomputed_chunk_order(self):
        argv = [*self.BASE]
        argv[argv.index("rlm")] = "skylight"
        assert "needs --mode rlm" in self.refused(argv)

    def test_search_is_refused(self):
        assert "no search()" in self.refused([*self.BASE, "--search-k", "5"])

    def test_a_chunk_source_is_refused(self):
        assert "replaces --chunks-source" in self.refused([*self.BASE, "--chunks-source", "bm25"])

    KVZIP = ["--sub-backend", "kvzip", "--memory-budget", "512", "--memory-budget-unit", "MB"]

    def test_a_kvzip_cell_needs_its_factor(self):
        assert "needs --compression-factor" in self.refused([*self.BASE, *self.KVZIP])

    def test_a_kvzip_cell_derives_its_own_read_cap(self):
        argv = [*self.BASE, *self.KVZIP, "--compression-factor", "4", "--chunk-call-tokens", "4096"]
        assert "drop --chunk-call-tokens" in self.refused(argv)

    def test_a_kvzip_cell_presses_every_read(self):
        argv = [*self.BASE, *self.KVZIP, "--compression-factor", "4", "--press-min-tokens", "100"]
        assert "drop --press-min-tokens" in self.refused(argv)

    def test_the_press_must_match_the_factor(self):
        assert "takes --press no_press" in self.refused([*self.BASE, *self.KVZIP, "--compression-factor", "1"])
        argv = [*self.BASE, *self.KVZIP, "--press", "no_press", "--compression-factor", "4"]
        assert "takes --press no_press" in self.refused(argv)

    def test_slice_sizing_flags_are_refused(self):
        argv = [*self.BASE, *self.KVZIP, "--compression-factor", "4", "--max-subcall-chars", "auto"]
        assert "takes only --memory-budget" in self.refused(argv)

    def test_the_call_budget_needs_an_order(self):
        argv = [
            "run_benchmark",
            "--dataset",
            "loft",
            "--data-dir",
            "nq_128k",
            "--mode",
            "rlm",
            "--chunk-call-tokens",
            "4096",
        ]
        assert "only applies with --chunk-order" in self.refused(argv)

    def test_an_organization_dir_needs_an_order(self):
        argv = [
            "run_benchmark",
            "--dataset",
            "loft",
            "--data-dir",
            "nq_128k",
            "--mode",
            "rlm",
            "--organization-dir",
            "/tmp/org",
        ]
        assert "only apply with --structured-order" in self.refused(argv)

    def test_ruler128k_needs_a_ruler_task(self):
        argv = ["run_benchmark", "--dataset", "ruler128k", "--data-dir", "nq_128k", "--mode", "rlm"]
        assert "requires --data-dir with one of" in self.refused(argv)


class TestDiagnostics:
    def test_where_the_gold_sat_and_whether_it_was_read(self):
        view = query_view("cello")  # ranks: B(1) B(2) A(3) A(4) C(5)
        assert rb.chunk_diagnostics(view, [3, 1, 3], ["glacier"]) == {
            "gold_first_position": 5,
            "gold_chunks": 1,
            "gold_in_read_chunks": False,
            "chunks_read": 2,
            "deepest_position_read": 3,
        }
        assert rb.chunk_diagnostics(view, [1], ["sonata"])["gold_in_read_chunks"] is True
        assert rb.chunk_diagnostics(view, [], ["nowhere"])["gold_first_position"] is None

    def test_run_level_aggregates(self):
        frame = pd.DataFrame(
            {
                "prompt_tokens": [100, 300],
                "completion_tokens": [10, 30],
                "chunks_read": [2, 4],
                "deepest_position_read": [2, 10],
                "gold_first_position": [1, None],
                "gold_in_read_chunks": [True, False],
                "metrics": [
                    dict(
                        chunks_total=5,
                        chunks_per_call_advertised=14,
                        chunk_payload_calls=1,
                        chunk_budget_refusals=0,
                        reader_input_tokens=[50],
                        reader_served_prompt_tokens=[50],
                        organization_sha256="d",
                        organization_build_cpu_seconds=1.5,
                        chunk_view_cpu_seconds=0.1,
                    ),
                    dict(
                        chunks_total=5,
                        chunks_per_call_advertised=14,
                        chunk_payload_calls=0,
                        chunk_budget_refusals=2,
                        reader_input_tokens=[],
                        reader_served_prompt_tokens=[],
                        organization_sha256="d",
                        organization_build_cpu_seconds=1.5,
                        chunk_view_cpu_seconds=0.3,
                    ),
                ],
            }
        )
        out = rb.chunk_order_runtime(frame)
        assert out["chunk_read_example_fraction"] == 0.5
        assert out["chunk_budget_refusals"] == 2
        assert out["average_deepest_position_read"] == 6.0
        assert out["gold_present_fraction"] == 0.5 and out["median_gold_first_position"] == 1.0
        assert out["gold_in_top5_fraction"] == 0.5 and out["gold_in_read_chunks_fraction"] == 0.5
        # One document, built once: its cost is counted once, not per query.
        assert out["organization_build_cpu_seconds"] == 1.5
        assert out["reader_token_count_mismatches"] == 0
        assert out["average_prompt_tokens"] == 200.0


class TestRuler128k:
    NIAH = (
        "A special magic number is hidden within the following text. Make sure to memorize it.\n"
        "The grass is green. One of the special magic numbers for foo-bar is: 12345. The grass is green.\n"
        "What is the special magic number for foo-bar mentioned in the provided text? "
        "The special magic number for foo-bar mentioned in the provided text is"
    )

    def test_it_is_registered_with_the_ruler_scorer(self):
        assert DATASET_REGISTRY["ruler128k"] == "lighteval/RULER-131072-Qwen-3"
        assert "ruler128k" in SCORER_REGISTRY

    def test_niah_splits_at_the_question(self):
        context, question, prefix = split_ruler_input(self.NIAH, QUESTION_PATTERNS["niah"], ANSWER_PATTERNS["niah"])
        assert "12345" in context and context.endswith("The grass is green.\n")
        assert question == "What is the special magic number for foo-bar mentioned in the provided text? "
        assert prefix == "The special magic number for foo-bar mentioned in the provided text is"

    def test_qa_takes_the_instruction_at_the_bottom_not_the_top(self):
        instruction = (
            "Answer the question based on the given documents. "
            "Only give me the answer and do not output any other words."
        )
        text = (
            f"{instruction}\n\nThe following are given documents.\n\nDocument 1:\nDanny Cannon\n"
            f"John Danny Cannon is a British director.\n\n{instruction}\n\nQuestion: Is Danny Cannon British? Answer:"
        )
        context, question, prefix = split_ruler_input(text, QUESTION_PATTERNS["qa"], ANSWER_PATTERNS["qa"])
        assert "Document 1:" in context and question.endswith("Question: Is Danny Cannon British? ")
        assert prefix == "Answer:"
        assert rb.chunk_order_query(question, "ruler128k") == "Is Danny Cannon British?"

    @pytest.mark.parametrize(
        "category,text,question_start,prefix_start",
        [
            (
                "vt",
                "Memorize the chains.\nVAR ABC = 12345 filler.\n\nQuestion: Find all variables that are assigned "
                "the value 12345 in the text above.Answer: According to the chain(s) of variable assignment",
                "Question: Find all variables",
                "Answer: According",
            ),
            (
                "cwe",
                "Below is a numbered list of words.\n1. powder 2. concrete\nQuestion: What are the 10 most common "
                "words in the above list? Answer: The top 10 words that appear most often in the list are:",
                "Question: What are the 10 most common words",
                "Answer: The top 10",
            ),
            (
                "fwe",
                "Read the coded text.\n.... abc .... abc\nQuestion: Do not provide any explanation. What are the "
                "three most frequently appeared words in the above coded text? Answer: According to the coded text",
                "Question: Do not provide any explanation.",
                "Answer: According",
            ),
        ],
    )
    def test_the_other_task_families(self, category, text, question_start, prefix_start):
        _, question, prefix = split_ruler_input(text, QUESTION_PATTERNS[category], ANSWER_PATTERNS[category])
        assert question.startswith(question_start) and prefix.startswith(prefix_start)

    def test_the_loader_builds_the_shared_columns(self):
        frame = pd.DataFrame({"index": [0], "input": [self.NIAH], "outputs": [["12345"]], "length": [131012]})
        dataset = mock.Mock()
        dataset.to_pandas.return_value = frame
        with mock.patch("evaluation.benchmarks.loaders.load_dataset", return_value=dataset) as load:
            df = _load_ruler128k("lighteval/RULER-131072-Qwen-3", "niah_single_1")
        load.assert_called_once_with("lighteval/RULER-131072-Qwen-3", split="niah_single_1")
        row = df.iloc[0]
        assert row["question"].startswith("What is the special magic number") and row["answer"] == ["12345"]
        assert row["answer_prefix"].startswith("The special magic") and "input" not in df
        assert row["task"] == "niah_single_1" and row["context_length"] == 131072

    def test_an_unknown_task_is_refused(self):
        with pytest.raises(ValueError, match="Unknown RULER128K task"):
            _load_ruler128k("lighteval/RULER-131072-Qwen-3", "nq_128k")
