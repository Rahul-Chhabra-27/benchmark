# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skylight RLM: protocol parsing and the root / scan loop."""

import pytest

from evaluation.rlm import skylight_rlm
from evaluation.rlm.organization import ordered_hits, organize
from evaluation.rlm.retrieval import reset_cache
from evaluation.rlm.rlm import MemoryBudget
from evaluation.rlm.skylight_rlm import SUB_SYSTEM_PROMPT, SkylightRLM, _rfind_order, parse_root_action

# This file was written against a newer skylight_rlm than apd10-dev pushed (it uses
# _is_no_answer, repetition penalties, a vanilla last step...). Until that version lands,
# its tests are expected to fail; non-strict, so they report XPASS once it does.
_is_no_answer = getattr(skylight_rlm, "_is_no_answer", None)
pytestmark = pytest.mark.xfail(
    _is_no_answer is None, reason="needs the newer skylight_rlm.py from apd10-dev", strict=False
)


class ScriptedClient:
    """Records chat() payloads and returns scripted replies."""

    def __init__(self, replies=()):
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if not self._replies:
            raise AssertionError(f"unexpected chat; last messages={messages!r}")
        return self._replies.pop(0)

    def chat_batch(self, jobs, **kwargs):
        return [self.chat(j, **kwargs) for j in jobs]


def _chunks_doc() -> str:
    # Three 2000-char chunks with a distinctive token in each (no overlap).
    return (
        ("alpha-only " * 100).ljust(2000, ".")
        + ("beta-only " * 100).ljust(2000, ".")
        + ("gamma-only " * 100).ljust(2000, ".")
    )


def _harness(root, sub=None, **kwargs):
    kwargs.setdefault("chunk_chars", 2000)
    kwargs.setdefault("chunk_overlap", 0)
    kwargs.setdefault("max_steps", 8)
    kwargs.setdefault("max_sub_calls", 50)
    kwargs.setdefault("token_counter", len)
    return SkylightRLM(root_client=root, sub_client=sub or root, **kwargs)


def setup_function(_fn):
    reset_cache()


def teardown_function(_fn):
    reset_cache()


class TestParse:
    def test_index_scan_fullscan_and_final(self):
        kind, payload = parse_root_action("QUERY: Who directed Jaws?\nSUB_CALL(STATE=ORDERED, LOOK=INDEX(0))")
        assert kind == "subcall"
        assert payload.state == "ORDERED" and payload.look == "INDEX" and payload.index == 0
        assert payload.query == "Who directed Jaws?"

        kind, payload = parse_root_action("QUERY: find birthplace\nSUB_CALL(STATE=UNORDERED, LOOK=SCAN(3))")
        assert kind == "subcall" and payload.look == "SCAN" and payload.index == 3 and payload.state == "UNORDERED"

        kind, payload = parse_root_action("QUERY: around hit\nSUB_CALL(STATE=ORDERED, LOOK=SCAN(5,10))")
        assert kind == "subcall" and payload.look == "SCAN" and payload.index == 5 and payload.end == 10

        kind, payload = parse_root_action("QUERY: nbrs\ni=2\nj=4\nspan=SCAN(i,j)\nSUB_CALL(STATE=ORDERED, LOOK=span)")
        assert kind == "subcall" and payload.index == 2 and payload.end == 4

        kind, payload = parse_root_action("QUERY: whole doc\nSUB_CALL(STATE=UNORDERED, LOOK=VANILLA)")
        assert kind == "subcall" and payload.look == "VANILLA"

        kind, payload = parse_root_action("final_answer={Cincinnati, Ohio}")
        assert kind == "final" and payload == "Cincinnati, Ohio"
        assert not _is_no_answer(payload)
        assert _is_no_answer("not found")
        assert _is_no_answer("['NOT_FOUND']")
        assert _is_no_answer("NOTHING RELEVANT")
        assert not _is_no_answer("Nottingham")

    def test_named_bindings_and_inline_query(self):
        text = "QUERY: hop\nst=ORDERED\ni=2\nspan=SCAN(i)\nSUB_CALL(STATE=st, LOOK=span)"
        kind, payload = parse_root_action(text)
        assert kind == "subcall"
        assert payload.state == "ORDERED" and payload.look == "SCAN" and payload.index == 2

        kind, payload = parse_root_action('SUB_CALL(STATE=UNORDERED, LOOK=INDEX(1), QUERY="what is here")')
        assert kind == "subcall" and payload.query == "what is here" and payload.index == 1

        kind, payload = parse_root_action('QUERY: find beta\nSUB_CALL(STATE=RFIND("beta-only"), LOOK=INDEX(0))')
        assert kind == "subcall"
        assert payload.state == "RFIND" and payload.find == "beta-only" and payload.index == 0

        kind, payload = parse_root_action(
            "QUERY: hop\nst=RFIND('Steven Spielberg')\ni=0\nspan=INDEX(i)\nSUB_CALL(STATE=st, LOOK=span)"
        )
        assert kind == "subcall" and payload.state == "RFIND" and payload.find == "Steven Spielberg"

    def test_final_wins_when_it_is_the_last_stop_form(self):
        kind, payload = parse_root_action(
            "QUERY: x\nSUB_CALL(STATE=ORDERED, LOOK=INDEX(0))\nfinal_answer={the real answer}"
        )
        assert kind == "final" and payload == "the real answer"

    def test_whole_document_looks_parse_without_an_explicit_look(self):
        # The root drops LOOK= for the looks that take no index; intent is clear.
        kind, payload = parse_root_action("QUERY: find it\nSUB_CALL(STATE=FULLSCAN)")
        assert kind == "subcall" and payload.look == "FULLSCAN" and payload.state == "ORDERED"

        kind, payload = parse_root_action("QUERY: find it\nSUB_CALL(LOOK=VANILLA)")
        assert kind == "subcall" and payload.look == "VANILLA"

        kind, _ = parse_root_action("SUB_CALL(STATE=FULLSCAN)")
        assert kind == "invalid"

    def test_final_survives_trailing_doubt(self):
        kind, payload = parse_root_action("final_answer={alpha}\nWait, but should I check chunk 4 again?")
        assert kind == "final" and payload == "alpha"

    def test_last_final_answer_is_the_committed_one(self):
        kind, payload = parse_root_action("final_answer={alpha}\nhmm, actually\nfinal_answer={beta}")
        assert kind == "final" and payload == "beta"

    def test_invalid_without_query_or_stop_form(self):
        kind, _ = parse_root_action("SUB_CALL(STATE=ORDERED, LOOK=INDEX(0))")
        assert kind == "invalid"
        kind, _ = parse_root_action("I think the answer is 42")
        assert kind == "invalid"

    def test_placeholder_finals_in_cot_are_ignored(self):
        text = (
            "Thinking Process:\n"
            "output only final_answer={value} or final_answer={...} when done.\n"
            "QUERY: when does monday night raw come on hulu\n"
            "SUB_CALL(STATE=BM25, LOOK=SCAN(4))\n"
            "Wait, example was final_answer={value}."
        )
        kind, payload = parse_root_action(text)
        assert kind == "subcall"
        assert payload.state == "BM25" and payload.look == "SCAN" and payload.index == 4

    def test_missing_query_filled_from_default(self):
        kind, payload = parse_root_action(
            "SUB_CALL(STATE=BM25, LOOK=SCAN(4))",
            default_query="when does monday night raw come on hulu",
        )
        assert kind == "subcall"
        assert payload.query == "when does monday night raw come on hulu"
        assert payload.state == "BM25" and payload.look == "SCAN" and payload.index == 4

    def test_untagged_think_still_parses_protocol(self):
        text = "I should write final_answer={...} only after FOUND.\n" "For now: SUB_CALL(STATE=BM25, LOOK=SCAN(4))"
        kind, payload = parse_root_action(text, default_query="q")
        assert kind == "subcall" and payload.query == "q"


class TestLoop:
    def test_index_returns_immediately_then_final_answer_is_the_eval_answer(self):
        root = ScriptedClient(
            [
                "QUERY: Who is named?\nSUB_CALL(STATE=UNORDERED, LOOK=INDEX(0))",
                "final_answer={alpha}",
            ]
        )
        sub = ScriptedClient(["passage mentions alpha"])
        result = _harness(root, sub).run(_chunks_doc(), "Who is named in the first chunk?")
        assert result.finished and result.end_reason == "final_answer"
        assert result.answer == "alpha"
        assert result.steps == 2
        assert len(sub.calls) == 1
        assert len(root.calls) == 2
        # Control returned after the single INDEX read: second root call sees that obs.
        second = "\n".join(m["content"] for m in root.calls[1])
        assert "[INDEX 0 | UNORDERED]" in second
        assert "passage mentions alpha" in second
        assert "SUMMARY:" in second
        assert "SUMMARY" in sub.calls[0][0]["content"]

    def test_root_history_drops_child_thinking_and_keeps_summary(self):
        root = ScriptedClient(
            [
                "QUERY: Who is named?\nSUB_CALL(STATE=UNORDERED, LOOK=INDEX(0))",
                "final_answer={alpha}",
            ]
        )
        sub = ScriptedClient(
            ["<think>I should SCAN again and guess</think>\nANSWER: the secret alpha\nSUMMARY: first chunk names alpha"]
        )
        _harness(root, sub).run(_chunks_doc(), "Who is named in the first chunk?")
        obs = "\n".join(m["content"] for m in root.calls[1] if m["role"] == "user")
        assert "SUMMARY: first chunk names alpha" in obs
        assert "<think>" not in obs
        assert "I should SCAN again" not in obs
        assert "ANSWER:" not in obs
        assert "the secret alpha" not in obs

    def test_only_text_after_the_last_summary_label_reaches_the_root(self):
        root = ScriptedClient(
            [
                "QUERY: Who is named?\nSUB_CALL(STATE=UNORDERED, LOOK=INDEX(0))",
                "final_answer={alpha}",
            ]
        )
        sub = ScriptedClient(
            [
                "Let me read this. SUMMARY: is what I must write, so I will reason first.\n"
                "The passage repeats one token over and over.\n"
                "SUMMARY: chunk 0 names alpha"
            ]
        )
        _harness(root, sub).run(_chunks_doc(), "Who is named in the first chunk?")
        obs = "\n".join(m["content"] for m in root.calls[1] if m["role"] == "user")
        assert "SUMMARY: chunk 0 names alpha" in obs
        assert "Let me read this" not in obs
        assert "so I will reason first" not in obs

    def test_unlabelled_child_reply_is_truncated_not_dumped(self):
        root = ScriptedClient(
            [
                "QUERY: Who is named?\nSUB_CALL(STATE=UNORDERED, LOOK=INDEX(0))",
                "final_answer={alpha}",
            ]
        )
        sub = ScriptedClient(["OPENING-MARKER " + ("filler " * 500) + "CLOSING-MARKER"])
        _harness(root, sub).run(_chunks_doc(), "Who is named in the first chunk?")
        obs = "\n".join(m["content"] for m in root.calls[1] if m["role"] == "user")
        assert "CLOSING-MARKER" in obs
        assert "OPENING-MARKER" not in obs

    def test_root_calls_carry_repetition_penalties(self):
        class RecordingClient(ScriptedClient):
            def __init__(self, replies):
                super().__init__(replies)
                self.kwargs: list[dict] = []

            def chat(self, messages, **kwargs):
                self.kwargs.append(kwargs)
                return super().chat(messages, **kwargs)

        root = RecordingClient(["final_answer={alpha}"])
        _harness(root, ScriptedClient([])).run(_chunks_doc(), "who?")
        extra = root.kwargs[0]["extra_body"]
        assert extra["repetition_penalty"] > 1.0
        assert extra["frequency_penalty"] > 0.0

    def test_root_sampling_does_not_clobber_client_extra_body(self):
        root = ScriptedClient(["final_answer={alpha}"])
        root.extra_body = {"chat_template_kwargs": {"enable_thinking": False}, "other": 1}
        harness = _harness(root, ScriptedClient([]), enable_thinking=False)
        extra = harness._root_chat_kwargs()["extra_body"]
        assert extra["chat_template_kwargs"]["enable_thinking"] is False
        assert extra["other"] == 1
        assert "repetition_penalty" in extra

    def test_thinking_on_sets_enable_thinking_and_strips_before_parse(self):
        root = ScriptedClient(
            [
                "<think>plan the call</think>\nQUERY: who?\nSUB_CALL(STATE=UNORDERED, LOOK=INDEX(0))",
                "I reasoned.\n</think>\nfinal_answer={alpha}",
            ]
        )
        sub = ScriptedClient(["SUMMARY: FOUND | answer=alpha | evidence=x | documents=y"])
        harness = _harness(root, sub, enable_thinking=True)
        extra = harness._root_chat_kwargs()["extra_body"]
        assert extra["chat_template_kwargs"]["enable_thinking"] is True
        assert harness._root_chat_kwargs()["max_tokens"] >= 1024
        result = harness.run(_chunks_doc(), "Who is named in the first chunk?")
        assert result.finished and result.answer == "alpha"

    def test_scan_is_one_sub_call_over_every_chunk(self):
        root = ScriptedClient(
            [
                "QUERY: anything?\nSUB_CALL(STATE=UNORDERED, LOOK=FULLSCAN)",
                "final_answer={from history}",
            ]
        )
        sub = ScriptedClient(["a", "b", "c"])
        result = _harness(root, sub, max_sub_calls=1).run(_chunks_doc(), "task")
        assert result.finished and result.answer == "from history"
        assert result.metrics["sub_calls"] == 1
        assert result.metrics["chunks_read"] == 3
        assert len(sub.calls) == 3

    def test_budget_exhaustion_ends_the_run_instead_of_raising(self):
        root = ScriptedClient(
            [
                "QUERY: x\nSUB_CALL(STATE=UNORDERED, LOOK=SCAN(1))",
                "QUERY: again\nSUB_CALL(STATE=UNORDERED, LOOK=FULLSCAN)",
                "QUERY: again\nSUB_CALL(STATE=UNORDERED, LOOK=FULLSCAN)",
            ]
        )
        sub = ScriptedClient(["a", "b", "c", "d", "e", "Final Answer: ['from vanilla']"])
        result = _harness(root, sub, max_sub_calls=2).run(_chunks_doc(), "task")
        assert result.finished and result.end_reason == "vanilla"
        assert "from vanilla" in result.answer
        assert result.metrics["vanilla_last_step"] == "ok"
        assert result.metrics["sub_calls"] == 3
        assert "Document:" in sub.calls[-1][-1]["content"]

    def test_not_found_final_answer_triggers_vanilla_after_failed_fullscan(self):
        root = ScriptedClient(
            [
                "QUERY: x\nSUB_CALL(STATE=UNORDERED, LOOK=FULLSCAN)",
                "final_answer={not found}",
            ]
        )
        sub = ScriptedClient(
            [
                "SUMMARY: NOT_FOUND | documents=a",
                "SUMMARY: NOT_FOUND | documents=b",
                "SUMMARY: NOT_FOUND | documents=c",
                "Final Answer: ['alpha']",
            ]
        )
        result = _harness(root, sub, max_steps=4).run(_chunks_doc(), "query: who?\n")
        assert result.finished and result.end_reason == "vanilla"
        assert "alpha" in result.answer
        assert result.metrics.get("abstained") is True
        assert "Document:" in sub.calls[-1][-1]["content"]

    def test_exhaustive_scan_counts_as_a_full_scan(self):
        # SCAN(n-1) covers the whole document even though it is not FULLSCAN.
        root = ScriptedClient(
            [
                "QUERY: x\nSUB_CALL(STATE=BM25, LOOK=SCAN(2))",
                "final_answer={not found}",
            ]
        )
        sub = ScriptedClient(
            [
                "SUMMARY: NOT_FOUND | documents=a",
                "SUMMARY: NOT_FOUND | documents=b",
                "SUMMARY: NOT_FOUND | documents=c",
                "Final Answer: ['alpha']",
            ]
        )
        result = _harness(root, sub, max_steps=4).run(_chunks_doc(), "query: who?\n")
        assert result.finished and result.end_reason == "vanilla"
        assert result.metrics["chunks_scanned"] == 3
        assert "alpha" in result.answer

    def test_partial_scans_that_together_cover_everything_count(self):
        root = ScriptedClient(
            [
                "QUERY: x\nSUB_CALL(STATE=UNORDERED, LOOK=SCAN(0,1))",
                "QUERY: x\nSUB_CALL(STATE=UNORDERED, LOOK=SCAN(2,2))",
                "final_answer={not found}",
            ]
        )
        sub = ScriptedClient(
            [
                "SUMMARY: NOT_FOUND | documents=a",
                "SUMMARY: NOT_FOUND | documents=b",
                "SUMMARY: NOT_FOUND | documents=c",
                "Final Answer: ['alpha']",
            ]
        )
        result = _harness(root, sub, max_steps=5).run(_chunks_doc(), "query: who?\n")
        assert result.finished and result.end_reason == "vanilla"
        assert result.metrics["chunks_scanned"] == 3

    def test_not_found_final_after_partial_scan_does_not_run_vanilla(self):
        root = ScriptedClient(
            [
                "QUERY: x\nSUB_CALL(STATE=UNORDERED, LOOK=INDEX(0))",
                "final_answer={not found}",
            ]
        )
        sub = ScriptedClient(["chunk-obs", "should-not-run"])
        result = _harness(root, sub).run(_chunks_doc(), "task")
        assert not result.finished
        assert result.end_reason == "no_answer"
        assert result.metrics["vanilla_last_step"] == "skipped_partial_scan"
        assert len(sub.calls) == 1

    def test_last_step_runs_vanilla_only_after_a_failed_fullscan(self):
        root = ScriptedClient(["QUERY: x\nSUB_CALL(STATE=UNORDERED, LOOK=FULLSCAN)"])
        sub = ScriptedClient(
            [
                "SUMMARY: NOT_FOUND | documents=a",
                "SUMMARY: NOT_FOUND | documents=b",
                "SUMMARY: NOT_FOUND | documents=c",
                "Final Answer: ['alpha']",
            ]
        )
        result = _harness(root, sub, max_steps=2).run(_chunks_doc(), "query: who?\n")
        assert result.finished and result.end_reason == "vanilla"
        assert "alpha" in result.answer
        assert len(root.calls) == 1
        assert "Document:" in sub.calls[-1][-1]["content"]

    def test_vanilla_last_step_is_skipped_when_coverage_is_partial(self):
        root = ScriptedClient(["QUERY: x\nSUB_CALL(STATE=UNORDERED, LOOK=INDEX(0))"])
        sub = ScriptedClient(["chunk-obs", "should-not-run"])
        result = _harness(root, sub, max_steps=2).run(_chunks_doc(), "task")
        assert not result.finished
        assert result.end_reason == "max_steps"
        assert result.metrics["vanilla_last_step"] == "skipped_partial_scan"
        assert len(sub.calls) == 1

    def test_vanilla_last_step_is_skipped_when_a_chunk_reported_found(self):
        root = ScriptedClient(["QUERY: x\nSUB_CALL(STATE=UNORDERED, LOOK=FULLSCAN)"])
        sub = ScriptedClient(
            [
                "SUMMARY: FOUND | answer=alpha | evidence=in text | documents=t (1)",
                "SUMMARY: NOT_FOUND | documents=b",
                "SUMMARY: NOT_FOUND | documents=c",
                "should-not-run",
            ]
        )
        result = _harness(root, sub, max_steps=2).run(_chunks_doc(), "task")
        assert not result.finished
        assert result.end_reason == "max_steps"
        assert result.metrics["vanilla_last_step"] == "skipped_found"
        assert len(sub.calls) == 3

    def test_vanilla_last_step_is_skipped_when_the_prompt_does_not_fit(self):
        root = ScriptedClient(["QUERY: x\nSUB_CALL(STATE=UNORDERED, LOOK=FULLSCAN)"])
        sub = ScriptedClient(["SUMMARY: NOT_FOUND | documents=a", "b", "c"])
        result = _harness(root, sub, max_steps=2, vanilla_max_prompt_tokens=8).run(_chunks_doc(), "task")
        assert not result.finished
        assert result.end_reason == "max_steps"
        assert result.metrics["vanilla_last_step"] == "skipped_too_long"
        assert len(sub.calls) == 3

    def test_nudge_shows_single_braces(self):
        root = ScriptedClient(
            [
                "I will think about it.",
                "final_answer={ok}",
            ]
        )
        harness = _harness(root, ScriptedClient([]))
        result = harness.run(_chunks_doc(), "task")
        nudge = next(m["content"] for m in result.transcript if m["role"] == "nudge")
        assert "final_answer={...}" in nudge
        assert "{{" not in nudge

    def test_scan_range_skips_chunks_before_i(self):
        root = ScriptedClient(
            [
                "QUERY: mid\nSUB_CALL(STATE=UNORDERED, LOOK=SCAN(1,2))",
                "final_answer={ok}",
            ]
        )
        sub = ScriptedClient(["B", "C"])
        result = _harness(root, sub).run(_chunks_doc(), "mid")
        assert result.answer == "ok"
        assert len(sub.calls) == 2
        first_pass = sub.calls[0][1]["content"]
        assert "beta-only" in first_pass
        assert "alpha-only" not in first_pass
        second = "\n".join(m["content"] for m in root.calls[1])
        assert "[SCAN 1 (1..2) | UNORDERED]" in second
        assert "[SCAN 2 (1..2) | UNORDERED]" in second
        assert "[SCAN 0 (1..2)" not in second

    def test_scan_reads_every_chunk_before_the_next_root_call(self):
        root = ScriptedClient(
            [
                "QUERY: list names\nSUB_CALL(STATE=UNORDERED, LOOK=SCAN(2))",
                "final_answer={alpha beta gamma}",
            ]
        )
        sub = ScriptedClient(["A", "B", "C"])
        result = _harness(root, sub).run(_chunks_doc(), "list names")
        assert result.answer == "alpha beta gamma"
        assert len(sub.calls) == 3
        assert len(root.calls) == 2  # no root turn until the scan finished
        second = "\n".join(m["content"] for m in root.calls[1])
        assert "[SCAN 0 (0..2) | UNORDERED]" in second
        assert "[SCAN 1 (0..2) | UNORDERED]" in second
        assert "[SCAN 2 (0..2) | UNORDERED]" in second

    def test_fullscan_covers_all_chunks(self):
        root = ScriptedClient(
            [
                "QUERY: anything?\nSUB_CALL(STATE=UNORDERED, LOOK=FULLSCAN)",
                "final_answer={done}",
            ]
        )
        sub = ScriptedClient(["a", "b", "c"])
        result = _harness(root, sub).run(_chunks_doc(), "anything?")
        assert result.answer == "done"
        assert len(sub.calls) == 3

    def test_ordered_uses_the_bm25_graph_walk(self):
        doc = _chunks_doc()
        artifact, _, _ = organize(doc, "bm25-graph")
        graph0 = ordered_hits(doc, artifact)[0].text
        root = ScriptedClient(
            [
                "QUERY: anything\nSUB_CALL(STATE=ORDERED, LOOK=INDEX(0))",
                "final_answer={ok}",
            ]
        )
        sub = ScriptedClient(["saw it"])
        _harness(root, sub).run(doc, "where is beta?")
        assert graph0 in sub.calls[0][1]["content"]

    def test_unordered_index_zero_is_source_order(self):
        doc = _chunks_doc()
        root = ScriptedClient(
            [
                "QUERY: x\nSUB_CALL(STATE=UNORDERED, LOOK=INDEX(0))",
                "final_answer={ok}",
            ]
        )
        sub = ScriptedClient(["saw it"])
        _harness(root, sub).run(doc, "task")
        assert "alpha-only" in sub.calls[0][1]["content"]
        assert "beta-only" not in sub.calls[0][1]["content"]

    def test_rfind_puts_matches_first_then_the_graph_order(self):
        doc = _chunks_doc()
        artifact, _, _ = organize(doc, "bm25-graph")
        graph = ordered_hits(doc, artifact)
        ranked = _rfind_order(graph, "beta-only")
        assert "beta-only" in ranked[0].text
        rest_graph = [(h.start, h.end) for h in graph if "beta-only" not in h.text.lower()]
        rest_ranked = [(h.start, h.end) for h in ranked[1:]]
        assert rest_ranked == rest_graph

        root = ScriptedClient(
            [
                'QUERY: what about beta?\nSUB_CALL(STATE=RFIND("BETA-ONLY"), LOOK=INDEX(0))',
                "final_answer={ok}",
            ]
        )
        sub = ScriptedClient(["saw it"])
        result = _harness(root, sub).run(doc, "task")
        assert "beta-only" in sub.calls[0][1]["content"]
        assert "alpha-only" not in sub.calls[0][1]["content"]
        second = "\n".join(m["content"] for m in root.calls[1])
        assert "[INDEX 0 | RFIND(BETA-ONLY)]" in second
        assert result.answer == "ok"

    def test_bm25_index_zero_is_the_best_query_match(self):
        doc = _chunks_doc()
        root = ScriptedClient(
            [
                "QUERY: beta-only\nSUB_CALL(STATE=BM25, LOOK=INDEX(0))",
                "final_answer={ok}",
            ]
        )
        sub = ScriptedClient(["saw it"])
        _harness(root, sub).run(doc, "where is beta?")
        user = sub.calls[0][1]["content"]
        assert "beta-only" in user
        assert "alpha-only" not in user

    def test_vanilla_sends_the_full_document_in_one_child_call(self):
        doc = _chunks_doc()
        root = ScriptedClient(
            [
                "QUERY: what is mentioned?\nSUB_CALL(STATE=UNORDERED, LOOK=VANILLA)",
                "final_answer={ok}",
            ]
        )
        sub = ScriptedClient(["everything"])
        result = _harness(root, sub).run(doc, "task")
        assert len(sub.calls) == 1
        blob = sub.calls[0][1]["content"]
        assert "alpha-only" in blob and "beta-only" in blob and "gamma-only" in blob
        assert "[VANILLA | full document]" in "\n".join(m["content"] for m in root.calls[1])
        assert result.answer == "ok"

    def test_document_never_enters_the_root_context(self):
        doc = _chunks_doc()
        root = ScriptedClient(
            [
                "QUERY: x\nSUB_CALL(STATE=UNORDERED, LOOK=INDEX(0))",
                "final_answer={ok}",
            ]
        )
        sub = ScriptedClient(["obs"])
        _harness(root, sub).run(doc, "task")
        for messages in root.calls:
            blob = "\n".join(m["content"] for m in messages)
            assert "alpha-only" not in blob
            assert doc not in blob

    def test_root_is_told_the_chunk_count_on_the_first_turn(self):
        root = ScriptedClient(
            [
                "QUERY: x\nSUB_CALL(STATE=UNORDERED, LOOK=INDEX(0))",
                "final_answer={ok}",
            ]
        )
        sub = ScriptedClient(["obs"])
        _harness(root, sub).run(_chunks_doc(), "task")
        first = root.calls[0]
        system = first[0]["content"]
        begin = first[1]["content"]
        assert first[0]["role"] == "system"
        assert "CHUNKS: there are 3 chunks" in system
        assert "valid indices: 0..2" in system
        assert "BM25-graph order" in system
        assert "STATE=BM25" in system
        assert "3 chunks" in begin
        assert "indices 0..2" in begin

    def test_sub_calls_do_not_receive_the_root_scan_api(self):
        root = ScriptedClient(
            [
                "QUERY: names?\nSUB_CALL(STATE=UNORDERED, LOOK=SCAN(1))",
                "final_answer={ok}",
            ]
        )
        sub = ScriptedClient(["a", "b"])
        _harness(root, sub).run(_chunks_doc(), "task")
        assert len(sub.calls) == 2
        for messages in sub.calls:
            assert messages[0] == {"role": "system", "content": SUB_SYSTEM_PROMPT}
            blob = "\n".join(m["content"] for m in messages)
            assert "SUB_CALL(STATE" not in blob
            assert "LOOK=" not in blob
            assert "FULLSCAN" not in blob
            assert "final_answer" not in blob
            assert "UNORDERED" not in blob

    def test_scan_observations_can_be_evicted_under_a_tight_budget(self):
        # SCAN(2) appends 3 chunk observations. A tiny budget + keep_recent_turns=1
        # must drop the earliest scan hits before the next root call.
        root = ScriptedClient(
            [
                "QUERY: list\nSUB_CALL(STATE=UNORDERED, LOOK=SCAN(2))",
                "final_answer={later}",
            ]
        )
        sub = ScriptedClient(["OBS-EARLY-AAAA", "OBS-MID-BBBB", "OBS-LATE-CCCC"])

        def tokens(text: str) -> int:
            return 100 if "OBS-" in text else 1

        budget = MemoryBudget(max_context_tokens=250, keep_recent_turns=1)
        result = _harness(root, sub, budget=budget, token_counter=tokens).run(_chunks_doc(), "list")
        assert result.answer == "later"
        second = "\n".join(m["content"] for m in root.calls[1])
        assert "OBS-LATE-CCCC" in second
        assert "OBS-EARLY-AAAA" not in second
        assert result.metrics["evictions"] >= 1

    def test_debug_prints_call_history_and_full_root_prompts(self, capsys):
        root = ScriptedClient(
            [
                "QUERY: names?\nSUB_CALL(STATE=UNORDERED, LOOK=SCAN(1))",
                "final_answer={ok}",
            ]
        )
        sub = ScriptedClient(["child-a", "child-b"])
        _harness(root, sub, debug=True).run(_chunks_doc(), "task")
        out = capsys.readouterr().out
        assert "ROOT(1)" in out
        assert "ROOT(1) -> CHILD(1)" in out
        assert "ROOT(1) -> CHILD(2)" in out
        assert "ROOT(2)" in out
        assert "----- CALL HISTORY -----" in out
        assert "----- ROOT(2) FULL PROMPT -----" in out
        assert "----- ROOT(2) GENERATION -----" in out
        assert "final_answer={ok}" in out
        assert "child-a" in out
        assert out.index("ROOT(1) -> CHILD(1)") < out.index("----- ROOT(2) FULL PROMPT -----")
