# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for LOOK routing and cost accounting in RLM experiment 1."""

import pytest

# The module these tests cover has not been pushed yet (see run_benchmark's optional import).
pytest.importorskip("evaluation.rlm.rlm_exp1")

from evaluation.rlm.rlm_exp1 import RLM, LookEndpointConfig, LookStrategy  # noqa: E402
from evaluation.rlm.rlm_exp1.costs import render_cost_table, token_placement  # noqa: E402
from evaluation.rlm.rlm_exp1.repl import extract_sub_summary  # noqa: E402
from evaluation.rlm.shared.client import Usage  # noqa: E402


class ScriptedClient:
    def __init__(self, replies, prompt_tokens=None, context_tokens=2):
        self.replies = list(replies)
        self.prompt_tokens = prompt_tokens
        self.context_tokens = context_tokens
        self.tokenize_calls = []
        self.calls = []
        self.index = 0
        self.max_tokens = 64
        self.usage = Usage()

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        reply = self.replies[min(self.index, len(self.replies) - 1)]
        self.index += 1
        self.usage.calls += 1
        self.usage.last_prompt_tokens = self.prompt_tokens
        return reply

    def count_tokens(self, text):
        self.tokenize_calls.append(text)
        return self.context_tokens


def code(body):
    return f"```python\n{body}\n```"


def test_default_look_endpoint_ports():
    endpoints = LookEndpointConfig().endpoints()

    assert endpoints == {
        LookStrategy.CHUNK: "http://localhost:8111/v1",
        LookStrategy.CHUNK_2X: "http://localhost:8222/v1",
        LookStrategy.CHUNK_4X: "http://localhost:8333/v1",
    }
    assert LookEndpointConfig(shared_port=8111).endpoints() == {
        LookStrategy.CHUNK: "http://localhost:8111/v1",
        LookStrategy.CHUNK_2X: "http://localhost:8111/v1",
        LookStrategy.CHUNK_4X: "http://localhost:8111/v1",
    }


def test_each_look_routes_and_returns_gpu_offload_costs():
    root = ScriptedClient(
        [
            code(
                'a = llm_query(LOOK-CHUNK, "q", context)\n'
                'b = llm_query("LOOK-CHUNK-2x", "q", context)\n'
                'c = llm_query("LOOK-CHUNK-4x", "q", context)\n'
                "print(a.text, b.text, c.text)\n"
                "print(a.offloaded_tokens, b.offloaded_tokens, c.offloaded_tokens)"
            ),
            code("FINAL(c.text)"),
        ]
    )
    full = ScriptedClient(["private reasoning\nSUMMARY <full>"], prompt_tokens=12)
    x2 = ScriptedClient(["analysis\nSUMMARY <two>"], prompt_tokens=12)
    x4 = ScriptedClient(["thinking\nSUMMARY <four>"], prompt_tokens=12)

    result = RLM(
        root_client=root,
        look_clients={
            LookStrategy.CHUNK: full,
            LookStrategy.CHUNK_2X: x2,
            LookStrategy.CHUNK_4X: x4,
        },
        look_memory_tokens=4,
        max_steps=3,
        exec_timeout=None,
    ).run("document", "task")

    assert result.answer == "four"
    assert [len(full.calls), len(x2.calls), len(x4.calls)] == [1, 1, 1]
    assert result.metrics["look_calls"] == {
        "LOOK-CHUNK": 1,
        "LOOK-CHUNK-2x": 1,
        "LOOK-CHUNK-4x": 1,
    }
    assert result.metrics["look_gpu_tokens"] == 11
    assert result.metrics["look_offloaded_tokens"] == 10
    assert result.metrics["look_cost"] == 1011
    assert root.tokenize_calls == ["document"]
    sent = "\n".join(message["content"] for call in root.calls for message in call)
    assert sent.count("[LOOK COST]") >= 3
    # CHUNK and 2x offload under m=4; 4x fits. Warning only on offload > 0.
    assert sent.count("the context was large so that we had to offload") == 2
    assert "try 2x and 4x compression" in sent
    assert "private reasoning" not in sent
    assert "analysis\n" not in sent
    assert "thinking\n" not in sent
    assert "mode: LOOK-CHUNK-2x" in root.calls[0][0]["content"]
    assert "mode: VANILLA" in root.calls[0][0]["content"]


def test_look_parser_accepts_prompt_strings_and_repl_alias_form():
    assert LookStrategy.parse("LOOK-CHUNK-2x") is LookStrategy.CHUNK_2X
    assert LookStrategy.parse("LOOK_CHUNK_4X") is LookStrategy.CHUNK_4X


def test_sub_summary_parser_keeps_only_text_after_last_marker():
    summary, ok = extract_sub_summary(
        "private chain of thought\nSUMMARY:\n<partial clue>\n" "more private revision\nSUMMARY:\n<exact\nanswer>"
    )

    assert ok
    assert summary == "exact\nanswer"


def test_sub_summary_parser_never_leaks_unmarked_output():
    summary, ok = extract_sub_summary("private reasoning and an accidental answer")

    assert not ok
    assert summary.startswith("[SUBQUERY FORMAT ERROR:")
    assert "accidental answer" not in summary


def test_missing_summary_is_reprompted_once_as_mandatory():
    from evaluation.rlm.rlm_exp1.prompts import SUB_SUMMARY_RETRY_PROMPT

    root = ScriptedClient(
        [
            code('a = llm_query(LOOK-CHUNK, "q", context)\nprint(a.text)\nFINAL(a.text)'),
        ]
    )
    look = ScriptedClient(
        ["private reasoning without a marker", "Greece; the mosque passage is incomplete."],
        prompt_tokens=8,
    )
    result = RLM(
        root_client=root,
        look_clients={
            LookStrategy.CHUNK: look,
            LookStrategy.CHUNK_2X: look,
            LookStrategy.CHUNK_4X: look,
        },
        look_memory_tokens=8192,
        max_steps=2,
        exec_timeout=None,
    ).run("document", "task")

    sent = "\n".join(message["content"] for call in root.calls for message in call)
    wrapped = "SUMMARY: <Greece; the mosque passage is incomplete.>"
    assert result.answer == wrapped
    assert len(look.calls) == 2
    assert look.calls[1][-1]["content"] == SUB_SUMMARY_RETRY_PROMPT
    assert "private reasoning" not in sent
    assert wrapped in result.transcript[0]["observation"]
    assert result.metrics["sub_summary_retries"] == 1
    assert result.metrics["sub_summary_format_errors"] == 0


def test_final_composed_from_observed_words_is_accepted_without_chunk_order():
    root = ScriptedClient(
        [
            code('a = llm_query(LOOK-CHUNK, "q", context)\nprint(a.text)'),
            code('FINAL("Final Answer: Al-Aqsa Mosque, ID: 410")'),
        ]
    )
    look = ScriptedClient(["reasoning\nSUMMARY: <ID: 410 | TITLE: Al-Aqsa Mosque>"], prompt_tokens=8)
    kwargs = dict(
        look_clients={
            LookStrategy.CHUNK: look,
            LookStrategy.CHUNK_2X: look,
            LookStrategy.CHUNK_4X: look,
        },
        look_memory_tokens=8192,
        max_steps=3,
        exec_timeout=None,
    )
    task = "query: What is the great mosque at the site of the old temple? Print the TITLE and ID."

    result = RLM(root_client=root, **kwargs).run("document", task, answer_format="Final Answer:")

    assert result.answer == "Final Answer: Al-Aqsa Mosque, ID: 410"
    assert result.end_reason != "ungrounded_final"

    guessing = ScriptedClient(
        [
            code('a = llm_query(LOOK-CHUNK, "q", context)\nprint(a.text)'),
            code('FINAL("Final Answer: Umayyad Mosque, ID: 999")'),
        ]
    )
    rejected = RLM(root_client=guessing, **kwargs).run("document", task, answer_format="Final Answer:")

    assert rejected.answer != "Final Answer: Umayyad Mosque, ID: 999"
    sent = "\n".join(message["content"] for call in guessing.calls for message in call)
    assert "REJECTED: FINAL(" in sent


def test_cost_table_uses_measured_chars_per_token_and_configured_factor():
    assert token_placement(20, LookStrategy.CHUNK_2X, memory_tokens=4, cost_factor=10) == (10, 4, 6, 64)

    table = render_cost_table(
        context_chars=100,
        context_tokens=20,
        memory_tokens=4,
        cost_factor=10,
    )

    assert "10 | 2 | 2 | 0 | 2" in table
    assert "100 | 20 | 4 | 16 | 164" in table


def test_cap_root_history_keeps_latest_pair_and_does_not_mutate_source():
    from evaluation.rlm.rlm_exp1.history import cap_root_history
    from evaluation.rlm.rlm_exp1.prompts import HISTORY_OVERFLOW_NOTICE

    def count(msgs):
        return sum(len(m["content"]) for m in msgs)

    base = [{"role": "system", "content": "SYS"}]
    latest = [{"role": "assistant", "content": "a2"}, {"role": "user", "content": "NEW-LATEST"}]
    pairs = [
        [{"role": "assistant", "content": "a1"}, {"role": "user", "content": "OLD" * 80}],
        latest,
    ]
    max_tokens = count(base + [{"role": "user", "content": HISTORY_OVERFLOW_NOTICE}] + latest)
    sent, dropped, truncated = cap_root_history(base, pairs, count, max_tokens=max_tokens)

    contents = [m["content"] for m in sent]
    assert truncated
    assert dropped == 1
    assert HISTORY_OVERFLOW_NOTICE in contents
    assert "OLD" not in "".join(contents)
    assert "NEW-LATEST" in contents
    assert latest[1]["content"] == "NEW-LATEST"


def test_root_history_overflow_injects_exact_notice_and_drops_large_print():
    huge = "X" * 8000
    root = ScriptedClient(
        [
            code('print("READY")\nprint(context)'),
            code("FINAL('READY')"),
        ]
    )
    look = ScriptedClient(["unused"], prompt_tokens=4)
    result = RLM(
        root_client=root,
        look_clients={
            LookStrategy.CHUNK: look,
            LookStrategy.CHUNK_2X: look,
            LookStrategy.CHUNK_4X: look,
        },
        look_memory_tokens=32,
        max_steps=3,
        exec_timeout=None,
    ).run(huge, "task")

    notice = "[HISTORY TRUNCATED due to overflow in GPU memory budget . avoid printing large chunks]"
    first = "\n".join(message["content"] for message in root.calls[0])
    second = "\n".join(message["content"] for message in root.calls[1])
    assert result.answer == "READY"
    assert not any(message.get("content") == notice for message in root.calls[0])
    assert any(message.get("content") == notice for message in root.calls[1])
    assert huge not in second
    assert "hard-capped at 32 tokens" in first
    assert result.metrics["history_truncations"] >= 1


def test_print_is_capped_at_32_tokens_and_keeps_look_cost():
    from evaluation.rlm.rlm_exp1.prompts import print_truncation_notice
    from evaluation.rlm.rlm_exp1.types import TokenCounter

    huge = "X" * 8000
    notice = print_truncation_notice(32)
    root = ScriptedClient(
        [
            code('print("READY")\nprint(context)'),
            code("FINAL('READY')"),
        ]
    )
    look = ScriptedClient(["unused"], prompt_tokens=4)
    result = RLM(
        root_client=root,
        look_clients={
            LookStrategy.CHUNK: look,
            LookStrategy.CHUNK_2X: look,
            LookStrategy.CHUNK_4X: look,
        },
        look_memory_tokens=8192,
        print_truncate_tokens=32,
        max_steps=3,
        exec_timeout=None,
    ).run(huge, "task")

    obs = [message["content"] for message in root.calls[1] if message["role"] == "user"][-1]
    inner = obs.split("```", 2)[1] if obs.count("```") >= 2 else obs
    body = "\n".join(line for line in inner.splitlines() if not line.startswith("[LOOK COST]"))
    clipped = body.replace(notice, "").strip()
    assert result.answer == "READY"
    assert notice in obs
    assert huge not in obs
    assert TokenCounter().count(clipped) <= 32
