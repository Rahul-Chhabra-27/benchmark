# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for the modular, vLLM-only RLM."""

import sys

import pytest

from evaluation.rlm.clean_rlm import RLM, Scratchpad, VLLMClient, vanilla_answer
from evaluation.rlm.client import LLMClient
from evaluation.rlm.rlm import RLM as ParentRLM
from evaluation.rlm.rlm import vanilla_answer as parent_vanilla_answer
from evaluation.rlm.run_benchmark import main


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.index = 0
        self.usage = None

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        reply = self.replies[min(self.index, len(self.replies) - 1)]
        self.index += 1
        return reply


def code(body):
    return f"```python\n{body}\n```"


def test_public_backend_is_explicitly_vllm_http_only():
    assert issubclass(VLLMClient, LLMClient)
    assert not hasattr(VLLMClient, "chat_split")
    with pytest.raises(ValueError, match=r"explicit http\(s\)"):
        VLLMClient(model="test", base_url="local-vllm")


@pytest.mark.parametrize("backend", ["kvzip", "nim"])
def test_clean_runner_rejects_non_http_sub_backend(backend, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmark",
            "--dataset",
            "niah",
            "--mode",
            "clean-rlm",
            "--sub-backend",
            backend,
            "--out",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        main()
    assert "--mode clean-rlm needs explicit --sub-backend http" in capsys.readouterr().err


def test_subcall_always_uses_openai_chat_and_embeds_the_slice():
    class SplitShapedClient(ScriptedClient):
        def chat_split(self, **kwargs):
            raise AssertionError("clean RLM must never select a split-context transport")

    root = ScriptedClient([code("print(llm_query('find it', context[0:8]))"), code("FINAL('reader')")])
    sub = SplitShapedClient(["reader"])

    result = RLM(root_client=root, sub_client=sub, max_steps=3, exec_timeout=None).run("document text", "task")

    assert result.answer == "reader"
    assert len(sub.calls) == 1
    assert "find it\n\ndocument" in sub.calls[0][-1]["content"]
    assert "sub_split_calls" not in result.metrics


def test_agentic_http_path_matches_parent_representatively():
    replies = [
        code("hits = search('Riverside winner 1983')\nprint(hits)"),
        code("ans = llm_query('Who won?', hits)\nprint(ans)"),
        code("note(ans)\nFINAL(ans)"),
    ]
    document = "padding " * 300 + "Riverside was won by Bignotti Cotter Racing in 1983."

    def run(cls):
        root = ScriptedClient(replies)
        sub = ScriptedClient(["Bignotti Cotter Racing"])
        return cls(
            root_client=root,
            sub_client=sub,
            max_steps=4,
            search_k=2,
            scratchpad=Scratchpad(),
            exec_timeout=None,
        ).run(document, "Who won the Riverside race in 1983?")

    parent = run(ParentRLM)
    clean = run(RLM)
    assert (clean.answer, clean.finished, clean.end_reason) == (
        parent.answer,
        parent.finished,
        parent.end_reason,
    )
    for key in (
        "steps",
        "sub_calls",
        "sub_cache_hits",
        "search_calls",
        "search_hits_read",
        "notes_saved",
        "document_coverage_fraction",
    ):
        assert clean.metrics[key] == parent.metrics[key]


def test_structured_http_path_matches_parent_representatively():
    from evaluation.rlm.retrieval import Hit

    document = "alpha fact. beta fact."
    chunks = [
        Hit(1, 0, 11, 1.0, document[0:11]),
        Hit(2, 12, len(document), 1.0, document[12:]),
    ]

    def run(cls):
        root = ScriptedClient(["combined answer"])
        sub = ScriptedClient(["first finding", "second finding"])
        return cls(root_client=root, sub_client=sub, exec_timeout=None).run_structured(document, "combine?", chunks)

    parent = run(ParentRLM)
    clean = run(RLM)
    assert (clean.answer, clean.finished, clean.end_reason) == (
        parent.answer,
        parent.finished,
        parent.end_reason,
    )
    for key in ("steps", "map_calls", "sub_calls", "chunks_with_findings", "document_coverage_fraction"):
        assert clean.metrics[key] == parent.metrics[key]


def test_vanilla_http_prompt_and_accounting_match_parent():
    document = "abcdef" * 100

    def run(fn):
        client = ScriptedClient(["answer"])
        stats = {}
        answer = fn(client, document, "question?", char_limit=120, answer_format="Final: x", stats=stats)
        return answer, client.calls, stats

    assert run(vanilla_answer) == run(parent_vanilla_answer)
