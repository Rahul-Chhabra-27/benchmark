# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import sys

import pytest

from evaluation.rlm.diagnostic import DiagnosticTrace
from tests.evaluation.test_rlm_chunk_order import graph_view, query_view, run_once


@pytest.mark.parametrize("arm,view", [("bm25-query", query_view), ("bm25-graph", graph_view)])
def test_observational_trace(tmp_path, arm, view):
    cells = [
        'def nested(hits):\n    def reader():\n        return llm_query("Which cello sonata?", hits)\n'
        "    return reader()\nans = nested(chunks[:2])\nprint(ans)",
        "FINAL(ans)",
    ]
    plain, prompt, root = run_once(arm, view(), cells=cells)
    with DiagnosticTrace(tmp_path / "trace"):
        traced, traced_prompt, traced_root = run_once(arm, view(), cells=cells)
    assert plain.answer == traced.answer
    assert plain.finished == traced.finished
    assert plain.transcript == traced.transcript
    assert prompt == traced_prompt
    assert root.calls == traced_root.calls
    # Wall time is the only expected measurement difference.
    assert {k: v for k, v in plain.metrics.items() if "time" not in k and "seconds" not in k} == {
        k: v for k, v in traced.metrics.items() if "time" not in k and "seconds" not in k
    }
    summary = json.loads((tmp_path / "trace/summary.json").read_text())
    names = {f["name"] for f in summary["functions"]}
    assert {"nested", "nested.<locals>.reader", "RLM._make_env.<locals>.llm_query"} <= names
    events = [json.loads(line) for line in (tmp_path / "trace/calls.jsonl").read_text().splitlines()]
    assert any(e.get("code", "").strip() == cells[0] for e in events)
    calls = {e["call"]: e for e in events if e["event"] == "call"}
    assert all(e["parent"] is None or e["parent"] in calls for e in calls.values())
    assert sys.getprofile() is None


def test_exception_restores_profiler(tmp_path):
    with pytest.raises(ValueError), DiagnosticTrace(tmp_path / "trace"):
        raise ValueError("deliberate")
    assert sys.getprofile() is None
    assert (tmp_path / "trace/summary.json").exists()


def test_model_exchange_allowlist(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from evaluation.rlm.client import LLMClient

    monkeypatch.setattr(LLMClient, "__post_init__", lambda self: None)
    client = LLMClient("fake-model")
    requests = []

    def create(**params):
        secret_not_for_trace = "PRIVATE_SENTINEL"
        requests.append(params)
        assert secret_not_for_trace
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="reader answer"))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
        )

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    messages = [{"role": "user", "content": "read this"}]
    with DiagnosticTrace(tmp_path / "trace"):
        assert client.chat(messages) == "reader answer"
    raw = (tmp_path / "trace/calls.jsonl").read_text()
    events = [json.loads(line) for line in raw.splitlines()]
    assert len(requests) == client.usage.calls == 1
    assert sum("messages" in e for e in events) == 1
    assert next(e["messages"] for e in events if "messages" in e) == messages
    assert next(e["response"] for e in events if "response" in e) == "reader answer"
    assert "PRIVATE_SENTINEL" not in raw


def test_generated_functions_with_same_location_are_distinct(tmp_path):
    with DiagnosticTrace(tmp_path / "trace"):
        env = {}
        exec("def generated():\n    return 1\ngenerated()", env)
        exec("def generated():\n    return 2\ngenerated()", env)
    summary = json.loads((tmp_path / "trace/summary.json").read_text())
    functions = [f for f in summary["functions"] if f["name"] == "generated"]
    assert len(functions) == 2
    assert functions[0]["key"] != functions[1]["key"]
