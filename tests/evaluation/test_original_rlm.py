# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original RLM: REPL + FINAL / FINAL_VAR, matching recursive-llm/."""

from evaluation.rlm.harness.original_rlm import OriginalRLM, extract_final, extract_final_var_name


class ScriptedClient:
    def __init__(self, replies=()):
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if not self._replies:
            raise AssertionError(f"unexpected chat; last messages={messages!r}")
        return self._replies.pop(0)


def test_final_is_the_whole_reply():
    assert extract_final('FINAL("Cincinnati, Ohio")') == "Cincinnati, Ohio"
    assert extract_final('FINAL("x")\nwait') is None
    assert extract_final_var_name("FINAL_VAR(ans)") == "ans"
    assert extract_final_var_name("print(1)\nFINAL_VAR(ans)") is None


def test_python_then_final():
    root = ScriptedClient(
        [
            "print(context[0:5])",
            'FINAL("alpha")',
        ]
    )
    result = OriginalRLM(root, max_steps=4).run("alphabet", "what letter")
    assert result.finished and result.end_reason == "FINAL"
    assert result.answer == "alpha"
    assert result.transcript[1]["role"] == "observation"
    assert "alpha" in result.transcript[1]["content"]


def test_final_var_reads_repl_state():
    root = ScriptedClient(
        [
            "ans = context.split()[0]\nprint(ans)",
            "FINAL_VAR(ans)",
        ]
    )
    result = OriginalRLM(root, max_steps=4).run("Melvil Dewey invented it", "who")
    assert result.finished and result.end_reason == "FINAL_VAR"
    assert result.answer == "Melvil"


def test_recursive_llm_is_a_plain_leaf_call():
    root = ScriptedClient(
        [
            'ans = recursive_llm("Who?", context[0:20])\nprint(ans)',
            'FINAL("Melvil Dewey")',
        ]
    )
    sub = ScriptedClient(["Melvil Dewey"])
    result = OriginalRLM(root, sub, max_steps=4).run("Melvil Dewey invented it", "who")
    assert result.finished and result.answer == "Melvil Dewey"
    assert result.metrics["sub_calls"] == 1
    leaf = sub.calls[0]
    assert "do not emit REPL code" in leaf[0]["content"]
    assert "Task:\nWho?" in leaf[1]["content"]
    assert "Melvil Dewey invente" in leaf[1]["content"]


def test_root_prompt_does_not_contain_the_document():
    root = ScriptedClient(['FINAL("x")'])
    doc = "SECRET_DOCUMENT_BODY " * 20
    OriginalRLM(root, max_steps=2).run(doc, "q")
    blob = "\n".join(m["content"] for m in root.calls[0])
    assert "SECRET_DOCUMENT_BODY" not in blob
    assert "recursive_llm" in blob
    assert "FINAL_VAR" in blob


def test_runaway_step_is_cut_off_and_the_loop_continues():
    root = ScriptedClient(
        [
            "while True:\n    pass",
            'FINAL("done")',
        ]
    )
    result = OriginalRLM(root, max_steps=4, repl_timeout=0.5).run("doc", "q")
    assert result.finished and result.answer == "done"
    assert "timed out" in result.transcript[1]["content"]


def test_max_iterations_without_final():
    root = ScriptedClient(["print(1)", "print(2)"])
    result = OriginalRLM(root, max_steps=2).run("doc", "q")
    assert not result.finished
    assert result.end_reason == "max_iterations"
