# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The clean, config-driven RLM (evaluation/rlm/core): config, prompt, REPL, loop, runner."""

import json
import time
from pathlib import Path

import pytest
import yaml

from evaluation.rlm.core import __main__ as cli
from evaluation.rlm.core.config import config_from_dict, load_config
from evaluation.rlm.core.loop import RLM, Limits
from evaluation.rlm.core.ordering import build_chunks, query_text
from evaluation.rlm.core.prompts import NO_CODE_BLOCK, system_prompt
from evaluation.rlm.core.readers import KVReader, Read
from evaluation.rlm.core.repl import Repl, extract_code
from evaluation.rlm.core.run import run_experiment

CONFIGS = Path(__file__).resolve().parents[2] / "evaluation" / "rlm" / "configs"


def code(body: str) -> str:
    return f"```python\n{body}\n```"


class ScriptedRoot:
    """A root model that replies from a script and records what it was sent."""

    def __init__(self, replies, fail_first=None):
        self.replies = list(replies)
        self.sent = []
        self.fail_first = fail_first

    def chat(self, messages, **kw):
        self.sent.append([dict(m) for m in messages])
        if self.fail_first is not None:
            exc, self.fail_first = self.fail_first, None
            raise exc
        return self.replies.pop(0) if self.replies else code("print('idle')")


class FakeReader:
    factor = 1

    def __init__(self, answer="Paris"):
        self.answer = answer
        self.reads = []

    def count_tokens(self, text):
        return len(text) // 4

    def read(self, question, text):
        self.reads.append((question, text))
        return Read(self.answer, len(text) // 4, len(text) // 4, 0.0)


def rlm(replies, reader=None, **limits):
    return RLM(ScriptedRoot(replies), reader or FakeReader(), Limits(**limits))


# --------------------------------------------------------------------------- config


def minimal(**extra):
    return {"name": "t", "data": {"tasks": ["hotpotqa"]}, **extra}


def test_defaults_and_loft_task_names():
    config = config_from_dict(minimal())
    assert config.ordering == "none" and config.compression.factor == 1
    assert config.data.task_dirs() == ["hotpotqa_128k"]
    assert config.read_tokens == 8192 and not config.compression.in_process


def test_budget_sets_reads_to_b_times_f():
    config = config_from_dict(minimal(compression={"factor": 4, "budget_mb": 1024}))
    assert config.compression.budget_tokens == 6944 and config.read_tokens == 4 * 6944
    assert config.compression.in_process


@pytest.mark.parametrize(
    "bad, message",
    [
        (minimal(ordering="random"), "ordering must be one of"),
        (minimal(compression={"factor": 3, "budget_mb": 1024}), "factor must be one of"),
        (minimal(compression={"factor": 2}), "needs compression.budget_mb"),
        (minimal(rlm={"max_stpes": 3}), "unknown key"),
        (minimal(colour="red"), "unknown key"),
        ({"name": "t"}, "at least `name` and `data`"),
    ],
)
def test_bad_configs_are_refused(bad, message):
    with pytest.raises(ValueError, match=message):
        config_from_dict(bad)


def test_extends_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({"data": {"tasks": ["nq"], "limit": 5}, "rlm": {"max_steps": 9}})
    )
    (tmp_path / "x.yaml").write_text(yaml.safe_dump({"extends": "base.yaml", "name": "x", "ordering": "bm25-graph"}))
    config = load_config(tmp_path / "x.yaml", **{"data.limit": 2, "name": "smoke"})
    assert (config.name, config.ordering, config.data.limit, config.rlm.max_steps) == ("smoke", "bm25-graph", 2, 9)
    assert config.data.tasks == ("nq",)


def test_every_shipped_config_loads():
    names = [load_config(p).name for p in sorted(CONFIGS.glob("[!_]*.yaml"))]
    assert len(names) == len(set(names)) >= 6  # distinct names, so no two share an output dir


def test_check_command_prints_the_plan(capsys):
    cli.main(["check", str(CONFIGS / "graph_x2.yaml"), "--set", "data.limit=3", "--show-prompt"])
    out = capsys.readouterr().out
    assert "B=6,944 tokens (1024 MB) x F=2" in out and "limit=3" in out and "grouped by topic" in out


# --------------------------------------------------------------------------- prompt


def test_prompt_adds_only_what_an_option_adds():
    common = dict(question="Q?", context_chars=1000, read_tokens=100, max_steps=5, output_chars=50)
    base = system_prompt(**common)
    graph = system_prompt(**common, ordering="bm25-graph", n_chunks=7, chunk_chars=2000)
    assert "chunks" not in base and "cut into 7 pieces" in graph and "grouped by topic" in graph
    listed = system_prompt(**common, answer_format="Final Answer:", list_answer=True)
    assert '["first", "second", "third"]' in listed


# --------------------------------------------------------------------------- repl


def test_repl_persists_echoes_and_reports_errors():
    repl = Repl({}, timeout=None)
    assert repl.run("x = 2", 100) == "[no output]"
    assert repl.run("x * 21", 100).strip() == "42"
    assert "ZeroDivisionError" in repl.run("1 / 0", 1000)
    assert "SyntaxError" in repl.run("def (", 1000)


def test_repl_truncates_long_output_in_the_middle():
    out = Repl({}, timeout=None).run("print('a' * 500 + 'b' * 500)", 100)
    assert out.startswith("a" * 50) and out.rstrip().endswith("b" * 40) and "characters cut" in out


def test_repl_stops_a_runaway_cell_but_not_a_slow_sub_call():
    repl = Repl({}, timeout=0.5)
    assert "[TIMEOUT]" in repl.run("while True: pass", 200)

    def slow():
        with repl.outside():
            time.sleep(0.8)  # longer than the timeout, but paused
        return "done"

    repl.namespace["slow"] = slow
    assert repl.run("slow()", 200).strip() == "done"


def test_what_a_sub_call_prints_never_reaches_the_root(capsys):
    repl = Repl({}, timeout=None)

    def noisy_read():
        with repl.outside():
            print("library warning")  # e.g. a transformers log line during a sub-call
        return "answer"

    repl.namespace["noisy_read"] = noisy_read
    assert repl.run("print(noisy_read())", 200).strip() == "answer"
    assert "library warning" in capsys.readouterr().out


def test_extract_code_takes_the_first_block():
    assert extract_code("text\n```python\na = 1\n```\n```python\nb = 2\n```") == "a = 1"
    assert extract_code("no code here") is None


# --------------------------------------------------------------------------- loop


def test_read_then_final():
    reader = FakeReader("Paris")
    result = rlm([code("ans = llm_query('Capital?', context[:400])\nprint(ans)"), code("FINAL(ans)")], reader).run(
        "x" * 1000, "What is the capital?"
    )
    assert (result.answer, result.end_reason, result.steps) == ("Paris", "final", 2)
    (call,) = result.calls
    assert (call.step, call.read_tokens, call.refused, call.answer) == (1, 100, "", "Paris")
    assert reader.reads == [("Capital?", "x" * 400)]


def test_reads_over_the_limit_and_past_the_call_cap_are_refused():
    loop = rlm(
        [code("print(llm_query('q', context))"), code("for _ in range(3): print(llm_query('q', 'abcd'))")],
        read_tokens=10,
        max_sub_calls=2,
        max_steps=2,
    )
    result = loop.run("y" * 400, "q")
    refused = [c.refused for c in result.calls]
    assert "100 tokens" in refused[0] and "at most 10" in refused[0]
    assert [bool(r) for r in refused] == [True, False, False, True]
    assert "limit of 2 llm_query calls" in refused[-1]


def test_a_reply_without_code_is_told_so_and_the_run_continues():
    root = ScriptedRoot(["I think it is Paris.", code("FINAL('Paris')")])
    result = RLM(root, FakeReader(), Limits()).run("doc", "q")
    assert result.answer == "Paris" and result.steps == 2
    assert NO_CODE_BLOCK in root.sent[1][-1]["content"]


def test_max_steps_ends_the_run_without_an_answer():
    result = rlm([], max_steps=3).run("doc", "q")
    assert (result.answer, result.end_reason, result.steps) == (None, "max_steps", 3)


def test_root_window_overflow_drops_the_oldest_turn():
    class Overflow(Exception):
        status_code = 400

    root = ScriptedRoot([code("print(1)"), code("FINAL('x')")])
    loop = RLM(root, FakeReader(), Limits())
    original = root.chat
    calls = {"n": 0}

    def chat(messages, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise Overflow("This model's maximum context length is 32768 tokens")
        return original(messages, **kw)

    root.chat = chat
    result = loop.run("doc", "q")
    assert result.answer == "x"
    assert len(root.sent[-1]) == 2  # system + begin: the one earlier turn was dropped


def test_chunk_lists_are_read_and_counted():
    chunks = build_chunks("a" * 2000 + "b" * 2000 + "c" * 2000, "q", "bm25-query", None)
    reader = FakeReader("ok")
    loop = RLM(ScriptedRoot([code("llm_query('q', chunks[0:2])"), code("FINAL('ok')")]), reader, Limits())
    result = loop.run("a" * 2000 + "b" * 2000 + "c" * 2000, "q", chunks=chunks, chunk_chars=2000)
    assert result.calls[0].chunks == 2 and len(reader.reads[0][1]) == 4002


# --------------------------------------------------------------------------- readers and ordering


def test_kv_reader_compresses_the_text_side_and_counts_kept_tokens():
    class Sub:
        def token_len(self, text):
            return len(text)

        def chat_split(self, question, context):
            self.call = (question, context)
            return "answer"

    sub = Sub()
    read = KVReader(sub, factor=4).read("Who?", "z" * 10)
    assert (read.answer, read.read_tokens, read.kept_tokens) == ("answer", 10, 3)
    assert sub.call[1] == "z" * 10 and "Who?" in sub.call[0]


def test_query_text_strips_the_loft_wrapper():
    assert query_text("Based on the documents above...\nquery: Who won?\n") == "Who won?"
    assert query_text("Plain question?") == "Plain question?"


def test_orderings_cover_the_whole_document(tmp_path):
    doc = "".join(f"filler{i} " * 250 for i in range(5))[:10_000] + "quasar " * 10
    ranked = build_chunks(doc, "query: quasar?", "bm25-query", None)
    assert ranked[0].text.count("quasar") == 10
    graph = build_chunks(doc, "anything", "bm25-graph", str(tmp_path))
    assert sorted((c.start, c.end) for c in graph) == sorted((c.start, c.end) for c in ranked)
    assert build_chunks(doc, "q", "none", None) == []


# --------------------------------------------------------------------------- runner


def examples():
    return [
        {
            "id": f"loft:hotpotqa_128k:{i}",
            "task": "hotpotqa_128k",
            "question": f"query: question {i}?",
            "context": "The capital of France is Paris. " * 20,
            "answers": ["Paris"],
            "answer_prefix": "Final Answer: ",
        }
        for i in range(2)
    ]


def test_run_experiment_writes_results_and_resumes(tmp_path):
    config = config_from_dict(minimal(output_dir=str(tmp_path), rlm={"max_steps": 3}))
    script = [code("a = llm_query('q', context[:200])"), code("FINAL(a)")] * 2
    root = ScriptedRoot(script)
    logs = []
    results = run_experiment(config, root=root, reader=FakeReader("Paris"), load=lambda *a: examples(), log=logs.append)
    metrics = results["hotpotqa_128k"]
    assert metrics["f1"] == 1.0 and metrics["runtime"]["scored"] == 2
    assert metrics["runtime"]["end_reasons"] == {"final": 2} and metrics["runtime"]["sub_calls_mean"] == 1.0
    run_dir = tmp_path / "t" / "hotpotqa_128k"
    assert {p.name for p in run_dir.iterdir()} >= {"config.yaml", "checkpoint.jsonl", "predictions.csv", "metrics.json"}
    record = json.loads((run_dir / "checkpoint.jsonl").read_text().splitlines()[0])
    assert record["calls"][0]["read_tokens"] == 50 and record["reads"]["sub_calls"] == 1

    again = ScriptedRoot([])
    run_experiment(config, root=again, reader=FakeReader(), load=lambda *a: examples(), log=logs.append)
    assert again.sent == []  # both questions were already answered


def test_a_different_config_never_resumes_into_an_old_directory(tmp_path):
    first = config_from_dict(minimal(output_dir=str(tmp_path)))
    run_experiment(
        first,
        root=ScriptedRoot([code("FINAL('x')")] * 2),
        reader=FakeReader(),
        load=lambda *a: examples(),
        log=lambda _: None,
    )
    changed = config_from_dict(minimal(output_dir=str(tmp_path), ordering="bm25-query"))
    with pytest.raises(SystemExit, match="different config"):
        run_experiment(
            changed, root=ScriptedRoot([]), reader=FakeReader(), load=lambda *a: examples(), log=lambda _: None
        )


def test_a_failing_sub_model_is_shown_to_the_root_not_fatal(tmp_path):
    class Broken(FakeReader):
        def read(self, question, text):
            raise RuntimeError("GPU fell over")

    config = config_from_dict(minimal(output_dir=str(tmp_path)))
    root = ScriptedRoot([code("llm_query('q', 'abc')")] * 2)
    # The reader error escapes llm_query inside the cell, so the REPL reports it and the
    # run continues: a sub-model failure is an observation, not a lost question.
    run_experiment(config, root=root, reader=Broken(), load=lambda *a: examples()[:1], log=lambda _: None)
    record = json.loads((tmp_path / "t" / "hotpotqa_128k" / "checkpoint.jsonl").read_text())
    assert record["error"] is None and record["end_reason"] == "max_steps"
