# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the recursive-llm LOFT-128k baseline runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from baselines.official_rlm.run_benchmark import (
    LOFT_128K_TASKS,
    build_query,
    build_rlm,
    in_recursive_llm_env,
    install_vllm_reasoning_split,
    litellm_model_id,
    load_done,
    loft_128k_tasks,
    normalize_endpoint,
    parse_args,
    protocol_text_from_completion,
    record_from_result,
    recursive_llm_python,
    recursive_llm_root,
    run_examples,
    split_vllm_reasoning,
    write_run_artifacts,
)


def test_normalize_endpoint_adds_v1():
    assert normalize_endpoint("localhost:8000") == "http://localhost:8000/v1"
    assert normalize_endpoint("http://127.0.0.1:8000/") == "http://127.0.0.1:8000/v1"
    assert normalize_endpoint("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000/v1"
    assert normalize_endpoint("http://host:8000/v1/chat/completions") == "http://host:8000/v1"


def test_litellm_model_id_prefixes_hf_ids():
    assert litellm_model_id("Qwen/Qwen3-4B-Instruct-2507") == "openai/Qwen/Qwen3-4B-Instruct-2507"
    assert litellm_model_id("openai/local") == "openai/local"
    assert litellm_model_id("hosted_vllm/Qwen/Qwen3-4B") == "hosted_vllm/Qwen/Qwen3-4B"
    assert litellm_model_id("gemini/gemini-2.5-flash") == "gemini/gemini-2.5-flash"
    assert litellm_model_id("anthropic/claude-3-5-sonnet-20241022") == "anthropic/claude-3-5-sonnet-20241022"


def test_loft_128k_task_selection():
    assert loft_128k_tasks(None) == LOFT_128K_TASKS
    assert loft_128k_tasks("loft_128k") == LOFT_128K_TASKS
    assert loft_128k_tasks("nq_128k") == ("nq_128k",)
    with pytest.raises(ValueError, match="unknown LOFT-128k task"):
        loft_128k_tasks("nq_32k")


def test_parse_args_options():
    args = parse_args(["--endpoint", "http://localhost:8000/v1", "--task", "nq_128k", "--limit", "1"])
    assert args.endpoint == "http://localhost:8000/v1"
    assert args.task == "nq_128k"
    assert args.limit == 1
    assert args.max_depth == 1

    # Direct Gemini API call without --endpoint
    args_gemini = parse_args(["--model", "gemini/gemini-2.5-flash", "--api-key", "my-gemini-key"])
    assert args_gemini.endpoint is None
    assert args_gemini.model == "gemini/gemini-2.5-flash"
    assert args_gemini.api_key == "my-gemini-key"


def test_build_query_appends_loft_prefix():
    example = {"question": "Who won?", "answer_prefix": "Final Answer: "}
    assert build_query(example, True) == "Who won?\nFinal Answer: "
    assert build_query(example, False) == "Who won?"


def test_record_and_artifacts_score_loft_frame(tmp_path: Path):
    result = SimpleNamespace(
        answer="Paris",
        succeeded=True,
        stats={
            "total_iterations": 3,
            "leaf_calls": 1,
            "total_tokens": 40,
            "prompt_tokens": 30,
            "completion_tokens": 10,
        },
        trajectory=(),
        error_type=None,
        error=None,
    )
    example = {
        "id": "loft:nq_128k:0",
        "task": "nq_128k",
        "answers": ["Paris"],
        "answer_prefix": "Final Answer: ",
        "context": "x",
        "scoring": {"answer": ["Paris"], "answers": ["Paris"]},
    }
    record = record_from_result(example, "Who?", result, 1.25)
    assert record["pred"] == "Paris"
    assert record["finished"] is True
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(json.dumps(record) + "\n")
    metrics = write_run_artifacts(checkpoint, tmp_path, {"backend": "recursive-llm"})
    assert metrics["num_samples"] == 1
    assert metrics["subspan_em"] == 1.0
    assert (tmp_path / "predictions.csv").exists()
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["backend"] == "recursive-llm"


def test_run_examples_uses_try_complete_result(tmp_path: Path):
    class StubRLM:
        def try_complete_result(self, query, context):
            assert "needle" in context
            assert "secret" in query
            return SimpleNamespace(
                answer="42",
                succeeded=True,
                stats={"total_iterations": 1, "leaf_calls": 0, "total_tokens": 8},
                trajectory=(),
            )

    checkpoint = tmp_path / "checkpoint.jsonl"
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    example = {
        "id": "loft:nq_128k:0",
        "task": "nq_128k",
        "context": "the needle is 42",
        "question": "what is the secret",
        "answers": ["42"],
        "answer_prefix": "",
        "scoring": {},
    }
    n, finished = run_examples(
        StubRLM(),
        [example],
        include_answer_prefix=True,
        checkpoint_path=checkpoint,
        transcript_dir=transcripts,
        save_trajectory=False,
        debug=False,
        done=set(),
    )
    assert (n, finished) == (1, 1)
    saved = json.loads(checkpoint.read_text())
    assert saved["pred"] == "42"
    assert load_done(checkpoint) == {"loft:nq_128k:0"}


def test_build_rlm_configurations():
    pytest.importorskip("litellm")
    args_vllm = parse_args(["--endpoint", "http://localhost:8000/v1", "--model", "Qwen/Qwen3-4B"])
    rlm_vllm = build_rlm(args_vllm)
    assert rlm_vllm.model == "openai/Qwen/Qwen3-4B"
    assert rlm_vllm.api_base == "http://localhost:8000/v1"
    assert rlm_vllm.api_key == "EMPTY"
    assert rlm_vllm.llm_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    args_nothink = parse_args(["--endpoint", "http://localhost:8000/v1", "--model", "Qwen/Qwen3-4B", "--no-thinking"])
    rlm_off = build_rlm(args_nothink)
    assert "extra_body" not in rlm_off.llm_kwargs

    args_gemini = parse_args(["--model", "gemini/gemini-2.5-flash", "--api-key", "test-key"])
    rlm_gemini = build_rlm(args_gemini)
    assert rlm_gemini.model == "gemini/gemini-2.5-flash"
    assert rlm_gemini.api_base is None
    assert rlm_gemini.api_key == "test-key"
    from rlm.core import RLM

    assert getattr(RLM._response_text_info, "_vllm_reasoning_split", False) is True


def test_split_vllm_reasoning_makes_final_and_repl_parseable():
    pytest.importorskip("litellm")
    from baselines.official_rlm.run_benchmark import ensure_recursive_llm_on_path

    ensure_recursive_llm_on_path()
    from rlm.parser import extract_final
    from rlm.repl import _extract_code

    tagged = '<think>\nThe answer is Paris.\n</think>\nFINAL("Paris")'
    reasoning, protocol = split_vllm_reasoning(tagged)
    assert "The answer is Paris" in reasoning
    assert protocol == 'FINAL("Paris")'
    assert extract_final(tagged) is None
    assert extract_final(protocol) == "Paris"

    # Qwen3.5 on vLLM often writes the trace then </think> with no opening tag.
    unfenced = "I will print a slice.\n</think>\nprint(context[:20])"
    reasoning, protocol = split_vllm_reasoning(unfenced)
    assert reasoning == "I will print a slice."
    assert protocol == "print(context[:20])"
    assert _extract_code(protocol) == "print(context[:20])"
    assert "I will print" in _extract_code(unfenced)

    already_split = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='FINAL("x")', reasoning_content="notes"))]
    )
    assert protocol_text_from_completion(already_split, 'FINAL("x")') == 'FINAL("x")'

    merged = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=tagged, reasoning_content=None))])
    assert protocol_text_from_completion(merged, tagged) == 'FINAL("Paris")'

    install_vllm_reasoning_split()
    from rlm.core import RLM

    text, _ = RLM._response_text_info(merged)
    assert text == 'FINAL("Paris")'


def test_recursive_llm_paths_point_at_local_checkout():
    root = recursive_llm_root()
    if not (root / "src" / "rlm").is_dir():
        pytest.skip(
            "the recursive-llm checkout is not present (baselines/official_rlm/recursive-llm is an empty submodule)"
        )
    assert root.name == "recursive-llm"
    assert (root / "src" / "rlm").is_dir()
    python = recursive_llm_python()
    assert python == root / ".venv" / "bin" / "python"
    assert in_recursive_llm_env() is False


def test_recursive_llm_env_exposes_rlm_query():
    python = recursive_llm_python()
    if not python.is_file():
        pytest.skip(f"recursive-llm venv missing at {python}")
    code = (
        "from rlm.core import RLM\n"
        "rlm = RLM(model='openai/local', api_base='http://localhost:8000/v1', "
        "api_key='EMPTY', max_depth=1)\n"
        "env = rlm._build_repl_env('q', 'document')\n"
        "assert env['recursive_llm'] is env['rlm_query']\n"
        "print('ok')\n"
    )
    import subprocess

    result = subprocess.run([str(python), "-c", code], check=True, capture_output=True, text=True)
    assert "ok" in result.stdout
