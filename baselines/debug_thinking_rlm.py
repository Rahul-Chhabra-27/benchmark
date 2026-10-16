# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot debug: dump a thinking completion and two RLM steps."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from openai import OpenAI

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "baselines" / "recursive-llm" / "src"))

ENDPOINT = os.environ.get("DEBUG_ENDPOINT", "http://127.0.0.1:8012/v1")
MODEL = os.environ.get("DEBUG_MODEL", "Qwen/Qwen3.5-4B")
OUT = Path(os.environ.get("DEBUG_OUT", str(REPO / "evaluation/results/rlm/recursive-llm/loft128k-n20-think/debug")))
OUT.mkdir(parents=True, exist_ok=True)


def dump_raw_completion() -> dict:
    client = OpenAI(base_url=ENDPOINT, api_key="EMPTY")
    extra = {"chat_template_kwargs": {"enable_thinking": True}}
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Recursive Language Model. Reply with either one Python "
                "statement or a standalone FINAL(\"answer\")."
            ),
        },
        {
            "role": "user",
            "content": "The secret is 42. Write Python to print context[:20], then you will FINAL later.",
        },
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=2048,
        extra_body=extra,
    )
    msg = resp.choices[0].message
    payload = {
        "finish_reason": resp.choices[0].finish_reason,
        "content": msg.content,
        "reasoning_content": getattr(msg, "reasoning_content", None),
        "model_extra": getattr(msg, "model_extra", None),
        "message_keys": list(msg.model_dump().keys()),
        "message_dump": msg.model_dump(),
    }
    (OUT / "raw_completion.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("=== raw completion ===")
    print("finish_reason", payload["finish_reason"])
    print("content_preview", repr((msg.content or "")[:500]))
    print("reasoning_preview", repr(str(payload["reasoning_content"] or "")[:500]))
    print("message_keys", payload["message_keys"])
    return payload


def dump_rlm_steps() -> None:
    from rlm import RLM
    from rlm.repl import _extract_code
    from rlm.parser import extract_final

    raw = json.loads((OUT / "raw_completion.json").read_text())
    content = raw.get("content") or ""
    from baselines.run_benchmark import install_vllm_reasoning_split, split_vllm_reasoning

    print("=== protocol parse of raw content ===")
    print("extract_final_raw", extract_final(content))
    code = _extract_code(content)
    print("extract_code_len", len(code), "starts_with_think", code.lstrip().startswith("<think>"))
    print("code_preview", repr(code[:400]))
    reasoning, protocol = split_vllm_reasoning(content)
    print("split_reasoning_preview", repr(reasoning[:300]))
    print("split_protocol", repr(protocol[:400]))
    print("extract_final_split", extract_final(protocol))
    print("extract_code_split_preview", repr(_extract_code(protocol)[:400]))

    install_vllm_reasoning_split()
    rlm = RLM(
        model=f"openai/{MODEL}",
        api_base=ENDPOINT,
        api_key="EMPTY",
        max_depth=1,
        max_iterations=3,
        capture_trajectory_content=True,
        temperature=0.0,
        max_tokens=4096,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    result = rlm.try_complete_result(
        'The document says "the capital of France is Paris". What is the capital of France?',
        "Irrelevant filler. " * 50 + "the capital of France is Paris. " + "more filler. " * 50,
    )
    dump = result.to_dict()
    (OUT / "rlm_tiny.json").write_text(json.dumps(dump, indent=2, default=str) + "\n")
    print("=== tiny RLM ===")
    print("succeeded", result.succeeded, "answer", getattr(result, "answer", None))
    print("error", getattr(result, "error_type", None), getattr(result, "error", None)[:200] if getattr(result, "error", None) else None)
    for event in dump.get("trajectory") or []:
        if event.get("kind") not in ("model_call_end", "repl_step", "final_answer", "rlm_error"):
            continue
        data = event.get("data") or {}
        preview = ""
        for key in ("response", "code", "output", "error", "answer"):
            if data.get(key):
                preview = str(data[key])[:300].replace("\n", "\\n")
                break
        print(f"  {event.get('kind')} { {k: data.get(k) for k in ('status', 'code_chars', 'response_chars') if k in data} } {preview!r}")


if __name__ == "__main__":
    dump_raw_completion()
    dump_rlm_steps()
