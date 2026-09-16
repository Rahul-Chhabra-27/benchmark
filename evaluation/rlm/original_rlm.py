# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paper-style Recursive Language Model (the recursive-llm loop).

The long document is not in the root prompt. It lives as `context` in a
persistent Python REPL. Each root turn is exactly one of:

  1. executable Python (no FINAL inside it)
  2. a standalone FINAL("answer") or FINAL_VAR(variable_name)

`recursive_llm(sub_query, sub_context)` is a plain LM call at the default
depth boundary (max_depth=1). Nested child RLMs exist only when max_depth > 1.

This is the original protocol from recursive-llm/: no search(), no SUB_CALL,
no memory budget, no answer-format injection, no extra REPL names.
"""

from __future__ import annotations

import ast
import contextlib
import io
import re
import signal
import threading
import time
from typing import Any, Optional

from .client import LLMClient
from .rlm import RLMResult

_RESULT_NAME = "_rlm_last_result"

LEAF_SYSTEM_PROMPT = (
    "Answer the subproblem using only the supplied context. "
    "Return the answer directly and do not emit REPL code or FINAL directives."
)

EMPTY_NUDGE = (
    "Your response was empty. Return one short executable Python step "
    "or a standalone final directive."
)


def build_system_prompt(context_size: int, depth: int = 0, max_depth: int = 1) -> str:
    """Build the minimal paper-style prompt for one RLM loop."""
    if max_depth == 0:
        recursive_api = "- LM subcalls are disabled for this run."
        depth_note = "No subcalls are available."
    else:
        recursive_api = (
            "- recursive_llm(sub_query, sub_context) -> str " "(recursively process sub-context)"
        )
        if depth + 1 >= max_depth:
            depth_note = "recursive_llm makes one plain LM call at this boundary."
        else:
            depth_note = "recursive_llm creates a child RLM."

    return f"""You are a Recursive Language Model. You interact with context through a Python REPL environment.

The context is stored in variable `context` (not in this prompt). Size: {context_size:,} characters.
IMPORTANT: You cannot see the context directly. You MUST write Python code to search and explore it.

Available in environment:
- context: str (the document to analyze)
- query: str (the question)
{recursive_api}
- re: already imported regex module (use re.findall, re.search, etc.)

Each response must be exactly one of these two forms:
1. Executable Python code for one REPL step. Do not wrap it in Markdown fences or include FINAL in it.
2. A standalone FINAL("answer") or FINAL_VAR(variable_name) directive after the REPL has shown you
   enough evidence. FINAL is a protocol directive, not a Python function: never call, print, assign,
   or place it inside Python code or a conditional block.

Imports are restricted. Use the objects already available in the environment. In particular,
use `re` directly without writing `import re`.

The last expression or print() output from a Python step will be shown to you exactly once. After
seeing that output, use a new response for either the next Python step or the standalone final.

Examples:
- print(context[:500])
- matches = re.findall(r'keyword.*', context); print(matches[:5])
- idx = context.find('search term'); print(context[idx:idx+200])

CRITICAL: Do NOT guess or make up answers. You MUST search the context first to find the actual information.
Only use a standalone FINAL("answer") after you have found concrete evidence in the context.

Depth: {depth}. max_depth: {max_depth}. {depth_note}"""


def extract_final(response: str) -> Optional[str]:
    match = re.fullmatch(r"\s*FINAL\s*\((.*)\)\s*", response, re.DOTALL)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(1).strip())
    except (SyntaxError, ValueError):
        return None
    if isinstance(value, str):
        return value.strip()
    return None


def extract_final_var_name(response: str) -> Optional[str]:
    match = re.fullmatch(r"\s*FINAL_VAR\s*\(\s*(\w+)\s*\)\s*", response)
    return match.group(1) if match else None


def _extract_code(text: str) -> str:
    for marker in ("```python", "```"):
        if marker in text:
            start = text.find(marker) + len(marker)
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
    return text


class _StepTimeout(Exception):
    """Raised when one REPL step outruns repl_timeout."""


def _raise_step_timeout(signum, frame):  # pragma: no cover - signal handler
    raise _StepTimeout()


def _execute(code: str, env: dict[str, Any], max_output_chars: int, timeout: Optional[float]) -> str:
    code = _extract_code(code)
    if not code.strip():
        return "No code to execute"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"Error: Compilation error: {exc.msg}"

    env.pop(_RESULT_NAME, None)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        final_expression = tree.body[-1]
        tree.body[-1] = ast.copy_location(
            ast.Assign(
                targets=[ast.Name(id=_RESULT_NAME, ctx=ast.Store())],
                value=final_expression.value,
            ),
            final_expression,
        )
        ast.fix_missing_locations(tree)

    # The original runs each step in a worker process it can kill. In-process,
    # SIGALRM is the closest equivalent: it interrupts a runaway Python loop, but
    # a C-level regex only raises once it returns -- a known CPython limitation.
    use_alarm = (
        bool(timeout) and hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    )
    buf = io.StringIO()
    old_handler = None
    try:
        with contextlib.redirect_stdout(buf):
            if use_alarm:
                old_handler = signal.signal(signal.SIGALRM, _raise_step_timeout)
                signal.setitimer(signal.ITIMER_REAL, timeout)
            exec(compile(tree, "<repl>", "exec"), env, env)
    except _StepTimeout:
        return f"Error: Execution timed out after {timeout:g} seconds"
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    output = buf.getvalue()
    result = env.pop(_RESULT_NAME, None)
    if result is not None:
        output += f"{result}\n"
    text = output.strip() or "Code executed successfully (no output)"
    if len(text) > max_output_chars:
        text = text[:max_output_chars] + "\n... [truncated]"
    return text


class OriginalRLM:
    """Root REPL loop matching recursive-llm/src/rlm/core.py."""

    def __init__(
        self,
        root_client: LLMClient,
        sub_client: LLMClient | None = None,
        max_steps: int = 30,
        max_depth: int = 1,
        repl_timeout: Optional[float] = 5.0,
        max_output_chars: int = 2000,
        run_timeout: Optional[float] = None,
        debug: bool = False,
        _current_depth: int = 0,
    ):
        if max_depth < 0:
            raise ValueError("max_depth must be zero or greater")
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        self.root = root_client
        self.sub = sub_client or root_client
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.repl_timeout = repl_timeout
        self.max_output_chars = max_output_chars
        self.run_timeout = run_timeout
        self.debug = debug
        self._current_depth = _current_depth

    def _chat_kwargs(self, client: LLMClient) -> dict:
        """The protocol reads the WHOLE reply, so the served model must not think.

        A reasoning block ahead of the reply is executed as Python, and it also
        stops FINAL(...) from matching, since the directive must stand alone.
        """
        extra = dict(getattr(client, "extra_body", None) or {})
        template = dict(extra.get("chat_template_kwargs") or {})
        template["enable_thinking"] = False
        extra["chat_template_kwargs"] = template
        return {"extra_body": extra}

    def run(self, context: str, task: str, answer_format: str | None = None) -> RLMResult:
        # LOFT's cue is literally "Final Answer: ", which collides with the FINAL
        # directive: asked for it, the root emits the cue line instead of FINAL.
        # Its scorer already reads the uncued str(FINAL(x)) this loop produces.
        del answer_format
        started = time.time()
        metrics: dict = {"sub_calls": 0, "depth": self._current_depth}
        transcript: list = []

        def timed_out() -> bool:
            return bool(self.run_timeout) and (time.time() - started) >= self.run_timeout

        env = self._repl_env(task, context, metrics)
        messages: list[dict] = [
            {
                "role": "system",
                "content": build_system_prompt(len(context), self._current_depth, self.max_depth),
            },
            {"role": "user", "content": task},
        ]

        for step in range(1, self.max_steps + 1):
            if timed_out():
                return RLMResult(
                    answer=None,
                    steps=step - 1,
                    finished=False,
                    transcript=transcript,
                    end_reason="run_timeout",
                    metrics=metrics,
                )
            response = (self.root.chat(messages, **self._chat_kwargs(self.root)) or "").strip()
            transcript.append({"step": step, "role": "assistant", "content": response})
            if self.debug:
                print(f"\n===== ORIGINAL ROOT({step}) =====\n{response}\n")

            if not response:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": EMPTY_NUDGE})
                transcript.append({"step": step, "role": "observation", "content": EMPTY_NUDGE})
                continue

            direct = extract_final(response)
            if direct is not None:
                return RLMResult(
                    answer=direct,
                    steps=step,
                    finished=True,
                    transcript=transcript,
                    end_reason="FINAL",
                    metrics=metrics,
                )

            var_name = extract_final_var_name(response)
            if var_name is not None:
                if var_name in env:
                    return RLMResult(
                        answer=str(env[var_name]),
                        steps=step,
                        finished=True,
                        transcript=transcript,
                        end_reason="FINAL_VAR",
                        metrics=metrics,
                    )
                obs = f"Error: variable {var_name!r} was not found"
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": obs})
                transcript.append({"step": step, "role": "observation", "content": obs})
                continue

            obs = _execute(response, env, self.max_output_chars, self.repl_timeout)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": obs})
            transcript.append({"step": step, "role": "observation", "content": obs})
            if self.debug:
                print(f"----- REPL -----\n{obs}\n")

        return RLMResult(
            answer=None,
            steps=self.max_steps,
            finished=False,
            transcript=transcript,
            end_reason="max_iterations",
            metrics=metrics,
        )

    def _repl_env(self, query: str, context: str, metrics: dict) -> dict[str, Any]:
        env: dict[str, Any] = {
            "context": context,
            "query": query,
            "re": re,
        }
        if self.max_depth > 0:
            env["recursive_llm"] = self._make_recursive_llm(metrics)
        return env

    def _make_recursive_llm(self, metrics: dict):
        def recursive_llm(sub_query: str, sub_context: str = "") -> str:
            metrics["sub_calls"] = metrics.get("sub_calls", 0) + 1
            if self._current_depth + 1 >= self.max_depth:
                return self._leaf(sub_query, sub_context)
            child = OriginalRLM(
                root_client=self.sub,
                sub_client=self.sub,
                max_steps=self.max_steps,
                max_depth=self.max_depth,
                repl_timeout=self.repl_timeout,
                max_output_chars=self.max_output_chars,
                run_timeout=self.run_timeout,
                debug=self.debug,
                _current_depth=self._current_depth + 1,
            )
            result = child.run(sub_context, sub_query)
            metrics["sub_calls"] = metrics.get("sub_calls", 0) + result.metrics.get("sub_calls", 0)
            return result.answer if result.answer is not None else ""

        return recursive_llm

    def _leaf(self, sub_query: str, sub_context: str) -> str:
        user = sub_query
        if sub_context:
            user = f"Task:\n{sub_query}\n\nContext:\n{sub_context}"
        messages = [
            {"role": "system", "content": LEAF_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return (self.sub.chat(messages, **self._chat_kwargs(self.sub)) or "").strip()
