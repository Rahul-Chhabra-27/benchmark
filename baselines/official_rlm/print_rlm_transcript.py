# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pretty-print a recursive-llm trajectory JSON for humans."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def clip(text: Any, limit: int = 400) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\r\n", "\n").strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)].rstrip() + f"… [{len(s)} chars]"


def unwrap_fence(code: str) -> str:
    s = (code or "").strip()
    match = re.match(r"^```(?:python)?\s*\n(.*)\n```$", s, re.DOTALL)
    return match.group(1) if match else s


def load_transcript(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "trajectory" in data:
        return data
    if isinstance(data, dict) and "schema_version" in data:
        return {
            "question": ((data.get("trajectory") or [{}])[0].get("data") or {}).get("query"),
            "answers": None,
            "pred": data.get("answer"),
            "end_reason": data.get("termination_reason"),
            "stats": data.get("stats") or {},
            "trajectory": data.get("trajectory") or [],
        }
    raise SystemExit(f"no trajectory in {path}")


def format_transcript(data: dict[str, Any], limit: int) -> str:
    lines: list[str] = []
    question = data.get("question") or ""
    answers = data.get("answers")
    pred = data.get("pred")
    end = data.get("end_reason") or "unknown"
    stats = data.get("stats") or {}
    traj = data.get("trajectory") or []

    ctx_chars = 0
    for event in traj:
        if event.get("kind") in ("run_start", "rlm_start"):
            ctx = (event.get("data") or {}).get("context")
            if isinstance(ctx, str):
                ctx_chars = len(ctx)
            if not question:
                question = (event.get("data") or {}).get("query") or question
            break

    q_short = clip(re.sub(r"\s+", " ", question), 240)
    m = re.search(r"query:\s*(.+?)(?:\n|$)", question, re.IGNORECASE | re.DOTALL)
    if m:
        q_short = clip(re.sub(r"\s+", " ", m.group(1)), 240)
    lines.append(f"Question: {q_short}")
    if answers is not None:
        lines.append(f"Gold: {clip(answers, 200)}")
    lines.append(f"Pred: {clip(pred, 200) if pred else '(none)'}")
    lines.append(
        f"Outcome: {end}  |  iterations={stats.get('total_iterations') or stats.get('iterations')}  "
        f"tokens={stats.get('total_tokens')}  context={ctx_chars:,} chars"
    )
    lines.append("")

    step = 0
    pending_model: str | None = None
    for event in traj:
        kind = event.get("kind")
        d = event.get("data") or {}
        if kind == "model_call_end" and d.get("response") is not None:
            pending_model = unwrap_fence(d["response"])
            continue
        if kind == "repl_step":
            step += 1
            code = unwrap_fence(d.get("code") or pending_model or "")
            pending_model = None
            lines.append(f"Step {step}  REPL {d.get('status') or '?'}")
            lines.append(clip(code, limit))
            lines.append("")
            lines.append("→ " + clip(d.get("output"), limit))
            if d.get("error"):
                lines.append("error: " + clip(d.get("error"), limit))
            lines.append("")
            continue
        if kind == "final_answer":
            lines.append("FINAL  " + clip(d.get("answer"), limit))
            lines.append("")
            continue
        if kind in ("rlm_error", "run_error"):
            msg = f"Error: {d.get('error_type') or kind}: {clip(d.get('error'), 240)}"
            if msg not in lines:
                lines.append(msg)
                lines.append("")
            continue
    if pending_model:
        step += 1
        lines.append(f"Step {step}  model (no REPL yet)")
        lines.append(clip(pending_model, limit))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="trajectory JSON from --save-trajectory or rlm_tiny.json")
    parser.add_argument("--limit", type=int, default=400, help="max chars per code/output block")
    args = parser.parse_args()
    print(format_transcript(load_transcript(args.path), args.limit), end="")


if __name__ == "__main__":
    main()
