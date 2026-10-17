# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tuning feature: our prompt tuning and guards, from the original RLM (evaluation/rlm/rlm.py).

The base RLM is the RLM idea with a short prompt. Everything we tuned on top of it over
the LOFT campaigns lives here, switched on with `tuning: true` in experiments.yaml, so
the same experiment can run with and without it and the difference is measured, not
assumed. The text and the grounding helpers are imported from rlm.py, not rewritten:

    prompt   the "never simulate output" rule; search coaching (distinctive terms, not
             the whole question); "zero matches means the pattern is wrong"; FINAL on a
             computed value; FINAL_NONE for a genuine absence; a worked example session
             (the one rlm.py shows for this ordering)
    guards   FINAL_NONE, which is refused until the run has actually read the document
             (1 read; 3 with ordered chunks); a FINAL on a literal that never appeared in
             any output is rejected as a guess (3 strikes end the question); 3 cells in a
             row printing nothing earn a nudge; 3 cells in a row raising end the question
"""

from __future__ import annotations

import ast
from typing import Any

from ..rlm import (
    CHUNK_READS_BEFORE_ABSTAIN,
    CHUNK_WORKED_SESSIONS,
    EXAMPLE_LLM_QUERY_SPLIT,
    PREMATURE_ABSTAIN_NUDGE,
    PREMATURE_ABSTAIN_NUDGE_CHUNKS,
    SEARCH_TERMS_CHUNKS,
    SEARCH_TERMS_FIND,
    UNGROUNDED_FINAL_CHUNKS,
    WORKED_SESSION_FIND,
    _mentions,
    _static_string,
    composed_answer_grounded,
)
from .rlm import Feature, Result

FINAL_NONE_TOOL = "  FINAL_NONE(reason)            end WITHOUT an answer, only after genuinely searching.\n"

RULES = """
Rules:
- CRITICAL: you CANNOT see the result of your code until the next message. NEVER write,
  guess, or simulate the REPL output yourself. NEVER state an answer you have not
  literally seen in a real REPL output.
{search_terms}
- If a search or extraction returns ZERO matches or fewer than the task implies, that is
  USUALLY a signal your pattern is wrong -- not that the answer is empty. Print a sample
  of the text around a likely keyword to see the actual format, then retry. When the
  task states how many items exist, verify you found that many before answering.
- Finish with FINAL on a VARIABLE or expression computed by your code. NEVER call FINAL
  with a literal guessed value -- FINAL("42") with a made-up constant will be rejected.
- If, after genuinely searching, the document does NOT contain the answer, end with
  FINAL_NONE("short reason") instead of guessing. An abstention backed by no llm_query
  read will be rejected: read some of the document before claiming the answer is absent.

{worked_session}
"""

UNGROUNDED_FINAL = (
    "REJECTED: FINAL({val!r}) is a literal constant that never appeared in any REPL output -- it looks "
    "like a guess. Find the real answer first (search `context`, read the slice with llm_query), "
    "print it, then call FINAL on the variable holding it. If you have genuinely searched and the "
    'document does not contain the answer, say so with FINAL_NONE("reason") instead.'
)
UNPRODUCTIVE_NUDGE = (
    "NOTE: your last {n} code cells printed nothing, so you learned nothing from them. Print what "
    "you find, or read a promising part of the document with llm_query and print the answer."
)

MAX_REJECTIONS = 2  # FINAL rejections forgiven before an ungrounded guess ends the question
MAX_ABSTAIN_NUDGES = 2  # premature FINAL_NONEs refused before one is accepted
LOOP_STEPS = 3  # cells in a row printing nothing (nudge) / raising (end)


def final_literals(code: str) -> list[str]:
    """String literals passed to FINAL in this cell: a guess, unless they appeared in output."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "FINAL" and node.args:
            value = _static_string(node.args[0])
            if value is not None:
                found.append(value)
    return found


class Tuning(Feature):
    """Our tuned prompt and guards, on top of the base RLM."""

    def __init__(self, ordering: str = "none"):
        self.ordering = ordering
        self.ordered = ordering != "none"

    def prepare(self, document: str, question: str) -> None:
        self.question = question
        self.rejections = self.abstain_nudges = 0
        self.empty_steps = self.error_steps = 0
        self.pending = ""  # a refusal to show the root after the current cell
        self.abstain_reason = ""

    # ---- prompt -------------------------------------------------------------
    def tool_help(self) -> str:
        return FINAL_NONE_TOOL

    def notes(self) -> str:
        if self.ordered:
            search, session = SEARCH_TERMS_CHUNKS, CHUNK_WORKED_SESSIONS[self.ordering]
        else:
            search, session = SEARCH_TERMS_FIND, WORKED_SESSION_FIND.format(example_llm_query=EXAMPLE_LLM_QUERY_SPLIT)
        return RULES.format(search_terms=search, worked_session=session)

    # ---- guards -------------------------------------------------------------
    def tools(self, result: Result) -> dict:
        def FINAL_NONE(reason: Any = "") -> str:  # noqa: N802 - the name the prompt uses
            reads = sum(1 for c in result.calls if not c.refused)
            needed = CHUNK_READS_BEFORE_ABSTAIN if self.ordered else 1
            if reads < needed and self.abstain_nudges < MAX_ABSTAIN_NUDGES:
                self.abstain_nudges += 1
                self.pending = PREMATURE_ABSTAIN_NUDGE_CHUNKS if self.ordered else PREMATURE_ABSTAIN_NUDGE
                return "FINAL_NONE refused (see below)."
            self.abstain_reason = str(reason)
            result.answer, result.done, result.ending = None, True, "abstained"
            return "abstained"

        return {"FINAL_NONE": FINAL_NONE}

    def _seen(self, result: Result) -> str:
        """Everything the run has observed: what its code printed and the sub-model's answers.

        Not the notes features add (a rejection quotes the guess it rejects).
        """
        outputs = [turn["printed"] for turn in result.transcript]
        return "\n".join(outputs + [a for a in result.read_answers if a])

    def check_final(self, answer: str, code: str, result: Result) -> str:
        if answer not in final_literals(code):
            return ""  # computed by the code, so grounded by construction
        seen = self._seen(result)
        grounded = answer in seen or _mentions(self.question, answer)
        if not grounded and self.ordered:  # an answer assembled from several reads
            grounded = composed_answer_grounded(answer, seen, self.question)
        if grounded:
            return ""
        self.rejections += 1
        if self.rejections > MAX_REJECTIONS:  # the root keeps guessing: end without an answer
            result.answer, result.done, result.ending = None, True, "ungrounded_final"
            return ""
        return (UNGROUNDED_FINAL_CHUNKS if self.ordered else UNGROUNDED_FINAL).format(val=answer)

    def after_turn(self, output: str, result: Result) -> str:
        note, self.pending = self.pending, ""
        self.error_steps = self.error_steps + 1 if "[ERROR]" in output else 0
        if self.error_steps >= LOOP_STEPS and not result.done:
            result.done, result.ending = True, "error_loop"
        self.empty_steps = self.empty_steps + 1 if output.strip() == "[no output]" else 0
        if self.empty_steps >= LOOP_STEPS:
            self.empty_steps = 0
            note = (note + "\n" if note else "") + UNPRODUCTIVE_NUDGE.format(n=LOOP_STEPS)
        return note

    def summary(self, result: Result) -> dict:
        return {
            "rejected_finals": self.rejections,
            "abstain_nudges": self.abstain_nudges,
            **({"abstain_reason": self.abstain_reason} if self.abstain_reason else {}),
        }
