# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static-answer parsing and grounding checks."""

from __future__ import annotations

import ast
import re
from typing import Optional


def _static_string(node: ast.AST) -> Optional[str]:
    """The value of `node` if it is knowable without running any code, else None.

    This is what separates a GUESS from a computed answer for the grounding guard.
    Matching only `ast.Constant` let three spellings of the same guess through --
    FINAL(f"42"), FINAL("4"+"2") and, once folded, any concatenation of literals --
    so the guard rejected the honest spelling and waved past the evasive ones.
    A node mentioning any variable is not static, and is grounded by construction:
    the value came from code that actually ran.
    """
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.JoinedStr):  # f-string: static only if nothing is interpolated
        parts = [_static_string(v) for v in node.values]
        return "".join(parts) if all(p is not None for p in parts) else None  # type: ignore[arg-type]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _static_string(node.left), _static_string(node.right)
        return left + right if left is not None and right is not None else None
    return None


def _mentions(haystack: str, needle: str) -> bool:
    """Word-boundary containment, for the `answer appears in the task text` check.

    Plain `needle in haystack` whitelisted almost every short answer: any question
    containing "12" anywhere licensed the ungrounded answer 12, and every
    classification/multiple-choice task enumerates its labels in the prompt, so that
    whole class of tasks had no grounding at all.
    """
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


# What a composed answer is made of (composed_answer_grounded): runs of letters and
# digits, lower-cased, with digit-group commas dropped so "2,548" and "2548" are one word.
_ANSWER_WORD_RE = re.compile(r"[a-z0-9]+")
_DIGIT_GROUP_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
# Words that carry no answer, and so never ground one.
COMPOSED_ANSWER_FILLER = frozenset(
    "a an and are as at be but by for from in into is it its of on or that the their there "
    "these they this those to was were which with".split()
)
# A literal that adds any of these is an abstention in an answer's clothes ("are none",
# "it is not mentioned"): it must appear word for word, or be said with FINAL_NONE.
COMPOSED_ANSWER_NEGATIONS = frozenset(
    "no not none nothing unknown never neither nor cannot aren couldn didn doesn don hasn "
    "haven isn wasn weren wouldn".split()
)


def _answer_words(text: str) -> set[str]:
    return set(_ANSWER_WORD_RE.findall(_DIGIT_GROUP_COMMA_RE.sub("", text.lower())))


def composed_answer_grounded(val: str, seen_output: str, task_text: str) -> bool:
    """Was a FINAL literal put together from what the run observed? (ordered-chunk arms)

    The word-for-word test rejects an answer assembled from several reads -- four values
    read on four calls and joined with commas, or RULER's answer template filled in with
    the value a reader returned -- although every part of it was printed: on the RULER
    pilot that discarded 9 of 78 answers, most of them right. So every content word the
    literal adds to the task text (question and answer template) must have appeared in
    REPL output. It must add at least one, and no negation. A value no read returned is
    still a word no output contains, so a guess is still rejected.
    """
    added = _answer_words(val) - _answer_words(task_text)
    if added & COMPOSED_ANSWER_NEGATIONS:
        return False
    content = added - COMPOSED_ANSWER_FILLER
    return bool(content) and content <= _answer_words(seen_output)


# --------------------------------------------------------------------------- #
# Retrieved-span arithmetic
# --------------------------------------------------------------------------- #
