# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The `ordering` option: hand the root the document as an ordered list of chunks.

    none        no `chunks`; the root searches `context` itself
    bm25-query  chunks ranked by BM25 against the question (query-aware)
    bm25-graph  chunks grouped by a similarity graph built once per document,
                before any question is seen (query-agnostic; cached on disk)

Both orders use the same grid: 2000-character chunks, no overlap.
"""

from __future__ import annotations

from typing import Optional

from ..organization import CONFIG as GRID
from ..organization import ordered_hits, organize
from ..retrieval import Hit, get_index


def query_text(question: str) -> str:
    """The question without LOFT's citation boilerplate (the text after a `query:` line).

    Left in, words like "documents" and "answer" match every chunk equally.
    """
    lines = question.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].lstrip()
        if stripped[:6].lower() == "query:":
            return "\n".join([stripped[6:], *lines[index + 1 :]]).strip()
    return question.strip()


def build_chunks(document: str, question: str, ordering: str, organization_dir: Optional[str]) -> list[Hit]:
    """Every chunk of the document, in the given order. Empty for `none`."""
    if ordering == "none":
        return []
    if ordering == "bm25-graph":
        artifact, _, _ = organize(document, "bm25-graph", organization_dir)
        return ordered_hits(document, artifact)
    if ordering == "bm25-query":
        index = get_index(document, GRID["chunk_chars"], GRID["overlap"])
        return index.rank_all(query_text(question))
    raise ValueError(f"unknown ordering {ordering!r}")


def chunk_chars() -> int:
    return int(GRID["chunk_chars"])
