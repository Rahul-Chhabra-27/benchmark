# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ordering feature: hand the root the document as an ordered list of chunks.

    none        no feature; the root searches `context` itself
    bm25-query  chunks ranked by BM25 against the question (query-aware)
    bm25-graph  chunks grouped by a similarity graph built once per document,
                before any question is seen (query-agnostic; cached on disk)

Both orders use the same grid: 2000-character chunks, no overlap. Switched on with
`ordering:` in experiments.yaml; it adds the `chunks` tool and a note on the order.
"""

from __future__ import annotations

from typing import Optional

from ..organization import CONFIG as GRID
from ..organization import ordered_hits, organize
from ..retrieval import Hit, get_index
from .rlm import Feature, Result

CHUNKS_TOOL = """\
  chunks                        the document cut into {n_chunks} pieces of ~{chunk_chars} characters,
                                in the order described below. Each has .text, .start, .end;
                                llm_query(question, chunks[i:j]) reads a list of chunks.
"""

ORDERING_NOTES = {
    "bm25-query": (
        "\n`chunks` is sorted by BM25 keyword relevance to the question: chunks[0] matches best."
        "\nThe answer is usually near the top.\n"
    ),
    "bm25-graph": (
        "\n`chunks` is grouped by topic, before the question was known: chunks sharing names and"
        "\nrare words sit next to each other. Find a chunk mentioning the question's distinctive"
        "\nterms, then read it together with its neighbours.\n"
    ),
}


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


class Ordering(Feature):
    """Adds `chunks`, in the chosen order, to every question."""

    def __init__(self, order: str, organization_dir: Optional[str] = None):
        if order not in ORDERING_NOTES:
            raise ValueError(f"unknown ordering {order!r}; choose from {sorted(ORDERING_NOTES)}")
        self.order = order
        self.organization_dir = organization_dir
        self.chunks: list[Hit] = []

    def prepare(self, document: str, question: str) -> None:
        self.chunks = build_chunks(document, question, self.order, self.organization_dir)

    def tools(self, result: Result) -> dict:
        return {"chunks": list(self.chunks)}

    def tool_help(self) -> str:
        # Before any question (e.g. `check`), say how many pieces a 128k document makes.
        return CHUNKS_TOOL.format(n_chunks=len(self.chunks) or 250, chunk_chars=chunk_chars())

    def notes(self) -> str:
        return ORDERING_NOTES[self.order]
