# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a picker may look at before a run: the question and the document, never a model.

The one signal is how SPREAD OUT the question's lexical evidence is. Every chunk of
the bm25-graph grid is scored by BM25 against the bare question; a question whose
strong matches sit in two chunks needs a narrow read, one whose matches are scattered
over thirty needs a wide one. On LOFT-128k test rows the signal tracks the task the
way one would hope: nq and hotpotqa questions concentrate in 1-3 chunks, quest and
musique spread over far more.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..organization import CONFIG as ORGANIZATION_CONFIG
from ..retrieval import get_index
from ..rlm import CHARS_PER_TOKEN_ESTIMATE
from ..run_benchmark import chunk_order_query


@dataclass(frozen=True)
class QuestionFeatures:
    """BM25 evidence profile of one question over its document."""

    query: str
    n_chunks: int
    # Average tokens in one grid chunk, so a chunk count converts to a read size.
    chunk_tokens: float
    # Positive BM25 scores, best first. Kept whole so calibration can sweep the
    # threshold without re-scoring the document.
    scores: tuple[float, ...]

    def evidence_chunks(self, ratio: float) -> int:
        """Chunks scoring at least `ratio` x the best chunk.

        A question sharing no term with the document carries no evidence of where its
        answer is, so it counts as spread over everything: the widest read.
        """
        if not 0 < ratio <= 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        if not self.scores:
            return self.n_chunks
        floor = ratio * self.scores[0]
        return sum(1 for score in self.scores if score >= floor)

    def evidence_tokens(self, ratio: float) -> float:
        return self.evidence_chunks(ratio) * self.chunk_tokens


def question_features(
    question: str,
    document: str,
    dataset: str = "loft",
    chars_per_token: float = CHARS_PER_TOKEN_ESTIMATE,
) -> QuestionFeatures:
    """Score every grid chunk against the bare question.

    The grid is the organization's (2000 chars, no overlap): the same chunks a
    bm25-graph read is made of. The query is what the bm25-query arm ranks by, with
    LOFT's citation boilerplate stripped -- left in, "documents" and "answer" match
    every chunk and every question looks maximally spread.
    """
    index = get_index(document, ORGANIZATION_CONFIG["chunk_chars"], ORGANIZATION_CONFIG["overlap"])
    query = chunk_order_query(question, dataset)
    scores = tuple(hit.score for hit in index.rank_all(query) if hit.score > 0)
    return QuestionFeatures(
        query=query,
        n_chunks=index.n_chunks,
        chunk_tokens=len(document) / max(index.n_chunks, 1) / chars_per_token,
        scores=scores,
    )
