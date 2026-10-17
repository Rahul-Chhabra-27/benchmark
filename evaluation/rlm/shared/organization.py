# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Question-free organization and lossless, tokenizer-budgeted reader groups.

`organize` sees the document and a frozen configuration and nothing else -- no
question, no answers -- so its output can be built once per corpus and reused by
every query that shares it. Two orders:

- `source`: chunks in document order. The control.
- `bm25-graph`: every chunk is scored as a BM25 query against every other chunk;
  the resulting lexical graph is reduced to a maximum-weight spanning forest and
  read depth-first, so related chunks land next to each other.

Chunks never overlap: overlapping windows share their boundary text verbatim, and
that duplicated text would dominate chunk-to-chunk similarity.

`pack` then cuts an order into reader groups under a token budget counted with the
served model's own tokenizer, on the COMPLETE reader message -- instructions,
question and source labels included. Both arms share it, so they differ in chunk
order and nothing else.

Like retrieval.py this module imports no torch; `ModelTokens` loads a tokenizer
lazily, and only the ordered structured path constructs one.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .retrieval import ChunkIndex, Hit

# Frozen before the pilot. Bump `revision` for any change that can alter an order:
# it is part of the cache key and of every run's config.
CONFIG = dict(revision=1, chunk_chars=2000, overlap=0, neighbors=5, k1=1.5, b=0.75)
ORDERS = ("source", "bm25-graph")


def graph_order(rows, neighbors=5):
    """Order chunks from a directional score matrix. Returns (order, edges, forest).

    Self scores are dropped before each row is normalized by its maximum, so a
    chunk's match with itself never sets its scale. The two directions are then
    averaged. Each node keeps its `neighbors` strongest positive edges (union over
    nodes), Kruskal builds a maximum-weight spanning forest, and each component is
    walked depth-first from its earliest chunk, strongest edge first. Every tie is
    broken by source index, so the order is a pure function of `rows`.
    """
    n = len(rows)
    normalized = []
    for i, row in enumerate(rows):
        maximum = max((v for j, v in enumerate(row) if j != i), default=0)
        normalized.append([v / maximum if maximum > 0 and i != j else 0 for j, v in enumerate(row)])
    weights = [[(normalized[i][j] + normalized[j][i]) / 2 for j in range(n)] for i in range(n)]
    edges = set()
    for i in range(n):
        for j in sorted((j for j in range(n) if weights[i][j] > 0), key=lambda j: (-weights[i][j], j))[:neighbors]:
            edges.add((min(i, j), max(i, j)))
    edges = sorted(((i, j, weights[i][j]) for i, j in edges), key=lambda e: (-e[2], e[0], e[1]))
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    forest, adjacency = [], [[] for _ in range(n)]
    for i, j, w in edges:
        a, b = root(i), root(j)
        if a != b:
            parent[b] = a
            forest.append((i, j, w))
            adjacency[i].append((j, w))
            adjacency[j].append((i, w))
    order, seen = [], set()
    for start in range(n):
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            order.append(node)
            # Pushed weakest first so the strongest (lowest index on a tie) pops next.
            weakest_first = sorted(adjacency[node], key=lambda e: (-e[1], e[0]), reverse=True)
            stack.extend(j for j, _ in weakest_first if j not in seen)
    return order, edges, forest


def organize(document, order, cache_dir=None):
    """Accepts only the document and frozen configuration; never a question.

    Returns (artifact, cpu_seconds_this_call, cache_hit). The artifact's own
    `build_cpu_seconds` is the one-off preprocessing cost to amortise over queries.
    """
    if order not in ORDERS:
        raise ValueError(f"unknown order: {order}")
    started = time.process_time()
    digest = hashlib.sha256(document.encode()).hexdigest()
    config = dict(CONFIG, order=order)
    key = hashlib.sha256(json.dumps([digest, config], sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f"{key}.json" if cache_dir else None
    if path and path.exists():
        artifact = json.loads(path.read_text())
        if artifact["document_sha256"] != digest or artifact["config"] != config:
            raise ValueError("organization cache mismatch")
        return artifact, time.process_time() - started, True
    size = CONFIG["chunk_chars"]
    spans = [(s, min(s + size, len(document))) for s in range(0, len(document), size)]
    edges, forest = [], []
    indices = list(range(len(spans)))
    if order == "bm25-graph" and spans:
        index = ChunkIndex(document, chunk_chars=size, overlap=CONFIG["overlap"], k1=CONFIG["k1"], b=CONFIG["b"])
        if index.spans != spans:
            raise ValueError("BM25 index and organization disagree on chunk spans")
        rows = []
        for s, e in spans:
            row = [0.0] * len(spans)
            for hit in index.search(document[s:e], k=len(spans)):
                row[hit.start // size] = hit.score
            rows.append(row)
        indices, edges, forest = graph_order(rows, CONFIG["neighbors"])
    artifact = dict(
        config=config,
        document_sha256=digest,
        document_chars=len(document),
        spans=spans,
        order=indices,
        edges=edges,
        forest=forest,
        build_cpu_seconds=time.process_time() - started,
    )
    text = json.dumps(artifact)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(text)
        temporary.replace(path)
    # Round-trip so a fresh build and a cache hit hand back identical types.
    return json.loads(text), time.process_time() - started, False


def ordered_hits(document, artifact):
    """The artifact's chunks as Hits, in its order. Rank is the position in that order."""
    if artifact["document_sha256"] != hashlib.sha256(document.encode()).hexdigest():
        raise ValueError("organization artifact belongs to a different document")
    spans = artifact["spans"]
    if sorted(artifact["order"]) != list(range(len(spans))):
        raise ValueError("organization order is not a permutation of its chunks")
    return [
        Hit(rank, spans[i][0], spans[i][1], 0.0, document[spans[i][0] : spans[i][1]])
        for rank, i in enumerate(artifact["order"], start=1)
    ]


@dataclass(frozen=True)
class OrderedPayload:
    """Chunks read together, in the given order, each labelled with its own offsets.

    Distant chunks are never described as one interval: every chunk carries its own
    `[source chars s:e]` label, even when two happen to be adjacent in the source.
    """

    hits: tuple[Hit, ...]

    def render(self, document):
        for h in self.hits:
            if not 0 <= h.start < h.end <= len(document) or h.text != document[h.start : h.end]:
                raise ValueError("ordered payload has invalid source offsets/text")
        return "\n\n".join(f"[source chars {h.start}:{h.end}]\n{h.text}" for h in self.hits)


def reader_messages(system, question, payload, document):
    """Exactly the messages llm_query sends for an ordered payload on a plain chat client."""
    return [dict(role="system", content=system), dict(role="user", content=f"{question}\n\n{payload.render(document)}")]


class ModelTokens:
    """Counts a chat request the way the server will: the model's own chat template."""

    def __init__(self, model):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    def count_messages(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        # transformers 5 may return a BatchEncoding rather than a list of ids.
        return len(ids["input_ids"] if hasattr(ids, "keys") else ids)


def pack(document, hits, question, system, counter, budget):
    """Sequentially pack full messages; bisect oversized chunks without text loss."""
    if budget <= 0:
        raise ValueError("group input budget must be positive")
    groups, current = [], []

    def size(items):
        return counter.count_messages(reader_messages(system, question, OrderedPayload(tuple(items)), document))

    pending = list(reversed(hits))
    while pending:
        hit = pending.pop()
        if size([*current, hit]) <= budget:
            current.append(hit)
        elif current:
            groups.append(OrderedPayload(tuple(current)))
            current = []
            pending.append(hit)
        else:
            if hit.end - hit.start <= 1:
                raise ValueError("reader instructions/question/labels leave no room for source text")
            middle = (hit.start + hit.end) // 2
            for s, e in ((middle, hit.end), (hit.start, middle)):
                pending.append(Hit(hit.rank, s, e, hit.score, document[s:e]))
    if current:
        groups.append(OrderedPayload(tuple(current)))
    return groups
