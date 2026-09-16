# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retrieved-span validation, grouping, and clipping helpers."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from ..retrieval import Hit, tokenize
from .prompts import STRUCTURED_MAP_GRANULARITIES

HIT_SEPARATOR = "\n...\n"


def _merge_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and coalesce overlapping or touching spans.

    Retrieval windows overlap by construction (2000-char windows on a 1600-char
    stride), so two adjacent hits share 400 characters. Joining them unmerged
    would repeat that text in the payload AND double-count it in coverage.
    """
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _cluster_hits(hits: Sequence[Hit]) -> list[list[Hit]]:
    """Group hits into runs that touch or overlap, in document order.

    Retrieval windows overlap by construction, and a document that straddles a
    window boundary is deliberately selected as an ADJACENT PAIR. Reading those as
    two separate passages re-reads the shared 400 characters and, worse, shows
    each call half a document -- so the half without the fact can honestly answer
    "nothing relevant". The agentic path never had this problem: it hands a whole
    hit list to one llm_query, where _merge_spans coalesces them.

    Returns lists rather than merged spans so each cluster can go to llm_query as
    hits, which is what keeps the payload attributable to real document offsets.
    """
    clusters: list[list[Hit]] = []
    for hit in sorted(hits, key=lambda h: (h.start, h.end)):
        if clusters and hit.start <= max(h.end for h in clusters[-1]):
            clusters[-1].append(hit)
        else:
            clusters.append([hit])
    return clusters


def _window_of(hit: Hit, chunk_chars: int, overlap: int) -> int:
    """Which grid window this hit is, derived exactly as the grid was built.

    `window_spans` strides by `chunk_chars - overlap` from 0, so the window id is the
    start offset divided by the stride. Recomputed rather than carried on the Hit,
    which has no room for it and is shared with the BM25 path.
    """
    stride = max(chunk_chars - overlap, 1)
    return hit.start // stride


def _map_groups(hits: Sequence[Hit], granularity: str) -> list[list[Hit]]:
    """The hits the map step reads, grouped one `llm_query` per group.

    Three granularities, so "how many windows go into one call" stops being welded
    to the pipeline:

      passage  contiguous runs, one call each (revision 2) -- see _cluster_hits
      window   one call per provided window (revision 1's granularity)
      all      ONE call over every window, which is the payload the AGENTIC root
               assembles: it passes the whole hit list to llm_query and _merge_spans
               folds the overlap. 45 of 50 nq examples and 40 of 49 hotpotqa ones of
               arm B did exactly that, so B is a single combined read rather than an
               elaborate search -- and this value is what removes that difference.

    Ordering is document order at EVERY granularity. Revision 1 emitted findings in
    RANK order, but re-introducing that here would put a third variable inside a
    two-way ablation, so `window` is revision 1's granularity on revision 2's
    ordering rather than a byte-exact revision-1 rerun.
    """
    if granularity == "passage":
        return _cluster_hits(hits)
    ordered = sorted(hits, key=lambda h: (h.start, h.end))
    if granularity == "window":
        return [[hit] for hit in ordered]
    if granularity == "all":
        return [ordered] if ordered else []
    raise ValueError(f"unknown map granularity {granularity!r}; choose from {list(STRUCTURED_MAP_GRANULARITIES)}")


def _truncate_hits(hits: Sequence[Hit], max_chars: int, document: str) -> list[Hit]:
    """The first `max_chars` characters of a cluster, still addressed by offsets.

    The exposure axis asks how much of a passage the map step should SEE, which is a
    different question from how wide the retrieval window is -- and the two are one
    knob today. Cutting here rather than inside `llm_query` keeps `Hit.text ==
    document[start:end]`, so a shortened payload still records the coverage it
    actually got instead of claiming the whole window.

    Spans are merged first: cluster members overlap by construction, so charging the
    budget per-hit would spend it twice on the shared 400 characters.
    """
    if max_chars <= 0 or not hits:
        return list(hits)
    ranks = {(h.start, h.end): (h.rank, h.score) for h in hits}
    kept: list[Hit] = []
    spent = 0
    for start, end in _merge_spans([(h.start, h.end) for h in hits]):
        if spent >= max_chars:
            break
        stop = min(end, start + (max_chars - spent))
        rank, score = ranks.get((start, end), (len(kept) + 1, 0.0))
        kept.append(Hit(rank=rank, start=start, end=stop, score=score, text=document[start:stop]))
        spent += stop - start
    return kept


# How many NEW terms a second hop borrows from what the first hop found. Small on
# purpose: the round-2 query keeps the original question and adds the bridge the
# reader surfaced, so a long tail of incidental nouns cannot outweigh the question.
HOP_QUERY_TERMS = 12


def _hop_query(task: str, findings: Sequence[str]) -> str:
    """A follow-up query: the question, plus the terms the FIRST hop actually found.

    Expanded from the map model's findings, never from the retrieved window text.
    That distinction is the whole design: blind pseudo-relevance feedback over raw
    windows was measured on this corpus and made recall WORSE everywhere (musique
    recall@5 47.5% -> 37.5%). Hop 2 has to be a decision taken after reading, which
    is what a finding is and what a window is not.

    Terms already in the question are dropped -- they are what round 1 searched on,
    so re-weighting them just returns round 1's windows.
    """
    asked = set(tokenize(task))
    counts: Counter = Counter()
    for text in findings:
        counts.update(term for term in tokenize(text) if term not in asked)
    borrowed = [term for term, _ in counts.most_common(HOP_QUERY_TERMS)]
    return f"{task} {' '.join(borrowed)}".strip() if borrowed else task


def _widen_spans(spans: Sequence[tuple[int, int]], target_chars: int, doc_len: int) -> list[tuple[int, int]]:
    """Grow retrieved spans symmetrically until they total `target_chars`.

    The `--min-subcall-chars` floor exists so a KV-compression arm has something
    to compress. `_expand_subcall` cannot serve a retrieved payload (there is no
    single slice to locate), so the floor is applied to the spans instead, before
    the payload is assembled.
    """
    spans = _merge_spans(spans)
    total = sum(end - start for start, end in spans)
    if not spans or doc_len <= 0 or total >= target_chars:
        return list(spans)
    grow = -(-(target_chars - total) // len(spans))  # ceil, so the floor is reached
    widened = []
    for start, end in spans:
        want = (end - start) + grow
        new_start = max(0, start - grow // 2)
        new_end = min(doc_len, new_start + want)
        new_start = max(0, new_end - want)
        widened.append((new_start, new_end))
    return _merge_spans(widened)


def _clip_spans(spans: Sequence[tuple[int, int]], kept_chars: int, separator_len: int) -> list[tuple[int, int]]:
    """The prefix of `spans` that survives a truncation to `kept_chars`.

    `_cap_subcall` truncates the payload from the right, so a payload that was cut
    must not go on claiming coverage for text the sub model never received. Walks
    left to right and drops or shortens the tail.
    """
    kept: list[tuple[int, int]] = []
    used = 0
    for index, (start, end) in enumerate(spans):
        if index:
            used += separator_len
        room = kept_chars - used
        if room <= 0:
            break
        length = end - start
        if length <= room:
            kept.append((start, end))
            used += length
        else:
            kept.append((start, start + room))
            break
    return kept


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #
def _validated_view(document: str, view: Sequence[Hit]) -> tuple[Hit, ...]:
    """An ordered-chunk view, checked before the root is shown any of it.

    Every chunk must be verbatim document text, ranked 1..n in list order, and the
    chunks together must cover every character exactly once: the view is the whole
    document in another order, never a selection from it.
    """
    chunks = tuple(view)
    if not chunks:
        raise ValueError("an ordered-chunk view needs at least one chunk")
    for position, hit in enumerate(chunks, start=1):
        if not isinstance(hit, Hit) or hit.rank != position or hit.text != document[hit.start : hit.end]:
            raise ValueError(f"ordered chunk {position} is not verbatim document text ranked by its position")
    spans = sorted((hit.start, hit.end) for hit in chunks)
    if spans[0][0] != 0 or spans[-1][1] != len(document) or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("an ordered-chunk view must cover every character of the document exactly once")
    return chunks
