# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunks chosen for the RLM by something other than the RLM.

The BM25 arm let the root FIND things; this lets a chunk set be handed to it
outright, so "who picked the text" becomes an axis instead of a fixed property of
the harness. The selector is external and offline -- in practice a much larger
model reading the corpus on its own time -- and its choices arrive as this file.

Deliberately free of torch/transformers imports, for the same reason
``retrieval.py`` and ``sizing.py`` are: ``run_benchmark.py`` must stay importable
in a venv without them, since the ``http`` sub-call path has no torch.

THREE PROPERTIES THE REST OF THE HARNESS DEPENDS ON:

1. **Selection is expressed as window ids on the retrieval grid.** The same
   fixed windows ``ChunkIndex`` builds, so a provided-chunk arm and a BM25 arm of
   the same k differ in WHICH windows were chosen and in nothing else. Offsets
   are then exact by construction rather than transcribed by the selector, which
   is the difference between a verifiable span and a plausible one.
2. **The document is verified, not assumed.** A chunk file built against a
   different corpus carries offsets that are still in range and still resolve to
   text, so the failure is silent and total: every example reads the wrong
   passage and the arm reports a real-looking score. ``document_sha256`` turns
   that into a refusal.
3. **The payload is capped at the same size the comparator sends.** Without it a
   provided-chunk arm reads more text than the BM25 arm it is compared against,
   and the result measures payload size rather than selection quality.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from evaluation.rlm.shared.retrieval import DEFAULT_CHUNK_CHARS, DEFAULT_OVERLAP, Hit

# A tag becomes a COMPONENT of the run-directory name, which is "__"-joined, so it
# must not contain a separator, a path character, or a doubled underscore that would
# fake a component boundary. Lowercase kebab keeps it readable next to `qbare` and
# `listfmt`, and the length cap keeps the directory name inside sane limits.
SELECTOR_TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,23}$")


def document_digest(document: str) -> str:
    """The identity a chunk file is checked against."""
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    """The identity of the chunk FILE itself, for the resume guard.

    A path is not identity: two runs pointed at the same filename with different
    contents would otherwise resume into one checkpoint and merge two selections.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def window_spans(document_chars: int, chunk_chars: int = DEFAULT_CHUNK_CHARS, overlap: int = DEFAULT_OVERLAP):
    """The window grid, derived exactly as ``ChunkIndex`` derives it.

    Kept as a free function over a LENGTH rather than a document so a chunk file
    can be validated without materialising the corpus, and so the two definitions
    cannot drift: ``ChunkIndex.spans`` is the same arithmetic on ``len(document)``.
    """
    if chunk_chars <= 0:
        raise ValueError(f"chunk_chars must be positive, got {chunk_chars}")
    if not 0 <= overlap < chunk_chars:
        raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_chars, got {overlap} with {chunk_chars}")
    stride = chunk_chars - overlap
    return [(start, min(start + chunk_chars, document_chars)) for start in range(0, max(document_chars, 1), stride)]


def _valid_tag(raw, path, lineno: int) -> str:
    """`selector_tag` if it is safe to put in a directory name, else a refusal."""
    tag = str(raw or "")
    if tag and not SELECTOR_TAG_PATTERN.match(tag):
        raise ChunkSetError(
            f"{path}:{lineno} has selector_tag {tag!r}, which is not lowercase-kebab within 24 "
            "characters. The tag becomes a run-directory component, so a separator or a path "
            "character in it would fake a component boundary"
        )
    return tag


@dataclass(frozen=True)
class ChunkRecord:
    """One example's selection, as written by the selector."""

    id: str
    windows: tuple[int, ...]
    document_sha256: str
    document_chars: int
    selector: str = ""
    selector_tag: str = ""
    note: str = ""
    spans: tuple[tuple[int, int], ...] = field(default=())


class ChunkSetError(ValueError):
    """A chunk file that cannot be trusted. Always fatal -- never a fallback."""


class ChunkSet:
    """The selections for one run, keyed by example id."""

    def __init__(
        self,
        records: Iterable[ChunkRecord],
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        overlap: int = DEFAULT_OVERLAP,
        max_payload_chars: Optional[int] = None,
    ):
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.max_payload_chars = max_payload_chars
        self._by_id: dict[str, ChunkRecord] = {}
        for record in records:
            if record.id in self._by_id:
                raise ChunkSetError(f"duplicate id {record.id!r} in chunk file")
            self._by_id[record.id] = record

    def __len__(self) -> int:
        return len(self._by_id)

    @property
    def ids(self) -> list[str]:
        return list(self._by_id)

    @property
    def selectors(self) -> list[str]:
        return sorted({r.selector for r in self._by_id.values() if r.selector})

    @property
    def tag(self) -> str:
        """The one selector tag this file describes, or "" if it names none.

        Absent from every chunk file written before the query-agnostic selectors
        existed, which is exactly what lets the run-directory stamp be conditional
        on it: an untagged file names the directory it always named, so the oracle
        and hand-made chunk sets on disk keep their checkpoints. A file carrying two
        tags is not one arm and is refused rather than silently named after whichever
        record was read last.
        """
        tags = {r.selector_tag for r in self._by_id.values() if r.selector_tag}
        if len(tags) > 1:
            raise ChunkSetError(
                f"chunk file mixes selector tags {sorted(tags)}; one file is one arm. "
                "Split it, or the run directory would name only one of them"
            )
        return tags.pop() if tags else ""

    @property
    def max_windows(self) -> int:
        """The most windows any one example selects.

        What a per-example call budget has to clear: the structured map step makes
        one `llm_query` per group, and a budget below this silently turns the tail
        of the sweep into notice strings that the findings filter counts as findings.
        """
        return max((len(r.windows) for r in self._by_id.values()), default=0)

    def max_cluster_chars(self) -> int:
        """The widest CONTIGUOUS run of selected text, in characters.

        `rlm._cluster_hits` merges windows that touch or overlap into ONE map call, and
        the retrieval grid overlaps by construction -- so a selection that tiles the
        document collapses into a single passage spanning the whole corpus, which
        `_cap_subcall` then truncates to the subcall cap. The run reports a full sweep
        and reads the first cap-worth of characters. This is what lets that be refused
        before the run starts rather than discovered in the numbers afterwards.

        The merge rule is duplicated rather than imported: `rlm` carries the heavy
        imports this module's docstring promises to stay free of. It is six lines, and
        a test pins the two against each other.
        """
        widest = 0
        for record in self._by_id.values():
            grid = window_spans(record.document_chars, self.chunk_chars, self.overlap)
            spans = sorted(grid[w] for w in record.windows if 0 <= w < len(grid))
            start = end = None
            for span_start, span_end in spans:
                if start is None:
                    start, end = span_start, span_end
                elif span_start <= end:  # touching or overlapping -- same map call
                    end = max(end, span_end)
                else:
                    widest = max(widest, end - start)
                    start, end = span_start, span_end
            if start is not None:
                widest = max(widest, end - start)
        return widest

    @classmethod
    def from_file(
        cls,
        path,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        overlap: int = DEFAULT_OVERLAP,
        max_payload_chars: Optional[int] = None,
    ) -> "ChunkSet":
        path = Path(path)
        if not path.exists():
            raise ChunkSetError(f"chunk file {path} does not exist")
        records = []
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChunkSetError(f"{path}:{lineno} is not valid JSON ({exc})")
            missing = {"id", "windows", "document_sha256", "document_chars"} - set(raw)
            if missing:
                raise ChunkSetError(f"{path}:{lineno} is missing {sorted(missing)}")
            records.append(
                ChunkRecord(
                    id=str(raw["id"]),
                    windows=tuple(int(w) for w in raw["windows"]),
                    document_sha256=str(raw["document_sha256"]),
                    document_chars=int(raw["document_chars"]),
                    selector=str(raw.get("selector", "")),
                    selector_tag=_valid_tag(raw.get("selector_tag", ""), path, lineno),
                    note=str(raw.get("note", "")),
                    spans=tuple((int(a), int(b)) for a, b in raw.get("spans", ())),
                )
            )
        return cls(records, chunk_chars=chunk_chars, overlap=overlap, max_payload_chars=max_payload_chars)

    def hits_for(self, example_id: str, document: str) -> list[Hit]:
        """The selected windows for this example, as verbatim slices of `document`.

        Every failure here is raised, never softened into an empty list: a
        provided-chunk arm that silently reads nothing scores like a model that
        found nothing, and the two are not the same result.
        """
        record = self._by_id.get(example_id)
        if record is None:
            raise ChunkSetError(
                f"no chunks provided for example {example_id!r}. A chunk file must cover every "
                "example the run evaluates -- restrict the run with --ids-file, or extend the file"
            )
        if record.document_chars != len(document):
            raise ChunkSetError(
                f"{example_id}: chunk file was built against a {record.document_chars}-char document, "
                f"this run's is {len(document)}. The offsets would resolve to unrelated text"
            )
        digest = document_digest(document)
        if record.document_sha256 != digest:
            raise ChunkSetError(
                f"{example_id}: chunk file names document sha256 {record.document_sha256[:12]}..., "
                f"this run's document is {digest[:12]}.... Same length, different text"
            )
        grid = window_spans(len(document), self.chunk_chars, self.overlap)
        spans = []
        for window in record.windows:
            if not 0 <= window < len(grid):
                raise ChunkSetError(
                    f"{example_id}: window {window} is outside the {len(grid)}-window grid for this document"
                )
            spans.append(grid[window])
        if record.spans and tuple(spans) != record.spans:
            raise ChunkSetError(
                f"{example_id}: the file's `spans` disagree with its `windows` resolved on the "
                f"{self.chunk_chars}/{self.overlap} grid. One of them was written by hand"
            )
        payload = sum(end - start for start, end in spans)
        if self.max_payload_chars is not None and payload > self.max_payload_chars:
            raise ChunkSetError(
                f"{example_id}: {payload} chars selected against a {self.max_payload_chars}-char cap. "
                "The comparator arm sends at most the cap, so a larger payload measures size, not selection"
            )
        return [
            Hit(rank=rank, start=start, end=end, score=0.0, text=document[start:end])
            for rank, (start, end) in enumerate(spans, start=1)
        ]
