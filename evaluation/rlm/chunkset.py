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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .retrieval import DEFAULT_CHUNK_CHARS, DEFAULT_OVERLAP, Hit


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


@dataclass(frozen=True)
class ChunkRecord:
    """One example's selection, as written by the selector."""

    id: str
    windows: tuple[int, ...]
    document_sha256: str
    document_chars: int
    selector: str = ""
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
