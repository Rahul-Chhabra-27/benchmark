# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One pass over the corpus with the question nowhere in it.

Every other arm in this harness reads the document THROUGH the query: BM25 ranks
windows on it, the map prompt asks each passage about it, the oracle uses the
answers. So the corpus is re-read from scratch for every question, and nothing
measures what a representation built without the question is worth.

LOFT makes that question cheap to ask. Each subset is ONE corpus and 100 queries
against it -- every row of `nq_128k` carries a byte-identical 475,516-char context
-- so a query-free pass costs ~298 sub-calls for the whole subset rather than 298
per example, and its per-query cost is the build divided by 100. That is under
three sub-calls per query, against the ~298 an exhaustive per-query sweep of the
same text spends.

WHAT THIS BUILDS. One note per window: whatever a reader who has not seen the
question would need to have written down. Notes are keyed on the document sha, the
window geometry and the prompt revision, and the loader refuses on any mismatch --
a notes file silently reused at a different --search-window would address real but
unrelated text, exactly the failure `spans` exists to prevent in a chunk file.

NO QUESTION, AND THAT IS CHECKABLE. This module reads `context` and nothing else
from an example. It never touches `question` or `answers`, and does not import
`is_correct`. The absence is the claim -- there is a test that greps for it.

THE RISK WORTH NAMING. LOFT is scored by string matching, so the way a lossy
representation fails here is by PARAPHRASING AWAY the exact strings the scorer
matches on. That is why the prompt demands titles, ids, names, dates and numbers
verbatim, and why `--audit` exists: it reports what fraction of gold answer
strings survive into the notes, which is the ceiling this arm can possibly reach.
The audit is the one part of this file that may look at answers, it runs offline,
and it never touches the notes it is auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from . import retrieval
from .chunkset import document_digest, window_spans
from .client import LLMClient
from .datasets import canonical_dataset_name, load_examples
from .rlm import STRUCTURED_NOTHING

# Bump when the prompt below changes. It is part of the filename AND hashed into the
# header, so a notes file built by a different prompt cannot be read as this one.
WINDOW_NOTE_REVISION = 1

WINDOW_NOTE_PROMPT = """Below is one passage from a large collection of documents.

Write compact factual notes on this passage for a reader who will later be asked \
questions about the whole collection and will NOT see this passage again.

Copy every document TITLE and ID exactly as written. Record the entities named, what \
is asserted about each, dates, numbers, and any cross-references to other documents. \
Names, titles and identifiers must appear verbatim -- do not paraphrase or abbreviate them.

Do not summarise what the passage is about. Do not speculate. Do not add anything you \
know from outside the passage. At most {max_words} words.

If the passage carries no factual content, reply with exactly: {nothing}

PASSAGE:
{passage}"""


def prompt_digest(max_words: int) -> str:
    """The identity of the prompt itself, so a changed one cannot be read as this one."""
    text = WINDOW_NOTE_PROMPT.format(max_words=max_words, nothing=STRUCTURED_NOTHING, passage="")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NotesIndex:
    """BM25 over the NOTES, not the document.

    Deliberately not `retrieval.ChunkIndex`: that class derives its spans from a
    document and a uniform grid, which is what makes its offsets exact. Notes are
    variable-length text ABOUT those windows, so the unit being scored is a note
    while the thing being returned is still a document span. Same BM25 arithmetic,
    different unit, and `ChunkIndex` is left byte-identical -- its `search()` output
    is what every earlier arm rests on.
    """

    def __init__(self, notes: dict[int, str], spans: list[tuple[int, int]], k1: float = 1.5, b: float = 0.75):
        self.windows = sorted(notes)
        self.spans = spans
        self.k1, self.b = k1, b
        self._tf: list[Counter] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        frequency: Counter = Counter()
        for position, window in enumerate(self.windows):
            counts = Counter(retrieval.tokenize(notes[window]))
            self._tf.append(counts)
            self._lengths.append(sum(counts.values()))
            for term in counts:
                frequency[term] += 1
                self._postings[term].append(position)
        self.n = len(self.windows)
        self._avg = (sum(self._lengths) / self.n) if self.n else 0.0
        self._idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    def search(self, query: str, k: int) -> list[int]:
        """The k best-scoring windows for `query`, best first, as window ids."""
        if k <= 0 or not self.n:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in retrieval.tokenize(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for position in self._postings[term]:
                freq = self._tf[position][term]
                denominator = freq + self.k1 * (1 - self.b + self.b * self._lengths[position] / (self._avg or 1))
                scores[position] += idf * freq * (self.k1 + 1) / denominator
        ranked = sorted(scores, key=lambda p: (-scores[p], p))[:k]
        return [self.windows[p] for p in ranked]


def notes_path(out_dir: Path, data_dir: str, window: int, overlap: int, model: str) -> Path:
    model_slug = model.replace("/", "_")
    return Path(out_dir) / f"{data_dir}.notes.w{window}o{overlap}.p{WINDOW_NOTE_REVISION}.{model_slug}.jsonl"


def load_notes(path, document: str, window: int, overlap: int) -> dict[int, str]:
    """The notes for this document and geometry, or a refusal.

    Every mismatch is fatal, in the shape `ChunkSet.from_file` uses: a notes file
    read against the wrong document or the wrong grid resolves to real text about
    unrelated windows, and the arm reports a plausible score for a corpus it never
    summarised.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"notes file {path} does not exist")
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"notes file {path} is empty")
    header = json.loads(lines[0])
    if header.get("schema") != "window_notes/1":
        raise ValueError(f"{path}: not a window_notes/1 file (schema={header.get('schema')!r})")
    if header.get("document_sha256") != document_digest(document):
        raise ValueError(
            f"{path}: built against document {str(header.get('document_sha256'))[:12]}..., "
            f"this run's is {document_digest(document)[:12]}.... The notes describe other text"
        )
    if (header.get("window"), header.get("overlap")) != (window, overlap):
        raise ValueError(
            f"{path}: built on a {header.get('window')}/{header.get('overlap')} grid, this run is "
            f"{window}/{overlap}. Window ids would address unrelated passages"
        )
    notes = {}
    for line in lines[1:]:
        record = json.loads(line)
        if not record.get("empty"):
            notes[int(record["window"])] = str(record["note"])
    return notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--window", type=int, default=retrieval.DEFAULT_CHUNK_CHARS)
    ap.add_argument("--overlap", type=int, default=retrieval.DEFAULT_OVERLAP)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-words", type=int, default=120, help="length budget offered to the note writer")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force", action="store_true", help="rebuild from scratch instead of resuming")
    args = ap.parse_args()

    examples = list(load_examples(canonical_dataset_name(args.dataset), args.data_dir, None, args.split))
    if not examples:
        raise SystemExit("no examples selected; check --data-dir/--split")
    # The whole economy of this arm is one corpus per subset. If that stops being
    # true the build is per-example and costs 100x, so say so rather than discover it.
    documents = {document_digest(e["context"]) for e in examples}
    if len(documents) != 1:
        raise SystemExit(
            f"{args.data_dir} has {len(documents)} distinct documents across {len(examples)} rows. "
            "A query-free index is built ONCE per corpus; this subset would need one per row"
        )
    document = examples[0]["context"]
    spans = window_spans(len(document), args.window, args.overlap)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = notes_path(out_dir, args.data_dir, args.window, args.overlap, args.model)

    header = {
        "schema": "window_notes/1",
        "document_sha256": document_digest(document),
        "document_chars": len(document),
        "window": args.window,
        "overlap": args.overlap,
        "n_windows": len(spans),
        "prompt_revision": WINDOW_NOTE_REVISION,
        "prompt_sha256": prompt_digest(args.max_words),
        "model": args.model,
        "max_words": args.max_words,
    }

    # Resume: a 300-call build that dies at 200 must not start over. The header is
    # compared field by field, so a resume across a changed prompt or grid is refused
    # rather than producing a file half-written by each.
    done: dict[int, dict] = {}
    if path.exists() and not args.force:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if lines:
            prior = json.loads(lines[0])
            mismatch = [k for k, v in header.items() if prior.get(k) != v]
            if mismatch:
                raise SystemExit(f"{path} was built with different {mismatch}; pass --force to rebuild")
            for line in lines[1:]:
                record = json.loads(line)
                done[int(record["window"])] = record

    client = LLMClient(model=args.model, base_url=args.base_url, max_tokens=512)
    records = dict(done)
    for window, (start, end) in enumerate(spans):
        if window in records:
            continue
        prompt = WINDOW_NOTE_PROMPT.format(
            max_words=args.max_words, nothing=STRUCTURED_NOTHING, passage=document[start:end]
        )
        reply = (client.chat([{"role": "user", "content": prompt}]) or "").strip()
        empty = not reply or reply.upper().lstrip().startswith(STRUCTURED_NOTHING)
        records[window] = {"window": window, "start": start, "end": end, "note": "" if empty else reply,
                           "empty": empty}
        if window % 25 == 0:
            print(f"  window {window}/{len(spans)}", flush=True)

    header["build_calls"] = len(spans) - len(done)
    header["build_prompt_tokens"] = client.usage.prompt_tokens
    header["build_completion_tokens"] = client.usage.completion_tokens
    with open(path, "w") as fh:
        fh.write(json.dumps(header) + "\n")
        for window in sorted(records):
            fh.write(json.dumps(records[window]) + "\n")

    kept = sum(1 for r in records.values() if not r["empty"])
    chars = sum(len(r["note"]) for r in records.values())
    print(
        f"{len(spans)} windows -> {path}\n"
        f"  notes kept:   {kept} ({len(spans) - kept} said nothing)\n"
        f"  notes chars:  {chars:,} against {len(document):,} of document "
        f"({chars / len(document):.1%} -- the compression this arm actually achieves)\n"
        f"  build cost:   {header['build_calls']} calls, "
        f"{client.usage.total_tokens:,} tokens "
        f"({client.usage.total_tokens // max(len(examples), 1):,} per query amortised over "
        f"{len(examples)} rows)"
    )


if __name__ == "__main__":
    main()
