# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The windows that provably carry the answer, as a chunk file the harness can read.

Every arm so far confounds two failures: the selector never surfaced the evidence,
and the 4B reader had it and could not use it. `gold_in_retrieved` splits them one
example at a time, but nothing has ever measured the second on its own. This builds
the arm that does -- retrieval is perfect BY CONSTRUCTION, so whatever score is
missing afterwards is the reader's ceiling and not the selector's.

That ceiling is not obviously high. Measured across 360M-7B with oracle passages,
exact match on questions needing external knowledge reached 14.6% at 7B, 10.0% at
1.5B and 0.0% at 360M -- most of a perfect retriever's output is wasted on a small
reader (arXiv 2603.11513). So read this arm as the number every other arm is a
fraction OF.

THREE PROPERTIES, matching chunkset.py's:

1. **Same window grid, same predicate.** Selection is expressed as window ids on the
   `ChunkIndex` grid, so an oracle arm and a BM25 arm at the same k differ in WHICH
   windows were chosen and in nothing else. "Carries the answer" is `is_correct`
   itself -- the same loose containment test `gold_in_retrieved` and
   `gold_windows_in_chunks` use -- so the three cannot disagree about what a gold
   window is.
2. **`spans` are always written.** The two hand-made chunk files carry none, which
   leaves `ChunkSet.hits_for`'s grid check unable to fire: window ids re-resolved on
   a DIFFERENT `--search-window` would silently address unrelated text. With spans
   present, a geometry mismatch is a refusal. The geometry is in the filename too.
3. **A padded or empty selection is recorded, never hidden.** An example with no
   gold-bearing window cannot have an oracle; it falls back to BM25 and says so in
   `note`, so the analysis can drop those rows instead of averaging a non-oracle
   into the ceiling.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import retrieval
from .chunkset import document_digest, window_spans
from .datasets import canonical_dataset_name, load_examples
from .run_benchmark import is_correct

SELECTOR = "oracle (windows containing a gold answer string)"


def gold_windows(document: str, answers: list[str], spans: list[tuple[int, int]]) -> dict[int, frozenset[int]]:
    """For each window that carries at least one gold answer, WHICH golds it carries.

    Returning the set rather than a count is what lets the greedy below cover as many
    distinct answers as possible: on a multi-value subset (qampari, quest) the gold is
    a list scattered over many passages, and k windows that all carry the same answer
    are a far worse oracle than k that carry k different ones.
    """
    carried: dict[int, frozenset[int]] = {}
    for index, (start, end) in enumerate(spans):
        text = document[start:end]
        found = frozenset(i for i, answer in enumerate(answers) if is_correct(text, [answer]))
        if found:
            carried[index] = found
    return carried


def select(carried: dict[int, frozenset[int]], k: int) -> list[int]:
    """Up to k windows, greedily maximising the number of DISTINCT golds covered.

    Ties break on window index so the selection is deterministic and reproducible
    from the same inputs -- an oracle that reshuffles between runs is not a ceiling.
    """
    chosen: list[int] = []
    covered: set[int] = set()
    remaining = dict(carried)
    while remaining and len(chosen) < k:
        best = min(remaining, key=lambda w: (-len(remaining[w] - covered), w))
        gain = remaining[best] - covered
        if not gain and chosen:
            # Nothing left adds a new answer. Keep filling by window order rather
            # than stopping: a wider oracle is still the honest k-sized read.
            best = min(remaining)
        chosen.append(best)
        covered |= remaining.pop(best)
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids-file", default=None, help="restrict to these example ids, one per line")
    ap.add_argument("--window", type=int, default=retrieval.DEFAULT_CHUNK_CHARS)
    ap.add_argument("--overlap", type=int, default=retrieval.DEFAULT_OVERLAP)
    ap.add_argument("--k", type=int, default=retrieval.DEFAULT_K, help="windows per example, matching --chunks-k")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ids = None
    if args.ids_file:
        lines = Path(args.ids_file).read_text().splitlines()
        ids = {ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")}

    examples = list(
        load_examples(
            canonical_dataset_name(args.dataset),
            args.data_dir,
            args.limit if ids is None else None,
            args.split,
            ids=ids,
        )
    )
    if not examples:
        raise SystemExit("no examples selected; check --data-dir/--split/--limit")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The geometry is IN THE NAME, not only in `spans`. A file silently reused at a
    # different --search-window is the one failure mode this arm cannot survive:
    # every offset stays in range and resolves to real text, so the run reports a
    # plausible score for windows it never chose.
    out_path = out_dir / f"{args.data_dir}.oracle.w{args.window}o{args.overlap}k{args.k}.jsonl"

    stats: Counter = Counter()
    covered_fractions: list[float] = []
    with open(out_path, "w") as fh:
        for example in examples:
            document = example["context"]
            answers = [str(a) for a in example["answers"]]
            spans = window_spans(len(document), args.window, args.overlap)
            carried = gold_windows(document, answers, spans)
            note = SELECTOR
            if carried:
                windows = select(carried, args.k)
                covered = set().union(*(carried[w] for w in windows))
                covered_fractions.append(len(covered) / len(answers) if answers else 0.0)
                if len(windows) < args.k:
                    # Pad with BM25 so the payload matches the comparator's size. Without
                    # it an oracle arm reads LESS text than the arm it is compared
                    # against and the difference is partly payload size, not selection.
                    index = retrieval.get_index(document, args.window, args.overlap)
                    by_span = {(start, end): i for i, (start, end) in enumerate(spans)}
                    for hit in index.search(example["question"], args.k * 3):
                        if len(windows) >= args.k:
                            break
                        candidate = by_span.get((hit.start, hit.end))
                        if candidate is not None and candidate not in windows:
                            windows.append(candidate)
                    windows.sort()
                    note += f"; padded to k with BM25 ({len(carried)} gold window(s) found)"
                    stats["padded"] += 1
                else:
                    stats["exact"] += 1
                note += f"; covers {len(covered)}/{len(answers)} gold answer(s)"
            else:
                # No window carries a gold string, so there is no oracle for this row.
                # BM25's own top-k, plainly labelled: averaging a non-oracle into the
                # ceiling would understate it and hide that the row was never testable.
                index = retrieval.get_index(document, args.window, args.overlap)
                by_span = {(start, end): i for i, (start, end) in enumerate(spans)}
                windows = sorted(
                    {
                        by_span[(h.start, h.end)]
                        for h in index.search(example["question"], args.k)
                        if (h.start, h.end) in by_span
                    }
                )
                note = "NO GOLD WINDOW; BM25 top-k fallback -- exclude from the ceiling"
                stats["no_gold"] += 1
                covered_fractions.append(0.0)

            fh.write(
                json.dumps(
                    {
                        "id": example["id"],
                        "windows": windows,
                        "spans": [list(spans[w]) for w in windows],
                        "document_sha256": document_digest(document),
                        "document_chars": len(document),
                        "selector": SELECTOR,
                        "note": note,
                    }
                )
                + "\n"
            )

    mean_covered = sum(covered_fractions) / len(covered_fractions) if covered_fractions else 0.0
    print(
        f"{len(examples)} examples -> {out_path}\n"
        f"  grid:      {args.window}/{args.overlap}, k={args.k}\n"
        f"  exact:     {stats['exact']} (k gold-bearing windows found)\n"
        f"  padded:    {stats['padded']} (fewer than k, filled with BM25)\n"
        f"  no gold:   {stats['no_gold']} (NOT an oracle row -- exclude)\n"
        f"  mean gold answers reachable: {mean_covered:.3f}"
    )


if __name__ == "__main__":
    main()
