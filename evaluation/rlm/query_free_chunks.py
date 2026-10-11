# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunk sets chosen WITHOUT the question, as the floor every retrieval arm is measured against.

Every arm in this harness selects text with the query: `--chunks-source bm25` ranks
windows on it, `--chunks-source file` inherits a selection someone else made with it,
and the oracle uses the answers. So there has never been a floor to read a score
against -- when nq moved 0.545 -> 0.800 by asking BM25 the question instead of LOFT's
instruction page, nothing said how much of the remaining 0.800 the query was buying
and how much was there for free.

These four selectors buy nothing. They are pure functions of `(n_windows, k, seed)`:
no document text, no question, no answers. Read through `--chunks-source file` they
share the BM25 arm's pipeline exactly, so `bm25_bare - random` at the same k is the
value of the query and nothing else.

  lead     the first k windows            -- the positional prior
  stride   k windows evenly spaced        -- uniform coverage
  random   k windows, seeded              -- the null model
  all      every window                   -- exhaustive; perfect recall, maximum distractors

THREE PROPERTIES, matching chunkset.py's and oracle_chunks.py's:

1. **No gold, no question, and that is checkable.** This module does not import
   `is_correct` and never reads `question` or `answers` -- the absence is the claim.
   `oracle_chunks.py` is the module that may look at answers, and keeping the two
   apart keeps the leakage boundary at a file boundary rather than a code path.
2. **`spans` are always written.** Without them `ChunkSet.hits_for`'s grid check
   cannot fire, and window ids re-resolved on a different `--search-window` stay in
   range and silently address unrelated text. The geometry is in the filename too.
3. **`selector_tag` is always written.** Every file-sourced arm at the same k is
   otherwise named `chunks-file<k>`, so oracle, lead, stride, random and exhaustive
   would collide on one run directory and the second arm to run would die on a
   resume conflict. The tag is what makes them distinct runs.

WHY THE RANDOM ARM DRAWS PER ROW. One draw shared by every example measures that
draw; an independent draw per example makes the run mean an estimator of the
expected score under random selection, which is the quantity the ladder wants. The
seed is derived from `--seed` and the example id, so it is reproducible and survives
a change in row order.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import retrieval
from .chunkset import document_digest, window_spans
from .datasets import canonical_dataset_name, load_examples

SELECTORS = ("lead", "stride", "random", "all")


def lead(n_windows: int, k: int) -> list[int]:
    """The first k windows. The positional prior, and LOFT's corpora are not sorted."""
    return list(range(min(k, n_windows)))


def stride(n_windows: int, k: int) -> list[int]:
    """k windows spread evenly over the grid, endpoints included.

    A set, then sorted: on a small grid two evenly-spaced indices can round to the
    same window, and a duplicate would be merged by `_cluster_hits` while still
    counting twice against the payload cap. Returning k-1 windows is honest; a
    duplicate is not.
    """
    if k <= 0 or n_windows <= 0:
        return []
    if k == 1:
        return [n_windows // 2]
    if k >= n_windows:
        return list(range(n_windows))
    return sorted({round(i * (n_windows - 1) / (k - 1)) for i in range(k)})


def random_windows(n_windows: int, k: int, seed) -> list[int]:
    """k windows drawn without replacement. The null model at this read budget."""
    return sorted(random.Random(seed).sample(range(n_windows), min(k, n_windows)))


def every_window(n_windows: int) -> list[int]:
    """The whole grid. Retrieval removed rather than replaced."""
    return list(range(n_windows))


def windows_for(selector: str, n_windows: int, k: int, seed) -> list[int]:
    if selector == "lead":
        return lead(n_windows, k)
    if selector == "stride":
        return stride(n_windows, k)
    if selector == "random":
        return random_windows(n_windows, k, seed)
    if selector == "all":
        return every_window(n_windows)
    raise ValueError(f"unknown selector {selector!r}; choose from {list(SELECTORS)}")


def tag_for(selector: str, seed: int) -> str:
    """The run-directory component. Seeded arms carry their seed or they merge."""
    return f"random-s{seed}" if selector == "random" else selector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids-file", default=None, help="restrict to these example ids, one per line")
    ap.add_argument("--selector", required=True, choices=SELECTORS)
    ap.add_argument("--window", type=int, default=retrieval.DEFAULT_CHUNK_CHARS)
    ap.add_argument("--overlap", type=int, default=retrieval.DEFAULT_OVERLAP)
    ap.add_argument(
        "--k",
        type=int,
        default=retrieval.DEFAULT_K,
        help="windows per example, matching --chunks-k. Ignored by --selector all, which "
        "takes every window and writes the resulting count into the filename",
    )
    ap.add_argument("--seed", type=int, default=0, help="--selector random only; part of the tag and the filename")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing chunk file. Off by default: rewriting one changes its content "
        "hash and invalidates every resume pointed at it",
    )
    ap.add_argument(
        "--ids-out",
        default=None,
        help="also write the selected example ids here, one per line, for --ids-file. Emitted from "
        "the loader rather than hand-written, so the ids cannot drift from the rows they name",
    )
    ap.add_argument(
        "--ids-out-range",
        default=None,
        metavar="A:B",
        help="restrict --ids-out to a half-open slice of the loaded order. With `--split test` and "
        "no --limit, `55:100` is the holdout the chunk-geometry campaign never ran",
    )
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

    # Every example in this run shares one grid size only if it shares one document
    # length; LOFT does (one corpus, many queries) but nothing here assumes it.
    n_windows_by_id = {}
    for example in examples:
        n_windows_by_id[example["id"]] = len(window_spans(len(example["context"]), args.window, args.overlap))
    widest = max(n_windows_by_id.values())

    # `all` has no free k: the payload cap `--chunks-k x --search-window` must clear the
    # whole grid, and the grid size differs per subset (nq 298 windows at 2000/400,
    # quest 302). Putting the EFFECTIVE k in the filename is what lets the driver read
    # it back instead of guessing a global value that silently fails on the widest subset.
    effective_k = widest if args.selector == "all" else args.k
    tag = tag_for(args.selector, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.data_dir}.{tag}.w{args.window}o{args.overlap}k{effective_k}.jsonl"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} already exists; pass --force to rewrite it (this changes its content hash)")

    selector_text = {
        "lead": f"lead ({effective_k} first windows, no query, no gold)",
        "stride": f"stride ({effective_k} windows evenly spaced, no query, no gold)",
        "random": f"random ({effective_k} windows, seed {args.seed}, no query, no gold)",
        "all": "exhaustive (every window on the grid, no query, no gold)",
    }[args.selector]

    sizes = []
    with open(out_path, "w") as fh:
        for example in examples:
            document = example["context"]
            spans = window_spans(len(document), args.window, args.overlap)
            windows = windows_for(args.selector, len(spans), args.k, f"{args.seed}:{example['id']}")
            sizes.append(len(windows))
            fh.write(
                json.dumps(
                    {
                        "id": example["id"],
                        "windows": windows,
                        "spans": [list(spans[w]) for w in windows],
                        "document_sha256": document_digest(document),
                        "document_chars": len(document),
                        "selector": selector_text,
                        "selector_tag": tag,
                        "note": (
                            f"grid {args.window}/{args.overlap}, {len(spans)} windows; "
                            f"k={len(windows)}; seed={args.seed if args.selector == 'random' else 'n/a'}"
                        ),
                    }
                )
                + "\n"
            )

    if args.ids_out:
        wanted = examples
        if args.ids_out_range:
            start, _, end = args.ids_out_range.partition(":")
            wanted = examples[int(start or 0) : int(end) if end else None]
        Path(args.ids_out).write_text("".join(f"{e['id']}\n" for e in wanted))

    payload = max(sizes) * args.window
    print(
        f"{len(examples)} examples -> {out_path}\n"
        f"  selector:  {selector_text}\n"
        f"  tag:       {tag}   (run directory gets `sel-{tag}`)\n"
        f"  grid:      {args.window}/{args.overlap}, {widest} windows on the widest document\n"
        f"  windows:   {min(sizes)}-{max(sizes)} per example\n"
        f"  run it with --chunks-k {effective_k} (payload cap {payload} chars)"
        + (f"\n  ids:       {len(wanted)} -> {args.ids_out}" if args.ids_out else "")
    )


if __name__ == "__main__":
    main()
