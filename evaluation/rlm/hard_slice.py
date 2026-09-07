# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pick the examples worth handing to a better selector, and lay out the corpus for it.

The interesting failures are not the ones where retrieval missed. On LOFT-128k's
nq and hotpotqa subsets BM25 puts the gold answer in the root's hands 98% and 91%
of the time and the arm still fails 42% and 62% of the time -- so on those rows
the evidence was found and then wasted. Those are the rows where "read the chunks
with a fixed pipeline instead of a REPL" has something to prove, and they are
selected here rather than by eye.

Selecting on "the arm got it wrong" would normally invite regression to the mean.
It does not here: LLMClient runs at temperature 0.0 and vLLM decodes greedily, so
re-running the same arm on these rows reproduces the same answers. They are
genuinely failures, not unlucky samples.

Emits three files:
  <subset>.hardslice.ids   one example id per line, for --ids-file
  questions.jsonl          id + question ONLY -- never the answers. A selector
                           shown the gold measures leakage and nothing else.
  windows.jsonl            the corpus on the retrieval grid, so a selection can be
                           expressed as window ids and resolved to exact offsets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import retrieval
from .chunkset import document_digest, window_spans
from .datasets import canonical_dataset_name, load_examples


def hard_ids(checkpoint: Path) -> list[str]:
    """Ids the search arm retrieved the gold for and answered wrongly anyway."""
    ids = []
    for line in checkpoint.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("gold_in_retrieved") is True and not row.get("correct"):
            ids.append(row["id"])
    if not ids:
        raise SystemExit(
            f"{checkpoint} has no rows with gold_in_retrieved=True and correct=False. "
            "Either it is not a --search-k run (only those record the diagnostic), or "
            "the arm has no retrieved-but-wrong failures to study"
        )
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="checkpoint.jsonl of a --search-k run")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="rows the source run evaluated")
    ap.add_argument("--window", type=int, default=retrieval.DEFAULT_CHUNK_CHARS)
    ap.add_argument("--overlap", type=int, default=retrieval.DEFAULT_OVERLAP)
    ap.add_argument("--out", required=True, help="directory for the three files")
    args = ap.parse_args()

    ids = hard_ids(Path(args.checkpoint))
    wanted = set(ids)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    examples = list(load_examples(canonical_dataset_name(args.dataset), args.data_dir, None, args.split, ids=wanted))
    # LOFT ships one corpus per subset, byte-identical across every row of it, which
    # is what makes ONE window file enough for the whole slice. Check rather than
    # assume: a per-example corpus would need a window file each, and silently
    # writing only the first row's would send the selector to the wrong document.
    documents = {ex["context"] for ex in examples}
    if len(documents) != 1:
        raise SystemExit(
            f"{len(documents)} distinct documents across {len(examples)} examples. This "
            "layout assumes one shared corpus (as LOFT has); a per-example corpus needs "
            "a window file per example"
        )
    document = examples[0]["context"]

    (out / f"{args.data_dir}.hardslice.ids").write_text("\n".join(ids) + "\n")
    with open(out / "questions.jsonl", "w") as fh:
        for ex in examples:
            fh.write(json.dumps({"id": ex["id"], "question": ex["question"]}) + "\n")
    spans = window_spans(len(document), args.window, args.overlap)
    with open(out / "windows.jsonl", "w") as fh:
        for index, (start, end) in enumerate(spans):
            fh.write(json.dumps({"window": index, "start": start, "end": end, "text": document[start:end]}) + "\n")
    print(
        f"{len(ids)} hard examples -> {out}\n"
        f"  document: {len(document)} chars, sha256 {document_digest(document)}\n"
        f"  windows:  {len(spans)} at {args.window}/{args.overlap}"
    )


if __name__ == "__main__":
    main()
