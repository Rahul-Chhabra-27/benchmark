# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download everything an ordered-chunk campaign reads, then prove it loads offline.

    python -m evaluation.rlm.campaigns.rlm_order_stage --dataset all             # LOFT-128k and RULER-128k
    python -m evaluation.rlm.campaigns.rlm_order_stage --dataset ruler128k --weights

Lanes run with HF_HUB_OFFLINE=1, so these must be in the Hugging Face cache before the
first lane starts:
  - the tokenizer, which prices every chunk read;
  - the datasets;
  - with --weights, the model weights. The vLLM server needs them, and serve_root.sh also
    runs offline.
Run this once, with network access and no offline flags, under the same HF_HOME (and
HF_DATASETS_CACHE, if you set one) that the lanes will use. It prints what each task
loads, then reloads one task per dataset and the tokenizer in a fresh interpreter with
the offline flags set, so a gap shows up here rather than in the first lane.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from evaluation.benchmarks.registry import RULER_32K_TASKS
from evaluation.rlm.shared.datasets import load_examples

LOFT_128K = ("nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k")
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def tasks_for(dataset: str) -> list[tuple[str, str]]:
    """The (dataset, task) pairs a campaign on `dataset` reads."""
    loft = [("loft", task) for task in LOFT_128K]
    ruler = [("ruler128k", task) for task in RULER_32K_TASKS]
    return {"loft": loft, "ruler128k": ruler, "all": loft + ruler}[dataset]


def _split(dataset: str) -> dict:
    # The campaign runs LOFT's test split; RULER has no split.
    return {"split": "test"} if dataset == "loft" else {}


def verify_offline(datasets: list[str], model: str) -> None:
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(model, local_files_only=True)
    print(f"offline: the {model} tokenizer loads")
    for name in datasets:
        dataset, task = tasks_for(name)[0]
        rows = list(load_examples(dataset, task, limit=2, **_split(dataset)))
        print(f"offline: {dataset}/{task} loads ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("loft", "ruler128k", "all"), default="all")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=55, help="rows per task to load and check (the campaign runs 55)")
    ap.add_argument("--weights", action="store_true", help="also download the model weights the vLLM server loads")
    ap.add_argument("--verify-offline", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    datasets = ["loft", "ruler128k"] if args.dataset == "all" else [args.dataset]
    if args.verify_offline:
        verify_offline(datasets, args.model)
        return
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("HF_DATASETS_OFFLINE") == "1":
        ap.error("staging needs the network: unset HF_HUB_OFFLINE and HF_DATASETS_OFFLINE")

    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(args.model)
    print(f"cached: the {args.model} tokenizer")
    if args.weights:
        from huggingface_hub import snapshot_download

        print(f"cached: the {args.model} weights, at {snapshot_download(args.model)}")
    for dataset, task in tasks_for(args.dataset):
        rows = list(load_examples(dataset, task, limit=args.limit, **_split(dataset)))
        gold = sum(all(str(a) in r["context"] for a in r["answers"]) for r in rows)
        chars = max(len(r["context"]) for r in rows)
        print(
            f"cached: {dataset}/{task}, {len(rows)} rows (up to {chars:,} characters of context; "
            f"every gold string found in the context on {gold})",
            flush=True,
        )

    offline = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, "-m", "evaluation.rlm.campaigns.rlm_order_stage", "--verify-offline"]
    done = subprocess.run(command + ["--dataset", args.dataset, "--model", args.model], env=offline)
    if done.returncode:
        sys.exit("OFFLINE LOAD FAILED: the lanes would fail the same way. Check HF_HOME and HF_DATASETS_CACHE.")
    print("staged: lanes can run with HF_HUB_OFFLINE=1")


if __name__ == "__main__":
    main()
