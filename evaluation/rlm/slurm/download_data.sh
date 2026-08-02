#!/bin/bash
# Run on the LOGIN node (needs internet). Populate the shared Hugging Face cache.
# Usage (from repository root): bash evaluation/rlm/slurm/download_data.sh
set -euo pipefail

export HF_HOME="$HOME/hf_cache"

source .venv/bin/activate

python - <<'EOF'
from datasets import load_dataset
from evaluation.benchmarks.registry import DATASET_REGISTRY, RULER_32K_TASKS

# These are the same IDs/configs consumed by KVPress's shared loader.
print("Caching shared LongBench-v2 test split ...")
longbench = load_dataset(DATASET_REGISTRY["longbench-v2"], split="test")
print(f"LongBench-v2: {len(longbench)} examples")

print("Caching shared RULER-32K subsets ...")
for task in RULER_32K_TASKS:
    dataset = load_dataset(
        DATASET_REGISTRY["ruler32k"],
        data_dir=task,
        split="test",
    )
    print(f"RULER-32K/{task}: {len(dataset)} examples")

# --- OOLONG ---
# The OOLONG benchmark splits live on the Hugging Face Hub; the repo/config
# names have changed since release — check https://huggingface.co/datasets?search=oolong
# and the official RLM repo (github.com/alexzhang13/rlm) for the exact loader,
# then write oolong.jsonl under RLM_DATA_DIR with fields:
#   {"id", "context", "question", "answers": [...]}.
EOF

echo "Done. Shared datasets cached under $HF_HOME"
