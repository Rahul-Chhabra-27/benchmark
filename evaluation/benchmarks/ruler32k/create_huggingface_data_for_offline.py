import os
import shutil
from datasets import load_dataset

huggingface_dataset_id = "xAlg-AI/att-hub-ruler-32k"

subsets_to_run = [
    "cwe",
    "fwe",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "qa_1",
    "qa_2",
    "vt",
]

base_cache_dir = os.path.expanduser(
    "~/.cache/huggingface/datasets/xAlg-AI___att-hub-ruler-32k"
)

for subset in subsets_to_run:
    # config_name == split == subset for this repo, so no data_dir/rename logic needed
    ds = load_dataset(
        huggingface_dataset_id,
        subset,
        split=subset,
    )
    print(f"cached: {subset} ({len(ds)} samples)")
 
print("Done. Cache folders are named after the config (e.g. 'cwe', 'fwe', ...) "
      "which is exactly what load_dataset(repo, subset, split=subset) expects to find.")
 