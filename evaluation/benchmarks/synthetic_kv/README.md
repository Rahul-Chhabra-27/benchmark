# Synthetic-KV dataset preparation

This directory contains the Synthetic-KV scorer, tests, and dataset preparation
utility. The evaluation registry uses these Hugging Face datasets:

| Variant | Dataset repository | Evaluation dataset name |
| --- | --- | --- |
| Native 32K | `ollamaweights/synthetickv_32l` | `synthetic_kv_32k` |
| 64K | `ollamaweights/synthetickv_formated` | `synthetic_kv` |

## Prepare the cache online

Run this on an internet-enabled login node from the `evaluation/` directory:

```bash
export HF_HOME=/home/rethinkingai-self/25m0820/.cache/huggingface
unset HF_DATASETS_OFFLINE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

python benchmarks/synthetic_kv/download_dataset.py --variant all --force-redownload
```

Use `--variant 32k` or `--variant 64k` to download only one dataset. Without
`--force-redownload`, an existing cached copy is reused.

## Verify that offline execution is ready

Still using the same `HF_HOME`, disable network access explicitly:

```bash
export HF_HOME=/home/rethinkingai-self/25m0820/.cache/huggingface
python benchmarks/synthetic_kv/download_dataset.py --variant all --offline-check
```

The offline check fails if either dataset is absent from the local Hugging Face
cache. It also validates the compact context, question/answer counts, required
columns, `Records:` marker, and generation metadata.

Slurm compute jobs must use the same `HF_HOME` and should set:

```bash
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```
