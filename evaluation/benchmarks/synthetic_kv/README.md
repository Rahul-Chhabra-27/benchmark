# Synthetic-KV dataset preparation

This directory contains the Synthetic-KV scorer, tests, and dataset preparation
utilities. The evaluation registry uses these Hugging Face datasets:

| Variant | Dataset repository | Evaluation dataset name |
| --- | --- | --- |
| Native 32K | `ollamaweights/synthetickv_32l` | `synthetic_kv_32k` |
| 64K | `ollamaweights/synthetickv_formated` | `synthetic_kv` |

## Formatted 64K dataset

[`create_huggingface_dataset.py`](create_huggingface_dataset.py) is the source of
truth for regenerating the 64K dataset. It reproduces the format currently
published at
[`ollamaweights/synthetickv_formated`](https://huggingface.co/datasets/ollamaweights/synthetickv_formated).
Do not use older `[KEY: VALUE]` generators for this repository.

The dataset has one compact row. Its `context` starts with retrieval instructions
and contains records formatted as:

```text
<record>
KEY: K_3900A3B1799D
VALUE: V_7D6246685257
</record>
```

The aligned question and answer are:

```text
Query KEY: K_3900A3B1799D
Return only its VALUE.
```

```text
V_7D6246685257
```

The published signature is one context, 1,707 questions, and 62,982 Qwen3-8B
context tokens. Generation uses seed `42`, 12 hexadecimal digits, and a 63,000
token ceiling.

### Install dependencies

From the repository root:

```bash
uv sync
source .venv/bin/activate
```

The tokenizer is downloaded from `Qwen/Qwen3-8B` unless it is already cached.

### Generate and validate locally

Generate first without touching Hugging Face:

```bash
python evaluation/benchmarks/synthetic_kv/create_huggingface_dataset.py \
  --output-dir /tmp/synthetickv_formated \
  --verify-published
```

`--verify-published` fails unless the output has exactly 1,707 pairs and 62,982
context tokens. On an offline machine, add `--local-files-only`.

Inspect the saved dataset before uploading:

```bash
python - <<'PY'
from datasets import load_from_disk

dataset = load_from_disk("/tmp/synthetickv_formated")
row = dataset[0]
print(row["context"][:800])
print(row["questions"][0])
print(row["answers"][0])
print(len(row["questions"]), row["context_tokens"])
PY
```

### Push to Hugging Face

Authenticate once on the machine performing the upload:

```bash
hf auth login
```

Then regenerate, validate, save a local copy, and push the `test` split:

```bash
python evaluation/benchmarks/synthetic_kv/create_huggingface_dataset.py \
  --output-dir /tmp/synthetickv_formated \
  --repo-id ollamaweights/synthetickv_formated \
  --verify-published \
  --push
```

Uploading to the existing repository updates its `test` split. To review a new
version without changing the benchmark dataset, pass a different `--repo-id`.
The script never uploads unless `--push` is explicitly present.

### Generate a different size

Use another token ceiling:

```bash
python evaluation/benchmarks/synthetic_kv/create_huggingface_dataset.py \
  --target-context-tokens 31000 \
  --output-dir /tmp/synthetickv_32k
```

Or use an exact number of records:

```bash
python evaluation/benchmarks/synthetic_kv/create_huggingface_dataset.py \
  --num-pairs 100 \
  --output-dir /tmp/synthetickv_100
```

Do not use `--verify-published` for intentionally different dataset sizes.

## Prepare the cache online

Run this on an internet-enabled login node from the `evaluation/` directory:

```bash
export HF_HOME=/home/rethinkingai-self/25m0820/.cache/huggingface
unset HF_DATASETS_OFFLINE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

python benchmarks/synthetic_kv/prepare_huggingface_cache.py --variant all --force-redownload
```

Use `--variant 32k` or `--variant 64k` to download only one dataset. Without
`--force-redownload`, an existing cached copy is reused.

## Verify that offline execution is ready

Still using the same `HF_HOME`, disable network access explicitly:

```bash
export HF_HOME=/home/rethinkingai-self/25m0820/.cache/huggingface
python benchmarks/synthetic_kv/prepare_huggingface_cache.py --variant all --offline-check
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
