# Synthetic-KV 64K Results: Non-quantized vs NF4 Qwen3-8B

This Intel-company-only presentation summarizes the completed Synthetic-KV 64K
evaluations for the non-quantized and bitsandbytes NF4-quantized Qwen3-8B
models. It reports every completed configuration: baseline, 1 GB, 2 GB, and
4 GB for both models, plus the completed NF4 512 MB run. The non-quantized
512 MB evaluation is still running, and 256 MB is outside the requested result
matrix.

## Dataset

The cached `ollamaweights/synthetic-kv-qwen3-8b` test split contains one
approximately 64K-token context with 2,340 synthetic key/value retrieval
questions. Keys and values use the forms `K_<12 hex characters>` and
`V_<12 hex characters>`. Each question asks for the value associated with one
key and requires the model to return only that value.

All reported configurations use the same:

- 2,340 questions and reference answers
- original context length of 62,998 tokens
- Qwen3-8B checkpoint
- KVzip cache compression
- seed 42
- maximum generation length of 32 tokens
- maximum context length of 65,536 tokens

NF4 quantizes the **model weights**, not the KV cache. Therefore, matching
budgets retain the same number of context tokens and report the same KV-cache
memory for both model variants.

## Evaluation metrics

- **Exact match:** the normalized prediction equals the complete reference.
- **String match:** the complete normalized reference occurs anywhere in the
  prediction.

The `V_` prefix is part of the reference. For example, `123456789ABC` does not
match `V_123456789ABC`. All values below are percentages over 2,340 questions.

## Results

| Model | KV-cache setting | Retained tokens | Average compression ratio | Exact match | String match |
|---|---|---:|---:|---:|---:|
| Non-quantized | Baseline (1% pruned) | 62,368 | 0.0100 | 7.99% | 9.23% |
| Non-quantized | 1 GB | 6,781 | 0.8924 | 0.13% | 0.47% |
| Non-quantized | 2 GB | 13,563 | 0.7847 | 9.79% | 10.04% |
| Non-quantized | 4 GB | 27,126 | 0.5694 | 9.32% | 10.68% |
| NF4 | Baseline (1% pruned) | 62,368 | 0.0100 | 1.71% | 2.09% |
| NF4 | 512 MB | 3,472 | 0.9449 | 0.00% | 0.00% |
| NF4 | 1 GB | 6,781 | 0.8924 | 0.34% | 0.43% |
| NF4 | 2 GB | 13,563 | 0.7847 | 3.38% | 4.36% |
| NF4 | 4 GB | 27,126 | 0.5694 | 1.37% | 1.92% |

The average compression ratio is the fraction of context tokens pruned, not
the fraction retained.

## Quantization comparison

| KV-cache setting | Non-quantized exact | NF4 exact | Non-quantized string | NF4 string |
|---|---:|---:|---:|---:|
| Baseline | 7.99% | 1.71% | 9.23% | 2.09% |
| 1 GB | 0.13% | 0.34% | 0.47% | 0.43% |
| 2 GB | 9.79% | 3.38% | 10.04% | 4.36% |
| 4 GB | 9.32% | 1.37% | 10.68% | 1.92% |

Except at the severely constrained 1 GB setting, the non-quantized model
substantially outperforms NF4. At 1 GB, both models are close to zero, so the
small ordering difference is not meaningful.

## Observations

- The non-quantized model has its highest exact-match score at 2 GB and its
  highest string-match score at 4 GB.
- NF4 has its highest exact- and string-match scores at 2 GB. Its non-monotonic
  results suggest that KVzip can remove distractors, but the 2 GB NF4 run should
  be repeated before treating the improvement as conclusive.
- NF4 scores 0.00% at 512 MB, where only 3,472 context tokens remain.
- The 1 GB budget retains only 6,781 of 62,998 tokens and performs poorly for
  both model variants.
- More retained KV cache does not guarantee a higher retrieval score: KVzip
  changes which tokens remain, and long synthetic contexts contain thousands
  of structurally similar key/value distractors.

## Result locations

| Model | Configuration | Results |
|---|---|---|
| Non-quantized | `evaluate_synthetic_kv_independent_config.yaml` | `results_synthetic_kv_qwen_yarn4_full_independent/` |
| NF4 | `evaluate_synthetic_kv_nf4_independent_config.yaml` | `results_synthetic_kv_qwen_yarn4_int4_nf4_full_independent/` |

Each completed run directory contains:

- `predictions.csv`: per-question predictions, references, and KV-cache data
- `metrics.json`: aggregate retrieval metrics and KV-cache statistics
- `config.yaml`: resolved evaluation configuration
- `README.md`: automatically generated summary for that individual run

The non-quantized 512 MB run is not included because it was still running when
this summary was prepared. The completed NF4 512 MB result is included. No
256 MB results are expected for this matrix.
