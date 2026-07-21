# LOFT Results: Non-quantized vs Quantized Qwen3-8B

This Intel-company presentation collects the completed LOFT results for the
non-quantized and quantized Qwen3-8B models at 32K and 128K context lengths.
Quantization uses bitsandbytes INT8 model weights; the KV cache is not
quantized.

## Evaluation setup

| Setting | Value |
|---|---|
| Benchmark | LOFT |
| Tasks | NQ, HotpotQA, MuSiQue, QAMPARI, QUEST |
| KV-cache method | KVzip |
| Models | Qwen3-8B non-quantized, Qwen3-8B quantized (INT8) |
| Baseline | `compression_ratio=0.01` |
| Memory budgets | 256 MB, 512 MB, 1 GB, 2 GB |
| Dataset fraction | 1.0 |
| Maximum new tokens | 100 |
| 128K extension | YaRN-4 with `max_context_length=131072` |

The tables use F1 for NQ, HotpotQA, and MuSiQue, and coverage for QAMPARI and
QUEST. All scores are percentages taken directly from the completed
`metrics.json` files.

## LOFT 32K

| Task | Model | 256 MB | 512 MB | 1 GB | 2 GB | Baseline |
|---|---|---:|---:|---:|---:|---:|
| NQ | Non-quantized | 2.83 | 3.92 | 49.19 | 58.35 | 57.96 |
| NQ | Quantized (INT8) | 3.23 | 6.62 | 36.77 | 45.53 | 53.64 |
| HotpotQA | Non-quantized | 8.66 | 13.07 | 46.59 | 61.76 | 57.62 |
| HotpotQA | Quantized (INT8) | 7.77 | 12.34 | 42.88 | 51.55 | 48.92 |
| MuSiQue | Non-quantized | 0.00 | 5.46 | 30.74 | 33.79 | 31.05 |
| MuSiQue | Quantized (INT8) | 0.94 | 8.97 | 26.87 | 32.21 | 29.53 |
| QAMPARI | Non-quantized | 0.57 | 3.43 | 40.05 | 50.33 | 53.76 |
| QAMPARI | Quantized (INT8) | 0.00 | 3.67 | 30.43 | 39.19 | 45.48 |
| QUEST | Non-quantized | 0.00 | 0.48 | 20.48 | 36.43 | 39.05 |
| QUEST | Quantized (INT8) | 0.00 | 1.90 | 14.76 | 34.76 | 32.86 |

At 32K, the non-quantized model generally performs better at 1 GB, 2 GB, and
baseline. INT8 is competitive in a few low-budget cells but loses more quality
at the larger budgets.

## LOFT 128K

| Task | Model | 256 MB | 512 MB | 1 GB | 2 GB | Baseline |
|---|---|---:|---:|---:|---:|---:|
| NQ | Non-quantized | 0.00 | 0.30 | 3.06 | 20.03 | 49.30 |
| NQ | Quantized (INT8) | 0.00 | 0.91 | 4.60 | 23.81 | 44.90 |
| HotpotQA | Non-quantized | 0.09 | 0.98 | 4.07 | 41.13 | 40.08 |
| HotpotQA | Quantized (INT8) | 0.18 | 2.85 | 6.64 | 36.22 | 40.15 |
| MuSiQue | Non-quantized | 0.00 | 0.00 | 1.50 | 15.25 | 21.82 |
| MuSiQue | Quantized (INT8) | 0.00 | 0.00 | 2.77 | 17.15 | 18.98 |
| QAMPARI | Non-quantized | 0.00 | 0.00 | 1.45 | 17.67 | 30.55 |
| QAMPARI | Quantized (INT8) | 0.18 | 0.00 | 2.00 | 20.97 | 30.52 |
| QUEST | Non-quantized | 0.00 | 0.00 | 0.00 | 4.09 | 11.97 |
| QUEST | Quantized (INT8) | 0.00 | 0.00 | 0.00 | 5.15 | 10.00 |

At 128K, INT8 performs better in several constrained-budget cells, while the
non-quantized model has stronger NQ and MuSiQue baselines. The 256 MB and
512 MB configurations are too restrictive for most 128K tasks.

## Result locations

| Context | Model | Configuration | Results |
|---|---|---|---|
| 32K | Non-quantized | `evaluate_loft32k_config.yaml` | `results_loft_qwen/` |
| 32K | Quantized (INT8) | `evaluate_loft32k_int8_config.yaml` | `results_loft32k_qwen_int8/` |
| 128K | Non-quantized | `evaluate_loft128k_config.yaml` | `results_loft128k_qwen_yarn4/` |
| 128K | Quantized (INT8) | `evaluate_loft128k_int8_config.yaml` | `results_loft128k_qwen_yarn4_int8/` |

Each completed run directory contains `predictions.csv`, `metrics.json`,
`config.yaml`, and an automatically generated per-run `README.md`.

The historical `results_loft_qwen/` directory also contains several older
128K runs. Only task directories ending in `_32k` are used for the 32K table;
`results_loft128k_qwen_yarn4/` is the canonical non-quantized 128K directory.
