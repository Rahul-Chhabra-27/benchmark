# Qwen3-8B Long-context Benchmark Results

This document summarizes the completed LOFT and RULER evaluation matrices for
Qwen3-8B. The LOFT results compare the non-quantized model with bitsandbytes
INT8 model-weight quantization at 32K and 128K context lengths. The available
RULER 32K results cover non-quantized inference.

## LOFT experimental setup

| Setting | Value |
|---|---|
| Model | Local `Qwen3-8B` checkpoint |
| Benchmark | LOFT |
| Tasks | Natural Questions (NQ), HotpotQA, MuSiQue, QAMPARI, QUEST |
| KV-cache method | KVzip |
| Baseline | KVzip `compression_ratio=0.01` |
| Memory-budget runs | 256 MB, 512 MB, 1 GB, and 2 GB |
| Dataset fraction | 1.0 |
| Maximum new tokens | 100 |
| 128K extension | YaRN-4; `max_context_length=131072` |
| INT8 method | bitsandbytes (`int8: true`) |

Here, **INT8 quantizes the model weights, not the KV cache**. Consequently, the
reported KV-cache memory for a given task and budget is the same in the two
model variants. The baseline compression ratio is the fraction pruned (1%),
whereas the budgeted runs prune enough tokens to fit the requested KV-cache
budget.

The tables report each task's most informative answer-quality metric: F1 for
NQ, HotpotQA, and MuSiQue; coverage for QAMPARI and QUEST. Values are percentages
and come directly from each run's `metrics.json`.

## 32K results

| Task | Model | 256 MB | 512 MB | 1 GB | 2 GB | Baseline |
|---|---|---:|---:|---:|---:|---:|
| NQ | Non-quantized | 2.83 | 3.92 | 49.19 | 58.35 | 57.96 |
| NQ | INT8 | 3.23 | 6.62 | 36.77 | 45.53 | 53.64 |
| HotpotQA | Non-quantized | 8.66 | 13.07 | 46.59 | 61.76 | 57.62 |
| HotpotQA | INT8 | 7.77 | 12.34 | 42.88 | 51.55 | 48.92 |
| MuSiQue | Non-quantized | 0.00 | 5.46 | 30.74 | 33.79 | 31.05 |
| MuSiQue | INT8 | 0.94 | 8.97 | 26.87 | 32.21 | 29.53 |
| QAMPARI | Non-quantized | 0.57 | 3.43 | 40.05 | 50.33 | 53.76 |
| QAMPARI | INT8 | 0.00 | 3.67 | 30.43 | 39.19 | 45.48 |
| QUEST | Non-quantized | 0.00 | 0.48 | 20.48 | 36.43 | 39.05 |
| QUEST | INT8 | 0.00 | 1.90 | 14.76 | 34.76 | 32.86 |

The 32K budgeted runs retain approximately 256, 512, 1,000, and 2,000 MB of KV
cache. The task-dependent baseline cache ranges from approximately 6.48 to
7.25 GB.

## 128K results

| Task | Model | 256 MB | 512 MB | 1 GB | 2 GB | Baseline |
|---|---|---:|---:|---:|---:|---:|
| NQ | Non-quantized | 0.00 | 0.30 | 3.06 | 20.03 | 49.30 |
| NQ | INT8 | 0.00 | 0.91 | 4.60 | 23.81 | 44.90 |
| HotpotQA | Non-quantized | 0.09 | 0.98 | 4.07 | 41.13 | 40.08 |
| HotpotQA | INT8 | 0.18 | 2.85 | 6.64 | 36.22 | 40.15 |
| MuSiQue | Non-quantized | 0.00 | 0.00 | 1.50 | 15.25 | 21.82 |
| MuSiQue | INT8 | 0.00 | 0.00 | 2.77 | 17.15 | 18.98 |
| QAMPARI | Non-quantized | 0.00 | 0.00 | 1.45 | 17.67 | 30.55 |
| QAMPARI | INT8 | 0.18 | 0.00 | 2.00 | 20.97 | 30.52 |
| QUEST | Non-quantized | 0.00 | 0.00 | 0.00 | 4.09 | 11.97 |
| QUEST | INT8 | 0.00 | 0.00 | 0.00 | 5.15 | 10.00 |

The 128K budgeted runs retain approximately 256, 512, 1,000, and 2,000 MB of KV
cache. The task-dependent baseline cache ranges from approximately 17.22 to
17.70 GB.

## RULER 32K results

The completed RULER 32K matrix uses the non-quantized Qwen3-8B model, KVzip,
50% of the cached evaluation examples, and a maximum context length of 32,768.
The baseline removes 1% of KV-cache tokens; the constrained run retains about
1 GB of KV cache, removing approximately 78-79% of tokens depending on the
task. RULER scores are the benchmark's `string_match` percentages.

| Task | 1 GB | Baseline |
|---|---:|---:|
| CWE | 79.70 | 79.80 |
| FWE | 96.00 | 95.33 |
| NIAH multikey 1 | 94.00 | 100.00 |
| NIAH multikey 2 | 74.00 | 76.00 |
| NIAH multikey 3 | 62.00 | 77.00 |
| NIAH multiquery | 99.25 | 100.00 |
| NIAH multivalue | 95.75 | 97.75 |
| NIAH single 1 | 100.00 | 100.00 |
| NIAH single 2 | 100.00 | 100.00 |
| NIAH single 3 | 100.00 | 100.00 |
| QA 1 | 71.00 | 72.00 |
| QA 2 | 60.00 | 58.00 |
| VT | 100.00 | 100.00 |
| **Macro average** | **87.05** | **88.91** |

No INT8 RULER configuration or INT8 RULER result directory is currently
present in this workspace, so an INT8 column is intentionally not reported.
This avoids treating an evaluation that has not been run as a zero score. The
LOFT INT8 results above remain the available model-quantization comparison.

## Result locations

| Benchmark | Context | Model | Configuration | Results |
|---|---|---|---|---|
| LOFT | 32K | Non-quantized | `evaluate_loft32k_config.yaml` | `results_loft_qwen/` |
| LOFT | 32K | INT8 | `evaluate_loft32k_int8_config.yaml` | `results_loft32k_qwen_int8/` |
| LOFT | 128K | Non-quantized | `evaluate_loft128k_config.yaml` | `results_loft128k_qwen_yarn4/` |
| LOFT | 128K | INT8 | `evaluate_loft128k_int8_config.yaml` | `results_loft128k_qwen_yarn4_int8/` |
| RULER | 32K | Non-quantized | `evaluate_ruler32k_1gb_config.yaml` | `results_ruler32k_qwen/` |

Each run directory contains:

- `README.md`: human-readable run summary
- `metrics.json`: aggregate metrics and KV-cache statistics
- `predictions.csv`: per-sample predictions and statistics
- `config.yaml`: resolved evaluation configuration

The historical `results_loft_qwen/` directory contains both the canonical 32K
non-quantized matrix and several older 128K runs. For the 32K comparison above,
only directories whose task name ends in `_32k` were used. The NQ 32K baseline
is in the run's nested `1/` directory. Use `results_loft128k_qwen_yarn4/` for the
canonical non-quantized 128K matrix.

## Reading the results

- At 32K, non-quantized inference generally scores higher than INT8, especially
  at 1 GB, 2 GB, and baseline settings, though a few low-budget cells favor
  INT8.
- At 128K, results are mixed: INT8 is better in several budgeted cells, while
  the non-quantized model has the stronger NQ and MuSiQue baselines.
- The 256 MB and 512 MB budgets remove roughly 92-99% of context tokens and
  sharply reduce answer quality. The 2 GB budget is consistently the strongest
  constrained setting.
- Compare answer quality and model memory separately: these files report
  KV-cache memory, but do not report total GPU memory or the model-weight memory
  saved by INT8 quantization.
