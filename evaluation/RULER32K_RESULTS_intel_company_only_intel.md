# RULER 32K Results: Qwen3-8B

This Intel-company-only presentation summarizes the completed RULER 32K
evaluation for the non-quantized Qwen3-8B model. The available matrix compares
the KVzip 1%-pruned baseline with a 1 GB KV-cache budget across 13 tasks.

## Evaluation setup

| Setting | Value |
|---|---|
| Benchmark | RULER 32K |
| Model | Qwen3-8B, non-quantized |
| KV-cache method | KVzip |
| Dataset fraction | 0.5 |
| Maximum context length | 32,768 tokens |
| Baseline | `compression_ratio=0.01` |
| Constrained configuration | 1 GB KV-cache budget |
| Metric | String match (%) |

No completed INT8 RULER 32K result directory is present, so this presentation
reports only the available non-quantized results.

## Results

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

## Observations

- The baseline macro average is 88.91%, compared with 87.05% at 1 GB.
- The 1 GB result remains within 1.86 percentage points of the baseline while
  reducing retained KV-cache memory to approximately 1 GB.
- Most single-needle and variable-tracking tasks retain perfect scores at
  1 GB. The largest reduction is on NIAH multikey 3.
- FWE and QA 2 score slightly higher at 1 GB, showing that individual task
  scores are not strictly monotonic with retained context.

## Result locations

| Configuration | File or directory |
|---|---|
| Baseline configuration | `evaluate_ruler32k_baseline_config.yaml` |
| 1 GB configuration | `evaluate_ruler32k_1gb_config.yaml` |
| Completed results | `results_ruler32k_qwen/` |

Each completed run directory contains `predictions.csv`, `metrics.json`,
`config.yaml`, and an automatically generated per-run `README.md`.
