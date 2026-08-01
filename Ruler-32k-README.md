# Qwen3-8B RULER-32K Benchmark Results

Benchmark results for the **RULER (32K context length)** dataset across multiple memory budget configurations and Qwen3-8B model.

## Dataset

- **Name:** `ruler32k`
- **Context length:** 32,000 tokens

### Tasks (`data_dir`)

| Task | Description |
|---|---|
| `cwe` | Common Words Extraction |
| `fwe` | Frequent Words Extraction |
| `niah_multikey_1` | Needle-in-a-Haystack — Multi-key, variant 1 |
| `niah_multikey_2` | Needle-in-a-Haystack — Multi-key, variant 2 |
| `niah_multikey_3` | Needle-in-a-Haystack — Multi-key, variant 3 |
| `niah_multiquery` | Needle-in-a-Haystack — Multi-query |
| `niah_multivalue` | Needle-in-a-Haystack — Multi-value |
| `niah_single_1` | Needle-in-a-Haystack — Single needle, variant 1 |
| `niah_single_2` | Needle-in-a-Haystack — Single needle, variant 2 |
| `niah_single_3` | Needle-in-a-Haystack — Single needle, variant 3 |
| `qa_1` | Question Answering, dataset 1 |
| `qa_2` | Question Answering, dataset 2 |
| `vt` | Variable Tracking |

## Memory Budgets

The benchmark includes an effectively uncompressed baseline (configured with
`compression_ratio: 0.01`) and five explicit KV-cache memory budgets.

| Configuration | Completed tasks |
|---|---:|
| Baseline | 13/13 |
| 0.2 GB | 12/13 |
| 0.4 GB | 13/13 |
| 0.6 GB | 13/13 |
| 0.8 GB | 13/13 |
| 1 GB | 13/13 |

## Results

Scores are `string_match` percentages. Each completed task-configuration result
contains 100 evaluated samples.

| Task | Baseline | 0.2 GB | 0.4 GB | 0.6 GB | 0.8 GB | 1 GB |
|---|---:|---:|---:|---:|---:|---:|
| `cwe` | 79.80 | 0.00 | 15.20 | 51.60 | 73.20 | 79.70 |
| `fwe` | 95.33 | 77.67 | 95.67 | 97.00 | 96.00 | 96.00 |
| `niah_multikey_1` | 100.00 | 32.00 | 58.00 | 81.00 | 90.00 | 94.00 |
| `niah_multikey_2` | 76.00 | 3.00 | 46.00 | 80.00 | 86.00 | 74.00 |
| `niah_multikey_3` | 77.00 | 0.00 | 2.00 | 18.00 | 38.00 | 62.00 |
| `niah_multiquery` | 100.00 | 17.50 | 87.75 | 98.25 | 99.50 | 99.25 |
| `niah_multivalue` | 97.75 | 24.25 | 68.25 | 86.00 | 91.25 | 95.75 |
| `niah_single_1` | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| `niah_single_2` | 100.00 | 8.00 | 91.00 | 100.00 | 100.00 | 100.00 |
| `niah_single_3` | 100.00 | 13.00 | 99.00 | 100.00 | 100.00 | 100.00 |
| `qa_1` | 72.00 | -- | 50.00 | 63.00 | 68.00 | 71.00 |
| `qa_2` | 58.00 | 26.00 | 52.00 | 62.00 | 64.00 | 60.00 |
| `vt` | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| **Average** | **88.91** | **33.45*** | **66.53** | **79.76** | **85.07** | **87.05** |

\* The 0.2 GB average covers the 12 completed tasks; `qa_1` is missing.

## Result Artifacts

- Baseline and 1 GB: `evaluation/results_ruler32k_qwen`
- 0.2, 0.4, 0.6, and 0.8 GB: `evaluation/results_new`
- Total completed result sets: 77
- Total sample evaluations: 7,700
