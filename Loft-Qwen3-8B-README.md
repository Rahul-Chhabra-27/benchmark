# Qwen3-8B LOFT Combined Benchmark Results

This document combines the Qwen3-8B KVzip results for LOFT 32K and the
post-YaRN-4 LOFT 128K evaluation. Results are grouped into a separate table for
each KV-cache memory budget.

## Configuration

- Model: Qwen3-8B
- Press: KVzip
- Tasks: NQ, HotPotQA, MuSiQue, Qampari, and Quest
- Context suites: LOFT 32K and LOFT 128K with YaRN-4
- 128K maximum context length: 131,072 tokens
- Budgets: baseline, 256 MB, 512 MB, 1 GB, and 2 GB
- Baseline compression ratio: 0.01
- F1 is reported for NQ, HotPotQA, and MuSiQue; coverage is reported for
  Qampari and Quest.
- Average rows are weighted by the number of evaluated samples. Averages in
  tables with pending rows cover completed rows only.

## Baseline cache summary

| Context suite | Samples | Avg. context tokens | Avg. compression | Avg. retained KV cache | Avg. original KV cache |
|---|---:|---:|---:|---:|---:|
| LOFT 32K | 470 | 47,218 | 0.0100 | 6.893 GB | 6.963 GB |
| LOFT 128K YaRN-4 | 550 | 119,491 | 0.0100 | 17.443 GB | 17.620 GB |
| **Combined weighted average** | **1,020** | **86,189** | **0.0100** | **12.582 GB** | **12.709 GB** |

## Baseline

| Context | Task | EM | Subspan EM | F1/Coverage | Samples | Avg. context tokens | Avg. compression | Avg. retained KV | Avg. original KV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | `nq` | 0.4818 | 0.6455 | 0.5796 | 110 | 44,955 | 0.0100 | 6.563 GB | 6.629 GB |
| 32K | `hotpotqa` | 0.4545 | 0.5727 | 0.5762 | 110 | 49,679 | 0.0100 | 7.252 GB | 7.325 GB |
| 32K | `musique` | 0.2091 | 0.2455 | 0.3105 | 110 | 47,969 | 0.0100 | 7.003 GB | 7.073 GB |
| 32K | `qampari` | 0.2286 | 0.2857 | 0.5376 | 70 | 48,548 | 0.0100 | 7.087 GB | 7.159 GB |
| 32K | `quest` | 0.1429 | 0.3286 | 0.3905 | 70 | 44,399 | 0.0100 | 6.481 GB | 6.547 GB |
| 128K YaRN-4 | `nq` | 0.4000 | 0.5182 | 0.4930 | 110 | 118,609 | 0.0100 | 17.315 GB | 17.490 GB |
| 128K YaRN-4 | `hotpotqa` | 0.3182 | 0.3545 | 0.4008 | 110 | 121,232 | 0.0100 | 17.698 GB | 17.876 GB |
| 128K YaRN-4 | `musique` | 0.1364 | 0.1636 | 0.2182 | 110 | 119,446 | 0.0100 | 17.437 GB | 17.613 GB |
| 128K YaRN-4 | `qampari` | 0.0273 | 0.0364 | 0.3055 | 110 | 120,192 | 0.0100 | 17.546 GB | 17.723 GB |
| 128K YaRN-4 | `quest` | 0.0182 | 0.0455 | 0.1197 | 110 | 117,977 | 0.0100 | 17.222 GB | 17.396 GB |
| **Weighted average** | **Completed rows** | **0.2461** | **0.3206** | **0.3876** | **1,020** | **86,189** | **0.0100** | **12.582 GB** | **12.709 GB** |

## 256 MB

| Context | Task | EM | Subspan EM | F1/Coverage | Samples | Avg. context tokens | Avg. compression | Avg. retained KV | Avg. original KV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | `nq` | 0.0182 | 0.0273 | 0.0283 | 110 | 44,955 | 0.9614 | 255.984 MB | 6.629 GB |
| 32K | `hotpotqa` | 0.0545 | 0.0909 | 0.0866 | 110 | 49,679 | 0.9651 | 255.984 MB | 7.325 GB |
| 32K | `musique` | 0.0000 | 0.0000 | 0.0000 | 110 | 47,969 | 0.9638 | 255.984 MB | 7.073 GB |
| 32K | `qampari` | 0.0000 | 0.0000 | 0.0057 | 70 | 48,548 | 0.9642 | 255.984 MB | 7.159 GB |
| 32K | `quest` | 0.0000 | 0.0000 | 0.0000 | 70 | 44,399 | 0.9609 | 255.984 MB | 6.547 GB |
| 128K YaRN-4 | `nq` | 0.0000 | 0.0000 | 0.0000 | 110 | 118,609 | 0.9854 | 255.984 MB | 17.490 GB |
| 128K YaRN-4 | `hotpotqa` | 0.0000 | 0.0000 | 0.0009 | 110 | 121,232 | 0.9857 | 255.984 MB | 17.876 GB |
| 128K YaRN-4 | `musique` | 0.0000 | 0.0000 | 0.0000 | 110 | 119,446 | 0.9855 | 255.984 MB | 17.613 GB |
| 128K YaRN-4 | `qampari` | 0.0000 | 0.0000 | 0.0000 | 110 | 120,192 | 0.9856 | 255.984 MB | 17.723 GB |
| 128K YaRN-4 | `quest` | 0.0000 | 0.0000 | 0.0000 | 110 | 117,977 | 0.9853 | 255.984 MB | 17.396 GB |
| **Weighted average** | **Completed rows** | **0.0078** | **0.0127** | **0.0129** | **1,020** | **86,189** | **0.9752** | **255.984 MB** | **12.709 GB** |

## 512 MB

| Context | Task | EM | Subspan EM | F1/Coverage | Samples | Avg. context tokens | Avg. compression | Avg. retained KV | Avg. original KV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | `nq` | 0.0091 | 0.0545 | 0.0392 | 110 | 44,955 | 0.9228 | 511.967 MB | 6.629 GB |
| 32K | `hotpotqa` | 0.0545 | 0.1182 | 0.1307 | 110 | 49,679 | 0.9301 | 511.967 MB | 7.325 GB |
| 32K | `musique` | 0.0273 | 0.0364 | 0.0546 | 110 | 47,969 | 0.9276 | 511.967 MB | 7.073 GB |
| 32K | `qampari` | 0.0000 | 0.0000 | 0.0343 | 70 | 48,548 | 0.9285 | 511.967 MB | 7.159 GB |
| 32K | `quest` | 0.0000 | 0.0000 | 0.0048 | 70 | 44,399 | 0.9218 | 511.967 MB | 6.547 GB |
| 128K YaRN-4 | `nq` | 0.0000 | 0.0091 | 0.0030 | 110 | 118,609 | 0.9707 | 511.967 MB | 17.490 GB |
| 128K YaRN-4 | `hotpotqa` | 0.0000 | 0.0000 | 0.0098 | 110 | 121,232 | 0.9714 | 511.967 MB | 17.876 GB |
| 128K YaRN-4 | `musique` | 0.0000 | 0.0000 | 0.0000 | 110 | 119,446 | 0.9709 | 511.967 MB | 17.613 GB |
| 128K YaRN-4 | `qampari` | 0.0000 | 0.0000 | 0.0000 | 110 | 120,192 | 0.9711 | 511.967 MB | 17.723 GB |
| 128K YaRN-4 | `quest` | -- | -- | -- | -- | -- | -- | -- | -- |
| **Weighted average** | **Completed rows** | **0.0110** | **0.0264** | **0.0317** | **910** | **82,346** | **0.9479** | **511.967 MB** | **12.142 GB** |

## 1 GB

| Context | Task | EM | Subspan EM | F1/Coverage | Samples | Avg. context tokens | Avg. compression | Avg. retained KV | Avg. original KV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | `nq` | 0.4091 | 0.5364 | 0.4919 | 110 | 44,955 | 0.8492 | 999.899 MB | 6.629 GB |
| 32K | `hotpotqa` | 0.3818 | 0.4636 | 0.4659 | 110 | 49,679 | 0.8635 | 999.899 MB | 7.325 GB |
| 32K | `musique` | 0.2182 | 0.2455 | 0.3074 | 110 | 47,969 | 0.8586 | 999.899 MB | 7.073 GB |
| 32K | `qampari` | 0.1143 | 0.1286 | 0.4005 | 70 | 48,548 | 0.8603 | 999.899 MB | 7.159 GB |
| 32K | `quest` | 0.0286 | 0.1000 | 0.2048 | 70 | 44,399 | 0.8473 | 999.899 MB | 6.547 GB |
| 128K YaRN-4 | `nq` | 0.0091 | 0.0364 | 0.0306 | 110 | 118,609 | 0.9428 | 999.899 MB | 17.490 GB |
| 128K YaRN-4 | `hotpotqa` | 0.0182 | 0.0364 | 0.0407 | 110 | 121,232 | 0.9441 | 999.899 MB | 17.876 GB |
| 128K YaRN-4 | `musique` | 0.0000 | 0.0182 | 0.0150 | 110 | 119,446 | 0.9432 | 999.899 MB | 17.613 GB |
| 128K YaRN-4 | `qampari` | 0.0000 | 0.0000 | 0.0145 | 110 | 120,192 | 0.9436 | 999.899 MB | 17.723 GB |
| 128K YaRN-4 | `quest` | -- | -- | -- | -- | -- | -- | -- | -- |
| **Weighted average** | **Completed rows** | **0.1363** | **0.1791** | **0.2117** | **910** | **82,346** | **0.8983** | **999.899 MB** | **12.142 GB** |

## 2 GB

| Context | Task | EM | Subspan EM | F1/Coverage | Samples | Avg. context tokens | Avg. compression | Avg. retained KV | Avg. original KV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | `nq` | 0.4727 | 0.6273 | 0.5835 | 110 | 44,955 | 0.6983 | 2.000 GB | 6.629 GB |
| 32K | `hotpotqa` | 0.4727 | 0.6091 | 0.6176 | 110 | 49,679 | 0.7270 | 2.000 GB | 7.325 GB |
| 32K | `musique` | 0.2364 | 0.2727 | 0.3379 | 110 | 47,969 | 0.7173 | 2.000 GB | 7.073 GB |
| 32K | `qampari` | 0.1571 | 0.1714 | 0.5033 | 70 | 48,548 | 0.7206 | 2.000 GB | 7.159 GB |
| 32K | `quest` | 0.1429 | 0.2857 | 0.3643 | 70 | 44,399 | 0.6945 | 2.000 GB | 6.547 GB |
| 128K YaRN-4 | `nq` | 0.1455 | 0.2182 | 0.2003 | 110 | 118,609 | 0.8856 | 2.000 GB | 17.490 GB |
| 128K YaRN-4 | `hotpotqa` | 0.3091 | 0.3636 | 0.4113 | 110 | 121,232 | 0.8881 | 2.000 GB | 17.876 GB |
| 128K YaRN-4 | `musique` | 0.0636 | 0.0818 | 0.1525 | 110 | 119,446 | 0.8865 | 2.000 GB | 17.613 GB |
| 128K YaRN-4 | `qampari` | 0.0091 | 0.0091 | 0.1767 | 110 | 120,192 | 0.8872 | 2.000 GB | 17.723 GB |
| 128K YaRN-4 | `quest` | -- | -- | -- | -- | -- | -- | -- | -- |
| **Weighted average** | **Completed rows** | **0.2297** | **0.2989** | **0.3665** | **910** | **82,346** | **0.7966** | **2.000 GB** | **12.142 GB** |

## Result artifacts

- LOFT 32K: `evaluation/results_loft_qwen`
- LOFT 128K YaRN-4: `evaluation/results_loft128k_qwen_yarn4`
- `--` denotes a result that had not completed when this README was generated.
- INT8 results are excluded because that benchmark is still running in
  `evaluation/results_loft128k_qwen_yarn4_int8`.
