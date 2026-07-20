# LOFT Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `hotpotqa_32k`
- Configuration: KVzip memory budget: 512 MB
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `em` | 0.054545 |
| `subspan_em` | 0.118182 |
| `f1` | 0.130745 |
| `num_samples` | 110 |
| `average_compression_ratio` | 0.930111 |
| `average_original_context_tokens` | 49679.000000 |
| `average_retained_context_tokens` | 3472.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 511.967232 |
| `average_retained_kv_memory_gb` | 0.511967 |
| `average_uncompressed_kv_memory_mb` | 7325.466624 |
| `average_uncompressed_kv_memory_gb` | 7.325467 |
| `memory_budget` | 512.000000 |
| `memory_budget_unit` | MB |
| `token_budget` | 3472 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
