# LOFT Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `hotpotqa_128k`
- Configuration: KVzip memory budget: 2 GB
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `em` | 0.309091 |
| `subspan_em` | 0.363636 |
| `f1` | 0.411320 |
| `num_samples` | 110 |
| `average_compression_ratio` | 0.888124 |
| `average_original_context_tokens` | 121232.000000 |
| `average_retained_context_tokens` | 13563.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 1999.945728 |
| `average_retained_kv_memory_gb` | 1.999946 |
| `average_uncompressed_kv_memory_mb` | 17876.385792 |
| `average_uncompressed_kv_memory_gb` | 17.876386 |
| `memory_budget` | 2.000000 |
| `memory_budget_unit` | GB |
| `token_budget` | 13563 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
