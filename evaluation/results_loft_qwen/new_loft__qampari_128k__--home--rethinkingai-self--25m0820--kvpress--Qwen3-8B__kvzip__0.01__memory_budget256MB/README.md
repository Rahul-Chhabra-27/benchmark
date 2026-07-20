# LOFT Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `qampari_128k`
- Configuration: KVzip memory budget: 256 MB
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `em` | 0.000000 |
| `subspan_em` | 0.000000 |
| `coverage` | 0.000000 |
| `num_samples` | 110 |
| `average_compression_ratio` | 0.985556 |
| `average_original_context_tokens` | 120192.000000 |
| `average_retained_context_tokens` | 1736.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 255.983616 |
| `average_retained_kv_memory_gb` | 0.255984 |
| `average_uncompressed_kv_memory_mb` | 17723.031552 |
| `average_uncompressed_kv_memory_gb` | 17.723032 |
| `memory_budget` | 256 |
| `memory_budget_unit` | MB |
| `token_budget` | 1736 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
