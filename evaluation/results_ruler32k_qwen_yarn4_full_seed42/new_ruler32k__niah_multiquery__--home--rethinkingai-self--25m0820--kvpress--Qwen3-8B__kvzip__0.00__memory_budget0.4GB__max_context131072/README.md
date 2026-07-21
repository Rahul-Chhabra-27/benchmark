# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `niah_multiquery`
- Configuration: KVzip memory budget: 0.4 GB
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `niah_multiquery` | {'string_match': 83.62} |
| `average_compression_ratio` | 0.916263 |
| `average_original_context_tokens` | 32387.195000 |
| `average_retained_context_tokens` | 2712.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 399.900672 |
| `average_retained_kv_memory_gb` | 0.399901 |
| `average_uncompressed_kv_memory_mb` | 4775.686226 |
| `average_uncompressed_kv_memory_gb` | 4.775686 |
| `memory_budget` | 0.400000 |
| `memory_budget_unit` | GB |
| `token_budget` | 2712 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
