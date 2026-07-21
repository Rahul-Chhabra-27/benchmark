# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `niah_multivalue`
- Configuration: KVzip memory budget: 0.6 GB
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `niah_multivalue` | {'string_match': 80.12} |
| `average_compression_ratio` | 0.874365 |
| `average_original_context_tokens` | 32387.480000 |
| `average_retained_context_tokens` | 4069.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 599.998464 |
| `average_retained_kv_memory_gb` | 0.599998 |
| `average_uncompressed_kv_memory_mb` | 4775.728251 |
| `average_uncompressed_kv_memory_gb` | 4.775728 |
| `memory_budget` | 0.600000 |
| `memory_budget_unit` | GB |
| `token_budget` | 4069 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
