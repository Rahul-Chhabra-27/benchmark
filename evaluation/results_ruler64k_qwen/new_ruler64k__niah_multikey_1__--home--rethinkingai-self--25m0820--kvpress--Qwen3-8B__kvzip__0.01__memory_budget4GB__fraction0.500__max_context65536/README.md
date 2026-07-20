# RULER64K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `niah_multikey_1`
- Configuration: KVzip memory budget: 4 GB
- Press: `kvzip`
- Dataset fraction: `0.5`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `niah_multikey_1` | {'string_match': 100.0} |
| `average_compression_ratio` | 0.000000 |
| `average_original_context_tokens` | 121.112000 |
| `average_retained_context_tokens` | 121.112000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 17.858691 |
| `average_retained_kv_memory_gb` | 0.017859 |
| `average_uncompressed_kv_memory_mb` | 17.858691 |
| `average_uncompressed_kv_memory_gb` | 0.017859 |
| `memory_budget` | 4.000000 |
| `memory_budget_unit` | GB |
| `token_budget` | 27126 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
