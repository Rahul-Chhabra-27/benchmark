# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `niah_multikey_3`
- Configuration: KVzip memory budget: 0.8 GB
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `niah_multikey_3` | {'string_match': 26.0} |
| `average_compression_ratio` | 0.870404 |
| `average_original_context_tokens` | 41860.825000 |
| `average_retained_context_tokens` | 5425.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 799.948800 |
| `average_retained_kv_memory_gb` | 0.799949 |
| `average_uncompressed_kv_memory_mb` | 6172.629811 |
| `average_uncompressed_kv_memory_gb` | 6.172630 |
| `memory_budget` | 0.800000 |
| `memory_budget_unit` | GB |
| `token_budget` | 5425 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
