# SYNTHETIC_KV Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `64k`
- Configuration: KVzip memory budget: 2 GB
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `synthetic_kv_64k` | {'exact_match': 9.79, 'string_match': 10.04, 'num_samples': 2340} |
| `average_compression_ratio` | 0.784707 |
| `average_original_context_tokens` | 62998.000000 |
| `average_retained_context_tokens` | 13563.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 1999.945728 |
| `average_retained_kv_memory_gb` | 1.999946 |
| `average_uncompressed_kv_memory_mb` | 9289.433088 |
| `average_uncompressed_kv_memory_gb` | 9.289433 |
| `memory_budget` | 2.000000 |
| `memory_budget_unit` | GB |
| `token_budget` | 13563 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
