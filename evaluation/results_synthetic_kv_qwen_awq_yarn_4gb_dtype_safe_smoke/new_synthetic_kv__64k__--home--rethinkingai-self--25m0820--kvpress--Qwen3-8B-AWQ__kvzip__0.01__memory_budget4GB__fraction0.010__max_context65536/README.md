# SYNTHETIC_KV Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B-AWQ`
- Task: `64k`
- Configuration: KVzip memory budget: 4 GB
- Press: `kvzip`
- Dataset fraction: `0.01`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `synthetic_kv_64k` | {'exact_match': 4.35, 'string_match': 4.35, 'num_samples': 23} |
| `average_compression_ratio` | 0.569415 |
| `average_original_context_tokens` | 62998.000000 |
| `average_retained_context_tokens` | 27126.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 3999.891456 |
| `average_retained_kv_memory_gb` | 3.999891 |
| `average_uncompressed_kv_memory_mb` | 9289.433088 |
| `average_uncompressed_kv_memory_gb` | 9.289433 |
| `memory_budget` | 4 |
| `memory_budget_unit` | GB |
| `token_budget` | 27126 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
