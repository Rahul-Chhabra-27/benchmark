# SYNTHETIC_KV Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `64k`
- Configuration: KVzip memory budget: 1024 MB
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `synthetic_kv_64k` | {'exact_match': 2.86, 'string_match': 2.91, 'num_samples': 2340} |
| `average_compression_ratio` | 0.881252 |
| `average_original_context_tokens` | 58477.000000 |
| `average_retained_context_tokens` | 6944.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 1023.934464 |
| `average_retained_kv_memory_gb` | 1.023934 |
| `average_uncompressed_kv_memory_mb` | 8622.784512 |
| `average_uncompressed_kv_memory_gb` | 8.622785 |
| `memory_budget` | 1024.000000 |
| `memory_budget_unit` | MB |
| `token_budget` | 6944 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
