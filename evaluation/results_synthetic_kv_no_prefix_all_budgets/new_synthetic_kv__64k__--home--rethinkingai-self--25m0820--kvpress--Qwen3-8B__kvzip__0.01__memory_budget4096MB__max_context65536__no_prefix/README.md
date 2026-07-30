# SYNTHETIC_KV Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `64k`
- Configuration: KVzip memory budget: 4096 MB
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `synthetic_kv_64k` | {'exact_match': 22.99, 'string_match': 22.99, 'num_samples': 2340} |
| `average_compression_ratio` | 0.524993 |
| `average_original_context_tokens` | 58477.000000 |
| `average_retained_context_tokens` | 27777.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 4095.885312 |
| `average_retained_kv_memory_gb` | 4.095885 |
| `average_uncompressed_kv_memory_mb` | 8622.784512 |
| `average_uncompressed_kv_memory_gb` | 8.622785 |
| `memory_budget` | 4096.000000 |
| `memory_budget_unit` | MB |
| `token_budget` | 27777 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
