# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `niah_multiquery`
- Configuration: KVzip baseline (compression ratio 0.0000)
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `niah_multiquery` | {'string_match': 99.38} |
| `average_compression_ratio` | 0.000000 |
| `average_original_context_tokens` | 32387.195000 |
| `average_retained_context_tokens` | 32387.195000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 4775.686226 |
| `average_retained_kv_memory_gb` | 4.775686 |
| `average_uncompressed_kv_memory_mb` | 4775.686226 |
| `average_uncompressed_kv_memory_gb` | 4.775686 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
