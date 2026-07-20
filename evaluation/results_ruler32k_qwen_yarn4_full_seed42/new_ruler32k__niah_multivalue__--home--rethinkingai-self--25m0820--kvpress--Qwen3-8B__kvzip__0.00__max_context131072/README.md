# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `niah_multivalue`
- Configuration: KVzip baseline (compression ratio 0.0000)
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `niah_multivalue` | {'string_match': 95.88} |
| `average_compression_ratio` | 0.000000 |
| `average_original_context_tokens` | 32387.480000 |
| `average_retained_context_tokens` | 32387.480000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 4775.728251 |
| `average_retained_kv_memory_gb` | 4.775728 |
| `average_uncompressed_kv_memory_mb` | 4775.728251 |
| `average_uncompressed_kv_memory_gb` | 4.775728 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
