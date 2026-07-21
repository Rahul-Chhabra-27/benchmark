# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `niah_multikey_3`
- Configuration: KVzip baseline (compression ratio 0.0000)
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `niah_multikey_3` | {'string_match': 54.5} |
| `average_compression_ratio` | 0.000000 |
| `average_original_context_tokens` | 41860.825000 |
| `average_retained_context_tokens` | 41860.825000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 6172.629811 |
| `average_retained_kv_memory_gb` | 6.172630 |
| `average_uncompressed_kv_memory_mb` | 6172.629811 |
| `average_uncompressed_kv_memory_gb` | 6.172630 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
