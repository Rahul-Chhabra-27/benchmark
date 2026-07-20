# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `vt`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `0.5`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `vt` | {'string_match': 100.0} |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 32601.170000 |
| `average_retained_context_tokens` | 32274.620000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 4759.086367 |
| `average_retained_kv_memory_gb` | 4.759086 |
| `average_uncompressed_kv_memory_mb` | 4807.238124 |
| `average_uncompressed_kv_memory_gb` | 4.807238 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
