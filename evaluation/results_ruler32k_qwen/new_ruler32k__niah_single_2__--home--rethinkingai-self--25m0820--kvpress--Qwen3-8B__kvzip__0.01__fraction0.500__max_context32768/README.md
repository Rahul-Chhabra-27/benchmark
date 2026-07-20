# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `niah_single_2`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `0.5`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `niah_single_2` | {'string_match': 100.0} |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 32321.880000 |
| `average_retained_context_tokens` | 31997.880000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 4718.279393 |
| `average_retained_kv_memory_gb` | 4.718279 |
| `average_uncompressed_kv_memory_mb` | 4766.055137 |
| `average_uncompressed_kv_memory_gb` | 4.766055 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
