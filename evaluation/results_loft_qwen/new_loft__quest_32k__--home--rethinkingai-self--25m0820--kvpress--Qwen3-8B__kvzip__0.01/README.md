# LOFT Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `quest_32k`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `em` | 0.142857 |
| `subspan_em` | 0.328571 |
| `coverage` | 0.390476 |
| `num_samples` | 70 |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 44399.000000 |
| `average_retained_context_tokens` | 43955.000000 |
| `kv_memory_per_token_kb` | 147.456 |
| `average_retained_kv_memory_mb` | 6481.428480 |
| `average_retained_kv_memory_gb` | 6.481428 |
| `average_uncompressed_kv_memory_mb` | 6546.898944 |
| `average_uncompressed_kv_memory_gb` | 6.546899 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
