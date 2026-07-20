# LOFT Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `quest_128k`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `em` | 0.000000 |
| `subspan_em` | 0.018182 |
| `coverage` | 0.065152 |
| `num_samples` | 110 |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 117977.000000 |
| `average_retained_context_tokens` | 116797.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 17222.418432 |
| `average_retained_kv_memory_gb` | 17.222418 |
| `average_uncompressed_kv_memory_mb` | 17396.416512 |
| `average_uncompressed_kv_memory_gb` | 17.396417 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
