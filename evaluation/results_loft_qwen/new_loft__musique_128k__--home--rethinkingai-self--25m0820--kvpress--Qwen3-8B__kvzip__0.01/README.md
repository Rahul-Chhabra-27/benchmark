# LOFT Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `musique_128k`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `em` | 0.090909 |
| `subspan_em` | 0.109091 |
| `f1` | 0.167508 |
| `num_samples` | 110 |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 119446.000000 |
| `average_retained_context_tokens` | 118251.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 17436.819456 |
| `average_retained_kv_memory_gb` | 17.436819 |
| `average_uncompressed_kv_memory_mb` | 17613.029376 |
| `average_uncompressed_kv_memory_gb` | 17.613029 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
