# LOFT Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `nq_32k`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `em` | 0.481818 |
| `subspan_em` | 0.645455 |
| `f1` | 0.579583 |
| `num_samples` | 110 |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 44955.000000 |
| `average_retained_context_tokens` | 44505.000000 |
| `kv_memory_per_token_kb` | 147.456 |
| `average_retained_kv_memory_mb` | 6562.529280 |
| `average_retained_kv_memory_gb` | 6.562529 |
| `average_uncompressed_kv_memory_mb` | 6628.884480 |
| `average_uncompressed_kv_memory_gb` | 6.628884 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
