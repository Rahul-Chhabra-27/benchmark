# LOFT Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `hotpotqa_128k`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `em` | 0.318182 |
| `subspan_em` | 0.354545 |
| `f1` | 0.400808 |
| `num_samples` | 110 |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 121232.000000 |
| `average_retained_context_tokens` | 120019.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 17697.521664 |
| `average_retained_kv_memory_gb` | 17.697522 |
| `average_uncompressed_kv_memory_mb` | 17876.385792 |
| `average_uncompressed_kv_memory_gb` | 17.876386 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
