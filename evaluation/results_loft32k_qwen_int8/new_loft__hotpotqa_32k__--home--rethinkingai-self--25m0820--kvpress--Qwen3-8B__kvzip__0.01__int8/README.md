# LOFT Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `hotpotqa_32k`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `em` | 0.418182 |
| `subspan_em` | 0.490909 |
| `f1` | 0.489242 |
| `num_samples` | 110 |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 49679.000000 |
| `average_retained_context_tokens` | 49182.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 7252.180992 |
| `average_retained_kv_memory_gb` | 7.252181 |
| `average_uncompressed_kv_memory_mb` | 7325.466624 |
| `average_uncompressed_kv_memory_gb` | 7.325467 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
