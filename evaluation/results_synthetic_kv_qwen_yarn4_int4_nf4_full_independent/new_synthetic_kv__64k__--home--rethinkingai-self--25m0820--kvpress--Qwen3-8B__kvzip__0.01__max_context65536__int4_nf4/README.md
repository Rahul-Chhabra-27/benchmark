# SYNTHETIC_KV Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `64k`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `synthetic_kv_64k` | {'exact_match': 1.71, 'string_match': 2.09, 'num_samples': 2340} |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 62998.000000 |
| `average_retained_context_tokens` | 62368.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 9196.535808 |
| `average_retained_kv_memory_gb` | 9.196536 |
| `average_uncompressed_kv_memory_mb` | 9289.433088 |
| `average_uncompressed_kv_memory_gb` | 9.289433 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
