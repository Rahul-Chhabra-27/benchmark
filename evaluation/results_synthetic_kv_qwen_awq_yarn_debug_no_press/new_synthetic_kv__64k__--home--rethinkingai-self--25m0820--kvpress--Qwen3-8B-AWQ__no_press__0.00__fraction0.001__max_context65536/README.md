# SYNTHETIC_KV Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B-AWQ`
- Task: `64k`
- Configuration: True no-press baseline (full KV cache)
- Press: `no_press`
- Dataset fraction: `0.0005`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `synthetic_kv_64k` | {'exact_match': 0.0, 'string_match': 0.0, 'num_samples': 1} |
| `average_compression_ratio` | 0.000000 |
| `average_original_context_tokens` | 62998.000000 |
| `average_retained_context_tokens` | 62998.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 9289.433088 |
| `average_retained_kv_memory_gb` | 9.289433 |
| `average_uncompressed_kv_memory_mb` | 9289.433088 |
| `average_uncompressed_kv_memory_gb` | 9.289433 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
