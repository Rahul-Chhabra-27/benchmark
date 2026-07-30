# SYNTHETIC_KV Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B-AWQ`
- Task: `64k`
- Configuration: True no-press baseline (full KV cache)
- Press: `no_press`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `synthetic_kv_64k` | {'exact_match': 11.84, 'string_match': 12.01, 'num_samples': 2340} |
| `average_compression_ratio` | 0.000000 |
| `average_original_context_tokens` | 58477.000000 |
| `average_retained_context_tokens` | 58477.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 8622.784512 |
| `average_retained_kv_memory_gb` | 8.622785 |
| `average_uncompressed_kv_memory_mb` | 8622.784512 |
| `average_uncompressed_kv_memory_gb` | 8.622785 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
