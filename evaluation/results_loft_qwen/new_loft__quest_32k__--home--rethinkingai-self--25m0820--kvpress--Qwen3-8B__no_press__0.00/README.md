# LOFT Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `quest_32k`
- Configuration: True no-press baseline (full KV cache)
- Press: `no_press`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `em` | 0.085714 |
| `subspan_em` | 0.242857 |
| `coverage` | 0.335714 |
| `num_samples` | 70 |
| `average_compression_ratio` | 0.000000 |
| `average_original_context_tokens` | 44399.000000 |
| `average_retained_context_tokens` | 44399.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 6546.898944 |
| `average_retained_kv_memory_gb` | 6.546899 |
| `average_uncompressed_kv_memory_mb` | 6546.898944 |
| `average_uncompressed_kv_memory_gb` | 6.546899 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
