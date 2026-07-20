# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `niah_multikey_3`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `0.5`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `niah_multikey_3` | {'string_match': 77.0} |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 32768.000000 |
| `average_retained_context_tokens` | 32440.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 4783.472640 |
| `average_retained_kv_memory_gb` | 4.783473 |
| `average_uncompressed_kv_memory_mb` | 4831.838208 |
| `average_uncompressed_kv_memory_gb` | 4.831838 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
