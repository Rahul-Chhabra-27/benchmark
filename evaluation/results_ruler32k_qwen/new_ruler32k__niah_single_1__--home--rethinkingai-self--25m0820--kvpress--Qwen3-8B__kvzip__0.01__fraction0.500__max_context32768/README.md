# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `niah_single_1`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `0.5`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `niah_single_1` | {'string_match': 100.0} |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 32451.960000 |
| `average_retained_context_tokens` | 32126.960000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 4737.313014 |
| `average_retained_kv_memory_gb` | 4.737313 |
| `average_uncompressed_kv_memory_mb` | 4785.236214 |
| `average_uncompressed_kv_memory_gb` | 4.785236 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
