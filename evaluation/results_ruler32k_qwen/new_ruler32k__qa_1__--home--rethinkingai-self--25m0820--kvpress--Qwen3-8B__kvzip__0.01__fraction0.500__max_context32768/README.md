# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `qa_1`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `0.5`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `qa_1` | {'string_match': 72.0} |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 32188.400000 |
| `average_retained_context_tokens` | 31866.080000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 4698.844692 |
| `average_retained_kv_memory_gb` | 4.698845 |
| `average_uncompressed_kv_memory_mb` | 4746.372710 |
| `average_uncompressed_kv_memory_gb` | 4.746373 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
