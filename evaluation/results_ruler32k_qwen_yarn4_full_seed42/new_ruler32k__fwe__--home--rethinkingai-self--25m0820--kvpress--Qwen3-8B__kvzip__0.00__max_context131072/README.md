# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `fwe`
- Configuration: KVzip baseline (compression ratio 0.0000)
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `fwe` | {'string_match': 91.67} |
| `average_compression_ratio` | 0.000000 |
| `average_original_context_tokens` | 31386.805000 |
| `average_retained_context_tokens` | 31386.805000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 4628.172718 |
| `average_retained_kv_memory_gb` | 4.628173 |
| `average_uncompressed_kv_memory_mb` | 4628.172718 |
| `average_uncompressed_kv_memory_gb` | 4.628173 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
