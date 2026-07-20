# RULER64K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `cwe`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `0.5`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `cwe` | {'string_match': 25.16} |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 65229.500000 |
| `average_retained_context_tokens` | 64576.716000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 9522.224234 |
| `average_retained_kv_memory_gb` | 9.522224 |
| `average_uncompressed_kv_memory_mb` | 9618.481152 |
| `average_uncompressed_kv_memory_gb` | 9.618481 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
