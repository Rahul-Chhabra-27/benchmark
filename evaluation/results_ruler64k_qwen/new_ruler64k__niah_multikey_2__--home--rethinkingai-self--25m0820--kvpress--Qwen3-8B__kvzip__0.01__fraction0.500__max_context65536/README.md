# RULER64K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `niah_multikey_2`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `0.5`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `niah_multikey_2` | {'string_match': 86.8} |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 65119.588000 |
| `average_retained_context_tokens` | 64467.896000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 9506.178073 |
| `average_retained_kv_memory_gb` | 9.506178 |
| `average_uncompressed_kv_memory_mb` | 9602.273968 |
| `average_uncompressed_kv_memory_gb` | 9.602274 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
