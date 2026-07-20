# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `fwe`
- Configuration: KVzip baseline (compression ratio 0.0100)
- Press: `kvzip`
- Dataset fraction: `0.5`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `fwe` | {'string_match': 95.33} |
| `average_compression_ratio` | 0.010000 |
| `average_original_context_tokens` | 31471.940000 |
| `average_retained_context_tokens` | 31156.680000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 4594.239406 |
| `average_retained_kv_memory_gb` | 4.594239 |
| `average_uncompressed_kv_memory_mb` | 4640.726385 |
| `average_uncompressed_kv_memory_gb` | 4.640726 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
