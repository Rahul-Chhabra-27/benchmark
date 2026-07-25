# LOFT Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `musique_32k`
- Configuration: KVzip memory budget: 4 GB
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `em` | 0.045455 |
| `subspan_em` | 0.063636 |
| `f1` | 0.048988 |
| `num_samples` | 110 |
| `average_compression_ratio` | 0.172180 |
| `average_original_context_tokens` | 32768.000000 |
| `average_retained_context_tokens` | 27126.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 3999.891456 |
| `average_retained_kv_memory_gb` | 3.999891 |
| `average_uncompressed_kv_memory_mb` | 4831.838208 |
| `average_uncompressed_kv_memory_gb` | 4.831838 |
| `memory_budget` | 4.000000 |
| `memory_budget_unit` | GB |
| `token_budget` | 27126 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
