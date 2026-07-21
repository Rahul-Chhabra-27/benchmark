# LOFT Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `musique_32k`
- Configuration: KVzip memory budget: 1 GB
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `em` | 0.181818 |
| `subspan_em` | 0.200000 |
| `f1` | 0.268713 |
| `num_samples` | 110 |
| `average_compression_ratio` | 0.858638 |
| `average_original_context_tokens` | 47969.000000 |
| `average_retained_context_tokens` | 6781.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 999.899136 |
| `average_retained_kv_memory_gb` | 0.999899 |
| `average_uncompressed_kv_memory_mb` | 7073.316864 |
| `average_uncompressed_kv_memory_gb` | 7.073317 |
| `memory_budget` | 1.000000 |
| `memory_budget_unit` | GB |
| `token_budget` | 6781 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
