# LOFT Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `qampari_32k`
- Configuration: KVzip memory budget: 4 GB
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `em` | 0.114286 |
| `subspan_em` | 0.114286 |
| `coverage` | 0.469048 |
| `num_samples` | 70 |
| `average_compression_ratio` | 0.441254 |
| `average_original_context_tokens` | 48548.000000 |
| `average_retained_context_tokens` | 27126.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 3999.891456 |
| `average_retained_kv_memory_gb` | 3.999891 |
| `average_uncompressed_kv_memory_mb` | 7158.693888 |
| `average_uncompressed_kv_memory_gb` | 7.158694 |
| `memory_budget` | 4.000000 |
| `memory_budget_unit` | GB |
| `token_budget` | 27126 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
