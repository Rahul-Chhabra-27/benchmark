# RULER32K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `qa_2`
- Configuration: KVzip memory budget: 0.6 GB
- Press: `kvzip`
- Dataset fraction: `1.0`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `qa_2` | {'string_match': 45.0} |
| `average_compression_ratio` | 0.876890 |
| `average_original_context_tokens` | 33080.030000 |
| `average_retained_context_tokens` | 4069.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 599.998464 |
| `average_retained_kv_memory_gb` | 0.599998 |
| `average_uncompressed_kv_memory_mb` | 4877.848904 |
| `average_uncompressed_kv_memory_gb` | 4.877849 |
| `memory_budget` | 0.600000 |
| `memory_budget_unit` | GB |
| `token_budget` | 4069 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
