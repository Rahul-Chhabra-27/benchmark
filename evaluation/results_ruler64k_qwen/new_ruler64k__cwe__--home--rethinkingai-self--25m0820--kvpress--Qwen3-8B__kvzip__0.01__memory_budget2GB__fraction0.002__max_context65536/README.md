# RULER64K Benchmark Result

- Model: `/home/rethinkingai-self/25m0820/kvpress/Qwen3-8B`
- Task: `cwe`
- Configuration: KVzip memory budget: 2 GB
- Press: `kvzip`
- Dataset fraction: `0.002`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
| `cwe` | {'string_match': 60.0} |
| `average_compression_ratio` | 0.792402 |
| `average_original_context_tokens` | 65333.000000 |
| `average_retained_context_tokens` | 13563.000000 |
| `kv_memory_per_token_kb` | 147.456000 |
| `average_retained_kv_memory_mb` | 1999.945728 |
| `average_retained_kv_memory_gb` | 1.999946 |
| `average_uncompressed_kv_memory_mb` | 9633.742848 |
| `average_uncompressed_kv_memory_gb` | 9.633743 |
| `memory_budget` | 2.000000 |
| `memory_budget_unit` | GB |
| `token_budget` | 13563 |

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
