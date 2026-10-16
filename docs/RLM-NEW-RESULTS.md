# RLM(new)

Results from the September 5–7, 2026 campaign after source commit
`c448c971e361d42c6628a607cfb551187cc8767d` (retrieval bookkeeping cleanup and zero-overlap window fix).

## Setup

- **Root and sub-model:** `Qwen/Qwen3-4B-Instruct-2507`.
- **Benchmark:** LOFT, `nq_128k`, `hotpotqa_128k`, and `musique_128k`, test split.
- **Samples:** 55 per configuration; all 27 runnable configurations produced metrics (1,485 scored predictions).
- **Method:** RLM with a 1,024-token scratchpad and fixed sub-model chunks; `search_k=0`.
- **Chunk factors:** 1× uses `no_press`; 2×, 4×, and 8× use KVzip. Chunk tokens = retained-token budget × factor.
- **Logical KV budgets:** decimal 2, 4, and 8 GB, converted at 147,456 bytes per token.

| Logical budget | Retained tokens | Runnable chunk factors |
|---|---:|---|
| 2 GB | 13,563 | 1×, 2×, 4×, 8× |
| 4 GB | 27,126 | 1×, 2×, 4× |
| 8 GB | 54,253 | 1×, 2× |

KVzip masks evicted KV entries without freeing the full chunk cache. These budgets
describe simulated KV retention, not measured GPU-memory consumption or savings.

## Results

The primary metric is **Subspan EM (%)**, with 55 examples in every cell.
The [CSV](data/RLM-new-loft128k.csv) also includes EM and F1 as **0–1 fractions**,
configuration details, and per-cell audit counts.

| Dataset | Logical KV budget | 1× (uncompressed) | 2× | 4× | 8× |
|---|---:|---:|---:|---:|---:|
| nq_128k | 2 GB | 41.8 | 40.0 | 41.8 | 16.4† |
| nq_128k | 4 GB | 40.0 | 40.0 | 21.8† | — |
| nq_128k | 8 GB | 34.5† | 29.1† | — | — |
| hotpotqa_128k | 2 GB | 20.0 | 29.1 | 20.0 | 20.0† |
| hotpotqa_128k | 4 GB | 27.3 | 27.3 | 29.1† | — |
| hotpotqa_128k | 8 GB | 23.6 | 32.7 | — | — |
| musique_128k | 2 GB | 1.8 | 3.6 | 1.8 | 0.0† |
| musique_128k | 4 GB | 3.6 | 1.8 | 1.8† | — |
| musique_128k | 8 GB | 1.8† | 0.0† | — | — |

**†** At least one transcript contains a GPU-fit error; the score includes those outcomes.
**—** The requested chunk exceeds the corpus length. At 8 GB × 8 it also exceeds the model window.
No smaller replacement chunks or padded configurations are reported.

## Run audit and interpretation

- GPU-fit errors occurred in **16 example-runs across 10 configurations**. These sub-model failures are visible in transcripts even though the top-level error count is zero.
- **14 predictions were marked unfinished**; they remain included in the reported scores. Per-configuration counts are in the CSV.
- Search was disabled, so this campaign does not evaluate the zero-overlap retrieval fix or establish a before/after improvement from that fix.
- These results use a 4B model and 55 examples per cell. Comparisons with other benchmark campaigns require matching the model, sample set, metric, and budget semantics.
- Workers were redistributed across hosts during the campaign. Timing is omitted because host load and root/sub-model placement changed.

The CSV was exported from the completed runs’ `metrics.json` and `config.yaml` files.
Fit-error counts come from transcripts containing `did not fit in GPU memory`.
Source campaign: `cleanup-loft128k-gb55-c448c97`; snapshot: September 7, 2026.
