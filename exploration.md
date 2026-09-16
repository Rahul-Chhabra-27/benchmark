# RLM exploration — LOFT-128k (n=20)

LOFT test split, first 20 rows of each RAG subset. Headline metric is `subspan_em` for nq / hotpotqa / musique and `coverage` for qampari / quest.

**Every number here has a denominator of 20.** An errored row gets no credit. The scorer's `metrics.json` averages over `runtime.scored`, which silently drops errored rows, so baseline numbers below are `metric × scored / 20`. Our RLM cells all have `scored: 20, errors: 0` and are unchanged.

Two systems are compared:

- **ours** — `python -m evaluation.rlm.run_benchmark --mode rlm`, thinking ON, `--max-steps 30`. Three chunk-order arms: `none` (search REPL), `bm25-graph` (query-agnostic), `bm25-query` (query-aware).
- **base** — `baselines/run_benchmark.py` → recursive-llm (`max_depth=1`), thinking ON, `--max-iterations 30`, client peels `</think>` out of vLLM `content` before FINAL/REPL.

Both hit the same vLLM servers: 4B GPU 0 `:8012`, 9B GPU 4 `:8014`, 27B GPU 3 `:8013`, `max-model-len` 32768, `--language-model-only`. Gemini ran through the API.

---

## Headline — all arms

Ours is **45/45**. Campaign finished; vLLM servers were left running.

| subset | metric | gemini-3.6-flash | gemini-3.5-flash-lite | ours 4B/none | ours 9B/none | ours 27B/none | ours 4B/graph | ours 9B/graph | ours 27B/graph | ours 4B/query | ours 9B/query | ours 27B/query | base 4B | base 9B | base 27B |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nq_128k | subspan_em | 0.900 | 0.850 | 0.750 | 0.800 | 0.800 | 0.750 | 0.750 | 0.900 | 0.650 | 0.800 | 0.850 | 0.050 | 0.300 | 0.900 |
| hotpotqa_128k | subspan_em | 0.850 | 0.900 | 0.600 | 0.800 | 0.900 | 0.700 | 0.800 | 0.900 | 0.700 | 0.850 | 0.850 | 0.200 | 0.650 | 0.850 |
| musique_128k | subspan_em | 0.700 | 0.550 | 0.300 | 0.300 | 0.400 | 0.100 | 0.350 | 0.300 | 0.200 | 0.450 | 0.350 | 0.000 | 0.200 | 0.350 |
| qampari_128k | coverage | 0.282 | 0.080 | 0.050 | 0.063 | 0.250 | 0.230 | 0.300 | 0.543 | 0.320 | 0.460 | 0.522 | 0.010 | 0.000 | 0.010 |
| quest_128k | coverage | 0.258 | 0.100 | 0.033 | 0.100 | 0.333 | 0.175 | 0.250 | 0.217 | 0.217 | 0.400 | 0.333 | 0.000 | 0.000 | 0.000 |

At matched size, ours beats the baseline by a wide margin at 4B and 9B — nq 0.750 vs 0.050 and 0.800 vs 0.300 — because the baseline exhausts its 30 iterations without emitting `FINAL` and those rows score zero. Only baseline 27B is competitive, and it is the one size that finished most of its rows.

**Caveat on the `none` column for qampari / quest.** The `none` cells ran without `answer_list_format`; both bm25 arms have `answer_list_format: auto` in `config.yaml`. The apparent ordering gain on those two subsets is confounded with that prompt change and is not yet a clean ordering result. The three QA subsets are unaffected.

---

## Ours — `evaluation.rlm`, `--mode rlm`

`evaluation/results/rlm/loft128k-qwen35-n20-order-grid/`

### chunk_order = none

| subset | metric | Qwen3.5-4B | Qwen3.5-9B | Qwen3.5-27B |
|---|---|---:|---:|---:|
| nq_128k | subspan_em | 0.750 | 0.800 | 0.800 |
| hotpotqa_128k | subspan_em | 0.600 | 0.800 | 0.900 |
| musique_128k | subspan_em | 0.300 | 0.300 | 0.400 |
| qampari_128k | coverage | 0.050 | 0.063 | 0.250 |
| quest_128k | coverage | 0.033 | 0.100 | 0.333 |

### chunk_order = bm25-graph

| subset | metric | Qwen3.5-4B | Qwen3.5-9B | Qwen3.5-27B |
|---|---|---:|---:|---:|
| nq_128k | subspan_em | 0.750 | 0.750 | 0.900 |
| hotpotqa_128k | subspan_em | 0.700 | 0.800 | 0.900 |
| musique_128k | subspan_em | 0.100 | 0.350 | 0.300 |
| qampari_128k | coverage | 0.230 | 0.300 | 0.543 |
| quest_128k | coverage | 0.175 | 0.250 | 0.217 |

### chunk_order = bm25-query

| subset | metric | Qwen3.5-4B | Qwen3.5-9B | Qwen3.5-27B |
|---|---|---:|---:|---:|
| nq_128k | subspan_em | 0.650 | 0.800 | 0.850 |
| hotpotqa_128k | subspan_em | 0.700 | 0.850 | 0.850 |
| musique_128k | subspan_em | 0.200 | 0.450 | 0.350 |
| qampari_128k | coverage | 0.320 | 0.460 | 0.522 |
| quest_128k | coverage | 0.217 | 0.400 | 0.333 |

### Detail

`unfinished` counts rows that hit `--max-steps 30`. Those rows are still scored, just on a truncated trajectory, which is where most of the remaining headroom sits.

| model | order | subset | unfinished / 20 | em | subspan_em | f1 / coverage | avg latency | avg tokens | listfmt |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 4B | none | nq_128k | 2 | 0.100 | 0.750 | f1 0.318 | 31.4s | 52664 | no |
| 4B | none | hotpotqa_128k | 0 | 0.200 | 0.600 | f1 0.322 | 16.6s | 22383 | no |
| 4B | none | musique_128k | 9 | 0.150 | 0.300 | f1 0.219 | 79.7s | 184895 | no |
| 4B | none | qampari_128k | 4 | 0.000 | 0.050 | coverage 0.050 | 48.2s | 158842 | no |
| 4B | none | quest_128k | 6 | 0.000 | 0.150 | coverage 0.033 | 83.3s | 164390 | no |
| 4B | bm25-graph | nq_128k | 3 | 0.300 | 0.750 | f1 0.491 | 46.5s | 42116 | no |
| 4B | bm25-graph | hotpotqa_128k | 3 | 0.250 | 0.700 | f1 0.419 | 59.6s | 52123 | no |
| 4B | bm25-graph | musique_128k | 7 | 0.000 | 0.100 | f1 0.036 | 70.8s | 105543 | no |
| 4B | bm25-graph | qampari_128k | 1 | 0.000 | 0.150 | coverage 0.230 | 61.6s | 84561 | yes |
| 4B | bm25-graph | quest_128k | 8 | 0.050 | 0.150 | coverage 0.175 | 82.4s | 225978 | yes |
| 4B | bm25-query | nq_128k | 2 | 0.050 | 0.650 | f1 0.212 | 51.5s | 42456 | no |
| 4B | bm25-query | hotpotqa_128k | 3 | 0.200 | 0.700 | f1 0.364 | 45.7s | 63388 | no |
| 4B | bm25-query | musique_128k | 7 | 0.050 | 0.200 | f1 0.143 | 90.3s | 127885 | no |
| 4B | bm25-query | qampari_128k | 2 | 0.050 | 0.150 | coverage 0.320 | 52.3s | 75029 | yes |
| 4B | bm25-query | quest_128k | 4 | 0.050 | 0.350 | coverage 0.217 | 66.6s | 112576 | yes |
| 9B | none | nq_128k | 1 | 0.450 | 0.800 | f1 0.589 | 40.9s | 53087 | no |
| 9B | none | hotpotqa_128k | 2 | 0.550 | 0.800 | f1 0.649 | 46.9s | 64812 | no |
| 9B | none | musique_128k | 8 | 0.200 | 0.300 | f1 0.310 | 108.1s | 223019 | no |
| 9B | none | qampari_128k | 2 | 0.000 | 0.300 | coverage 0.063 | 57.9s | 111922 | no |
| 9B | none | quest_128k | 4 | 0.000 | 0.300 | coverage 0.100 | 117.1s | 269436 | no |
| 9B | bm25-graph | nq_128k | 0 | 0.500 | 0.750 | f1 0.644 | 68.0s | 39248 | no |
| 9B | bm25-graph | hotpotqa_128k | 1 | 0.500 | 0.800 | f1 0.690 | 64.8s | 42977 | no |
| 9B | bm25-graph | musique_128k | 4 | 0.100 | 0.350 | f1 0.239 | 218.2s | 222220 | no |
| 9B | bm25-graph | qampari_128k | 0 | 0.000 | 0.050 | coverage 0.300 | 72.9s | 34981 | yes |
| 9B | bm25-graph | quest_128k | 1 | 0.150 | 0.250 | coverage 0.250 | 146.6s | 157624 | yes |
| 9B | bm25-query | nq_128k | 0 | 0.400 | 0.800 | f1 0.539 | 75.4s | 43842 | no |
| 9B | bm25-query | hotpotqa_128k | 0 | 0.600 | 0.850 | f1 0.722 | 70.3s | 60714 | no |
| 9B | bm25-query | musique_128k | 7 | 0.350 | 0.450 | f1 0.422 | 172.3s | 226567 | no |
| 9B | bm25-query | qampari_128k | 0 | 0.150 | 0.350 | coverage 0.460 | 100.3s | 84698 | yes |
| 9B | bm25-query | quest_128k | 2 | 0.150 | 0.350 | coverage 0.400 | 122.9s | 157392 | yes |
| 27B | none | nq_128k | 1 | 0.400 | 0.800 | f1 0.508 | 66.7s | 40510 | no |
| 27B | none | hotpotqa_128k | 1 | 0.650 | 0.900 | f1 0.761 | 62.3s | 37458 | no |
| 27B | none | musique_128k | 5 | 0.350 | 0.400 | f1 0.525 | 176.1s | 143325 | no |
| 27B | none | qampari_128k | 1 | 0.150 | 0.150 | coverage 0.250 | 75.7s | 41447 | no |
| 27B | none | quest_128k | 3 | 0.200 | 0.450 | coverage 0.333 | 198.4s | 175300 | no |
| 27B | bm25-graph | nq_128k | 0 | 0.300 | 0.900 | f1 0.455 | 137.3s | 48653 | no |
| 27B | bm25-graph | hotpotqa_128k | 1 | 0.550 | 0.900 | f1 0.655 | 105.6s | 40967 | no |
| 27B | bm25-graph | musique_128k | 6 | 0.200 | 0.300 | f1 0.306 | 317.3s | 229395 | no |
| 27B | bm25-graph | qampari_128k | 0 | 0.100 | 0.250 | coverage 0.543 | 200.7s | 98733 | yes |
| 27B | bm25-graph | quest_128k | 5 | 0.100 | 0.300 | coverage 0.217 | 313.4s | 217109 | yes |
| 27B | bm25-query | nq_128k | 2 | 0.200 | 0.850 | f1 0.402 | 117.6s | 87526 | no |
| 27B | bm25-query | hotpotqa_128k | 2 | 0.400 | 0.850 | f1 0.537 | 142.7s | 116137 | no |
| 27B | bm25-query | musique_128k | 4 | 0.250 | 0.350 | f1 0.402 | 208.9s | 212911 | no |
| 27B | bm25-query | qampari_128k | 0 | 0.150 | 0.400 | coverage 0.522 | 176.6s | 66698 | yes |
| 27B | bm25-query | quest_128k | 3 | 0.250 | 0.550 | coverage 0.333 | 247.9s | 184339 | yes |

Ordering helps most where the model is weakest and the answer is a list. At 9B, qampari `coverage` goes 0.063 → 0.300 → 0.460 across none → graph → query, but that jump is partly the `answer_list_format` change noted above. On the QA subsets, where no such confound exists, ordering is roughly neutral: nq at 9B is 0.800 / 0.750 / 0.800. What ordering does buy on QA is EM and F1 — 4B nq EM goes 0.100 → 0.300 with `bm25-graph` at similar `subspan_em`, so the answers get tighter rather than more often correct.

---

## Baseline — recursive-llm

### gemini-3.6-flash

`evaluation/results/rlm/recursive-llm/loft128k-gemini-3.6-flash-n20/`

| subset | scored / 20 | errors | em | subspan_em | f1 / coverage | avg latency | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| nq_128k | 20 | 0 | 0.100 | 0.900 | f1 0.278 | 57.6s | 59659 |
| hotpotqa_128k | 20 | 0 | 0.050 | 0.850 | f1 0.212 | 27.7s | 16837 |
| musique_128k | 20 | 0 | 0.150 | 0.700 | f1 0.284 | 50.4s | 56436 |
| qampari_128k | 19 | 1 | 0.050 | 0.550 | coverage 0.282 | 44.1s | 51332 |
| quest_128k | 17 | 3 | 0.050 | 0.400 | coverage 0.258 | 72.2s | 93497 |

### gemini-3.5-flash-lite

`evaluation/results/rlm/recursive-llm/loft128k-gemini-3.5-flash-lite-n20/`

| subset | scored / 20 | errors | em | subspan_em | f1 / coverage | avg latency | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| nq_128k | 20 | 0 | 0.150 | 0.850 | f1 0.312 | 10.4s | 12430 |
| hotpotqa_128k | 20 | 0 | 0.000 | 0.900 | f1 0.183 | 9.6s | 9796 |
| musique_128k | 18 | 2 | 0.000 | 0.550 | f1 0.161 | 21.4s | 57247 |
| qampari_128k | 19 | 1 | 0.000 | 0.600 | coverage 0.080 | 15.4s | 37727 |
| quest_128k | 13 | 7 | 0.050 | 0.400 | coverage 0.100 | 31.6s | 84548 |

Flash-lite is 3–6× faster and uses fewer tokens on nq / hotpotqa. It matches 3.6-flash on hotpotqa `subspan_em`, trails on musique, and is much weaker on qampari / quest `coverage`. Quest is its noisy cell: 7/20 errors.

### Qwen3.5 local vLLM

`evaluation/results/rlm/recursive-llm/loft128k-n20-think-split/`

#### Qwen3.5-4B

| subset | scored / 20 | errors | em | subspan_em | f1 / coverage | avg latency | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| nq_128k | 2 | 18 | 0.000 | 0.050 | f1 0.010 | 20.3s | 32024 |
| hotpotqa_128k | 6 | 14 | 0.000 | 0.200 | f1 0.071 | 44.2s | 44911 |
| musique_128k | 2 | 18 | 0.000 | 0.000 | f1 0.000 | 31.7s | 98894 |
| qampari_128k | 4 | 16 | 0.000 | 0.100 | coverage 0.010 | 38.2s | 77832 |
| quest_128k | 0 | 20 | 0.000 | 0.000 | coverage 0.000 | — | — |

#### Qwen3.5-9B

| subset | scored / 20 | errors | em | subspan_em | f1 / coverage | avg latency | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| nq_128k | 7 | 13 | 0.000 | 0.300 | f1 0.042 | 13.2s | 12113 |
| hotpotqa_128k | 13 | 7 | 0.150 | 0.650 | f1 0.225 | 25.9s | 17795 |
| musique_128k | 5 | 15 | 0.000 | 0.200 | f1 0.040 | 39.0s | 48525 |
| qampari_128k | 6 | 14 | 0.000 | 0.150 | coverage 0.000 | 26.2s | 38193 |
| quest_128k | 0 | 20 | 0.000 | 0.000 | coverage 0.000 | — | — |

#### Qwen3.5-27B

| subset | scored / 20 | errors | em | subspan_em | f1 / coverage | avg latency | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| nq_128k | 19 | 1 | 0.000 | 0.900 | f1 0.120 | 21.3s | 8736 |
| hotpotqa_128k | 17 | 3 | 0.100 | 0.850 | f1 0.176 | 17.7s | 5068 |
| musique_128k | 10 | 10 | 0.100 | 0.350 | f1 0.143 | 43.9s | 21103 |
| qampari_128k | 18 | 2 | 0.000 | 0.600 | coverage 0.010 | 36.3s | 23588 |
| quest_128k | 10 | 10 | 0.000 | 0.300 | coverage 0.000 | 66.4s | 52297 |

The baseline's problem is completion, not accuracy on the rows it finishes. 4B errored 18/20 on nq and 20/20 on quest, almost all `MaxIterationsError` (no `FINAL` within 30 iterations) plus some `ContextWindowExceededError` on qampari / quest. Over 20 rows that collapses 4B and 9B. Only 27B, which finished 17–19 of 20 on nq / hotpotqa, stays in the Gemini band. Latency and token averages are still over scored rows only, since a row that errored has no meaningful cost to average.

---

## Open items

- Re-run the `none` arm on qampari / quest with `answer_list_format: auto` to remove the confound before claiming an ordering effect on the list subsets.
- `write_run_artifacts` in `baselines/run_benchmark.py` scores `frame[~errored]`, so `metrics.json` will keep reporting scored-only averages. Anything read out of it needs the `× scored / 20` rescale applied here.
