# Benchmark Results

This document reports benchmark results for **meta-llama/Llama-3.1-8B and Qwen3-8B** across multiple long-context QA datasets, grouped by memory budget.

**LOFT** dataset
**Datasets:** `nq`, `hotpotqa`, `musique`, `qampari`, `quest`
**Context lengths:** `32k`, `128k`
**Memory budgets:** `256 MB`, `512 MB`, `768 MB`, `1024 MB`, `2048 MB` *(assumes a 1 GB / 2 GB total budget — adjust if different)*
**Metrics:** `EM`, `Subspan EM`, `F1`, `Num Samples`

> \* Datasets marked with an asterisk (`qampari*`, `quest*`) report **Coverage** instead of F1 in the F1/Coverage column, since they don't have an F1 metric. `num_samples` also differs for these: 70 for the `_32k` variants, 110 for the `_128k` variants.

---
## Loft Dataset
## meta-llama/Llama-3.1-8B

### Memory Budget: 256 MB

| Dataset | EM | Subspan EM | F1/Coverage* | Num Samples |
|---|---|---|---|---|
| nq_32k | 0.0727 | 0.1273 | 0.1015 | 110 |
| nq_128k | 0.0000 | 0.0000 | 0.0000 | 110 |
| hotpotqa_32k | 0.0273 | 0.0273 | 0.0333 | 110 |
| hotpotqa_128k | 0.0000 | 0.0000 | 0.0000 | 110 |
| musique_32k | 0.0000 | 0.0000 | 0.0088 | 110 |
| musique_128k | 0.0000 | 0.0000 | 0.0000 | 110 |
| qampari_32k* | 0.0000 | 0.0000 | 0.0057 | 70 |
| qampari_128k* | 0.0000 | 0.0000 | 0.0000 | 110 |
| quest_32k* | 0.0000 | 0.0000 | 0.0000 | 70 |
| quest_128k* | 0.0000 | 0.0000 | 0.0000 | 110 |

### Memory Budget: 512 MB

| Dataset | EM | Subspan EM | F1/Coverage* | Num Samples |
|---|---|---|---|---|
| nq_32k | 0.2818 | 0.3364 | 0.3300 | 110 |
| nq_128k | 0.0000 | 0.0000 | 0.0000 | 110 |
| hotpotqa_32k | 0.1364 | 0.1364 | 0.1597 | 110 |
| hotpotqa_128k | 0.0000 | 0.0000 | 0.0000 | 110 |
| musique_32k | 0.0455 | 0.0455 | 0.0764 | 110 |
| musique_128k | 0.0000 | 0.0000 | 0.0000 | 110 |
| qampari_32k* | 0.0000 | 0.0000 | 0.1076 | 70 |
| qampari_128k* | 0.0000 | 0.0000 | 0.0000 | 110 |
| quest_32k* | 0.0286 | 0.0286 | 0.0571 | 70 |
| quest_128k* | 0.0000 | 0.0000 | 0.0000 | 110 |

### Memory Budget: 768 MB

| Dataset | EM | Subspan EM | F1/Coverage* | Num Samples |
|---|---|---|---|---|
| nq_32k | 0.3818 | 0.5000 | 0.4693 | 110 |
| nq_128k | 0.0091 | 0.0273 | 0.0262 | 110 |
| hotpotqa_32k | 0.2727 | 0.3273 | 0.3378 | 110 |
| hotpotqa_128k | 0.0273 | 0.0273 | 0.0273 | 110 |
| musique_32k | 0.0909 | 0.1000 | 0.1836 | 110 |
| musique_128k | 0.0000 | 0.0000 | 0.0045 | 110 |
| qampari_32k* | 0.0857 | 0.1000 | 0.2238 | 70 |
| qampari_128k* | 0.0000 | 0.0000 | 0.0000 | 110 |
| quest_32k* | 0.0429 | 0.0571 | 0.1619 | 70 |
| quest_128k* | 0.0000 | 0.0000 | 0.0000 | 110 |

### Memory Budget: 1024 MB

| Dataset | EM | Subspan EM | F1/Coverage* | Num Samples |
|---|---|---|---|---|
| nq_32k | 0.5182 | 0.6455 | 0.6113 | 110 |
| nq_128k | 0.0364 | 0.0545 | 0.0490 | 110 |
| hotpotqa_32k | 0.3727 | 0.4545 | 0.4663 | 110 |
| hotpotqa_128k | 0.0455 | 0.0545 | 0.0532 | 110 |
| musique_32k | 0.1000 | 0.1091 | 0.1901 | 110 |
| musique_128k | 0.0000 | 0.0000 | 0.0005 | 110 |
| qampari_32k* | 0.1286 | 0.1714 | 0.4348 | 70 |
| qampari_128k* | 0.0000 | 0.0000 | 0.0145 | 110 |
| quest_32k* | 0.0714 | 0.2000 | 0.2929 | 70 |
| quest_128k* | 0.0000 | 0.0000 | 0.0030 | 110 |

### Memory Budget: 2048 MB

| Dataset | EM | Subspan EM | F1/Coverage* | Num Samples |
|---|---|---|---|---|
| nq_32k | 0.4727 | 0.6364 | 0.5808 | 110 |
| nq_128k | 0.1909 | 0.2727 | 0.2341 | 110 |
| hotpotqa_32k | 0.4182 | 0.4818 | 0.4847 | 110 |
| hotpotqa_128k | 0.1909 | 0.2091 | 0.2365 | 110 |
| musique_32k | 0.1727 | 0.1727 | 0.2739 | 110 |
| musique_128k | 0.0727 | 0.0818 | 0.1338 | 110 |
| qampari_32k* | 0.3571 | 0.4571 | 0.5948 | 70 |
| qampari_128k* | 0.0000 | 0.0000 | 0.1021 | 110 |
| quest_32k* | 0.1857 | 0.3571 | 0.3976 | 70 |
| quest_128k* | 0.0000 | 0.0000 | 0.0212 | 110 |
