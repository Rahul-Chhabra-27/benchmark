## Ruler dataset in almost completed, will be updated before 3pm

# RULER 32K Dataset

Benchmark results for the **RULER (32K context length)** dataset across multiple memory budget configurations.

## Dataset

- **Name:** `ruler32k`
- **Context length:** 32,000 tokens

### Tasks (`data_dir`)

| Task | Description |
|---|---|
| `cwe` | Common Words Extraction |
| `fwe` | Frequent Words Extraction |
| `niah_multikey_1` | Needle-in-a-Haystack — Multi-key, variant 1 |
| `niah_multikey_2` | Needle-in-a-Haystack — Multi-key, variant 2 |
| `niah_multikey_3` | Needle-in-a-Haystack — Multi-key, variant 3 |
| `niah_multiquery` | Needle-in-a-Haystack — Multi-query |
| `niah_multivalue` | Needle-in-a-Haystack — Multi-value |
| `niah_single_1` | Needle-in-a-Haystack — Single needle, variant 1 |
| `niah_single_2` | Needle-in-a-Haystack — Single needle, variant 2 |
| `niah_single_3` | Needle-in-a-Haystack — Single needle, variant 3 |
| `qa_1` | Question Answering, dataset 1 |
| `qa_2` | Question Answering, dataset 2 |
| `vt` | Variable Tracking |

## Memory Budgets

Budgets are configured in GB and converted to MB (1 GB = 1024 MB) below.

```shellscript
BUDGETS=(0.2 0.4 0.6 0.8)  # GB
```

| Budget (GB) | Budget (MB) |
|---|---|
| 0.2 | 200 |
| 0.4 | 400 |
| 0.6 | 600 |
| 0.8 | 800 |

## Results

Results will be added below as they come in, one budget at a time.

## Results

| Task              | 200 MB | 400 MB | 600 MB | 800 MB |
|--------------------|--------|--------|--------|--------|
| cwe                | 0.0    | 15.2   | 51.6   | 73.2   |
| fwe                | 77.67  | 95.67  | 97.0   | 96.0   |
| niah_multikey_1    | 32.0   | 58.0   | 81.0   | 90.0   |
| niah_multikey_2    | 3.0    | 46.0   | 80.0   | 86.0   |
| niah_multikey_3    | 0.0    | 2.0    | 18.0   | 38.0   |
| niah_multiquery    | 17.5   | 87.75  | 98.25  | 99.5   |
| niah_multivalue    | 24.25  | 68.25  | 86.0   | 91.25  |
| niah_single_1      | 100.0  | 100.0  | 100.0  | 100.0  |
| niah_single_2      | 8.0    | 91.0   | 100.0  | 100.0  |
| niah_single_3      | 13.0   | 99.0   | 100.0  | 100.0  |
| qa_1               | 50.0   | 63.0   | 68.0   | --    |
| qa_2               | 26.0   | 52.0   | 62.0   | 64.0   |
| vt                 | 100.0  | 100.0  | 100.0  | 100.0  |
| **Avg**            | **34.72** | **67.53** | **80.14** |  |
