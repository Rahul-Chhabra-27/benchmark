# slurm — batch jobs for `sbatch`

Submit from the repository root: `sbatch slurm/<job>`. Each job's `#SBATCH` header
names its partition and GPU; edit it (or pass `sbatch` flags) for another cluster.

| job | what it runs |
|---|---|
| `run-eval.sh` | any KVPress `run_matrix.py` evaluation: `sbatch slurm/run-eval.sh yml/<config>.yaml <profile> <workers>` |
| `loft128k_a100.slurm` | LOFT-128k: vanilla vs RLM vs RLM + scratchpad, Qwen3-4B-Instruct-2507 |
| `direct_baseline/qwen35_27b_gptq_int4_loft128k_dgx.sh` | uncompressed HF baseline on LOFT-128k (no kvpress) |
| `direct_baseline/qwen35_27b_gptq_int4_loft32k_dgx.sh` | the same on LOFT-32k |
| `direct_baseline/qwen35_27b_gptq_int4_loft32k_smoke_dgx.sh` | a short smoke test of that baseline |

Interactive launchers (`bash …`) are in [`../scripts/`](../scripts/).
