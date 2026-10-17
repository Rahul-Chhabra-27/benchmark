# KVPress + RLM long-context benchmarking

KV-cache compression (kvpress) and Recursive Language Models (RLM), measured on the same
datasets and scorers. [Results dashboard](https://effortless-cupcake-470e7a.netlify.app/)

## Layout
- `evaluation/rlm/configurable_rlm/`: the RLM to run; every setting is in `experiments.yaml`
- `evaluation/rlm/harness/`: original runner (`run_benchmark.py`) and tuned RLM (`rlm.py`)
- `evaluation/rlm/{shared,analysis,campaigns,chunk_sources}/`: building blocks, result tools
- `evaluation/benchmarks/`: datasets and scorers
- `scripts/rlm/`: server and sweep launchers
- `kvpress/`: the compression library
- `docs/`: detailed guide (`RUNNING.md`), past campaigns, design

## Infolab rules
- Hosts: `ant bee cat dog elk fox` (`<host>.cse.iitb.ac.in`); no scheduler, GPUs shared.
- Work in `tmux`. Take one free GPU (`nvidia-smi`: ~0 MiB and 0 %).
- Code in `$HOME`; environments, caches and results in `/tmp/$USER`; archive to `/mnt/nas/$USER`.
- Write as group `infolab` (`exec newgrp infolab`).

## Setup (once per host)
```bash
git clone git@github.com:skylight-org/agentic-longcontext.git ~/rlm && cd ~/rlm
git checkout cleanup/clean-rlm
export S=/tmp/$USER/rlm && mkdir -p $S
export VLLM=/mnt/nas/gautammahale/benchmark/.venv/bin/vllm           # server env (vLLM 0.8.5)
export PY=/mnt/nas/gautammahale/benchmark/.venv-kvpress/bin/python   # repo env (kvpress, openai)
export HF_HOME=/mnt/nas/gautammahale/hf_cache                        # model and datasets
$PY -m pytest tests/evaluation -q                                    # 3 known failures, all on master too
```
- Build your own envs instead: `docs/RUNNING.md` §2.

## Start the server
```bash
GPU=0 PORT=8002 GPU_UTIL=0.5 VLLM=$VLLM HF_HOME=$HF_HOME LOG=$S/logs/server.log SESSION=rlm_root \
  bash scripts/rlm/serve_root.sh
```
- Vanilla (whole document in the window): add `MAX_LEN=139264 GPU_UTIL=0.65`.
- Ready check and stopping it: `docs/RUNNING.md` §4.

## Run experiments
```bash
$PY -m evaluation.rlm.configurable_rlm list                      # all experiments
$PY -m evaluation.rlm.configurable_rlm check tuned_graph         # what it runs, no model loaded
$PY -m evaluation.rlm.configurable_rlm run tuned_graph --set data.limit=3
```
- Settings in `experiments.yaml`: `loop` (base | tuned), `ordering` (none | bm25-query | bm25-graph),
  `compression` (`budget_mb`, `factor`).
- `loop: tuned` is the original `rlm.py`, run with the earlier campaigns' flags.
- Another port: `--set models.base_url=http://localhost:<PORT>/v1`.
- Compression experiments: pick the card with `CUDA_VISIBLE_DEVICES`.
- Re-running resumes.

## `configurable_rlm` files
- `experiments.yaml`: every experiment; shared `defaults`, then what each one changes
- `config.py`: reads and checks `experiments.yaml` (unknown keys are refused)
- `rlm.py`: the base RLM: short prompt, Python REPL, loop, `llm_query`, `FINAL`, plain reads
- `ordering.py`: adds the `chunks` list in bm25-query or bm25-graph order
- `compression.py`: KVzip reads under a budget B; factor per read, offloading, cost
- `tuned.py`: `loop: tuned`; runs the original `harness/rlm.py` unchanged, with the earlier campaigns' flags
- `run.py`: runs an experiment over its questions; writes results; resumes
- `__main__.py`: the `list` / `check` / `run` commands
- `__init__.py`: package entry points

## Sweeps
```bash
LIMIT=55 SUBSETS="nq hotpotqa" BUDGETS_MB="512 1024" FACTORS="1 2 4" ARMS=bm25-graph bash scripts/rlm/run_bxf.sh
SUBSETS="nq hotpotqa" LIMIT=55 OUT=$S/hard PY=$PY bash scripts/rlm/run_hard_set.sh   # server with MAX_LEN=139264
```
- `run_bxf.sh`: B × F cells, then a results table. Add `SPLIT=dev` for the dev rows.
- `run_hard_set.sh`: questions vanilla solves but RLM + ordering misses →
  `$S/hard/hard_set/ids/`. Run them with `--set data.ids_file=<file>`.

## Results
- `evaluation/results/rlm/configurable_rlm/<experiment>/<task>/`: `metrics.json`, `predictions.csv`.
- Score: F1 (nq, hotpotqa, musique), coverage (qampari, quest).

## More
- `docs/RUNNING.md`: full guide · `docs/CAMPAIGNS.md`: past results · `docs/KNOWN_ISSUES.md`.
