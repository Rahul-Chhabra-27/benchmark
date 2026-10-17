# KVPress + RLM long-context benchmarking

A fork of [NVIDIA kvpress](https://github.com/NVIDIA/kvpress) (KV-cache compression for
🤗 transformers) extended with a **Recursive Language Model (RLM)** harness. It measures the
two approaches to long context, KV compression and RLM, and their combination, on the same
datasets with the same scorers. All inference is local: a vLLM server for the root model,
and kvpress in-process for compressed sub-calls.

[**Results dashboard**](https://effortless-cupcake-470e7a.netlify.app/) ·
[LOFT-128k RLM results](docs/RLM-NEW-RESULTS.md)

This page is the guide to **running it on the infolab GPU servers**, from login to scored
results. Every command runs from the repository root.

---

## 1. What is where

| path | what it is |
|---|---|
| `kvpress/` | the compression library (presses, `KVPressTextGenerationPipeline`) |
| `evaluation/benchmarks/` | datasets and their scorers (LOFT, RULER, …) |
| `evaluation/rlm/configurable_rlm/` | **the RLM to run**: every setting comes from one file, `experiments.yaml` |
| `evaluation/rlm/harness/` | the original runner `run_benchmark.py` and our tuned RLM `rlm.py` (plus the RLMs merged from apd10-dev) |
| `evaluation/rlm/shared/` | building blocks: datasets, model client, KVzip sub-calls, B × F sizing, BM25, bm25-graph order |
| `evaluation/rlm/analysis/` | reading results: hard questions, paired comparisons, B × F tables, compression picker |
| `scripts/rlm/` | launchers: `serve_root.sh` (root model server), `run_bxf.sh` (B × F sweep), lanes |
| `slurm/` | batch jobs, for clusters with a scheduler (infolab has none) |
| `docs/` | the detailed guide (`RUNNING.md`), finished campaigns (`CAMPAIGNS.md`), design (`RLM.md`) |

---

## 2. Before you start (infolab rules)

- **Hosts:** `ant bee cat dog elk fox`, at `<host>.cse.iitb.ac.in` (use the full name). There is
  no scheduler, and the GPUs are shared and unreserved.
- **Work inside `tmux`**, so a dropped ssh session doesn't kill your server or runs.
- **Take one free GPU.** In `nvidia-smi`, a free card shows near-0 MiB **and** 0 %. Free it when
  you're done.
- **Storage:** `$HOME` (NFS, small quota) holds code only. Environments, caches and results go
  on local disk (`/tmp/$USER/…` or `/mnt/$(hostname)/data/$USER`). Archive to `/mnt/nas/$USER`.
  A large write that stalls on NFS can hang the host.
- **Write as the `infolab` group:** run `exec newgrp infolab` once per shell, or wrap commands
  in `sg infolab -c "…"`. The launchers do this for you.
- **GitHub:** you need SSH access to `skylight-org/agentic-longcontext`
  (`ssh -T git@github.com` should greet you).

---

## 3. One-time setup on a host

```bash
ssh bee.cse.iitb.ac.in            # any host from §2
tmux new -s rlm

# code
git clone git@github.com:skylight-org/agentic-longcontext.git ~/rlm && cd ~/rlm
git checkout cleanup/clean-rlm    # this branch

# caches on local disk (put these lines in ~/.bashrc, or source them in every shell)
export S=/tmp/$USER/rlm && mkdir -p $S
export HF_HOME=$S/hf UV_CACHE_DIR=$S/cache/uv PIP_CACHE_DIR=$S/cache/pip
export CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
```

**Two Python environments**, never one: vLLM pins its own torch and transformers, and kvpress
needs a newer transformers. Use the campaign's shared copies on NAS, if your account can
read them:

```bash
export VLLM=/mnt/nas/gautammahale/benchmark/.venv/bin/vllm              # server: vLLM 0.8.5.post1
export PY=/mnt/nas/gautammahale/benchmark/.venv-kvpress/bin/python      # repo env: kvpress, openai, scorers
export HF_HOME=/mnt/nas/gautammahale/hf_cache                           # model + datasets already cached
```

Or build your own on local disk:

```bash
L=/tmp/$USER/envs
UV_PROJECT_ENVIRONMENT=$L/repo uv sync --extra eval                     # repo env (Python 3.11 or 3.12)
python3 -m venv $L/vllm && $L/vllm/bin/pip install vllm==0.8.5.post1
$L/vllm/bin/pip install "transformers==4.51.3"                          # last: vLLM 0.8.5 breaks on newer
export PY=$L/repo/bin/python VLLM=$L/vllm/bin/vllm
$L/vllm/bin/huggingface-cli download Qwen/Qwen3-4B-Instruct-2507       # weights, ~9 GB, into $HF_HOME
```

Check that it works (no GPU needed, about a minute):

```bash
$PY -m pytest tests/evaluation -q      # expect 3 failures (below)
```

Three tests fail on `master` as well, and predate this branch: `test_kvzip_backend` (fixed-grid
cap), `test_rlm_guards` (scratchpad cap) and `test_rlm_sizing` (smallest cap). Any other
failure is new.

---

## 4. Start the root model server

Pick a free GPU and port, then start Qwen3-4B-Instruct-2507:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
GPU=0 PORT=8002 GPU_UTIL=0.5 VLLM=$VLLM HF_HOME=$HF_HOME LOG=$S/logs/server.log SESSION=rlm_root \
  bash scripts/rlm/serve_root.sh
```

- It refuses a card already holding more than 1 GB, or a port already in use.
- It runs under tmux, or, on a host without tmux, as a detached process with a pid file
  beside the log.
- `GPU_UTIL` is the share of the **whole** card. 0.5 suits the RLM runs, whose root never holds
  the document (the default window is `MAX_LEN=65536`). Lower it (about 0.35) when a
  compressed sub-model shares the same card (§5).
- For a **vanilla** run, which puts the whole 128k document in the window, start it with
  `MAX_LEN=139264 GPU_UTIL=0.65`.

Wait for a **real** answer, not just `/v1/models`:

```bash
until curl -s --max-time 60 localhost:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B-Instruct-2507","messages":[{"role":"user","content":"ok"}],"max_tokens":4}' \
  | grep -q '"content"'; do sleep 15; done; echo READY
```

**Stop it** with `tmux kill-session -t rlm_root`. Killing only the PID that `nvidia-smi` shows
leaves the API process accepting requests that never run.

---

## 5. Run experiments

Every experiment is a few lines in **`evaluation/rlm/configurable_rlm/experiments.yaml`**.
Three switches decide what runs:

```yaml
loop: tuned                                 # base (minimal RLM) | tuned (our original rlm.py, unchanged)
ordering: bm25-graph                        # none | bm25-query | bm25-graph
compression: {budget_mb: 1024, factor: 2}   # leave out for no compression
```

```bash
$PY -m evaluation.rlm.configurable_rlm list                    # every experiment
$PY -m evaluation.rlm.configurable_rlm check tuned_graph_x2    # what it will run (no model loaded)
$PY -m evaluation.rlm.configurable_rlm run tuned_graph --set data.limit=3    # quick try: 3 questions per task
$PY -m evaluation.rlm.configurable_rlm run tuned_graph                       # the full run (55 per task)
```

The experiments already defined:

| experiment | loop | ordering | compression (B = 1024 MB) |
|---|---|---|---|
| `tuned` / `base` | tuned / base | — | — |
| `tuned_query`, `tuned_graph` / `order_query`, `order_graph` | tuned / base | bm25-query, bm25-graph | — |
| `tuned_x2` / `compress_x2` | tuned / base | — | ×2 |
| `tuned_graph_x1`, `tuned_graph_x2` / `graph_x1`, `graph_x2` | tuned / base | bm25-graph | ×1 (control), ×2 |
| `graph_cost_aware` | base | bm25-graph | the root picks ×1/2/4/8 per read, offloading counted, cost shown |
| `graph_cost_aware_real` | base | bm25-graph | the same, with **real** offloading (KV on the CPU, measured GPU peak) |

- **`loop: tuned`** runs our original `harness/rlm.py` with the same flags our earlier campaigns
  used, so its numbers are comparable with theirs. `check` prints the exact command.
  **`loop: base`** is the minimal RLM. The difference between them is what our prompt tuning
  is worth.
- **Point the runs at your server**, if it isn't on port 8002:
  `--set models.base_url=http://localhost:<PORT>/v1`.
- **Compression experiments** load the sub-model on a GPU in-process. Choose the card with
  `CUDA_VISIBLE_DEVICES=<GPU>` (unset, it picks a card with enough free memory), and leave
  room for it: about 13 GiB at 1024 MB × 2, and 40 GiB at 2048 MB × 8.
- **Re-running the same command resumes.** Questions already answered are skipped.
- **Override any setting for one run** with `--set`, for example `--set data.tasks=[nq]`,
  `--set data.split=dev`, or `--set name=my_try` (a changed experiment needs a new name).
- **Add an experiment** by copying a block in `experiments.yaml` and giving it a new name.

### A B × F sweep with one command

`run_bxf.sh` starts the server, runs every (B, F) cell with our original RLM, prints a table,
and stops the server:

```bash
LIMIT=55 SUBSETS="nq hotpotqa musique qampari quest" BUDGETS_MB="512 1024 2048" FACTORS="1 2 4" \
  ARMS=bm25-graph GPU=0 OUT=$S/bxf bash scripts/rlm/run_bxf.sh
```

Add `SPLIT=dev` to run the same cells on the dev rows. Running it again skips finished
questions. The settings and the table are explained in `docs/RUNNING.md` §7.

### The hard questions (the validation set)

The questions vanilla gets right and RLM + ordering gets wrong, found in one command. Start
the server with a window that holds a whole 128k document, so it serves both vanilla and the
RLM:

```bash
GPU=0 PORT=8002 MAX_LEN=139264 GPU_UTIL=0.65 VLLM=$VLLM HF_HOME=$HF_HOME LOG=$S/logs/server.log \
  SESSION=rlm_root bash scripts/rlm/serve_root.sh            # wait for READY (§4)
SUBSETS="nq hotpotqa musique qampari quest" LIMIT=55 OUT=$S/hard PY=$PY bash scripts/rlm/run_hard_set.sh
```

It runs vanilla (with the whole document, `--vanilla-char-limit 600000`) and `tuned_graph` on
the same questions, then `analysis/hard_questions.py`. The result is `$S/hard/hard_set/`:
`summary.md`, and `ids/<task>__validation_set.txt`. Add `RLM="tuned_graph tuned_graph_x2"`
to also find the questions only compression rescues (that needs GPU room for the
sub-model). Then run any experiment on just those questions:

```bash
$PY -m evaluation.rlm.configurable_rlm run graph_cost_aware \
  --set data.ids_file=$S/hard/hard_set/ids/hotpotqa_128k__validation_set.txt --set name=hard_cost_aware
```

### Offloading: counted or real

With `compression: {offload: true}`, a read may be larger than B × F. The KV beyond B is
offloaded to the CPU and costed.

- **`offload_mode: count`** (the default) only counts it. kvpress keeps the KV on the GPU.
- **`offload_mode: real`** runs such a read with an offloaded KV cache: each layer's KV lives
  on the CPU and moves to the GPU only while that layer computes. Each read records its
  real peak GPU memory (`gpu_peak_mb`).

Try it first on 1–2 questions (`--set data.limit=1`) on a free GPU:

```bash
CUDA_VISIBLE_DEVICES=1 $PY -m evaluation.rlm.configurable_rlm run graph_cost_aware_real \
  --set data.tasks=[hotpotqa] --set data.limit=1 --set name=offload_smoke
```

---

## 6. Where results go

| run | results |
|---|---|
| `configurable_rlm run <name>` | `evaluation/results/rlm/configurable_rlm/<name>/` (change with `--set output_dir=…`) |
| `run_bxf.sh` | `$OUT/results/bxf_<B>MB_x<F>/…`, plus the table it prints at the end |
| `hard_questions` | `<out>/summary.md`, `hard_questions.csv`, `ids/<task>__validation_set.txt` |

Each run folder has `metrics.json` (the official score: F1 for nq, hotpotqa and musique;
coverage for qampari and quest), `predictions.csv`, `config.yaml` and the transcripts. Copy
finished results to `/mnt/nas/$USER` as the `infolab` group.

---

## 7. When something goes wrong

| symptom | fix |
|---|---|
| the server never answers | check `$S/logs/server.log`; the first start takes minutes (weights load from NFS) |
| "card already in use" or out of memory | pick another GPU, or lower `GPU_UTIL`; a compressed sub-model needs room on its own card |
| `PermissionError` under `/mnt/nas/...` | your account can't write there: keep outputs and caches under `/tmp/$USER` |
| a run stopped part-way | run the same command again; it resumes |
| "loop: tuned runs our original rlm.py, which has no …" | `choices`, `offload` and `cost` exist only in `loop: base` |
| `--mode rlm-exp1 needs evaluation/rlm/rlm_exp1/` | that module hasn't been pushed by apd10-dev yet |
| tests fail | 3 failures are expected (§3); anything else is new |

---

## More

- [`docs/RUNNING.md`](docs/RUNNING.md): every experiment in detail (environments, serving,
  ordered arms, B × F cells, KVzip).
- [`docs/CAMPAIGNS.md`](docs/CAMPAIGNS.md): every finished campaign, its setup and its numbers.
- [`docs/RLM.md`](docs/RLM.md): how RLM and kvpress fit together, and B × F sizing.
- [`evaluation/rlm/README.md`](evaluation/rlm/README.md): the RLM harness's flags and guards.
- [`AGENTS.md`](AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md): conventions (SPDX headers,
  DCO sign-off, `make style`).
