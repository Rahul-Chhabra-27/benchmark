# Running the RLM experiments

One guide for every RLM experiment in this repository, from a fresh account to a scored
run. It covers an infolab GPU host and any other GPU machine. Run every command from the
repository root unless a step says otherwise.

- **What each campaign found:** [`CAMPAIGNS.md`](CAMPAIGNS.md).
- **The harness's flags and guards:** [`evaluation/rlm/README.md`](evaluation/rlm/README.md).
- **The design:** [`RLM.md`](RLM.md).

## 0. Pick your experiment

| you want to run | model | environments (§2) | section |
|---|---|---|---|
| vanilla vs RLM on any LOFT subset, the baseline | any served model; the example is Qwen3.5-4B | server A + driver | §5 |
| the ordered-chunk arms (`bm25-query` / `bm25-graph`) on LOFT-128k or RULER-128k, with gate and report | Qwen3-4B-Instruct-2507 | server B + repo env | §6 |
| B × F cells: the ordered arms reading through a KVzip-compressed reader | Qwen3-4B-Instruct-2507 | server B + repo env | §7 |
| the KVzip sub-call arms on LOFT (4a/4b), and the fixed-chunk grid on longbench128k | Qwen3-4B-Instruct-2507 | server B + repo env | §8 |

Then:
- **Off infolab** (RunPod, a cloud VM, or SLURM): read §9 after your section.
- **Reading any run's output:** §10.
- **When something fails:** §11.

---

## 1. Before you start

- **Hosts.** The infolab GPU hosts are `ant bee cat dog elk fox`, at `.cse.iitb.ac.in`. Ssh
  with the full name: there is no short alias. There is no scheduler, and the GPUs are shared
  and unreserved.
- **Run everything under `tmux`.** An ssh drop otherwise kills the server and the lanes.
- **Take one free card.** Check `nvidia-smi` first. A free card shows near-0 MiB **and** 0 %:
  a card with little memory in use can still be at 99 % from someone's small job. Take one
  card per campaign, and free it when you are done.
- **The driver decides the server stack.** Server A (§2) needs **CUDA 12.8 or newer**
  (`nvidia-smi`, top right): bee and elk qualify, and fox, at 12.5, does not. Server B works
  on all of them except ant, whose Blackwell (sm_120) cards need a cu128 build.
- **Follow the lab storage policy.**
  - `$HOME` (`/mnt/home/$USER`: NFS, backed up nightly, small quota) holds code and condensed
    output only.
  - Venvs, caches, weights and live results go on the host's local disk: `/tmp/$USER/…` (per
    host, not durable), or `/mnt/$(hostname)/data/$USER` if your advisor has allocated it.
  - Archive to `/mnt/nas/$USER`.
  - Write to scratch and archive as the `infolab` group: `sg infolab -c "…"`, or
    `exec newgrp infolab`.
  - A large write that stalls on NFS can hang the host.
- **GitHub access.** You need `skylight-org/agentic-longcontext` (formerly
  `intel-longcontext-pc`) over SSH: `ssh -T git@github.com` should greet you. This guide lives
  on the `gautam-dev` branch.

---

## 2. Environments

Four environments. §0 says which ones your experiment needs.

| environment | what it is | used by | how to build it |
|---|---|---|---|
| **server A** | `vllm==0.19.0` (torch 2.10, CUDA 12.8) | Qwen3.5 and newer models (§5) | §2.2 |
| **server B** | `vllm==0.8.5.post1` + `transformers==4.51.3` | every reported Qwen3-4B-Instruct-2507 number (§6–8) | §2.3 |
| **driver** | a light client with no torch: `run_benchmark` and `compare` over HTTP | §5 | §2.2 |
| **repo env** | `uv sync --extra eval`: torch, kvpress, pytest and friends | tests (§3), gates and reports (§6), the in-process KVzip reader (§7–8) | §2.3 |

**The rules:**
- **Never share one venv between a server and a client.** vLLM pins its own torch and
  transformers. kvpress needs `transformers>=4.56`, while vLLM 0.8.5 breaks on anything past
  `4.51.3`. The two work together only because they run as separate processes.
- **Never `pip install` into a venv a live server runs from.**
- **Keep every venv and cache on local disk.**

### 2.1 Shell environment (every new shell)

Write this once per host:

```bash
mkdir -p /tmp/$USER/rlm
cat > /tmp/$USER/rlm/env <<'EOF'
export S=/tmp/$USER/rlm
export HF_HOME=$S/hf XDG_CACHE_HOME=$S/cache
export UV_CACHE_DIR=$S/cache/uv UV_PYTHON_INSTALL_DIR=$S/uv-python PIP_CACHE_DIR=$S/cache/pip
export VLLM_CACHE_ROOT=$S/cache/vllm TRITON_CACHE_DIR=$S/cache/triton TORCHINDUCTOR_CACHE_DIR=$S/cache/inductor
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA's GPU numbering = nvidia-smi's
export UV_USE_IO_URING=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
EOF
```

Then, in **every** shell or tmux window you open:

```bash
source /tmp/$USER/rlm/env
```

§5 assumes `$S` is set. §6–9 set their own paths.

### 2.2 Server A and the driver (for §5): about 5 minutes on elk

```bash
source /tmp/$USER/rlm/env

# code
git clone -b gautam-dev git@github.com:skylight-org/agentic-longcontext.git $S/skylight   # this guide lives on gautam-dev
cd $S/skylight            # all later commands run from here

# venv 1: the model server (vLLM 0.19, CUDA 12.8 build)
uv venv -p 3.11 $S/venv-vllm
uv pip install -p $S/venv-vllm -r evaluation/rlm/requirements-server.txt --torch-backend=cu128

# venv 2: the benchmark driver (plain Python, no torch)
uv venv -p 3.11 $S/venv-drv
uv pip install -p $S/venv-drv -r evaluation/rlm/requirements-driver.txt

# weights (~9 GB)
$S/venv-vllm/bin/hf download Qwen/Qwen3.5-4B
```

Check before moving on. Each line should print a version:

```bash
$S/venv-vllm/bin/vllm --version                                  # 0.19.0
$S/venv-drv/bin/python -c "import pandas, scipy, openai, tiktoken; print('driver ok')"
```

### 2.3 Server B and the repo env (for §6–8)

The campaigns' own environments are on NAS. You can use them if your account can read them:

```bash
export PY=/mnt/nas/gautammahale/venvs/rlm/bin/python             # client: Python 3.11, the harness's dependencies
export VLLM=/mnt/nas/gautammahale/benchmark/.venv/bin/vllm       # server: vLLM 0.8.5.post1
```

To build your own instead, put them on the host's local disk, never in `$HOME`. Point the
package caches there as well:

```bash
L=/mnt/$(hostname)/data/$USER           # or /tmp/$USER
export UV_CACHE_DIR=$L/cache/uv PIP_CACHE_DIR=$L/cache/pip
cd $REPO && UV_PROJECT_ENVIRONMENT=$L/venvs/rlm uv sync --extra eval
python3 -m venv $L/venvs/vllm && $L/venvs/vllm/bin/pip install vllm==0.8.5.post1
$L/venvs/vllm/bin/pip install "transformers==4.51.3"   # LAST: vLLM 0.8.5 dies at tokenizer init on transformers 5.x
export PY=$L/venvs/rlm/bin/python VLLM=$L/venvs/vllm/bin/vllm
```

`$REPO` is your checkout (§6 sets it). The repo env needs Python 3.12 or 3.11. On 3.13,
`fire` fails to import.

---

## 3. Unit tests (no GPU)

```bash
pip install uv                        # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra eval                  # .venv with the harness and the dev tools (pytest, black, ...)
.venv/bin/python -m pytest tests/evaluation -q
```

These tests use fake model clients and a character-count tokenizer, so they need no GPU,
network or weights. Every one should pass.

`python -m pytest tests/` also runs kvpress's own suite, which reports about 42 failures on a
CPU-only machine. All of them predate the RLM work and are explained in
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

The style checks the code was held to:

```bash
.venv/bin/python -m black --check evaluation/rlm tests/evaluation     # line length 120, from pyproject.toml
.venv/bin/python -m isort --check-only evaluation/rlm tests/evaluation
.venv/bin/python -m flake8 evaluation/rlm tests/evaluation
```

---

## 4. Serve a model

### 4.1 Pick a GPU and a port

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
```

Pick a card showing near-0 MiB **and** 0 %. Then pick a port nobody is listening on:

```bash
GPU=2 PORT=8013
ss -ltn | grep -q ":$PORT " && echo "port $PORT is taken, pick another"
```

### 4.2 How much of the card to ask for

`--gpu-memory-utilization` is a fraction of the **whole card**, not of the free part. The
server needs the weights plus the KV for **one** sequence of `--max-model-len` tokens. It
refuses to start otherwise, and says how much it needs.

| model | KV per token | fits a 128k document at | on a 48 GB card |
|---|---|---|---|
| Qwen3.5-4B | ~32 KiB (8 full-attention layers × 4 KV heads × 256) | `--max-model-len 139264` | `0.5` (~25 GB), about 3 full-length requests at once |
| Qwen3-4B-Instruct-2507 | 144 KiB (147,456 bytes: 36 layers × 2 × 8 × 128 × 2) | `--max-model-len 139264`: ~19.1 GiB of KV + ~7.5 GiB of weights ≈ 31 GB | `0.65` |
| Qwen3-4B-Instruct-2507, for the ordered arms | same | `--max-model-len 65536` (the root never holds the document) | `0.5`, a KV cache of about 104k tokens |

### 4.3 Server A: Qwen3.5 (for §5)

```bash
tmux new -s rlm
source /tmp/$USER/rlm/env
GPU=2 PORT=8013 MODEL=Qwen/Qwen3.5-4B

CUDA_VISIBLE_DEVICES=$GPU $S/venv-vllm/bin/vllm serve $MODEL \
  --served-model-name $MODEL --port $PORT \
  --max-model-len 139264 --gpu-memory-utilization 0.5 \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

`--limit-mm-per-prompt` switches off Qwen3.5's image and video towers. Text-only models
reject the flag, so drop it for them. The first start takes about 5 minutes (loading, then
compiling).

### 4.4 Server B: Qwen3-4B-Instruct-2507 (for §6–8)

`serve_root.sh` refuses a card that already holds more than 1 GB, and a port that is already
listening:

```bash
cd $REPO
GPU=0 PORT=8002 GPU_UTIL=0.5 CPUS=2-5 VLLM=$VLLM HF_HOME=$HF_HOME \
LOG=$S/logs/server.log SESSION=ro_root bash evaluation/rlm/serve_root.sh
```

- `GPU` takes a device index or a UUID. vLLM 0.8.5 crashes on a UUID, so the script resolves
  it to an index for you.
- `CPUS` pins the server to a few cores, which keeps it polite on a shared host.
- For a vanilla arm at 128k, the server needs `--max-model-len 139264` and about 0.65 of the
  card (§4.2). `evaluation/rlm/loft128k/run_infolab.sh serve` (§8) sizes this for you.

### 4.5 Is it ready?

Wait for a **real generation**, not just `/v1/models`. A wedged vLLM still answers
`/v1/models`, and a campaign pointed at it looks exactly like one that is running.

```bash
until curl -s --max-time 60 localhost:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B-Instruct-2507","messages":[{"role":"user","content":"ok"}],"max_tokens":4}' \
  | grep -q '"content"'; do sleep 15; done; echo READY
```

Change the port and model to match your server.

**Stopping a server.** Kill its whole tmux session, for example
`tmux kill-session -t ro_root`. Killing only the PID `nvidia-smi` shows kills vLLM's engine
core. The API process keeps listening and accepts requests that never run.

---

## 5. Base run: vanilla and RLM

The worked example is LOFT-128k (`nq`, `hotpotqa`) on `Qwen/Qwen3.5-4B`, from server A (§4.3).
Sections 2.1, 2.2, 4 and 5 were tested verbatim on elk on 2026-09-14, from a clean
`/tmp/$USER/rlm`. They ran on master `c92f4e2` and again on `gautam-dev` `eba22c9`. Setup
took about 4 minutes, the server was ready in about 5, and the smoke test finished with 0
errors in both arms.

### 5.1 Smoke test: 1 row per arm (tmux window 2: `Ctrl-b c`)

Put the flags every run shares into one array, so each run line shows only what differs:

```bash
source /tmp/$USER/rlm/env; cd $S/skylight
PORT=8013 MODEL=Qwen/Qwen3.5-4B
RUN=($S/venv-drv/bin/python -m evaluation.rlm.run_benchmark
     --base-url http://localhost:$PORT/v1 --root-model $MODEL --sub-model $MODEL
     --max-steps 50 --max-sub-calls 40 --exec-timeout 60 --run-timeout 900
     --vanilla-char-limit 700000 --vanilla-max-prompt-tokens 134000
     --no-think)

"${RUN[@]}" --dataset loft --data-dir nq_128k --split dev --limit 1 --mode vanilla --out $S/smoke
"${RUN[@]}" --dataset loft --data-dir nq_128k --split dev --limit 1 --mode rlm     --out $S/smoke
grep -H -E '"(errors|subspan_em)"' $S/smoke/*/metrics.json
```

Pass: both runs finish with `"errors": 0`. A single row's score means nothing. A vanilla error
almost always means the server window is smaller than the prompt (§11).

### 5.2 The real run: 55 test rows, nq + hotpotqa, both arms

Same window, same `RUN` array. All four lanes run in parallel against the one server:

```bash
mkdir -p $S/logs
for ds in nq hotpotqa; do for mode in vanilla rlm; do
  "${RUN[@]}" --dataset loft --data-dir ${ds}_128k --split test --limit 55 \
    --mode $mode --out $S/results > $S/logs/${ds}_${mode}.log 2>&1 &
done; done
wait; echo ALL DONE
```

This takes about 35 minutes. The RLM lanes are the slow ones, at about 36 s per row on nq.
To watch, run `tail -f $S/logs/nq_rlm.log` in another window.

**Resuming.** Each run directory has a `checkpoint.jsonl`. If a lane dies, or you Ctrl-C,
rerun the same line and it skips finished rows. Change nothing else: a different flag is a
different run directory, or a refused resume.

### 5.3 Scores, and cleaning up

```bash
grep -H '"subspan_em"' $S/results/*/metrics.json      # headline score per run
$S/venv-drv/bin/python -m evaluation.compare --results-dir $S/results
```

**Reference numbers** (master `c92f4e2`, the same 55 rows, greedy decoding): RLM nq
**0.527**, hotpotqa **0.491**, with 0 errors. A correct setup lands within a question or two of
these. Before believing a vanilla score, check that `truncated_fraction` in its `metrics.json`
is 0.

**Keeping the results.** Stop the server with Ctrl-C in window 1, and check the card is free
(`nvidia-smi`). Copy what you want to keep to NAS; writes there need the `infolab` group.
Then close tmux:

```bash
sg infolab -c "mkdir -p /mnt/nas/$USER/loft128k_qwen35 && cp -r $S/results $S/logs /mnt/nas/$USER/loft128k_qwen35/"
tmux kill-session -t rlm
```

### 5.4 Adapting it

- **Other LOFT subsets and lengths.** `--data-dir` is `<subset>_<length>`, with subset in `nq
  hotpotqa musique qampari quest` and length in `32k 128k 1m`. The data downloads from the
  Hub (`f20180301/loft-rag-*`, a third-party namespace) on first use.
- **How many rows there are.** At 128k and 1m, each subset has 10 dev and 100 test rows, so
  `--split test --limit 100` is the whole test split. At 32k, qampari and quest have 60 test
  rows. On `master`, test row ids run from 10.
- **Always pass `--split`.** Without it, a `--limit` of 10 or less evaluates dev rows only,
  and the result lands in the tree looking like a real run.
- **1m.** No window here holds a 1m document, so run `--mode rlm` only. The root holds only
  REPL output, so a smaller server window is fine.
- **Other models.** Set `MODEL` in both windows, and adjust:
  - `--no-think`: keep it for thinking models (Qwen3, Qwen3.5), and **drop** it for
    non-thinking ones such as `Qwen3-4B-Instruct-2507`.
  - `--max-model-len` / `--vanilla-max-prompt-tokens`: the prompt cap must stay below the
    window minus the answer (139264 / 134000 at 128k). Check the model's
    `max_position_embeddings` covers the window.
  - `--gpu-memory-utilization`: see §4.2.

---

## 6. Ordered-chunk arms

The agentic RLM root (Qwen3-4B-Instruct-2507) gets every 2000-character chunk of a 128k
document, in a fixed order, and reads them through `llm_query`. The two arms differ only in
that order:

| arm | flag | how `chunks` is ordered |
|---|---|---|
| query-aware | `--chunk-order bm25-query` | every chunk, ranked by BM25 against the question |
| query-agnostic | `--chunk-order bm25-graph` | a BM25-graph organization built before the question is known |

A campaign runs both arms on LOFT-128k (the five subsets, first 55 test rows each) and on
RULER-128k (all 13 tasks, first 55 rows each). Results: [`CAMPAIGNS.md` §7](CAMPAIGNS.md#7-ordered-chunk-arms-bm25-query-vs-bm25-graph).

**How a run fits together.**
- **One server does everything.** The root and the chunk reader are the same model on the
  same server (§4.4, with `--max-model-len 65536`).
- **Run many lanes at once.** The GPU batches their requests. LOFT (550 rows, 10 lanes) took
  about 40 minutes on one A6000, and RULER (1,430 rows, 14 lanes) about 46.
- **Runs are resumable.** Rerunning a lane skips rows already in `checkpoint.jsonl`.
- **Every stage goes to a fresh results directory.** The harness refuses to resume into one
  whose config differs in a resume-critical key.
- **Always pilot first:** `LIMIT=3`, gate it, read a few transcripts, then `LIMIT=55`.

### 6.1 Paths and setup

```bash
# The code is the only thing that lives in $HOME.
mkdir -p ~/rlmorder && cd ~/rlmorder
git clone git@github.com:skylight-org/agentic-longcontext.git checkout   # or the lab's NAS mirror, if you can read it
cd checkout && git checkout gautam-dev

# Paths used in the rest of this section. Keep them in your shell.
export REPO=$HOME/rlmorder/checkout
export S=/tmp/$USER/rlmorder            # or /mnt/$(hostname)/data/$USER/rlmorder, if allocated
export HF_HOME=$S/hf_cache              # the model, the tokenizer and the datasets, on local disk
sg infolab -c "mkdir -p $S/logs $HF_HOME"
```

Then set `PY` and `VLLM` (§2.3), and start the server (§4.4).

### 6.2 Stage the model and the data (once, with network access)

The lanes and the server run offline (`HF_HUB_OFFLINE=1`), so everything must be cached
first. This downloads the tokenizer, both benchmarks and the weights, then proves they load
offline:

```bash
cd $REPO
sg infolab -c "HF_HOME=$HF_HOME PYTHONPATH=. $PY -m evaluation.rlm.rlm_order_stage --dataset all --weights"
```

It ends with `staged: lanes can run with HF_HUB_OFFLINE=1`. On cat, LOFT's five subsets
alone took about 10 minutes, printing nothing while each one downloaded. `/tmp` does not
survive a reboot; if it is wiped, run the command again.

### 6.3 Pilot, gate, full run

Each lane is one arm on one or more tasks, pinned to one CPU core, in its own tmux session
named `ro_<arm>_<lane>`.

**RULER-128k pilot** (3 rows per task: 14 lanes, 78 rows, about 4 minutes):

```bash
cd $REPO
R=$S/results_ruler128k_pilot                    # a fresh directory for every stage
sg infolab -c "mkdir -p $R"
core=6
for arm in bm25-graph bm25-query; do
  i=1
  for tasks in "niah_single_1 cwe" "niah_single_2 fwe" "niah_single_3 vt" "niah_multikey_1 qa_1" \
               "niah_multikey_2 qa_2" "niah_multikey_3 niah_multivalue" "niah_multiquery"; do
    tmux new-session -d -s "ro_${arm#bm25-}_p$i" \
      "sg infolab -c \"DATASET=ruler128k LIMIT=3 REPO=$REPO PY=$PY HF_HOME=$HF_HOME RESULTS=$R \
         bash $REPO/evaluation/rlm/rlm_order_lane.sh $arm $core $tasks > $S/logs/ruler_pilot_${arm}_p$i.log 2>&1\""
    core=$((core+1)); i=$((i+1))
  done
done
tmux ls
```

**LOFT-128k pilot** (10 lanes, one per arm and subset):

```bash
R=$S/results_loft_pilot
sg infolab -c "mkdir -p $R"
core=6
for arm in bm25-graph bm25-query; do
  for s in nq hotpotqa musique qampari quest; do
    tmux new-session -d -s "ro_${arm#bm25-}_$s" \
      "sg infolab -c \"DATASET=loft LIMIT=3 REPO=$REPO PY=$PY HF_HOME=$HF_HOME RESULTS=$R \
         bash $REPO/evaluation/rlm/rlm_order_lane.sh $arm $core $s > $S/logs/loft_pilot_${arm}_$s.log 2>&1\""
    core=$((core+1))
  done
done
```

On LOFT, the graph arm reads its order from `ORG`. This defaults to the structured
graph-order campaign's cache on NAS, `/mnt/nas/gautammahale/archive/graphorder/organization`,
so both campaigns read exactly the same artifact.

**Watch the progress:**

```bash
watch -n 60 "cat $R/*/checkpoint.jsonl 2>/dev/null | wc -l; tmux ls 2>/dev/null | grep -c '^ro_'"
tail -f $S/logs/ruler_pilot_bm25-query_p1.log     # every lane ends with '################ DONE'
```

**Gate the pilot** once every lane has printed `DONE`:

```bash
# RULER
sg infolab -c "cd $REPO && HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 PYTHONPATH=. $PY -m evaluation.rlm.rlm_order_gate \
  $S/results_ruler128k_pilot --dataset ruler128k --pilot --expect 3"
# LOFT: this also checks rows, questions and the graph order against the structured campaign
sg infolab -c "cd $REPO && HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 PYTHONPATH=. $PY -m evaluation.rlm.rlm_order_gate \
  $S/results_loft_pilot --dataset loft --pilot --expect 3 \
  --structured-results /mnt/nas/gautammahale/archive/graphorder/results \
  --organization-dir /mnt/nas/gautammahale/archive/graphorder/organization"
```

- **Integrity problems fail the gate outright:** errors, token counts that disagree with the
  server, missing transcripts, or limit notices.
- **Behaviour rows are listed under `REVIEW`.** These are rows a harness guard ended, or that
  never read a chunk. A run WARNS once more than 10% of its rows are listed, and FAILS past
  25%.
- **To see why each row was flagged:**
  `PYTHONPATH=. $PY -m evaluation.rlm.rlm_order_review <results dir>`.

**The full run** is the same loop, with four changes:
- `LIMIT=55`;
- a fresh `RESULTS`: `$S/results_ruler128k` for RULER, or `$S/results` for LOFT, which are the
  post-run script's defaults;
- lane sessions named `ro_<arm>_r$i`;
- logs named `ruler_full_*` or `full_*`.

### 6.4 Post-run: gate, report and archive

Once every lane has printed `DONE`, run the post-run script. If you changed code, pull first:
bash reads a script while it runs it, so never pull while a lane is running.

```bash
cd $REPO && git pull --ff-only
sg infolab -c "DATASET=ruler128k STAGE=ruler_full SCRATCH=$S PY=$PY HF_HOME=$HF_HOME \
  ARCHIVE=/mnt/nas/$USER/archive/rlmorder SERVER=ro_root bash evaluation/rlm/rlm_order_postrun.sh"
# LOFT: DATASET=loft STAGE=full
```

It does four things:
1. **Gates the run.**
2. **Stops the server** if the gate passes. If the gate fails, the server stays up.
3. **Writes the report**, to `$S/report_ruler128k/ruler_full/report.md` or
   `$S/report/full/report.md`. Every comparison is repeated with rows 0–2 left out, because
   those rows tuned the harness.
4. **Archives everything**, checksum-verified.

**The report** (`report.md`, next to `report.csv`, `paired.csv` and `campaign.json`) has:
- the headline pairs: query − graph, with a 95% bootstrap CI, W/L/T and a sign test;
- the same pairs on the held-out rows;
- cost and behaviour;
- the harness's interventions;
- where the gold sat in each order.

`python -m evaluation.rlm.rlm_order_replay <results> <dataset>` shows what revision 6's
grounding rule would have changed in a revision-5 run.

### 6.5 One row by hand, for debugging

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 $PY -m evaluation.rlm.run_benchmark \
  --dataset ruler128k --data-dir niah_single_1 --limit 1 \
  --mode rlm --chunk-order bm25-query --organization-dir $S/organization_debug --max-steps 50 \
  --root-model Qwen/Qwen3-4B-Instruct-2507 --sub-model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://localhost:8002/v1 --sub-backend http --out $S/debug --debug
```

- `--debug` prints every step: the root's code, and what it printed.
- For LOFT, use `--dataset loft --data-dir nq_128k --split test`.
- The ordered arms require `--mode rlm` and `--sub-backend http`, and refuse `--search-k` and
  `--chunks-source`.

### 6.6 Reproducing the reported numbers

- **Which code produced which numbers.** LOFT-128k ran on harness revision 5 (`2b62ef2`), and
  RULER-128k on revision 6 (`c466401`). The branch head runs revision 6.
- **Reproducing LOFT exactly.** Check out `2b62ef2` in a separate directory and point `REPO`
  at it, while running the lane and post-run scripts from the current checkout. Gate its
  results with `--revision 5`; the post-run script defaults to that for LOFT.
- **What a rerun gives.** Decoding is greedy, but vLLM's batching numerics vary with the lane
  count, so individual rows can differ. Compare means, against the one-row step in §10.

---

## 7. B × F cells

The ordered arms can read through an **in-process KVzip model** instead of the served one.
- **B** is how much KV one read may keep, in MB. For Qwen3-4B, 512 MB is 3,472 tokens.
- **F** is the exact factor every read is pressed by (ratio 1 − 1/F). F = 1 is the
  uncompressed control, with `--press no_press`.
- A read is capped at **N = B × F** tokens: 13,888 for 512 MB × 4.
- A cell the GPU or the model window cannot hold is refused before any example runs.

### 7.1 The smoke test: one command (about 5 minutes)

2 questions, both arms, B = 512 MB, F = 1 and 4. On `cat`:

```bash
ssh cat.cse.iitb.ac.in
cd ~/rlmorder/checkout          # your clone of the repository
git checkout gautam-dev && git pull
tmux new -s bxf                 # optional, but it keeps the run alive if your connection drops
bash evaluation/rlm/run_bxf.sh
```

The script:
1. starts the server on GPU 0 and waits for it;
2. runs the 4 cells one after another;
3. prints a results table;
4. stops the server.

It ends with `4 of 4 cells passed their checks`, and exits 0.

What it printed on 2026-09-12 (cat, GPU 0, 4 minutes):

```
B      F  arm         subset   rows  score  N      kept_max  B_tok  F_real  pressed  check
512MB  1  bm25-graph  nq_128k  2     0.500  3472   3094      3472   1.000   0.000    PASS
512MB  1  bm25-query  nq_128k  2     1.000  3472   2488      3472   1.000   0.000    PASS
512MB  4  bm25-graph  nq_128k  2     1.000  13888  1020      3472   4.000   1.000    PASS
512MB  4  bm25-query  nq_128k  2     0.500  13888  1020      3472   4.000   1.000    PASS
REVIEW loft__nq_128k__…__cf1__order-bm25-graph__ctok3472: loft:nq_128k:1:error_loop_broken+no_reads

4 of 4 cells passed their checks
stopped the server (tmux bxf_root)
```

The scores mean nothing on 2 questions: the smoke test only proves the pipeline works.
**A `REVIEW` line is model behaviour, not a failure.**

### 7.2 A real run

Put the settings you want in front of the same command:

```bash
LIMIT=55 SUBSETS="nq hotpotqa musique qampari quest" BUDGETS_MB="512 1024" FACTORS="1 2 4 8" \
  bash evaluation/rlm/run_bxf.sh
```

The cells run one at a time. **If it stops part-way, run the same command again:** it skips
every question already answered.

| setting | default | what it means |
|---|---|---|
| `BUDGETS_MB` | `512` | the B values, in MB (space-separated) |
| `FACTORS` | `1 4` | the F values (keep `1`, the no-compression baseline) |
| `ARMS` | `bm25-query bm25-graph` | which chunk orders to run |
| `DATASET` | `loft` | `loft` or `ruler128k` |
| `SUBSETS` | `nq` | LOFT: `nq hotpotqa musique qampari quest`; RULER: task names such as `niah_single_1 cwe` |
| `LIMIT` | `2` | questions per subset (the standard campaign uses 55) |
| `GPU` | `0` | which GPU to use (check `nvidia-smi` first) |
| `GPU_UTIL` | `0.5` | the share of the GPU the model server takes; lower it for big B × F |
| `OUT` | `/tmp/$USER/bxf` | where the results and logs go |
| `KEEP_SERVER` | `0` | set it to `1` to leave the model server running afterwards |

To run one cell by hand, give a lane `BUDGET_MB` and `FACTOR`:

```bash
sg infolab -c "BUDGET_MB=512 FACTOR=4 CUDA_VISIBLE_DEVICES=0 DATASET=loft LIMIT=2 RESULTS=$S/results_bxf \
  HF_HOME=$HF_HOME bash $REPO/evaluation/rlm/rlm_order_lane.sh bm25-query 6 nq"
```

Each cell writes to `$RESULTS/bxf_<B>MB_x<F>`. Gate one cell's directory at a time, as in
§6.3.

### 7.3 How big can B × F be?

A read's full uncompressed KV stays on the GPU while it is read (KVzip masks, it never frees).
A cell needs about 8 GiB of weights plus 2 × N × 147,456 bytes, on top of the root server:

| B | F = 4: N, GPU | F = 8: N, GPU |
|---|---|---|
| 512 MB | 13,888, ~13 GiB: fits | 27,776, ~17 GiB: fits |
| 1024 MB | 27,776, ~17 GiB: fits | 55,552, ~25 GiB: lower `GPU_UTIL` to about 0.35 |
| 2048 MB | 55,552, ~25 GiB: lower `GPU_UTIL` to about 0.35 | 111,104, ~40 GiB: a GPU of its own |

### 7.4 The results table

| column | meaning |
|---|---|
| `score` | the answer score for that subset (higher is better) |
| `rows` | how many questions were answered |
| `N` | the longest a single read was allowed to be (B × F, in tokens) |
| `B_tok` | B in tokens |
| `kept_max` | the most tokens any single read actually kept; it must be ≤ `B_tok` |
| `F_real` | the compression actually applied; it should equal F |
| `pressed` | the share of reads that were compressed: 1.0 for F > 1, 0.0 for F = 1 |
| `check` | `PASS` if the cell got exactly the B and F it asked for, with no errors |

Results go to `$OUT/results/bxf_<B>MB_x<F>/`, and logs to `$OUT/logs/`, one per cell plus
`server.log`. `/tmp` is wiped on reboot, so copy anything you want to keep to
`/mnt/nas/$USER/`.

---

## 8. KVzip sub-call arms and the fixed-chunk grid

With `--sub-backend kvzip`, the REPL tool becomes `llm_query(question, context_text)`. The
slice is prefilled under `KVzipPress`, and the question decodes against the compressed cache.

**What to know first.**
- **The driver is not the vLLM venv.** The driver hosts the sub model in-process, so it needs
  `transformers>=4.56`: the repo env (§2.3), never the vLLM 0.8.5 venv. It talks to the vLLM
  root over HTTP.
- **Compression is logical.** Evicted keys are masked, not freed, so a slice costs its full
  uncompressed KV whatever the budget.
- **Size `--sub-min-free-gib` for the uncompressed context.** The backend pins itself to a
  GPU, refuses to load without that much free memory, and returns an oversized slice as a
  retry notice rather than an OOM.

### 8.1 The LOFT arms, through `run_infolab.sh`

`run_infolab.sh` picks cards by live utilisation, sizes each server from the memory actually
free, and holds a lock per dataset. Keep its caches and its roughly 15 GB venv off `$HOME` and
NAS:

```bash
tmux new -s rlm
export RLM_SCRATCH=/tmp/$USER VENV=/tmp/$USER/venvs/rlm-vllm085   # caches and venv on local disk
bash evaluation/rlm/loft128k/run_infolab.sh setup
bash evaluation/rlm/loft128k/run_infolab.sh auto
```

Smoke first, cheaply, at 32k:

```bash
DATASETS="nq" LENGTH=32k LIMIT=3 SERVERS=1 bash evaluation/rlm/loft128k/run_infolab.sh auto
```

Then repeat at `LENGTH=128k LIMIT=2` for the real memory check. To drive the two halves by
hand instead, in two tmux windows:

```bash
GPU=1 bash evaluation/rlm/loft128k/run_infolab.sh serve
DATASETS="nq" bash evaluation/rlm/loft128k/run_infolab.sh run
```

The KV-compression arms:

```bash
# one-example smoke, arm 4a then 4b, at 32k
KVPRESS_ARMS=1 KV_RATIOS=0.5 KV_ARMS=4a DATASETS=nq LENGTH=32k LIMIT=1 SERVERS=1 \
    bash evaluation/rlm/loft128k/run_infolab.sh auto
KVPRESS_ARMS=1 KV_RATIOS=0.5 KV_ARMS=4b DATASETS=nq LENGTH=32k LIMIT=1 SERVERS=1 \
    bash evaluation/rlm/loft128k/run_infolab.sh auto

# the real grid: 5 datasets x {0.5, 0.75} x {4a, 4b}
KVPRESS_ARMS=1 LENGTH=128k bash evaluation/rlm/loft128k/run_infolab.sh auto

# cell 5: press-only baseline through evaluate.py (smoke with FRACTION=0.02)
DATASETS=nq KV_RATIOS=0.5 FRACTION=0.02 bash evaluation/rlm/loft128k/run_infolab.sh kvzip-baseline
bash evaluation/rlm/loft128k/run_infolab.sh kvzip-baseline
```

`KVPRESS_ARMS=1` runs **only** the arm-4 lanes, and puts two models on one card:

| tenant | memory |
|---|---|
| vLLM root, `--max-model-len 65536`, capped at `ROOT_BUDGET_MIB` | ~21 GB |
| HF sub model | ~8.1 GB of weights |
| one 34k-token sub prefill (the peak is the full KV) | ~4.9 GB |
| KVzip scoring transient | ~2.5 GB |
| **total** | **~37 GB**, about 11 GB of margin |

The colocated root has a reduced window. That is why vanilla never runs against it: a vanilla
arm there would shrink-retry and merge truncated rows into existing checkpoints.

### 8.2 One KVzip run, directly

```bash
python -m evaluation.rlm.run_benchmark \
  --dataset loft --data-dir nq_128k --split test --limit 50 \
  --mode rlm --scratchpad --max-steps 50 \
  --base-url http://localhost:8000/v1 \
  --root-model Qwen/Qwen3-4B-Instruct-2507 \
  --sub-model  Qwen/Qwen3-4B-Instruct-2507 \
  --sub-backend kvzip --press kvzip \
  --memory-budget 1 --memory-budget-unit GB \
  --max-subcall-chars auto --target-compression-ratio 0.5 \
  --sub-max-tokens 128 --sub-min-free-gib 14 \
  --exec-timeout 60 --run-timeout 3600 \
  --out evaluation/results
```

- **`--press no_press` is the control.** It keeps the identical load and generate path, with
  no pruning.
- **`--max-subcall-chars auto` needs two things:** `--sub-backend kvzip`, and one of
  `--target-compression-ratio` / `--compression-factor`.
- **Auto sizing reads live GPU free memory.** It is not reproducible across hosts. The
  resolved size is saved as `max_subcall_chars_resolved`, and a resume that would drift by
  more than 2% aborts.

### 8.3 The fixed-chunk grid (longbench128k)

`--fixed-chunk` makes a cell exact:
- every sub-call is token-truncated to N = B × F;
- the ratio is exactly 1 − 1/F;
- the run fails hard if the model window or the GPU would shrink the cell.

Build the six-subset data pack once. The build reads `$RLM_DATA_DIR/longbench.jsonl`, the
standard LongBench dump:

```bash
python -m evaluation.benchmarks.longbench128k.build_dataset --per-subset 60
```

The root server sits on its own GPU, with `--max-model-len 139264`. Every cell's sub-model
runs on a **different** GPU (`GPU=`). Two values derive from N (Qwen3-4B is 144 KiB per
token):

| N | `--max-sub-calls` | `--sub-min-free-gib` |
| ---: | ---: | ---: |
| 1024 | 64 | 14 |
| 2048 | 64 | 14 |
| 4096 | 33 | 14 |
| 8192 | 17 | 14 |
| 16384 | 9 | 15 |
| 32768 | 5 | 18 |
| 65536 | 3 | 23 |
| 131072 | 2 | 34 |

`run_grid.sh` computes both for you. It also locks each cell (with `mkdir`, because `flock`
does not lock across hosts on this NAS), and skips a cell whose `metrics.json` exists. The
commands below use four variables:
- `REPO` is your checkout.
- `KVPY` is the repo env's python (`$PY` from §2.3).
- `RLM_DATA_DIR` holds `longbench128k/<subset>/data.parquet`. The loader reads it from the
  environment, so export it.
- `BASE` is where results go. For lanes on several hosts, use one shared absolute path such
  as `/mnt/nas/$USER`.

```bash
cd "$REPO"
DATASETS=hotpotqa BUDGETS=4096 FACTORS=8 LIMIT=10 \
  GPU=1 PORT=8000 \
  REPO_ROOT="$REPO" KVZIP_PYTHON="$KVPY" RLM_DATA_DIR="$RLM_DATA_DIR" HF_HOME="$HF_HOME" \
  RESULTS=$BASE/fixedgrid/myrun \
  bash evaluation/rlm/fixedgrid/run_grid.sh
```

| var | default | note |
| --- | --- | --- |
| `DATASETS` | all six | `--data-dir` values |
| `BUDGETS` | `1024 2048 4096 8192` | `B`, in retained KV tokens |
| `FACTORS` | `1 2 4 8 16` | `F` |
| `LIMIT` | `50` | rows per cell = first `LIMIT` of the 60-row pack |
| `GPU` | `0` | **sub-model** GPU (not the server's) |
| `PORT` | `8000` | where the vLLM root is served |
| `MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | root and sub |
| `RESULTS` | `evaluation/results/fixedgrid` | **use a fresh dir per `LIMIT`** |
| `MIN_FREE_GIB` | derived | overrides `--sub-min-free-gib`; its flat 12 GiB headroom is sized for a bigger model |

To run one cell with full control, call `run_benchmark` directly:

```bash
cd "$REPO"
B=4096; F=8; SUBSET=hotpotqa; NEX=10; GPU=1
N=$((B*F)); CALLS=$(( (131072 + N - 1)/N + 1 )); [ $CALLS -gt 64 ] && CALLS=64
MINFREE=$(awk -v n=$N 'BEGIN{x=n*147456*1.2/(2^30);m=int(x);if(x>m)m++;m+=12;if(m<14)m=14;print m}')
PRESS=kvzip; [ "$F" -eq 1 ] && PRESS=no_press

CUDA_VISIBLE_DEVICES=$GPU "$KVPY" -m evaluation.rlm.run_benchmark \
  --dataset longbench128k --data-dir "$SUBSET" --limit "$NEX" \
  --base-url http://localhost:8000/v1 \
  --root-model Qwen/Qwen3-4B-Instruct-2507 --sub-model Qwen/Qwen3-4B-Instruct-2507 \
  --mode rlm --scratchpad --max-steps 50 --exec-timeout 60 --run-timeout 3600 \
  --sub-backend kvzip --press "$PRESS" \
  --memory-budget "$B" --memory-budget-unit tokens \
  --max-subcall-chars auto --compression-factor "$F" --fixed-chunk \
  --max-sub-calls "$CALLS" --sub-max-tokens 128 --sub-min-free-gib "$MINFREE" \
  --out $BASE/fixedgrid/myrun
```

**A healthy cell** has:
- `errors == 0`;
- `average_sub_context_tokens` within about 2% of N;
- `average_sub_compression_ratio` within 0.01 of 1 − 1/F;
- `subcall_sizing_binding == "budget"`.

`document_coverage_fraction` is a diagnostic, not a pass/fail. Audit a tree for cells whose
sub-model never really ran, then render the tables:

```bash
python evaluation/rlm/fixedgrid/audit_cells.py $BASE/fixedgrid/myrun --quiet   # exits 1 on any flagged cell
python evaluation/rlm/fixedgrid/grid_table.py --results $BASE/fixedgrid/myrun
```

To go faster, run more lanes with **disjoint** `DATASETS`, `BUDGETS` or `FACTORS`. Give each
lane its own `GPU=`, and point them all at the **same absolute `RESULTS`**. `--split` does not
apply to longbench128k: `--limit N` takes the first N rows.

### 8.4 Reading a KV arm's results

These numbers in `metrics.json → runtime` decide whether the arm measured anything:
- **`sub_split_call_fraction`:** how often the root used the two-argument form. Near 0 means
  the arm fell back to dense one-argument calls, and measured nothing about the press.
- **`average_sub_retained_context_tokens` vs `average_sub_context_tokens`:** what the budget
  actually bought. It is not the ratio you asked for unless the root filled the chunk.
- **`average_peak_context_tokens`:** the cost axis, reported as `max(root, sub)`. The sub
  model holds a whole slice of unfreed KV, so the root-only figure
  (`average_peak_context_tokens_root`) flatters the arm.

---

## 9. Off infolab: RunPod, other VMs, SLURM

The recipe is the same everywhere. What changes is the paths, there is no `sg infolab`, and
there is no structured campaign on disk to compare LOFT against.

**Choose a machine.**
- **GPU:** at least 24 GB. A 48 GB card (RTX A6000, A40 or L40S) matches the reported runs
  most closely. On a 24 GB card, raise `GPU_UTIL` to 0.85.
- **Image:** a CUDA 12 PyTorch template on Ubuntu, with at least 8 vCPUs. Server A needs a
  CUDA 12.8 driver.
- **Storage:** a volume of at least 50 GB mounted at `/workspace`, holding the code, caches,
  environments and results. On RunPod the container disk is not preserved when a pod is
  recreated, but the volume is.

**One-time setup, onto the volume** (this builds server B and the repo env):

```bash
apt-get update && apt-get install -y tmux git curl rsync   # most images lack tmux
cd /workspace
git clone https://github.com/skylight-org/agentic-longcontext.git benchmark   # needs access; use a token or an SSH key
cd benchmark && git checkout gautam-dev

cat > /workspace/rlm_env.sh <<'EOF'
export REPO=/workspace/benchmark
export S=/workspace/rlmorder
export HF_HOME=/workspace/hf_cache
export PY=/workspace/benchmark/.venv/bin/python
export VLLM=/workspace/venvs/vllm/bin/vllm
export UV_CACHE_DIR=/workspace/cache/uv PIP_CACHE_DIR=/workspace/cache/pip
EOF
source /workspace/rlm_env.sh          # run this again in every new shell
mkdir -p $S/logs $HF_HOME

pip install uv
uv sync --extra eval                                          # the client: /workspace/benchmark/.venv
python3 -m venv /workspace/venvs/vllm
/workspace/venvs/vllm/bin/pip install vllm==0.8.5.post1       # the server
```

**Stage, and start the server:**

```bash
cd $REPO && source /workspace/rlm_env.sh
PYTHONPATH=. $PY -m evaluation.rlm.rlm_order_stage --dataset all --weights
```

```bash
GPU=0 PORT=8002 GPU_UTIL=0.5 VLLM=$VLLM HF_HOME=$HF_HOME LOG=$S/logs/server.log SESSION=ro_root \
  bash evaluation/rlm/serve_root.sh
```

Then wait for a real generation (§4.5). Keep `--max-model-len 65536 --enforce-eager -O0`, so
your runs are comparable with the reported ones.

**Lanes, gate and report** are the §6.3 loops, without `sg infolab`. Lanes are light, so they
can share a core:

```bash
cd $REPO && source /workspace/rlm_env.sh
R=$S/results_ruler128k_pilot; mkdir -p $R
core=0; n=$(nproc)
for arm in bm25-graph bm25-query; do
  i=1
  for tasks in "niah_single_1 cwe" "niah_single_2 fwe" "niah_single_3 vt" "niah_multikey_1 qa_1" \
               "niah_multikey_2 qa_2" "niah_multikey_3 niah_multivalue" "niah_multiquery"; do
    tmux new-session -d -s "ro_${arm#bm25-}_p$i" \
      "DATASET=ruler128k LIMIT=3 REPO=$REPO PY=$PY HF_HOME=$HF_HOME RESULTS=$R \
       bash $REPO/evaluation/rlm/rlm_order_lane.sh $arm $((core % n)) $tasks > $S/logs/ruler_pilot_${arm}_p$i.log 2>&1"
    core=$((core+1)); i=$((i+1))
  done
done
```

- **LOFT's graph order is rebuilt.** The default `ORG` points at the lab's NAS, so set
  `ORG=$S/organization_loft` on each LOFT lane.
- **Gate LOFT with `--no-comparator`.** The structured campaign's results are not on the VM;
  the flag skips only the checks against them.

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 $PY -m evaluation.rlm.rlm_order_gate $S/results_ruler128k_pilot --dataset ruler128k --pilot --expect 3
PYTHONPATH=. HF_HUB_OFFLINE=1 $PY -m evaluation.rlm.rlm_order_gate $S/results_loft_pilot --dataset loft --pilot --expect 3 --no-comparator
```

```bash
DATASET=ruler128k STAGE=ruler_full SCRATCH=$S PY=$PY HF_HOME=$HF_HOME ARCHIVE=$S/archive SERVER=ro_root \
  bash evaluation/rlm/rlm_order_postrun.sh
```

**Take the results with you, then stop the pod.** A running pod bills by the hour, and a volume
bills for as long as it exists.

```bash
rsync -avz -e "ssh -p <port> -i <key>" root@<pod-ip>:/workspace/rlmorder/archive/ ./rlmorder_archive/
```

**B × F cells off infolab.** `run_bxf.sh`'s defaults point at shared environments on the lab
NAS. Set `HF_HOME` (the download folder), `KVZIP_PYTHON` (the repo env's python) and `VLLM`
(a `vllm==0.8.5.post1` install), then run the same command as §7.

**SLURM.**
- **Stage on the login node,** where there is internet. Compute nodes are often air-gapped,
  so run them with `HF_HUB_OFFLINE=1`.
- **`evaluation/rlm/slurm/download_data.sh`** prefetches the shared datasets into
  `$HF_HOME`.
- **`evaluation/rlm/loft128k/slurm/loft128k_a100.slurm`** is an array job that starts one vLLM
  per dataset. It has **never been submitted**, so verify the partition with `sinfo -s`
  before trusting it.
- **LOFT's data comes from the third-party `f20180301` namespace** on the Hub, which must
  stay public for staging to work.

---

## 10. Reading results

Every run, from either backend, writes its own directory:

| file | what it holds |
|---|---|
| `metrics.json` | **the score**, from the benchmark's canonical scorer, plus the harness's aggregates under `runtime` |
| `predictions.csv` | one row per question: prediction, gold, per-row metrics |
| `config.yaml` | every flag the run used; it also guards resumes |
| `checkpoint.jsonl` | raw per-row records: end reason, tokens, latency, chunk reads; it drives resume |
| `transcripts/` | (RLM) every root step, code block and sub-call, per question |

- **The score is `metrics.json`.** The `correct` column in `predictions.csv`
  (`progress_match_loose`) is a looser progress flag, and will not match it.
- **Never rescore from `predictions.csv` with `pd.read_csv` alone.** `answers` comes back as
  a string, and multi-gold rows collapse into one bogus answer. `ast.literal_eval` it, or use
  `checkpoint.jsonl`.
- **The metrics' scales differ.**
  - LOFT runs 0–1: `subspan_em` for nq, hotpotqa and musique, and `coverage` for qampari and
    quest.
  - RULER's `string_match` runs 0–100.
  - At n=55, one row moves a mean by 0.018 on LOFT, or 1.8 points on RULER. Deltas smaller
    than that are noise.
- **Two LOFT scoring quirks are faithful to LOFT and deliberately kept.**
  - On qampari and quest, a list repr (`['France', 'Spain']`) scores and prose does not.
  - A bracketed citation, as in `Final Answer: Paris [1]`, extracts as `["1"]`.
- **Vanilla runs must have `truncated_fraction` 0.** Otherwise the comparison is invalid.
- **Only LOFT's RAG family has a loader:** nq, hotpotqa, musique, qampari and quest. Its
  retrieval, ICL and SQL tasks do not.

Joining runs, and paired statistics:

```bash
python -m evaluation.compare --dataset loft --backend rlm --csv comparison.csv
```

`python -m evaluation.rlm.paired --a <run_dir> --b <run_dir>` gives a paired bootstrap CI,
W/L/T and a sign test. It refuses arms that ran different rows.

---

## 11. Troubleshooting

| symptom | cause and fix |
|---|---|
| `…/venv-vllm/bin/vllm: No such file or directory` | `$S` is unset (`source /tmp/$USER/rlm/env`), or §2.2 was not done **on this host**. `/tmp` is per host. A venv holding only `bin/python` means the install step failed, so rerun it. |
| `import vllm` / CUDA driver error | The host's driver is older than the build needs: fox (12.5) with server A, or ant's Blackwell cards with server B. Use another host. |
| vLLM dies at start-up with a GPU UUID in `CUDA_VISIBLE_DEVICES` | vLLM 0.8.5 needs a device index. `serve_root.sh` resolves the UUID for you. |
| server exits during start-up with a KV-cache or memory error | `--gpu-memory-utilization` is too low for the window, or a co-tenant took memory meanwhile. Pick an emptier card, or see §4.2. |
| server dies ~5 min into start-up with `Qwen2Tokenizer has no attribute all_special_tokens_extended` | transformers 5.x in the vLLM 0.8.5 venv. Install `transformers==4.51.3` into it, last. |
| `address already in use` / `port … is already listening` | An old server is still up. Kill its tmux session, or pick another port. |
| `PREFLIGHT FAILED` at a lane's start, or a campaign that makes no rows | The server never completed a generation: it is still loading, or it is wedged. Test a real generation (§4.5), and restart it by killing its whole tmux session. |
| vanilla rows error with HTTP 400 | The prompt is longer than the window. Lower `--vanilla-max-prompt-tokens`, or raise `--max-model-len`. |
| runs are very slow, and transcripts show `<think>` | A thinking model without `--no-think`. |
| `Launch this lane with primary group infolab` | Wrap the command in `sg infolab -c "…"`. |
| `couldn't be found on the Hugging Face Hub (offline mode is enabled)` | Nothing is staged under this `HF_HOME`. Rerun §6.2 with the same variables the lanes use. |
| Gate: `chunk reads priced differently from the server` | The client's tokenizer and the server's disagree. Serve the same model snapshot you staged. |
| Gate: `expected chunk_order_revision N` | The results come from another harness revision. Gate them with `--revision`. |
| A run refuses to resume | Its directory was written with a different resume-critical setting. Use a fresh results directory for every stage, and every `LIMIT`. |
| B × F: `GPU 0 already holds … MiB` | Someone else is on that GPU. Pass `GPU=<n>`, or wait. |
| B × F: `B x F asks for reads of N=… GiB free` | The cell doesn't fit. Lower `GPU_UTIL`, B or F (§7.3). |
| `OSError: [Errno 122] Disk quota exceeded` | Something is writing to `$HOME`. Source the env file before *every* command, including `uv` and `hf`. |
| `sg: failed`, or permission denied on `/mnt/nas` | You are not in the `infolab` group on this host. Ask the admins. |
| lanes crawl | Too many lanes for the KV cache, or someone else is on the GPU. Watch `nvidia-smi`, and the server log's `Running:` and `Waiting:` counts. |
