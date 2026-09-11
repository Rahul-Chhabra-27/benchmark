# How to run the ordered-chunk RLM evaluation

This guide takes you from a fresh clone to a gated, reported run. It covers an infolab
server (cat, fox, …) and any other GPU machine, such as a RunPod pod.

**What gets run.** The harness in `evaluation/rlm/` gives an agentic RLM root
(Qwen3-4B-Instruct-2507, writing Python in a REPL) every 2000-character chunk of a
128k-token document, in a fixed order. The root reads the chunks through `llm_query`.
There are two arms, which differ only in that order:

| arm | flag | how `chunks` is ordered |
|---|---|---|
| query-aware | `--chunk-order bm25-query` | every chunk, ranked by BM25 against the question |
| query-agnostic | `--chunk-order bm25-graph` | a BM25-graph organization built before the question is known |

A campaign runs both arms on two benchmarks:
- **LOFT-128k:** nq, hotpotqa, musique, qampari and quest, on the first 55 test rows of each.
- **RULER-128k:** all 13 tasks, on the first 55 rows of each.

The design, every decision and the results are in
[`evaluation/rlm/RLM_ORDER_STATE.md`](evaluation/rlm/RLM_ORDER_STATE.md). This file is only
about running it. Every command below runs from the repository root unless it says otherwise.

**The short version:**
1. Set up two Python environments: the client, and vLLM.
2. Stage the model and the data once, with network access (`rlm_order_stage`).
3. Start one vLLM server.
4. Run a 3-row pilot, and gate it.
5. Run the full campaign, one lane per arm.
6. Run the post-run script. It gates the run, writes the report, and archives everything.

---

## 0. What you need

**For the unit tests:** any machine with Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).
No GPU, model or network is needed after the install.

**For a campaign:**
- **One NVIDIA GPU with at least 24 GB.** The reported runs used a 48 GB RTX A6000 with
  vLLM at `--gpu-memory-utilization 0.5`, which is about 24 GB. Qwen3-4B's bf16 weights
  are about 8 GB, and its KV cache takes about 147 KB per token.
- **A CUDA 12 driver.** The runs used driver 570, with CUDA 12.8.
- **About 25 GB of free disk:**
  - about 8 GB of model weights;
  - about 4.5 GB for RULER-128k (a 1.5 GB download, prepared into about 3 GB);
  - about 0.25 GB for LOFT-128k's five subsets;
  - a few GB of results.
- **Network access once,** to download the model and the datasets. The runs themselves are
  offline.
- **`tmux`, `curl` and `taskset`,** which are standard on Linux. A bare container may need
  `apt-get install -y tmux`.
- **Two Python environments:**
  - **The client** runs the harness, scorers, gate, report and tests. It is the repo's own
    environment: `uv sync --extra eval`.
  - **The server** is `vllm==0.8.5.post1`, in its own venv. That is the version the
    reported runs used. vLLM pins its own torch and transformers, so keep it apart from the
    client.

---

## 1. Unit tests (no GPU)

```bash
pip install uv                        # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra eval                  # .venv with the harness and the dev tools (pytest, black, ...)
.venv/bin/python -m pytest tests/evaluation -q
```

These tests use fake model clients and a character-count tokenizer, so they need no GPU,
network or weights. Every one should pass. `tests/test_pipeline.py` exercises KVPress on
real models, and is not part of this harness.

Style checks, the same ones the code was held to:

```bash
.venv/bin/python -m black --check evaluation/rlm tests/evaluation     # line length 120, from pyproject.toml
.venv/bin/python -m isort --check-only evaluation/rlm tests/evaluation
.venv/bin/python -m flake8 evaluation/rlm tests/evaluation
```

---

## 2. How a run fits together

```
                 ┌──────────────────────── one GPU ────────────────────────┐
                 │  vLLM: Qwen3-4B-Instruct-2507 on :8002 (root AND reader) │
                 └──────────────────────────────────────────────────────────┘
                        ▲ HTTP        ▲ HTTP                   ▲ HTTP
   lane: graph × tasks ─┘   lane: query × tasks          ...  (10–14 lanes at once)
      evaluation/rlm/rlm_order_lane.sh  →  python -m evaluation.rlm.run_benchmark --mode rlm --chunk-order <arm>
      writes  $RESULTS/<run dir>/  checkpoint.jsonl · transcripts/ · config.yaml · predictions.csv · metrics.json

   gate      python -m evaluation.rlm.rlm_order_gate     integrity checks, then the rows that need review
   report    python -m evaluation.rlm.rlm_order_report   paired query − graph comparisons, cost, diagnostics
   post-run  evaluation/rlm/rlm_order_postrun.sh         gate → stop the server if it passes → report → archive
```

- **One server does everything.** The root and the chunk reader are the same model on the
  same server.
- **Run many lanes at once.** The GPU batches their requests, so adding lanes is nearly
  free. LOFT (550 rows, 10 lanes) took about 40 minutes on one A6000. RULER (1,430 rows,
  14 lanes) took about 46.
- **Runs are resumable.** Re-running a lane skips every row already in `checkpoint.jsonl`.
- **Every stage goes to a fresh results directory.** The harness refuses to resume into a
  directory whose config differs in a resume-critical key, such as the harness revision.
- **Always pilot first.** Run with `LIMIT=3`, gate it, read a few transcripts, and only
  then launch `LIMIT=55`.

---

## 3. On an infolab server (cat, fox, …)

### 3.1 The rules that apply here

- **Follow the lab storage policy.**
  - `$HOME` (`/mnt/home/$USER`, NFS, backed up nightly) is for code and condensed output
    only. Never put raw results, logs, caches or reports there.
  - Live output goes to the host's data disk, `/mnt/$(hostname)/data/$USER`, if your advisor
    has allocated it. Otherwise it goes to host-local `/tmp/$USER`, which is not durable, so
    archive it after every stage.
  - Archives go to `/mnt/nas/$USER`.
- **Write as the `infolab` group.** Wrap lanes and the post-run script in
  `sg infolab -c "…"`, or run `exec newgrp infolab` in your shell. On any host that has an
  `infolab` group, both scripts refuse to run otherwise.
- **Take a GPU only when it is free.** Check `nvidia-smi` first. `serve_root.sh` refuses a
  card that already holds more than 1 GB.
- **ssh with the full host name,** such as `ssh cat.cse.iitb.ac.in`. There is no short alias.
- **Hosts.** cat has a 48 GB RTX A6000 as GPU 0, alongside smaller cards. fox has five
  48 GB cards and full internet access.

### 3.2 One-time setup

```bash
# The code is the only thing that lives in $HOME.
mkdir -p ~/rlmorder && cd ~/rlmorder
git clone git@github.com:skylight-org/intel-longcontext-pc.git checkout   # or the lab's NAS mirror, if you can read it
cd checkout && git checkout gautam-dev

# Paths used in the rest of this section. Keep them in your shell.
export REPO=$HOME/rlmorder/checkout
export S=/tmp/$USER/rlmorder            # or /mnt/$(hostname)/data/$USER/rlmorder, if allocated
export HF_HOME=$S/hf_cache              # the model, the tokenizer and the datasets, on local disk
sg infolab -c "mkdir -p $S/logs $HF_HOME"
```

**Environments.** The campaign's own environments are on NAS:

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
export PY=$L/venvs/rlm/bin/python VLLM=$L/venvs/vllm/bin/vllm
```

### 3.3 Stage the model and the data (once, with network access)

The lanes and the server run offline (`HF_HUB_OFFLINE=1`), so everything must be cached
first. This downloads the tokenizer, both benchmarks and the weights, then proves they
load offline:

```bash
cd $REPO
sg infolab -c "HF_HOME=$HF_HOME PYTHONPATH=. $PY -m evaluation.rlm.rlm_order_stage --dataset all --weights"
```

It prints one line per task, and ends with `staged: lanes can run with HF_HUB_OFFLINE=1`.
The first run downloads everything. On cat, LOFT's five subsets alone took about 10
minutes, printing nothing while each one downloads, so let it run. `/tmp` does not survive
a reboot. If it is wiped, run the command again.

### 3.4 Start the server

```bash
cd $REPO
GPU=0 PORT=8002 GPU_UTIL=0.5 CPUS=2-5 VLLM=$VLLM HF_HOME=$HF_HOME \
LOG=$S/logs/server.log SESSION=ro_root bash evaluation/rlm/serve_root.sh
```

- `GPU` takes a device index or a UUID. vLLM 0.8.5 crashes on a UUID, so the script
  resolves it to an index for you.
- `CPUS` pins the server to a few cores, which keeps it polite on a shared host.
- Loading takes a minute or two. Then wait for a real generation, not just `/v1/models`:

```bash
until curl -s --max-time 60 localhost:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B-Instruct-2507","messages":[{"role":"user","content":"ok"}],"max_tokens":4}' \
  | grep -q '"content"'; do sleep 15; done; echo READY
```

### 3.5 Run the pilot, gate it, then run the full campaign

**Lanes.** Each lane is one arm on one or more tasks, pinned to one CPU core, and runs in
its own tmux session. The session names follow `ro_<arm>_<lane>`, which the post-run script
uses to tell whether any lane is still running.

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
graph-order campaign's own cache on NAS, `/mnt/nas/gautammahale/archive/graphorder/organization`,
so both campaigns read exactly the same artifact. Leave it as it is unless you cannot read
that path.

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

- Integrity problems fail the gate outright: errors, token counts that disagree with the
  server, missing transcripts, or limit notices.
- Rows that a harness guard ended, or that never read a chunk, are listed under `REVIEW`. A
  run WARNS once more than 10% of its rows are listed, and FAILS past 25%.
- To see why each flagged row is there, run
  `PYTHONPATH=. $PY -m evaluation.rlm.rlm_order_review <results dir>`.

**The full run** is the same loop, with four changes:
- `LIMIT=55`;
- a fresh `RESULTS`: `$S/results_ruler128k` for RULER, or `$S/results` for LOFT, which are
  the post-run script's defaults;
- lane sessions named `ro_<arm>_r$i`;
- logs named `ruler_full_*` or `full_*`.

### 3.6 Post-run: gate, report and archive

Once every lane has printed `DONE`, run the post-run script. If you changed code, pull
first: bash reads a script while it runs it, so never pull while any lane is still running.

```bash
cd $REPO && git pull --ff-only
sg infolab -c "DATASET=ruler128k STAGE=ruler_full SCRATCH=$S PY=$PY HF_HOME=$HF_HOME \
  ARCHIVE=/mnt/nas/$USER/archive/rlmorder SERVER=ro_root bash evaluation/rlm/rlm_order_postrun.sh"
# LOFT: DATASET=loft STAGE=full
```

The script:
1. **gates** the run;
2. **stops the server** (`tmux kill-session -t $SERVER`) if the gate passes. If it fails, the
   server stays up in case rows need a rerun;
3. **writes the report** to `$S/report_ruler128k/ruler_full/report.md`, or to
   `$S/report/full/report.md` for LOFT. Every comparison is repeated with rows 0–2 left out,
   because those rows were used to tune the harness;
4. **archives** results, report, RULER's graph organizations and logs to `$ARCHIVE`, and
   verifies the copy with checksums.

**Stopping the server by hand:** `tmux kill-session -t ro_root`. Kill the whole session.
Killing only the process `nvidia-smi` shows leaves a server that answers `/v1/models`, but
never completes a generation.

---

## 4. On RunPod, or any other GPU VM

The recipe is the same as on infolab. The differences are the paths, no `sg infolab`, and
no structured campaign to compare LOFT against.

### 4.1 Choose a machine

- **GPU.** Take one with at least 24 GB. A 48 GB card (RTX A6000, A40 or L40S) matches the
  reported runs most closely. On a 24 GB card such as the RTX 4090, raise `GPU_UTIL` to 0.85.
- **Image.** A CUDA 12 PyTorch template on Ubuntu, with at least 8 vCPUs.
- **Storage.** Attach a volume of at least 50 GB, mounted at `/workspace`. Keep the code,
  caches, environments and results there. On RunPod, the container disk (including `/root`
  and `~/.bashrc`) is not preserved when a pod is recreated, but the volume is.
- **Access.** Connect with SSH, or with the web terminal.

### 4.2 One-time setup, onto the volume

```bash
apt-get update && apt-get install -y tmux git curl rsync   # most images lack tmux
cd /workspace
git clone https://github.com/skylight-org/intel-longcontext-pc.git benchmark   # needs access; use a token or an SSH key
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

### 4.3 Stage the model and the data (once)

```bash
cd $REPO && source /workspace/rlm_env.sh
PYTHONPATH=. $PY -m evaluation.rlm.rlm_order_stage --dataset all --weights
```

### 4.4 Start the server

Use the same script as on infolab. `CPUS` is optional on a machine you have to yourself:

```bash
GPU=0 PORT=8002 GPU_UTIL=0.5 VLLM=$VLLM HF_HOME=$HF_HOME LOG=$S/logs/server.log SESSION=ro_root \
  bash evaluation/rlm/serve_root.sh
```

Or run the equivalent command directly:

```bash
tmux new-session -d -s ro_root "HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  $VLLM serve Qwen/Qwen3-4B-Instruct-2507 --served-model-name Qwen/Qwen3-4B-Instruct-2507 --port 8002 \
  --max-model-len 65536 --gpu-memory-utilization 0.5 --enforce-eager -O0 2>&1 | tee $S/logs/server.log"
```

Then wait for a real generation, with the `until curl …` loop from §3.4. Keep
`--max-model-len 65536 --enforce-eager -O0`, so your runs are comparable with the reported
ones.

### 4.5 Lanes, gate and report

These are the §3.5 loops, without `sg infolab`. Pick the cores with `$(nproc)` in mind:
lanes are light, and can share a core.

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

- **The infolab group check does not apply.** It only runs on hosts that have an `infolab`
  group.
- **LOFT's graph order is rebuilt.** The default `ORG` points at the lab's NAS, so set
  `ORG=$S/organization_loft` on each LOFT lane. The graph organization is then built from
  each document on first use, and cached there.
- **Gate LOFT with `--no-comparator`.** The structured campaign's results are not on the
  VM. The flag skips only the checks against them, and prints a warning saying so; every
  other check still runs. RULER has no comparator, so its gate command is unchanged:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 $PY -m evaluation.rlm.rlm_order_gate $S/results_ruler128k_pilot --dataset ruler128k --pilot --expect 3
PYTHONPATH=. HF_HUB_OFFLINE=1 $PY -m evaluation.rlm.rlm_order_gate $S/results_loft_pilot --dataset loft --pilot --expect 3 --no-comparator
```

- **The post-run script works unchanged.** For LOFT it switches to `--no-comparator` by
  itself when the structured results are missing, and says so:

```bash
DATASET=ruler128k STAGE=ruler_full SCRATCH=$S PY=$PY HF_HOME=$HF_HOME ARCHIVE=$S/archive SERVER=ro_root \
  bash evaluation/rlm/rlm_order_postrun.sh
```

### 4.6 Take the results with you, then stop the pod

The archive (`$S/archive`) holds the results, report and logs. Copy it off the pod before
you terminate it. From your own machine, using the SSH details the pod's "Connect" panel
shows:

```bash
rsync -avz -e "ssh -p <port> -i <key>" root@<pod-ip>:/workspace/rlmorder/archive/ ./rlmorder_archive/
```

`runpodctl send` / `runpodctl receive` also work. Then stop the pod. A running pod bills
by the hour, and a volume bills for as long as it exists.

### 4.7 Other clouds and clusters

The same steps work on any Linux GPU machine, such as a cloud VM or a bare-metal box: only
the paths change. On a SLURM cluster, stage on the login node (§4.3), then start the server
and the lanes inside one job on a GPU node. `evaluation/rlm/README.md` describes that layout.

---

## 5. One row by hand, for debugging

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 $PY -m evaluation.rlm.run_benchmark \
  --dataset ruler128k --data-dir niah_single_1 --limit 1 \
  --mode rlm --chunk-order bm25-query --organization-dir $S/organization_debug --max-steps 50 \
  --root-model Qwen/Qwen3-4B-Instruct-2507 --sub-model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://localhost:8002/v1 --sub-backend http --out $S/debug --debug
```

- `--debug` prints every step: the root's code, and what that code printed.
- For LOFT, use `--dataset loft --data-dir nq_128k --split test`. For the other arm, use
  `--chunk-order bm25-graph`.
- `python -m evaluation.rlm.run_benchmark --help` lists every flag. The ordered-chunk arms
  require `--mode rlm` and `--sub-backend http`, and refuse `--search-k` and
  `--chunks-source`.

---

## 6. Reading the results

Each run directory, for example `ruler128k__niah_single_1__Qwen_Qwen3-4B-Instruct-2507__rlm__order-bm25-query`,
holds these files:

| file | what it holds |
|---|---|
| `checkpoint.jsonl` | one JSON line per row: prediction, end reason, tokens, latency, chunk reads, where the gold sat |
| `transcripts/<id>.json` | the row's whole REPL session: every cell, and what it printed |
| `metrics.json` | the benchmark's own scorer over the run; `runtime` holds the harness's aggregates |
| `predictions.csv` · `config.yaml` | the shared result contract; the config also guards resumes |

- **The metrics' scales differ.** LOFT's run 0–1: `subspan_em`, or `coverage` for qampari
  and quest. RULER's `string_match` runs 0–100. At n=55, one row moves a mean by 0.018 on
  LOFT, or by 1.8 points on RULER.
- **The report** is `report.md`, alongside `report.csv`, `paired.csv` and `campaign.json`
  (the code revision, the ids and every config). It contains:
  - the headline pairs (query − graph, with a 95% bootstrap CI, wins/losses/ties and a
    sign test);
  - the same pairs on the rows held out from tuning;
  - cost and behaviour;
  - the harness's interventions;
  - where the gold sat in each order.
- **Other analyses:**
  - `python -m evaluation.rlm.rlm_order_review <results>` gives the cause of every flagged row.
  - `python -m evaluation.rlm.rlm_order_replay <results> <dataset>` shows what revision 6's
    grounding rule would have changed in a revision-5 run.

---

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| `PREFLIGHT FAILED` at a lane's start | The server did not complete a generation within 90 s: it is still loading, or it is wedged. A wedged vLLM still answers `/v1/models`. Test a real generation, and restart by killing its whole tmux session. |
| `Launch this lane with primary group infolab` | You are on an infolab host: wrap the command in `sg infolab -c "…"`. |
| `couldn't be found on the Hugging Face Hub (offline mode is enabled)`, or a tokenizer error mentioning `local_files_only` | Nothing is staged under this `HF_HOME`, or `HF_DATASETS_CACHE` differs. Rerun §3.3 or §4.3 with the same variables the lanes use. |
| vLLM dies at start-up with a GPU UUID in `CUDA_VISIBLE_DEVICES` | vLLM 0.8.5 needs a device index. `serve_root.sh` resolves a UUID to its index. |
| Gate: `chunk reads priced differently from the server` | The client's tokenizer and the server's disagree. Serve the same model snapshot that you staged. |
| Gate: `expected chunk_order_revision N` | The results come from another harness revision. Gate them at their own revision with `--revision`; LOFT's reported run is revision 5. |
| Gate: `LOFT requires --structured-results` | Pass the structured campaign's results, as on infolab, or `--no-comparator` elsewhere. |
| A run refuses to resume | Its directory was written with a different resume-critical setting. Use a fresh `RESULTS` directory for every stage. |
| `OSError: [Errno 122] Disk quota exceeded` on infolab | Something is writing under `$HOME`. Move it to scratch; see §3.1. |
| Lanes crawl | Too many lanes for the KV cache, or someone else is on the GPU. Watch `nvidia-smi`, and the server log's `Running:` and `Waiting:` counts. |

---

## 8. Reproducing the reported numbers

- **Which code produced which numbers:**
  - LOFT-128k was run on harness revision 5, at commit `2b62ef2`;
  - RULER-128k was run on revision 6, at commit `c466401`;
  - the current branch head runs revision 6.
- **Reproducing LOFT exactly.** Check out `2b62ef2` in a separate directory, and point
  `REPO` at it, while running the lane and post-run scripts from the current checkout. The
  lanes then run that commit's code, and the new scripts work off infolab.
- **Gating LOFT's revision-5 results.** Use `--revision 5`. The post-run script defaults to
  that for LOFT.
- **What to expect from a rerun.** Decoding is greedy (temperature 0), but vLLM's batching
  numerics vary with how many lanes run at once, so individual rows can differ from run to
  run. Compare means against the one-row step in §6.
- **The reported scores and their caveats** are in `evaluation/rlm/RLM_ORDER_STATE.md`,
  under **Results** (LOFT) and **RULER-128k → Scores**.
