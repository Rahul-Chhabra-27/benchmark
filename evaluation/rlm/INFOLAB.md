# Running RLM benchmarks on the infolab hosts

The shortest path from an infolab account to a scored run: a model served by vLLM on one
GPU, and the vanilla and RLM arms run against it. No SLURM, no launcher script — every
step is a command you can read and rerun. The worked example is LOFT-128k (`nq`,
`hotpotqa`) on `Qwen/Qwen3.5-4B`; section 8 covers other models and datasets.

Sections 1–5 tested verbatim on elk, 2026-09-14, from a clean `/tmp/$USER/rlm` with `master`
at `c92f4e2` and again on `gautam-dev` at `eba22c9` (setup ~4 min, server ready ~5 min, smoke: both
arms 0 errors). Section 6 is the same command at `--split test --limit 55`. The reference scores in
section 7 came from a 55-row run on `master` `c92f4e2` with these flags.

## 0. Before you start

- **Host.** Any infolab GPU host whose driver reports **CUDA 12.8 or newer**
  (`nvidia-smi`, top right). bee and elk qualify. fox reports 12.5, so the vLLM build
  used here will not load there.
- **GitHub access** to `skylight-org/agentic-longcontext` (formerly `intel-longcontext-pc`) over SSH from that host
  (`ssh -T git@github.com` should greet you).
- **Where things go.** Everything below lives in `/tmp/$USER/rlm` (`$S`), the host's local disk:
  venvs (~12 GB), weights (~9 GB), caches, results. **Not `$HOME`**: it is a small,
  nightly-backed-up NFS quota, and a large write that stalls there can hang the
  host. Its own subdirectory keeps it clear of anything else you keep in `/tmp/$USER`.
  `/tmp` is per host and not backed up, so do section 2 once **on each host** you
  use, and copy results you want to keep to NAS (section 7).
- **Sharing.** GPUs are shared and nobody reserves them. Take one card, run under
  `tmux` (an ssh drop otherwise kills everything), and free the card when done.

## 1. Environment (every new shell)

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

Every command below assumes `$S` is set. A command that fails with a path like
`/venv-vllm/bin/...` means you skipped this.

## 2. One-time setup per host (~5 min on elk; almost all of it downloads)

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

**Why two venvs.** The server needs vLLM's exact torch/transformers stack. The driver
only speaks HTTP, needs pandas, scipy and tiktoken, and must never be the reason the
server's pins move. Do not `pip install` extras into `venv-vllm`.

## 3. Pick a GPU and a port

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
```

Pick a card showing near-0 MiB **and** 0 %. A card with little memory in use can still be
at 99 % utilisation from someone's small job. Then pick a port nobody is listening on:

```bash
GPU=2 PORT=8013
ss -ltn | grep -q ":$PORT " && echo "port $PORT is taken, pick another"
```

## 4. Start the server (tmux window 1)

```bash
tmux new -s rlm
source /tmp/$USER/rlm/env
GPU=2 PORT=8013 MODEL=Qwen/Qwen3.5-4B

CUDA_VISIBLE_DEVICES=$GPU $S/venv-vllm/bin/vllm serve $MODEL \
  --served-model-name $MODEL --port $PORT \
  --max-model-len 139264 --gpu-memory-utilization 0.5 \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

- `--max-model-len 139264` holds one full 128k document plus the answer.
- `--gpu-memory-utilization` is a fraction of the **whole card**, not of the free part.
  At 0.5 this takes ~25 GB of a 48 GB card. Qwen3.5 stores only ~32 KiB of KV per token,
  so that still fits ~3 full-length requests at once.
- `--limit-mm-per-prompt` switches off Qwen3.5's image/video towers, since the benchmark
  sends text only.

The first start takes ~5 minutes (loading, then compiling). It is ready when this prints
the model id (run it from any other shell):

```bash
curl -s localhost:8013/v1/models
```

## 5. Smoke test: 1 row per arm (tmux window 2: `Ctrl-b c`)

Put the flags shared by every run into one array, so each run line shows only what
differs:

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

Pass: both runs finish with `"errors": 0`. A single row's score means nothing, since it is
one question. A vanilla error almost always means the server window is smaller than the
prompt (see Troubleshooting).

## 6. The real run: 55 test rows, nq + hotpotqa, both arms

Same window, same `RUN` array. All four lanes run in parallel against the one server:

```bash
mkdir -p $S/logs
for ds in nq hotpotqa; do for mode in vanilla rlm; do
  "${RUN[@]}" --dataset loft --data-dir ${ds}_128k --split test --limit 55 \
    --mode $mode --out $S/results > $S/logs/${ds}_${mode}.log 2>&1 &
done; done
wait; echo ALL DONE
```

This takes ~35 min. The RLM lanes are the slow ones (~36 s/row on nq). To watch, open another
window and run `tail -f $S/logs/nq_rlm.log`.

**Resuming.** Each run directory has a `checkpoint.jsonl`. If a lane dies or you
Ctrl-C, rerun the same line and it skips finished rows. Change nothing else. A
different flag is a different run directory.

## 7. Results

```bash
grep -H '"subspan_em"' $S/results/*/metrics.json      # headline score per run
$S/venv-drv/bin/python -m evaluation.compare --results-dir $S/results
```

Each run directory
`loft__<ds>_128k__Qwen_Qwen3.5-4B__<mode>__split-test/` contains:

| file | what |
|---|---|
| `metrics.json` | the score (`subspan_em` for LOFT) plus runtime/cost counters |
| `predictions.csv` | one row per question: prediction, gold, per-row metrics |
| `config.yaml` | every flag the run used |
| `checkpoint.jsonl` | raw per-row records; drives resume |
| `transcripts/` | (rlm) every root step, code block and sub-call, per question |

The score is **`metrics.json`**, produced by the canonical scorer. The `correct` column in
`predictions.csv` is a looser progress flag and will not match it.

**Reference** (`master` `c92f4e2`, same 55 rows, greedy): RLM `nq` **0.527**, `hotpotqa`
**0.491**, 0 errors. A correct setup lands within a question or two of these.

**Keep and clean up.** Stop the server with Ctrl-C in window 1 and check that the card is
free (`nvidia-smi`). Copy what you want to keep to NAS archival: writes there need the
`infolab` group. Then close tmux:

```bash
sg infolab -c "mkdir -p /mnt/nas/$USER/loft128k_qwen35 && cp -r $S/results $S/logs /mnt/nas/$USER/loft128k_qwen35/"
tmux kill-session -t rlm
```

## 8. Adapting it

**Other LOFT subsets and lengths.** `--data-dir` is `<subset>_<length>`, with subset in
`nq hotpotqa musique qampari quest` and length in `32k 128k 1m`. The data downloads from the Hub
(`f20180301/loft-rag-*`) on first use. At 128k and 1m each subset has 10 dev + 100 test
rows, so `--split test --limit 100` is the whole test split. On `master`, test row ids run
10–109.

**Always pass `--split`.** Without it, `--limit 10` or less evaluates dev rows only, and
the result lands in the tree looking like a real run.

**1m.** A 1m document does not fit any window here, so run `--mode rlm` only. The root only
ever holds REPL output (peak ~2–3k tokens), so a smaller server window is fine for
RLM-only runs.

**Other models.** Set `MODEL` in both windows. Then adjust three things:

| what | rule |
|---|---|
| `--no-think` | keep it for thinking models (Qwen3, Qwen3.5); **drop** it for non-thinking ones (`Qwen3-4B-Instruct-2507`). Without it, a thinking model reasons out loud and runs far slower |
| `--max-model-len` / `--vanilla-max-prompt-tokens` | the prompt cap must stay below the window minus the answer: 139264 / 134000 for 128k. Check the model's `max_position_embeddings` covers the window |
| `--gpu-memory-utilization` | weights + one full-window sequence of KV, as a fraction of the card. `Qwen3-4B-Instruct-2507` has 144 KiB/token of KV, so 128k needs ~31 GB, which is **0.65** on a 48 GB card. Too low: the server refuses to start and says how much it needs |
| `--limit-mm-per-prompt` | Qwen3.5 only (it is multimodal). Text-only models reject the flag, so drop it |

**KV-compression arms (`--sub-backend kvzip`).** These load a second model in-process and
need the full repo environment rather than `venv-drv`. See
[`loft128k/README.md`](loft128k/README.md) and [`REPRODUCE.md`](../../REPRODUCE.md).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `…/venv-vllm/bin/vllm: No such file or directory` | `$S` unset (run `source /tmp/$USER/rlm/env`), or section 2 not done **on this host**. `/tmp` is per host, and a venv holding only `bin/python` means the install step failed, so rerun it |
| `import vllm` / CUDA driver error | host driver older than CUDA 12.8 (fox). Use another host |
| server exits during startup with a KV-cache/memory error | `--gpu-memory-utilization` too low for the window, or a co-tenant took memory in the meantime. Pick an emptier card or raise the fraction |
| server exits with `address already in use` | port taken. Pick another |
| vanilla rows error with HTTP 400 | prompt longer than the window. Lower `--vanilla-max-prompt-tokens` or raise `--max-model-len` |
| runs are very slow, transcripts show `<think>` | thinking model without `--no-think` |
| `OSError: [Errno 122] Disk quota exceeded` | something is writing to `$HOME`. Source the env file before *every* command, including `uv` and `hf` |
| `sg: failed` / permission denied on `/mnt/nas` | you're not in the `infolab` group on this host. Ask the admins |
