# How to run the B × F tests

This page is for anyone who needs to run the B × F experiments. You don't need to know how
the code works. For most people it comes down to one command.

---

## What is being tested (a 30-second version)

- A small model (Qwen3-4B) answers a question about a very long document, about 128,000
  tokens. It can't read everything at once, so it reads the document in **chunks**, and a
  second copy of the model (the **reader**) reads each chunk it asks for.
- In a **B × F** test, the reader **compresses** what it reads, using a method called KVzip:
  - **B** is how much memory one read may keep, in megabytes. For this model, 512 MB is
    3,472 tokens.
  - **F** is how hard each read is squeezed. F = 4 keeps a quarter of what was read.
  - So one read may be up to **B × F** tokens long, which is 13,888 for 512 MB × 4.
  - **F = 1 means no compression.** It is the baseline that the other values of F are
    compared against.
- There are two ways the chunks are ordered, called **arms**:
  - `bm25-query`: chunks sorted by how well they match the question;
  - `bm25-graph`: chunks grouped by topic before the question is known.

Every run tries both arms.

---

## 1. Check that the code works (no GPU, about 1 minute)

Do this on any machine, such as a laptop, after cloning the repository:

```bash
pip install uv
uv sync --extra eval
.venv/bin/python -m pytest tests/evaluation -q
```

Every test should pass. These tests use fake models, so they need no GPU and no internet.

---

## 2. The smoke test: one command (about 5 minutes)

The smoke test checks, end to end, that a GPU run works. It is small on purpose: 2 questions,
both arms, B = 512 MB, and F = 1 and F = 4.

On the lab server `cat`:

```bash
ssh cat.cse.iitb.ac.in
cd ~/rlmorder/checkout          # your clone of the repository
git checkout gautam-dev && git pull
tmux new -s bxf                 # optional, but it keeps the run alive if your connection drops
bash evaluation/rlm/run_bxf.sh
```

That is all. The script does the following, in order:
1. starts the model server on GPU 0 and waits for it (a few minutes the first time);
2. runs the 4 test cells one after another (2 arms × 2 values of F);
3. prints a results table;
4. shuts the model server down again.

It ends with `4 of 4 cells passed their checks`, and exits with status 0 when everything
worked.

**What the smoke test printed on 2026-09-12** (cat, GPU 0, 4 minutes from start to finish):

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

How to read it:
- **Every cell got exactly its F.** The `F_real` column equals F, and at F = 4 every read was
  compressed (`pressed` 1.0).
- **No read kept more than B.** `kept_max` is at most `B_tok` everywhere.
- **The `REVIEW` line is model behaviour, not a failure.** On one question the model got stuck
  and was stopped before it read a chunk.
- **The scores mean nothing on 2 questions.** The smoke test only proves the pipeline works.
- **A `kept_max` well below `B_tok` is normal.** It means the model chose to read fewer chunks
  per call than it was allowed to. Here, about 4,000 of the 13,888 tokens a read could hold.

---

## 3. A real run

Put the settings you want in front of the same command. For example, all five LOFT subsets,
the 55 standard questions each, two budgets and four factors:

```bash
LIMIT=55 SUBSETS="nq hotpotqa musique qampari quest" BUDGETS_MB="512 1024" FACTORS="1 2 4 8" \
  bash evaluation/rlm/run_bxf.sh
```

The cells run one at a time. Start with a small `LIMIT` to see how long one cell takes, then
scale up. **If it stops part-way, run the same command again:** it skips every question
already answered.

### Settings you can change

Every setting is optional. Leave one out to get its default.

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

### How big can B × F be?

A read's full, uncompressed memory sits on the GPU while it is being read. On one 48 GB GPU,
next to the model server:

| B | F = 4 | F = 8 |
|---|---|---|
| 512 MB | fits | fits |
| 1024 MB | fits | lower `GPU_UTIL` to about `0.35` |
| 2048 MB | lower `GPU_UTIL` to about `0.35` | needs a GPU of its own |

If a cell doesn't fit, the run stops before it starts and says so. It never quietly produces
a wrong result.

---

## 4. Reading the results table

One row per cell (B, F, arm, subset):

| column | meaning |
|---|---|
| `score` | the answer score for that subset (higher is better) |
| `rows` | how many questions were answered |
| `N` | the longest a single read was allowed to be (B × F, in tokens) |
| `B_tok` | B in tokens |
| `kept_max` | the most tokens any single read actually kept; it must be ≤ `B_tok` |
| `F_real` | the compression that was actually applied; it should equal F |
| `pressed` | the share of reads that were compressed; 1.0 for F > 1 and 0.0 for F = 1 |
| `check` | `PASS` if the cell got exactly the B and F it asked for, with no errors |

Lines starting with `REVIEW` are questions where the model got stuck or never read a chunk.
That is model behaviour, not a bug. Lines starting with `FAIL` are real problems, and say what
went wrong.

**Where things are saved:**
- results: `$OUT/results/bxf_<B>MB_x<F>/`, one folder per run, holding every answer and the
  model's full working (`transcripts/`);
- logs: `$OUT/logs/`, one log per cell, plus `server.log`.

`/tmp` is wiped on reboot. Copy anything you want to keep to `/mnt/nas/$USER/`.

---

## 5. If something goes wrong

| you see | what to do |
|---|---|
| `GPU 0 already holds … MiB -- someone else is on it` | someone else is using that GPU. Pick a free one with `nvidia-smi` and pass `GPU=<n>`, or wait |
| `port 8002 is already listening` | an old server is still up. Stop it with `tmux kill-session -t bxf_root`, then re-run |
| `the server did not come up` | read `$OUT/logs/server.log` |
| `B x F asks for reads of N=… GiB free` | that cell doesn't fit. Lower `GPU_UTIL`, B or F |
| `!! this cell failed` | the last lines of its log are printed; the whole log is in `$OUT/logs/` |
| `FAIL …` in the summary | the line says what the cell didn't get right; read that cell's log |
| you want to start over | delete `$OUT/results` (or set a new `OUT=`), then re-run |

---

## 6. Running somewhere other than the lab servers

The defaults point at shared environments on the lab NAS. Anywhere else, you need:
- **a GPU with at least 48 GB**, or 24 GB for small cells with a lowered `GPU_UTIL`;
- **the model and datasets downloaded once**: see section 3.3 of
  [`how_to_run.md`](how_to_run.md) (`python -m evaluation.rlm.rlm_order_stage --dataset all --weights`);
- **three settings pointing at your own copies**: `HF_HOME` (the download folder),
  `KVZIP_PYTHON` (a Python with this repository's requirements plus torch and kvpress), and
  `VLLM` (a `vllm==0.8.5.post1` install).

Then run the same command. How the pieces fit together is explained in
[`how_to_run.md`](how_to_run.md) and [`evaluation/rlm/README.md`](evaluation/rlm/README.md).
