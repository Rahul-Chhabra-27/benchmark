# factor_picker — one compression factor per question, chosen up front

At a fixed memory budget **B**, the compression factor **F** decides how much text each
RLM sub-call reads (**N = B × F** tokens) and how hard kvpress (KVzip) compresses it.
**F = 1 means no kvpress at all**: plain, uncompressed reads.

Until now a whole run used one F. Here a **picker** chooses F **once per question,
before the RLM starts**, from the question alone (no model call). The question then
runs start to finish at that F, and we measure accuracy.

```
                 ┌──────────── 1. plan (CPU, seconds) ────────────┐
  questions ───▶ │ picker looks at each question → F ∈ {1,2,4,8}  │ ──▶ plan.jsonl
                 └────────────────────────────────────────────────┘     ids/<task>__x<F>.txt
                                         │
                 ┌──────────── 2. run (GPU) ──────────────────────┐
                 │ each ids file = an ordinary B × F cell          │ ──▶ results/bxf_<B>MB_x<F>/…
                 └────────────────────────────────────────────────┘
                                         │
                 ┌──────────── 3. assemble (CPU) ─────────────────┐
                 │ each answer at its own F → score → compare     │ ──▶ comparison.md
                 └────────────────────────────────────────────────┘
```

## The pickers

| picker | what it decides from | example |
|---|---|---|
| `fixed:F` | nothing: every question at F | `fixed:1` (plain), `fixed:4` — today's B × F cells, the baselines |
| `spread[:RATIO]` | how **spread out** the question's evidence is (below) | `spread:0.3` |
| `per-task:T=F,…` | the task | `per-task:nq=4,hotpotqa=1` |
| `calibrated:FILE` | whatever `calibrate` chose on dev | `calibrated:cal_1024MB.json` |

**`spread`**, the main algorithm. Score every 2000-char chunk of the document by BM25
against the bare question (LOFT's citation boilerplate stripped). The *evidence* is
every chunk scoring at least `RATIO` × the best one. If those chunks fit in B tokens,
pick **F = 1** — compressing could only lose detail. Otherwise pick the smallest F
whose read (B × F) is wide enough to hold them.

On LOFT-128k test rows at 1024 MB this already separates the tasks the way one would
expect: nq and hotpotqa questions are concentrated (mostly ×1), quest and musique are
spread out (more ×2/×4/×8). `RATIO` sets how eagerly it reaches for wide reads —
it is the one setting, and the one `calibrate` tunes.

## Usage

**One command** (plan → run → assemble; starts and stops the root vLLM server):

```bash
PICKER=spread:0.3 SUBSETS="hotpotqa nq" BUDGET_MB=1024 LIMIT=55 \
  bash evaluation/rlm/run_picked.sh

DRY_RUN=1 PICKER=spread:0.3 SUBSETS="hotpotqa" bash evaluation/rlm/run_picked.sh   # plan only
```

Add `BASELINES=<results root of the full fixed-F sweep>` to compare against every fixed F
and the oracle on the same rows. Each plan gets its own `OUT/plans/<name>` and
`OUT/results/<name>`; re-running resumes, and a changed plan needs a new `NAME`.

**The protocol** (choose on dev, report on test):

```bash
# a. fixed-F runs of the dev rows, one per factor (LOFT ships 10 dev rows per task)
SPLIT=dev LIMIT=10 SUBSETS="nq hotpotqa musique qampari quest" BUDGETS_MB=1024 \
  FACTORS="1 2 4 8" ARMS=bm25-graph bash evaluation/rlm/run_bxf.sh

# b. choose the picker on those dev runs
python -m evaluation.rlm.factor_picker calibrate --tasks nq hotpotqa musique qampari quest \
  --budget-mb 1024 --dev-results <dev results root> --out cal_1024MB.json

# c. plan, run and score the test rows with the chosen picker
PICKER=calibrated:cal_1024MB.json SUBSETS="nq hotpotqa musique qampari quest" BUDGET_MB=1024 \
  LIMIT=55 BASELINES=<test sweep root> bash evaluation/rlm/run_picked.sh
```

**Without any new GPU run**: if the full fixed-F cells already exist for the same rows,
the picked result is just each question's answer looked up in the cell of its F:

```bash
python -m evaluation.rlm.factor_picker plan --tasks hotpotqa --budget-mb 1024 --limit 55 \
  --picker spread:0.3 --out plans/spread
python -m evaluation.rlm.factor_picker assemble --plan plans/spread \
  --results <sweep root> --baselines <sweep root>
```

## Reading the result

`comparison.md` has one row per task plus the task average, all on the **same rows**:

| column | meaning |
|---|---|
| `picks` | how many questions got each factor, e.g. `x1:46 x2:8 x4:1` |
| `picked` | the score with each question at its picked F |
| `fixed xF` | the score with every question at F |
| `oracle` | the best F per question in hindsight: the ceiling for ANY picker over these factors |

The score is the dataset's official metric: F1 for nq / hotpotqa / musique, coverage
for qampari / quest. A question with no answer at its picked F is reported as missing,
never scored as 0.

## Files

| file | what it holds |
|---|---|
| `budget.py` | B in tokens from MB (1024 MB → 6,944 tokens on Qwen3-4B), the factor menu |
| `features.py` | the evidence profile of a question (BM25 over the bm25-graph chunk grid) |
| `pickers.py` | `FixedPicker`, `SpreadPicker`, `PerTaskPicker`, and `make_picker("spread:0.3")` |
| `plan.py` | step 1: pick and write `plan.jsonl` + ids files |
| `results.py` | read finished runs by their `config.yaml`, score each question |
| `assemble.py` | step 3: merge, compare with fixed F and the oracle |
| `calibrate.py` | choose the picker on dev |
| `__main__.py` | the `plan` / `calibrate` / `assemble` commands |
| `../run_picked.sh` | the one-command wrapper; `../root_server.sh` starts/stops vLLM |
