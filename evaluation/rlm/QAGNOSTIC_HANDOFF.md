# Handoff — query-agnostic performance & generalization, 2026-09-09

Picking up from [`HANDOFF.md`](HANDOFF.md) (the chunk-geometry campaign). Everything
here is measured; where something is a guess it says so.

> **Archived on 2026-09-11.** The lab storage policy keeps raw output out of `$HOME`. So
> `results`, `logs`, `chunksets`, `notes`, `smoke` and `smoke_ladder` moved to
> `/mnt/nas/gautammahale/archive/qagnostic/`; the copies were checksum-verified, then deleted
> from `/mnt/home/gautammahale/qagnostic/`. The checkouts, scripts and `ids/` stay in `$HOME`.
> Wherever this file names `/mnt/home/gautammahale/qagnostic/{results,logs,chunksets,notes}`,
> read that archive path instead.

**The question this campaign answers:** the chunk-geometry campaign ended with every
number being a function of *what BM25 was asked* — nq moved 0.545 → 0.800 by stripping
LOFT's instruction wrapper off the retrieval query. Nothing said how much of the
remaining 0.800 the query was **buying** and how much was there for free, because
nothing in the harness ever selected context without the question. This builds that
floor, and holds out rows to price the argmax that picked the headline cells.

---

## State right now — 15 lanes running on elk, NOT fox

**fox is unusable: all five cards carry another user's process.** Probed every
infolab host; `elk` has seven RTX A6000s and had two completely free.

| where | what |
|---|---|
| host | **elk.cse.iitb.ac.in, GPU 2** (`tmux: root_gpu2`, port 8001, 21.9 GB of 48 GB, util ~98%) |
| branch | `feature/query-agnostic` @ `b21fd7f`, off `feature/structured-chunks` |
| code (campaign) | elk `/mnt/home/gautammahale/qagnostic/checkout` — the ladder/exhaustive lanes run this |
| code (notes) | elk `/mnt/home/gautammahale/qagnostic/checkout-notes` — a **separate** copy, so the notes work could be synced without swapping code under a running campaign |
| results | elk `/mnt/home/gautammahale/qagnostic/results` |
| chunk sets | elk `/mnt/home/gautammahale/qagnostic/chunksets` (60 files) |
| holdout ids | elk `/mnt/home/gautammahale/qagnostic/ids` (5 files, 45 rows each) |
| notes | elk `/mnt/home/gautammahale/qagnostic/notes` (building) |
| logs | elk `/mnt/home/gautammahale/qagnostic/logs` |

**Results are on `/mnt/home` (11 TB free), not `/mnt/nas` (99% full, 756 GB).** Every
infolab host mounts both, so nothing is host-specific.

```bash
ssh elk.cse.iitb.ac.in 'tmux ls'          # 15 lanes + root_gpu2
ssh elk.cse.iitb.ac.in 'ls /mnt/home/gautammahale/qagnostic/results/*/metrics.json | wc -l'
```

Lanes: `ladder_{5 subsets}`, `exh_{5 subsets}`, `notes_{5 subsets}`. Each writes
`logs/<stage>_<subset>.log`. All are resumable — a killed lane re-run skips completed
cells via the checkpoint, and a partly-built notes file resumes from its last window.

---

## The result so far — at k=2, the query is the entire score

16 of ~85 ladder cells are in. `subspan_em`, except qampari/quest which are `coverage`.
k=2 is a 4,000-char read out of ~475,000.

| subset | bm25 **task** | bm25 **bare** | lead-2 | stride-2 |
|---|---|---|---|---|
| nq | 0.000 | **0.782** | 0.000 | — |
| hotpotqa | 0.000 | **0.345** | 0.000 | 0.000 |
| musique | 0.000 | **0.109** | 0.000 | — |
| qampari | 0.000 | **0.220** | 0.004 | 0.004 |
| quest *(held out)* | 0.000 | **0.258** | — | — |

Three things to carry:

1. **The query-agnostic floor at k=2 is zero on every subset.** `gold_seen` (the
   fraction of examples where a retrieved window contained a gold string) is 0.00–0.05
   for lead and stride against 0.38–0.95 for bare BM25. At this budget the method has
   no performance that is not the query's.
2. **`bm25-task` is also 0.000 at k=2** — the published default is, at depth 2,
   indistinguishable from reading two arbitrary windows. That is the recall@2 = 0.00
   finding from the previous campaign showing up end-to-end for the first time.
3. **quest has a number for the first time** (0.258 at k=2). It was never in the
   geometry sweep, so it is the cleanest generalization estimate in the study.

The interesting part is what happens as k rises — k=5 and k=10 are still running, and
the exhaustive arm (every window, no retrieval at all) is the ceiling that floor is
measured against.

**Do not read these as final.** n=55, one example is 0.018, and the deltas that matter
later are small. Everything reportable goes through `paired.py` (below).

---

## What was built

| file | what it is |
|---|---|
| `evaluation/rlm/query_free_chunks.py` | the four selectors — `lead`, `stride`, `random` (seeded per row), `all` — as pure functions of `(n_windows, k, seed)`. No document text, no question, no answers; it does not import `is_correct`, and the absence is the claim. Writes `spans` and a `selector_tag`. |
| `evaluation/rlm/window_notes.py` | the query-free index: one note per window, written by a pass with the question nowhere in it. Plus `NotesIndex` (BM25 over notes, not the document) and a strict loader. |
| `evaluation/rlm/paired.py` | the statistics the repo never had — per-row rescoring, paired bootstrap CI, win/loss/tie, sign test. |
| `--chunks-source notes` | reads that index instead of running the map step. `--notes-select all` is query-free until the reduce; `bm25` is query-aware **lookup** over a query-agnostic **index** and must be labelled that way. |
| `structured_grid.sh` stages | `ladder`, `exhaustive`, `holdout`. |
| `chunkset.py` | `selector_tag`, `ChunkSet.tag`, `max_windows`, `max_cluster_chars()`. |

430 tests pass; flake8 and mypy clean. `tests/test_pipeline.py`'s 42 failures are
**pre-existing** (stale kvpress log assertion) — do not chase them.

---

## Traps found, and what now refuses them

Each of these produces a **plausible number**, not an error. That is why they are guards.

1. **Every file-sourced arm at the same k was named `chunks-file<k>`.** Oracle, lead,
   stride, random and exhaustive would collide on one run directory and the second to
   run would die on a resume conflict — the ladder could not have run at all. Chunk
   files now carry `selector_tag` and the harness stamps `sel-<tag>`, **only when a tag
   exists**, so all 119 directories of the previous campaign keep byte-identical names.
   *Verified by rebuilding all 119 names from their configs — 119/119 unchanged.*
2. **`--max-sub-calls` defaults to 40, and overflow manufactures evidence.** Past the
   budget `llm_query` returns a `[SUB-CALL LIMIT REACHED]` **string**, which does not
   start with the "nothing relevant" sentinel — so the findings filter counts every
   remaining window as a *finding* and pastes the notice into the reduce prompt. A
   truncated 298-window sweep reports **more** findings than a healthy one and passes
   every gate that looks for a starved cell. Now refused unless the budget clears the
   chunk set. **The exhaustive lane runs `--max-sub-calls 0`.**
3. **Exhaustive at `passage` granularity collapses to one call.** `_cluster_hits` merges
   touching windows and the grid overlaps by construction, so every window becomes ONE
   passage spanning the corpus, which `_cap_subcall` truncates to ~24k chars. The run
   reports a full sweep having read the first 5%. Refused up front via
   `ChunkSet.max_cluster_chars()`. **Everything in the ladder runs
   `--structured-map-granularity window`** — which is also the only setting where the
   rungs are genuinely read-budget-matched (at `passage`, `lead-10` is one call of
   16,400 chars while `stride-10` is ten calls of 2,000).
4. **`predictions.csv` round-trips `answers` as a string.** `calculate_metrics` wraps a
   non-list in `[value]`, so gold becomes one answer that is a Python literal. nq's
   0.800 rescores to **0.6545** — a plausible wrong number, not an obvious zero.
   `paired.py` prefers `checkpoint.jsonl` and `literal_eval`s the CSV path.
5. **`--chunks-k` on a file source is a payload cap of `k × window`, and overlap is
   double-counted.** A global `--chunks-k 300` passes nq (298 windows) and **fails quest
   (302) fatally**. The writer puts the effective k in the filename and the driver
   parses it back rather than guessing.
6. **vLLM needs KV for one max-length sequence, not for the working set.** `GPU_UTIL=0.35`
   at `MAX_LEN=65536` fails to start (9 GiB needed, 7.22 available). Running at
   `GPU_UTIL=0.45 MAX_LEN=40960`, which fits and leaves 26 GB of the card for others.

---

## What to do next, in order

**1. Let the lanes finish**, then gate every cell before it enters a table:

```bash
ssh elk.cse.iitb.ac.in 'grep -rl "SUB-CALL LIMIT REACHED" /mnt/home/gautammahale/qagnostic/results | wc -l'   # must be 0
```
Also require, per cell: `errors == 0`; `average_map_calls == average_map_spans`; and for
exhaustive, `average_map_calls == the subset's window count` (75/68/72/72/76 at w=8000)
and `document_coverage_fraction == 1.0`. A cell failing any of these is not a result.

**2. Read the notes arm.** The builds are running at w=8000. Once a notes file exists:

```bash
ssh elk.cse.iitb.ac.in 'cd /mnt/home/gautammahale/qagnostic/checkout-notes && \
  HF_HOME=/mnt/nas/gautammahale/hf_cache PYTHONPATH=. \
  /mnt/nas/gautammahale/venvs/rlm/bin/python -m evaluation.rlm.run_benchmark \
    --dataset loft --data-dir nq_128k --split test --limit 55 --mode structured \
    --chunks-source notes --notes-file /mnt/home/gautammahale/qagnostic/notes/nq_128k.notes.w8000o1600.p1.Qwen_Qwen3-4B-Instruct-2507.jsonl \
    --notes-select all --search-window 8000 --search-overlap 1600 \
    --reduce-finding-chars 1600 --max-sub-calls 0 \
    --base-url http://localhost:8001/v1 --out /mnt/home/gautammahale/qagnostic/results'
```
**This path has not been run end to end yet** — it is unit-tested and imports cleanly,
but no cell exists. Smoke it at `--limit 3` first. Then run `--notes-select bm25
--notes-k 5` as the paired ablation, and label it query-*aware lookup* over a
query-*agnostic index* — not "query-free".

**3. Run the holdout.** `structured_grid.sh holdout <subset>` — the promoted arms on
test rows 55–99, which no configuration choice in either campaign has seen.

```bash
ssh elk.cse.iitb.ac.in 'for s in nq hotpotqa musique qampari quest; do
  tmux new-session -d -s "hold_$s" "/mnt/home/gautammahale/qagnostic/lane.sh holdout $s"; done'
```

**4. The A/B/C table — this is the deliverable.** For each subset:
- **A** = per-subset argmax, scored on the rows it was picked on (dev 0–54) — the overfit number
- **B** = that same config on holdout 55–99
- **C** = ONE config chosen by *mean rank across subsets*, on holdout

`A − B` is the per-subset overfitting cost; `B − C` is the price of one config
everywhere; **C on quest** is the cleanest generalization estimate in the study.
Use rank, not mean score: qampari's coverage spans 0.08–0.59 while musique's spans
0.11–0.22, so a mean-score winner is just whoever is good at qampari.

Every delta goes through:
```bash
python -m evaluation.rlm.paired --a <run_dir> --b <run_dir>
```
It refuses arms that ran different rows or report different metrics, and prints the CI
against the ±1-example noise floor (0.018 at n=55, 0.022 at n=45).

**5. Still unbuilt:** `structured_decompose._cell()` needs the new axes or it will
render one arm on top of another — the Trap-3 failure, a fourth time. It must gain
`chunks_tag`, `structured_map_granularity`, `ids_sha256`, `notes_*`. Refactor it from a
positional tuple to a frozen dataclass first; it is currently unpacked positionally in
four places. A cross-subset aggregator (`config_ladder.py`) for step 4 does not exist.

---

## Things worth knowing

- **Each LOFT subset is ONE corpus and 100 queries.** All 100 rows of `nq_128k` carry a
  byte-identical 475,516-char context. That is why the notes index is built once per
  subset (~75 calls at w=8000, ~298 at w=2000) rather than per example, and why its
  per-query cost is the build ÷ 100.
- **Rows 55–99 are genuinely untouched.** All 119 cells of the previous campaign were
  `limit: 55, split: test` (= `df.head(55)`), and the only `limit: null` runs were
  `--ids-file` restricted to ids drawn from those same first 55. Verified on disk.
- **Measured cost:** exhaustive is ~40 s/example at w=8000 (75 map calls), so ~37 min
  per 55-row subset. Ladder cells at k=5 are ~3 s/example. With 5 lanes the card sat at
  42% util; at 15 lanes it is at 98% — **more lanes are close to free, more GPUs are not.**
- **Do not run exhaustive at w=2000 first.** 298 windows is 3–4× the cost for the same
  claim, and `--reduce-finding-chars` must drop to 400 there or the reduce prompt
  (298 × 2000 chars ≈ 149k tokens) exceeds the served window and the server returns a
  400 — which `client.py` does **not** retry.

## Reporting rules still in force

- **Show loft128k only.** qampari and quest score on `coverage`, never `subspan_em`.
- Vanilla is n=110 — a reference line, never a paired comparison.
- `notes-bm25` is query-aware lookup over a query-agnostic index. Only `notes-all` and
  the `lead`/`stride`/`random`/`all` rungs are query-free selection.
- The `--chunks-source file` cost asymmetry in `README.md:110-121` does **not** apply to
  these selectors: unlike the oracle they are cheap and deployable, and
  `context_tokens_held` *is* comparable across them. That paragraph needs the carve-out.
