# RLM Long-Context Benchmarking

This directory contains the Recursive Language Model (RLM) harness within the
KVPress evaluation repository. RLM and KVPress keep independent inference paths,
but share the benchmark registry, Hugging Face loaders, canonical scorers, and
the `predictions.csv` / `metrics.json` / `config.yaml` result contract. Run all
commands below from the repository root.

**To run anything, follow [`RUNNING.md`](../../docs/RUNNING.md) at the repository root.** It
covers the base vanilla/RLM run, the ordered-chunk arms (the agentic root handed every
chunk in BM25 order, on LOFT-128k and RULER-128k), B × F cells and the KVzip arms, on an
infolab server or on RunPod. What each finished campaign found is in
[`CAMPAIGNS.md`](../../docs/CAMPAIGNS.md).

Either ordered arm can also run as a **B × F cell**, reading through an in-process KVzip
model instead of the served one: `--sub-backend kvzip --memory-budget <B>
--memory-budget-unit MB --compression-factor <F>`. B is the KV one chunk read may keep, F
is the factor every read is pressed by (F = 1 is the uncompressed control and takes
`--press no_press`), and a read is capped at N = B × F tokens. For Qwen3-4B, 512 MB is
3,472 tokens, so 512 MB × 4 caps a read at 13,888 tokens, each pressed to a quarter. A cell
the GPU or the model window cannot hold is refused before any example runs. The lane takes
the same cell as `BUDGET_MB=512 FACTOR=4`. To run cells with one command, see
[`RUNNING.md` §7](../../docs/RUNNING.md#7-b--f-cells) (`bash scripts/rlm/run_bxf.sh`).

## How this folder is arranged

| folder | holds |
|---|---|
| `shared/` | building blocks: `datasets.py`, `client.py` (vLLM), `kvzip_backend.py` (KVzip sub-calls), `sizing.py` (B × F), `retrieval.py` (BM25), `organization.py` (bm25-graph) |
| `harness/` | the original runner `run_benchmark.py` and the RLMs it runs with `--mode` (below) |
| `configurable_rlm/` | a clean base RLM; ordering, tuning and compression are features switched on in one `experiments.yaml` — see its [README](configurable_rlm/README.md) |
| `chunk_sources/` | chunks handed to the RLM by something else: `chunkset.py`, `oracle_chunks.py`, `query_free_chunks.py`, `window_notes.py` |
| `analysis/` | reading finished runs: `paired.py`, `bxf_summary.py`, `hard_questions.py`, `compression_picker.py`, `structured_decompose.py`, `diagnostic.py` |
| `campaigns/` | tooling for finished campaigns: `rlm_order_*.py` and `fixed_chunk_grid/` |

## The RLM implementations

| code | what it is | run it with |
|---|---|---|
| `harness/rlm.py` | our original, tuned RLM (long prompt, guards, KVzip B × F cells) | `python -m evaluation.rlm.harness.run_benchmark --mode rlm` |
| `harness/clean_rlm/` | the same RLM split into modules, KVzip removed (from apd10-dev) | `… run_benchmark --mode clean-rlm` |
| `harness/original_rlm.py` | the RLM paper's protocol inside this harness (from apd10-dev) | `… run_benchmark --mode original` |
| `harness/skylight_rlm.py` | an agent that reads chunks by command, without a REPL (from apd10-dev) | `… run_benchmark --mode skylight` |
| `configurable_rlm/` | a clean base RLM with ordering, our tuning and compression as features | `python -m evaluation.rlm.configurable_rlm run <experiment>` |

**`analysis/hard_questions.py`** finds the questions vanilla solves but RLM + ordering does
not, and those only a compressed RLM solves, and writes them as ids files — the small
validation set: `python -m evaluation.rlm.analysis.hard_questions --results <root> --out hard_set`,
then `configurable_rlm run <experiment> --set data.ids_file=hard_set/ids/<task>__validation_set.txt`.

**`analysis/compression_picker.py`** chooses one compression factor per question before the
run (`bash scripts/rlm/run_picked.sh`). Each module's docstring has the details.

## Goal
Quantify how Recursive Language Models (RLMs) compare against vanilla long-context
inference on standard long-context benchmarks, with all inference served locally:
a vLLM (or any OpenAI-compatible) server on the GPU host for the root and vanilla
arms, and optionally an in-process KVzip-compressed sub model (`--sub-backend kvzip`).

## Architecture
The harness talks to whatever `--base-url` points at (default
`http://localhost:8000/v1`). The two supported deployment shapes:

- **Bare-metal GPU host** (infolab-style, no scheduler):
  `scripts/rlm/campaigns/loft128k_vanilla_vs_rlm/run_infolab.sh` launches one vLLM server per GPU and
  drives `run_cells.sh`, which runs the benchmark arms against it.
- **SLURM**: `slurm/loft128k_a100.slurm` is an array job
  that starts one vLLM server per dataset on its allocated GPU and runs the same
  cells. `scripts/rlm/download_data.sh` prefetches HF datasets on a
  login node for air-gapped compute nodes.

## Pipeline
```
[login node]                          [GPU node]
download datasets (HF) ──> $HF_HOME ──> vllm serve <model> --port 8000
                                        ──> run_benchmark.py --base-url http://localhost:8000/v1
                                              ├── condition A: vanilla (full context in prompt)
                                              ├── condition B: RLM (context as REPL variable)
                                              └── per-example JSONL results + resumable checkpoints
                                        ──> compare.py ──> results table
```

## Steps
**On an infolab host, follow [`RUNNING.md` §5](../../docs/RUNNING.md#5-base-run-vanilla-and-rlm)
instead**: one GPU, two venvs, plain commands, from a fresh account to a scored LOFT run.
The steps below are the generic outline.

1. **Environment**: `uv sync --extra eval`, then `source .venv/bin/activate`.
2. **Data prep** (login node if compute nodes lack internet):
   `bash scripts/rlm/download_data.sh` — caches the shared datasets
   under `$HF_HOME`. Synthetic NIAH/RULER-style tasks are generated locally.
3. **Serve a model**: `vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000`
   (or use `run_infolab.sh`, which does this per GPU with the right flags).
4. **Smoke test** (~5 examples):
   `python -m evaluation.rlm.harness.run_benchmark --dataset niah --limit 5 --mode both`
   For a shared RULER-32K subset:
   `python -m evaluation.rlm.harness.run_benchmark --dataset ruler32k --data-dir niah_single_1 --limit 5 --mode both`.
5. **Full runs**: `bash scripts/rlm/campaigns/loft128k_vanilla_vs_rlm/run_infolab.sh` (bare metal) or
   `sbatch slurm/loft128k_a100.slurm` (SLURM).
6. **Score & compare**: `python -m evaluation.compare --backend rlm`

Each RLM run gets its own directory containing a resumable `checkpoint.jsonl`,
the three common result artifacts, and (for RLM mode) per-example transcripts.
The canonical benchmark scorer—not the harness's quick progress-match flag—is
the authoritative score in `metrics.json`.

## Comparing against KVPress
Run both backends on the same `dataset` / `data_dir` / `model`, then join them:

```bash
python -m evaluation.compare --dataset ruler32k --csv comparison.csv
```

`compare.py` walks `evaluation/results/`, reads each run's `config.yaml` and
`metrics.json`, and prints one row per run. What makes the rows commensurable:

- **Same scorer.** Both backends call `benchmarks/results.score_prediction_frame`,
  so `score` is literally the same function applied to the same gold answers.
- **Same cost axis.** `context_tokens_held` is KVPress's
  `average_retained_context_tokens` and RLM's `average_peak_context_tokens` —
  both answer "how much context was live at once", which is what KV memory is
  proportional to. Plot score against this column to put a compression sweep and
  a `--max-context-tokens` sweep on one chart.

For an arm with KV-compressed sub-calls, `context_tokens_held` is
`max(root_peak, sub_peak)`, with the root-only figure kept beside it as
`context_tokens_root`. Counting only the root would flatter that arm badly: while
the root holds ~2k tokens, the sub model holds a whole slice of KV on the GPU,
and `KVzipPress` masks it rather than freeing it.

Three asymmetries the harness records rather than hides, because ignoring any of
them silently flatters RLM:

- **Errors.** An RLM example that dies on a server/harness error is dropped from
  the score and counted in `runtime.errors`. KVPress runs in-process and cannot
  fail this way, so scoring an outage as a zero would understate RLM for a reason
  that has nothing to do with the method. Treat a run with many errors as
  incomplete, not as a result — check the `errors` column before believing a score.
- **Truncation.** KVPress compresses the full context; vanilla truncates it. The
  `context_retained` column is the fraction that survived, so a low vanilla score
  can be read as a property of the char/token limit rather than of the model.
  Set `--vanilla-max-prompt-tokens` to `(max-model-len - max-tokens - margin)`;
  without it, densely-tokenising subsets (RULER `cwe`, `niah_multikey_3`)
  overflow the served window and score 0.0 from a harness error.
- **Abstentions.** An RLM example can end with `FINAL_NONE` — the model searched
  and reports finding nothing. That scores 0 exactly like a wrong answer, so
  `runtime.abstained` is reported separately: an arm that abstains honestly and
  one that hallucinates confidently are not the same result, and the score column
  alone cannot tell them apart. Vanilla has no equivalent ending.
- **Retrieval.** A `--search-k > 0` arm has an information channel the others do
  not: a BM25 index over the whole document. Vanilla truncates, KVPress
  compresses, and this arm *selects*. The only fair comparator is a `--search-k 0`
  run of the same commit, which is why the flag is stamped into the run directory
  and `config.yaml` rather than becoming a default.
  `runtime.gold_in_retrieved_fraction` is a loose answer-string presence
  diagnostic among scored examples with recorded retrieved spans. It is not a
  score ceiling: the root can read elsewhere, and a matching string does not
  establish that the retrieved text supplies sufficient evidence to answer.
- **Provided chunks.** A `--chunks-source file` arm had its chunks chosen by an
  external model with full document access and an unbounded budget, outside the
  measured run. That selector's cost appears in no column, so
  `context_tokens_held` is **not** comparable across this line: the arm is an
  upper bound on "what if selection were solved", not a deployable method. Its
  two fair comparators are `--chunks-source bm25` at the same `k` (same
  pipeline, cheap selector) and the agentic arms of the same commit. The payload
  is capped at `--chunks-k x --search-window` characters, the same budget the
  BM25 arm spends, so the arms differ in *which* text is read and not how much --
  without that cap the better-chosen chunk set is also the bigger one and the
  result measures size. The selector is given the question and never the
  answers; a chunk file built otherwise measures leakage, not retrieval.
  Two caveats travel with it. The treatment is not one variable: the primitive,
  the worked example, the locate bullet and the abstention gate all move together.
  And `document_coverage_fraction` changed meaning — spans are now resolved before
  the dense fold, so it was structurally `0.0` on every earlier `--sub-backend
  http` run. `config.yaml` records `coverage_attribution: spans_v2`; do not
  compare a coverage number across that line.

Search-enabled RLM runs with `--search-overlap 0` now record
`search_index_revision: 2`: an earlier cleanup loop incorrectly discarded all
but the first window. These runs require a fresh results directory when an
existing configuration lacks that revision. Default overlapping windows are
unchanged.

## The structured arm (`--mode structured`)

`rlm` gives the root a REPL and lets it decide what to read and when to stop.
`structured` reads a FIXED set of chunks with a fixed pipeline -- one
`llm_query` per chunk, then one synthesis call -- and has no REPL at all. The
agentic loop is removed, not assisted, so a difference between the two is
attributable to the pipeline rather than to a luckier search.

| Flag | What it does |
|---|---|
| `--chunks-source bm25` | the top `--chunks-k` windows, selected in-harness. The control: same chunks a `--search-k` arm gets, different pipeline. |
| `--chunks-source file` | chunks from `--chunks-file`, chosen offline (see `chunkset.py`). Isolates selection quality, at the cost of the asymmetry above. |
| `--ids-file` | restrict the run to named example ids, so several arms score the same rows. Applied BEFORE `--limit`, and refuses to combine with it. |

### Taking the revision-2 treatment apart

Revision 2 changed two things in one commit -- the map step began reading one
contiguous *passage* per call instead of one window, and the reduce step began
*requiring* the answer shape instead of offering it -- and a third defect was found
afterwards. Each is now its own flag, in the shape of `--search-ablate`. **Every
default reproduces revision 2 exactly**, so a run that passes none of them is
byte-identical to the campaign already on disk, down to the directory name.

| Flag | Default | What the non-default value does |
|---|---|---|
| `--structured-map-question` | `task` | `query` strips the benchmark's wrapper down to the query inside it. LOFT's `task` is not a question but a page of citation instructions (*"Print out the TITLE and ID…"*), which derails a 4B sub-model reading one 2,000-char window. The **reduce** step always sees the full task -- it carries the output contract the scorer matches on. |
| `--structured-map-granularity` | `passage` | `window` reads each provided window alone (revision 1's granularity); `all` reads every window in ONE call, which is the payload the agentic root assembles -- 45 of 50 nq examples of the `--search-k` arm passed the whole hit list to a single `llm_query`. |
| `--structured-answer-shape` | `require` | `offer` restores revision 1's reduce wording ("Be concise." plus *"Format your answer as: …"*), so the shape change can be scored apart from the clustering change it shipped with. |

Ordering is held constant across the granularity axis: passages are emitted in
**document order** at every setting. Revision 1 emitted them in rank order, so
`window` is revision 1's granularity on revision 2's ordering rather than a
byte-exact revision-1 rerun -- re-introducing rank order would put a third variable
inside a two-way ablation. The byte-exact revision-1 numbers are the revision-1 runs.

`average_map_spans` is what to read for the granularity axis: `all` pins it at 1,
and `average_chunks_with_findings` then becomes "did the single call find anything"
rather than "how many of five did".

### The chunk geometry, as three separate axes

`--search-window` used to set two things at once: the unit BM25 indexes, and how
much text a map call is shown. They are different questions with different answers,
so they are now three flags.

| Flag | Default | What it controls |
|---|---|---|
| `--search-window` / `--search-overlap` | `2000` / `400` | the RETRIEVAL unit. 2000 chars is ~512 tokens at 20% overlap, which is where the chunking literature lands and where a multi-dataset sweep puts NQ's own optimum -- so this is the one knob that is probably already right. |
| `--structured-map-chars` | `0` (all of it) | the READ size: characters of each passage the map call actually sees. Truncation happens on the `Hit`s, so a shortened payload still records the coverage it got rather than claiming the whole window. |
| `--reduce-finding-chars` | `2000` | how much of each finding survives into the reduce prompt. **Raise it with the window.** Left at 2000 a 4000- or 8000-char window is truncated back here, and the top of a window sweep measures this cap instead of the window. |

`win<W>o<O>` joins the run directory name whenever the geometry is non-default --
in `--mode structured` as well as `--mode rlm`. It used to be stamped only when
`--search-k` was set, which `--mode structured` rejects, so every cell of a window
sweep named the same directory: the first ran and the rest aborted on a resume
conflict. A structured run at the defaults keeps the name it already had.

Both cells of a sweep are refused up front, never silently shrunk: a per-call
payload that `--max-subcall-chars` would truncate is an argparse error, in the shape
of the `--search-k x --search-window` guard the agentic arms already had.

### A second retrieval round (`--structured-hops 2`)

One follow-up round, whose query is the question plus the distinctive terms **the
first round's map step reported**. Windows round 1 already read are excluded.

The expansion source is the point. Blind pseudo-relevance feedback over retrieved
*window text* was measured on this corpus and made recall worse everywhere (musique
recall@5 47.5% -> 37.5%), so hop 2 has to be a decision taken after reading, which
is what a finding is and a window is not. `hop2_chunks` records how many fresh
windows the round added; `hop2_error` records a retriever that raised rather than
voiding the example.

There is deliberately no hop 3. A second routing level is a documented
non-improvement, not an untried idea.

## Serving it (`serve_root.sh`)

```bash
GPU=4 scripts/rlm/serve_root.sh          # one card, one endpoint, every lane
```

The whole structured campaign is HTTP against **one** vLLM: root calls, sub-calls and
every lane. Measured on fox, 80 cells / 4,400 examples finished in **77 minutes with
four lanes against a single RTX A6000** at 3.28x effective concurrency, so the card
was never the bottleneck. More lanes are close to free; more GPUs mostly are not.

Qwen3-4B bf16 is ~8 GB of weights and KV is 147,456 B/token, so eight of the widest
cell's ~20k-token sequences add ~6 GB: **the real floor is ~14-16 GB.**
`--gpu-memory-utilization` is what vLLM PRE-ALLOCATES, not what it needs.

Two things the script refuses rather than discovers later:

- **A card someone else is on.** It checks `memory.used` first and exits.
- **A port that is already listening.** A wedged vLLM still answers `/v1/models` and
  still accepts chat requests -- it logs "Added request" and completes none of them.
  A campaign pointed at one looks exactly like a campaign that is running: processes
  alive, no errors, no rows. `structured_grid.sh` asks for eight tokens before it
  launches for the same reason.

`CUDA_DEVICE_ORDER=PCI_BUS_ID` is set and is load-bearing: fox mixes RTX 6000 Ada
(GPU 0,1) with RTX A6000 (GPU 2,3,4), CUDA orders devices FASTEST_FIRST while
nvidia-smi orders by PCI bus, and without it the index you measured as idle is not
necessarily the index CUDA gives you.

## The reader's ceiling (`oracle_chunks.py`)

```bash
python -m evaluation.rlm.chunk_sources.oracle_chunks --dataset loft --data-dir musique_128k \
  --split test --limit 55 --window 2000 --overlap 400 --k 5 --out-dir <dir>
```

Writes a chunk file of the windows that provably carry a gold answer, greedily
chosen to cover as many DISTINCT gold answers as possible -- which is what makes it
an oracle on qampari and quest, where the gold is a list scattered across passages
rather than one string. Read through `--chunks-source file`, retrieval is perfect by
construction and whatever score is missing afterwards is the reader.

Two things it always does, and both matter:

- **`spans` are written.** The two hand-made chunk files carry none, so
  `ChunkSet.hits_for`'s grid check cannot fire and window ids re-resolved on a
  different `--search-window` would silently address unrelated text. The geometry is
  in the filename too.
- **A row with no gold-bearing window is labelled, not hidden.** It falls back to
  BM25 and says so in `note`, so the analysis drops it instead of averaging a
  non-oracle into the ceiling.

## Reading a sweep back (`structured_decompose.py`)

```bash
python -m evaluation.rlm.analysis.structured_decompose --results <tree> --csv cells.csv
```

Renders score by window x k, then walks k upward and splits each step into the
windows it added that carry a gold string and the ones that do not
(`average_gold_windows_in_chunks` / `average_nongold_windows_in_chunks`, recorded
beside `gold_in_retrieved`). Raising k adds signal and distractors together, so an
undecomposed k-curve turning over cannot say which side moved. Cells with errors, or
with `average_chunks_with_findings` near zero, are flagged as suspect and must not
enter a table.

## Provided chunks read by the AGENTIC root

`--mode rlm --search-k <k> --chunks-source file --chunks-file <…>` gives the REPL
arm a chunk set chosen offline. The chunks arrive through `search()`, which returns
them for **any** query, so the loop, the protocol and the system prompt are all
byte-identical to the `--search-k` arm and the only difference between the two runs
is which windows come back. That is what separates chunk quality from the pipeline:
`--chunks-source file` in `--mode structured` changes both at once.

**A frozen retriever traps the loop, and this is the arm's dominant effect.** The
root's main move is to rephrase and search again; when every rephrasing returns the
same windows it cannot tell, and it keeps going. Measured on nq: **17 search calls
per example frozen against 1.5 adaptive**, with the read arriving so late that at
`--max-steps 12` the run ends before it can answer. Budget the steps accordingly,
and never compare a frozen arm against an adaptive `--search-k` one -- the
difference is mostly the trap.

That is why `--chunks-source bm25` is **allowed** in `--mode rlm`: a frozen BM25
top-k is the only fair control for a frozen `file` arm, since both sides then pay
the same cost and only the choice of windows differs. It also reads exactly the
windows `--mode structured --chunks-source bm25` reads, so `file`-vs-`bm25` means
the same thing on both pipelines. The adaptive arm remains `--search-k` alone, and
lands in its own directory (`…bm25k5` against `…bm25k5__chunks-bm255`).

The provided-chunk asymmetry above applies unchanged, so `context_tokens_held` is
still not comparable across that line. `search_fixed: true` and `chunks_provided`
are recorded so no reader mistakes a frozen arm for an adaptive one; read
`average_search_calls` beside them.

Read `average_chunks_with_findings` beside the score. Near zero means every chunk
answered "nothing relevant" and the reduce step had nothing to work with -- this
arm's version of `sub_pressed_call_fraction` near zero, and the tell that the
pipeline ran without measuring anything.

A chunk file names the sha256 and length of the document it was built against,
and is refused if either disagrees. Stale offsets are all still in range, so
without that check every example reads real text from the wrong corpus and the
arm reports a plausible score.

One documented departure: when the reduce step's answer appears in none of the
findings, it is recorded as `reduce_answer_not_quoted` rather than rejected.
`run`'s grounding guard exists to catch a root answering from memory without
reading; this arm shows the root nothing but text quoted out of real chunks, so
that failure cannot arise -- and there is no loop to nudge a rejection back into.

## Runaway guards and the scratchpad
All four are toggleable and independent; pass `0` to disable one without
affecting the others. Defaults are on because an unguarded run can burn a whole
SLURM allocation on one pathological example, but a debugging session that wants
to watch a single example run to completion can switch any of them off.

| Flag | Default | Off | What it bounds |
|---|---|---|---|
| `--exec-timeout` | `60` s | `0` | Pure-Python time for ONE code block, so a model-generated infinite loop can't hang the sweep. Time inside `llm_query` is excluded — the watchdog is paused around sub-calls, so a slow-but-legitimate server call is never mistaken for a runaway loop. |
| `--run-timeout` | `900` s | `0` | Wall-clock for ONE example, ending it with `end_reason=run_timeout`. Needed because `--exec-timeout` deliberately does not bound `llm_query` time. |
| `--max-sub-calls` | `40` | `0` | `llm_query` calls per example. Past the cap, calls return a notice instead of hitting the server, so the model degrades gracefully rather than erroring. |
| `--scratchpad` | off | (omit) | Opt-in `note(text)` REPL tool. Notes are re-shown every turn and survive budget eviction, giving the model a durable place to keep findings. Size it with `--max-notes-tokens`. |
| `--search-k` | `0` (off) | `0` | Opt-in `search(query)` REPL tool: BM25 over fixed overlapping windows of the document, returning this many ranked windows. Pass them to `llm_query(question, hits)` so the payload stays attributable. The model may ask for fewer windows but never more, so a worked example cannot override a swept grid cell. Size the windows with `--search-window` / `--search-overlap`. |

`--exec-timeout` relies on `SIGALRM`, so it is a no-op off the main thread or on
platforms without it; it also cannot interrupt a C-level regex. `--run-timeout`
has neither limitation and is the backstop.

Because `--scratchpad` and `--max-context-tokens` both change results, both
appear in the run directory name — otherwise two configurations would share a
`checkpoint.jsonl` and silently merge. Nothing parses that name; `compare.py`
reads `config.yaml`.

The name cannot carry *every* result-affecting knob without becoming unreadable,
so the rest (`--limit`, `--split`, `--max-steps`, the timeouts, the sub model,
…) are enforced on **resume** instead: `config.yaml` is written before the first
example runs, and a later run whose settings disagree with it aborts rather than
appending into someone else's checkpoint.

`--split dev|test|all` picks a split where the dataset has them. It matters most
for LOFT, which ships 10 dev rows followed by 100 test rows: with `--split all`
(the default) a `--limit 10` smoke run silently evaluates *only* dev, and lands
in the results tree looking like a real run that scored 0.000. Use `--split dev`
for smoke runs — it also gets its own directory — and `--split test` for real ones.

## Benchmarks (in order of effort)
| Benchmark | Why | Source |
|---|---|---|
| Synthetic NIAH / RULER-style | Sanity check; both vanilla & RLM should ace it | generated locally |
| LongBench v2 | Standard, broad long-context QA | HF: `THUDM/LongBench-v2` |
| OOLONG | The benchmark where the RLM paper showed the biggest gap | HF (see download script) |
| ∞Bench / BrowseComp-Plus | 100k–10M token stress tests (RLM-only territory) | HF |

## Experimental conditions
- **Vanilla**: full context stuffed into the prompt of the served long-context
  model (truncate at model limit; record truncation).
- **RLM**: same base model as the REPL "root", context held as a Python variable,
  `llm_query()` available for recursive sub-calls (depth 1 to start).
- Hold the base model constant across both conditions.

## Metrics
- Accuracy / F1 per task (exact-match for NIAH, substring/F1 for QA)
- Tokens used per query (prompt + completion, summed over all recursive calls)
- Wall-clock latency per query
- REPL steps used; failure modes (max-steps exhausted, code errors, server errors)

## Practical constraints to engineer around
- Vanilla at 100k+ tokens is slow even served locally; start with `--limit 50`.
- Set `--time` in SLURM generously: RLM runs are many sequential server calls.
- JSONL checkpointing means an interrupted run resumes with the same command.
