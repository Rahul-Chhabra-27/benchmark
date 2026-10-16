# core — a clean RLM, one config file per experiment

A small, readable RLM that every experiment runs through. The base is the RLM idea
and nothing more; ordering and compression are **options switched on in a YAML
file**, so "ordering only", "compression only" and "ordering + compression" are
different config files, not different code.

```bash
python -m evaluation.rlm.minimal_rlm check evaluation/rlm/minimal_rlm/configs/graph_x2.yaml --show-prompt  # what it will do
python -m evaluation.rlm.minimal_rlm run   evaluation/rlm/minimal_rlm/configs/graph_x2.yaml                # run it (resumes)
python -m evaluation.rlm.minimal_rlm run   evaluation/rlm/minimal_rlm/configs/base.yaml --set data.limit=3 --set name=smoke
```

The root model needs an OpenAI-compatible server at `models.base_url`
(`bash scripts/rlm/serve_root.sh`). A config with `compression.budget_mb` also loads
the sub model in-process on the GPU (kvpress).

## A config

```yaml
extends: _defaults.yaml        # shared data / model / limits; this file changes only what it is about
name: graph_x2_1024mb          # results go to <output_dir>/<name>/<task>/
ordering: bm25-graph           # none | bm25-query | bm25-graph
compression:
  factor: 2                    # 1 = no kvpress | 2 | 4 | 8
  budget_mb: 1024              # B: KV one read may keep. Each read is B x F tokens.
```

| key | values | what it does |
|---|---|---|
| `ordering` | `none` | the root searches the `context` string itself |
| | `bm25-query` | the root also gets `chunks`, ranked by BM25 against the question |
| | `bm25-graph` | `chunks` grouped by topic, built once per document before any question |
| `compression.factor` | `1` | plain reads, no kvpress |
| | `2` `4` `8` | every read pressed by KVzip at that factor (needs `budget_mb`) |
| `compression.budget_mb` | e.g. `1024` | B; reads go through the in-process model and carry at most B × F tokens. With `factor: 1` this is the no-press control at the same B |
| `data` | `dataset`, `tasks`, `split`, `limit` | which questions (LOFT task names may be bare: `hotpotqa`) |
| `models` | `root`, `sub`, `base_url`, `thinking` | which models, where |
| `rlm` | `max_steps`, `max_sub_calls`, `exec_timeout_s`, `run_timeout_s`, `output_chars`, `read_tokens` | limits (`read_tokens` caps a read when no budget is set) |

Unknown keys are refused, and a run never resumes into a directory written by a
different config — give a changed experiment a new `name`.

`evaluation/rlm/minimal_rlm/configs/` ships the starting matrix:

| config | ordering | compression |
|---|---|---|
| `base.yaml` | none | none |
| `order_query.yaml` | bm25-query | none |
| `order_graph.yaml` | bm25-graph | none |
| `compress_x2.yaml` | none | ×2 at 1024 MB |
| `graph_x1.yaml` | bm25-graph | ×1 at 1024 MB (the control) |
| `graph_x2.yaml` | bm25-graph | ×2 at 1024 MB |
| `graph_cost_aware.yaml` | bm25-graph | root chooses ×1–×8 per read, offloading, cost shown |

## What the root is told

`check --show-prompt` prints it. About twenty lines: the document is `context` in a
REPL; reply with one code block; `llm_query(question, text)` sends text to a sub-model
that sees nothing else, up to the read limit; `FINAL(answer)` ends the run. An
ordering adds `chunks` and two lines on how they are ordered. That is all: no
worked examples, no strategy coaching, no guards. Those are what made the older
RLM (`evaluation/rlm/rlm.py`) hard to separate from the RLM idea; anything of that
kind should come back only as a measured option.

## What a run writes

Per task, under `<output_dir>/<name>/<task>/`: `config.yaml`, `checkpoint.jsonl`,
`predictions.csv`, `metrics.json`, and `transcripts/` with every root turn.

Every `llm_query` leaves a record in the question's checkpoint line (`calls`):

| field | meaning |
|---|---|
| `step` | which root turn made it |
| `chunks` | how many chunks it read (ordering on) |
| `read_tokens` / `limit_tokens` | how much it read / the most it could have |
| `kept_tokens` / `budget_tokens` | KV kept after compression / B |
| `factor` | the compression factor |
| `refused` | why it was refused (too long, or the call limit), else empty |
| `seconds`, `answer` | how long it took, what came back |

`metrics.json` has the official score (F1 for nq/hotpotqa/musique, coverage for
qampari/quest) and `runtime`: endings (`final` / `max_steps` / `run_timeout`), mean
steps, mean reads, refused reads, and `read_fill_mean` — how much of each read's
limit the root actually used. Near 0 means the root reads tiny slices and a bigger
(compressed) read buys it nothing.

## Files

| file | holds |
|---|---|
| `config.py` | the experiment file and its validation |
| `prompts.py` | every word the models see |
| `ordering.py` | `none` / `bm25-query` / `bm25-graph` → the `chunks` list |
| `readers.py` | how `llm_query` reads: `HttpReader` (plain) or `KVReader` (kvpress, factor F) |
| `repl.py` | the persistent REPL: run a cell, echo, timeout, capture output |
| `loop.py` | the RLM loop and the `llm_query` / `FINAL` tools |
| `stats.py` | the per-call record and its per-question summary |
| `run.py` | config → every question → result files |

## Choosing compression, offloading and cost

Three more keys, all off by default (so every config above renders the same prompt):

```yaml
compression:
  budget_mb: 1024
  factor: 1                # the default factor
  choices: [1, 2, 4, 8]    # the root may choose per read: llm_query(q, text, factor=4)
  offload: true            # reads larger than B x F are allowed; KV beyond B counts as offloaded
cost:
  gpu_token: 0             # per KV token kept on the GPU
  offloaded_token: 100     # per KV token fetched back from CPU
  show_root: true          # tell the root each read's cost and the running total
```

A read of n tokens at factor F keeps n / F tokens of KV: up to B on the GPU, the rest
offloaded; its cost is GPU tokens x `gpu_token` + offloaded tokens x `offloaded_token`.
One in-process kvpress model serves every factor (its press is set per read, F = 1 is no
press), so choosing a factor needs no extra servers. Offloading is accounting only for now:
kvpress keeps the KV on the GPU, so a read is capped at `max_read_tokens` (default B x the
largest factor -- an x4 read with 2x offloaded reaches as far as an x8 read).

`graph_cost_aware.yaml` is this setting with bm25-graph ordering. Every read's record then
also has `gpu_tokens`, `offloaded_tokens`, `cost`, `compression_needed` (F > 1: would it NOT
have fit in B uncompressed?) and `contributed` (did its answer feed FINAL's?), and
`metrics.json` adds `cost_mean`, `offloaded_tokens_mean`, `factors` (reads per factor),
`compressed_unneeded` and `contributing_calls`.

The questions to judge it on are the hard set: see `../hard_questions/`.
