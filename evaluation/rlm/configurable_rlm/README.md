# configurable_rlm

A clean base RLM with features added on top of it. Each feature is switched on in one
config file, `experiments.yaml`. This is the setup the professor asked for: *"a clean RLM,
and then we start adding all of this stuff, so now we can run any setting of RLM."*

## Files

| file | what it holds |
|---|---|
| `rlm.py` | **the base RLM**: short prompt, Python REPL, loop, `llm_query`, `FINAL`, plain reads. It imports none of the features. |
| `ordering.py` | feature: the `chunks` list, ordered by `bm25-query` or `bm25-graph` |
| `tuning.py` | feature: our prompt tuning and guards from the original `evaluation/rlm/harness/rlm.py` (rules, search coaching, worked example, `FINAL_NONE`, rejection of guessed answers, loop nudges) |
| `compression.py` | feature: KV-compressed reads under a memory budget B, a compression factor per read, offloading, and cost |
| `config.py` | the settings, read from `experiments.yaml` |
| `run.py` | runs an experiment over its questions and writes the result files |
| `experiments.yaml` | every experiment: shared `defaults`, then what each named experiment changes |

The base loop only calls two kinds of thing. A **reader** decides how `llm_query` reads
(plain reads, or the compression feature). **Features** add tools, prompt lines and checks
(ordering, tuning). Turning a feature off removes its prompt lines and its behaviour completely.

## Commands

Run from the repository root. The root model needs an OpenAI-compatible server at
`models.base_url` (`bash scripts/rlm/serve_root.sh`). Experiments with
`compression.budget_mb` also load the sub-model on the GPU.

```bash
python -m evaluation.rlm.configurable_rlm list                      # every experiment
python -m evaluation.rlm.configurable_rlm check order_graph_tuned   # what it will do + its prompt (no model)
python -m evaluation.rlm.configurable_rlm run order_graph           # run it; running again resumes

# change any setting for one run
python -m evaluation.rlm.configurable_rlm run graph_x2 --set data.limit=3 --set name=smoke

# run only a set of questions, e.g. the hard set below
python -m evaluation.rlm.configurable_rlm run graph_cost_aware \
  --set data.ids_file=hard_set/ids/hotpotqa_128k__validation_set.txt --set name=hard_cost_aware
```

Finding the hard questions (vanilla right, RLM + ordering wrong) from finished runs:
```bash
python -m evaluation.rlm.analysis.hard_questions --results <results root> --out hard_set
```

## Experiments

| experiment | ordering | tuning | compression |
|---|---|---|---|
| `base` | — | — | — |
| `base_tuned` | — | ✅ | — |
| `order_query` | bm25-query | — | — |
| `order_graph` | bm25-graph | — | — |
| `order_graph_tuned` | bm25-graph | ✅ | — |
| `compress_x2` | — | — | ×2 at 1024 MB |
| `graph_x1` | bm25-graph | — | ×1 at 1024 MB (control: same budget, no compression) |
| `graph_x2` | bm25-graph | — | ×2 at 1024 MB |
| `graph_cost_aware` | bm25-graph | — | root picks ×1/2/4/8 per read, offloading, cost shown |
| `rlm_like`, `rlm_like_graph`, `rlm_like_query` | —, bm25-graph, bm25-query | ✅ | — |
| `rlm_like_graph_x2` | bm25-graph | ✅ | ×2 at 1024 MB |

The `rlm_like*` experiments are **the closest to our original `harness/rlm.py`** as the LOFT
campaigns ran it: tuning on, 50 turns, 6,000 characters of output per turn. They are not
identical, because `rlm.py` also has BM25 `search()`, a repetition breaker and its own prompt
wording. To reproduce old numbers exactly, run `python -m evaluation.rlm.harness.run_benchmark
--mode rlm` (or `scripts/rlm/run_bxf.sh`).

To add an experiment, add a few lines under `experiments:`. The keys are:

- `ordering`: `none` | `bm25-query` | `bm25-graph`
- `tuning`: `true` | `false`
- `compression`: `budget_mb`, `factor`, `choices`, `offload`, `max_read_tokens`
- `cost`: `gpu_token`, `offloaded_token`, `show_root`
- `rlm`: `max_steps`, `max_sub_calls`, timeouts, `output_chars`, `read_tokens`
- `data`: `tasks`, `split`, `limit`, `ids_file`
- `models`: `root`, `sub`, `base_url`

Unknown keys are refused. A run never resumes into a folder written by a different
config, so give a changed experiment a new `name`.

## What a run writes

For each task, under `<output_dir>/<experiment>/<task>/`:

- `config.yaml`: the settings used
- `checkpoint.jsonl`: one line per question, including every read
- `predictions.csv`
- `metrics.json`: the official score plus `runtime`
- `transcripts/`

Every `llm_query` is recorded. For every reader, the record has the turn, the tokens read
against the limit, whether the read was refused, and whether it fed the final answer. With
compression it also has the factor, KV kept, GPU vs offloaded tokens, cost, and whether
compression was needed. With tuning, the question records rejected guesses and refused
abstentions. `metrics.json` averages all of these over the questions.

## The professor's requests: status

| asked in the meeting | status |
|---|---|
| "first thing is to just get a clean code" (our own code) | ✅ this package, plus the reorganized repository |
| a clean base RLM, separate from our prompt tuning ("what is the base RLM, what is your RLM") | ✅ `rlm.py` is the base; our tuning is the `tuning` feature, so it can be measured on or off |
| "then we start adding all of this stuff" | ✅ ordering, tuning and compression are features on top of the base |
| a config file: ordering or not, BM25 or graph, compression or not, 2x / 4x / 8x | ✅ `experiments.yaml` |
| configs for compression only, ordering only, compression + ordering | ✅ `compress_x2`, `order_query` / `order_graph`, `graph_x1` / `graph_x2` |
| offloading happens on its own when a read is bigger than GPU memory | ✅ `compression.offload`. ⚠️ Counted only: kvpress keeps the KV on the GPU |
| a cost for every call (GPU tokens vs fetched tokens) that the root tries to keep low | ✅ `cost.*`, with `show_root` telling the root the cost of each read |
| statistics on what it did, whether it was necessary, whether it was useful | ✅ per read: factor, whether compression was needed, whether it fed the answer |
| a small dataset where vanilla works but RLM + ordering fails, as a validation set | ✅ `hard_questions`. ⏳ Needs the vanilla and RLM runs on infolab |
| code formatting | ✅ black, isort, flake8 and mypy are clean |
| their cost code ("I will push it to GitHub") | ❌ `rlm_exp1` is not pushed yet |

Not run yet: none of this has been run on a GPU with a real model. The next step is
`run order_graph --set data.limit=3` on infolab, then the hard set.
