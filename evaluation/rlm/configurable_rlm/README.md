# configurable_rlm

A clean base RLM with features added on top of it. Each feature is switched on in one
config file, `experiments.yaml`. This is the setup the professor asked for: *"a clean RLM,
and then we start adding all of this stuff, so now we can run any setting of RLM."*

## Files

| file | what it holds |
|---|---|
| `rlm.py` | **the base RLM**: short prompt, Python REPL, loop, `llm_query`, `FINAL`, plain reads. It imports none of the features. |
| `ordering.py` | feature: the `chunks` list, ordered by `bm25-query` or `bm25-graph` |
| `tuned.py` | `loop: tuned`: runs **our original, fully tuned RLM** (`harness/rlm.py`) unchanged, through its own runner with the same flags our launchers used |
| `compression.py` | feature: KV-compressed reads under a memory budget B, a compression factor per read, offloading, and cost |
| `config.py` | the settings, read from `experiments.yaml` |
| `run.py` | runs an experiment over its questions and writes the result files |
| `experiments.yaml` | every experiment: shared `defaults`, then what each named experiment changes |

The base loop only calls two kinds of thing. A **reader** decides how `llm_query` reads
(plain reads, or the compression feature). **Features** add tools, prompt lines and checks
(ordering). Turning a feature off removes its prompt lines and its behaviour completely.

There are two loops. `loop: base` is the minimal RLM in `rlm.py`. `loop: tuned` is our original
`harness/rlm.py`, with every prompt and guard, run exactly as it is. The same `ordering` and
`compression` switches work for both, so "base vs tuned" is a single setting.

## Commands

Run from the repository root. The root model needs an OpenAI-compatible server at
`models.base_url` (`bash scripts/rlm/serve_root.sh`). Experiments with
`compression.budget_mb` also load the sub-model on the GPU.

```bash
python -m evaluation.rlm.configurable_rlm list                      # every experiment
python -m evaluation.rlm.configurable_rlm check tuned_graph_x2      # what it will do + the command it runs (no model)
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

| experiment | loop | ordering | compression |
|---|---|---|---|
| `base` | base | — | — |
| `order_query` / `order_graph` | base | bm25-query / bm25-graph | — |
| `compress_x2` | base | — | ×2 at 1024 MB |
| `graph_x1` / `graph_x2` | base | bm25-graph | ×1 (control) / ×2 at 1024 MB |
| `graph_cost_aware` | base | bm25-graph | root picks ×1/2/4/8 per read, offloading, cost shown |
| `tuned` | tuned | — | — |
| `tuned_query` / `tuned_graph` | tuned | bm25-query / bm25-graph | — |
| `tuned_x2` | tuned | — | ×2 at 1024 MB |
| `tuned_graph_x1` / `tuned_graph_x2` | tuned | bm25-graph | ×1 (control) / ×2 at 1024 MB |

The `tuned*` experiments run **our original `harness/rlm.py`** with the flags our launchers
used (`scripts/rlm/rlm_order_lane.sh`, `run_bxf.sh`): 50 turns, `--chunk-order`, and for
KVzip `--sub-backend kvzip --press --memory-budget --compression-factor --sub-min-free-gib`
(the lane's formula). Settings equal to the runner's defaults are left off, as they were.
`check` prints the exact command. The results are comparable with the earlier campaigns.

To add an experiment, add a few lines under `experiments:`. The keys are:

- `ordering`: `none` | `bm25-query` | `bm25-graph`
- `loop`: `base` | `tuned` (tuned refuses `choices`, `offload` and `cost.show_root`, which our `rlm.py` doesn't have)
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
compression was needed. A `tuned` run writes our original runner's files
(`predictions.csv`, `metrics.json`, `config.yaml`, transcripts) into
`<output_dir>/<experiment>/`. `metrics.json` averages all of these over the questions.

## The professor's requests: status

| asked in the meeting | status |
|---|---|
| "first thing is to just get a clean code" (our own code) | ✅ this package, plus the reorganized repository |
| a clean base RLM, separate from our prompt tuning ("what is the base RLM, what is your RLM") | ✅ `loop: base` is the minimal RLM; `loop: tuned` is our original `rlm.py` unchanged. The same switches work for both, so the tuning's value is one comparison |
| "then we start adding all of this stuff" | ✅ ordering and compression are switches for both loops; factor choice, offloading and cost are on the base loop |
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
