# RLM over BM25-ordered chunks: state of the work

> **Read this file first.** It is updated at every milestone: code, pilot, launch, each
> finished stage, gate and report. Where two statements disagree, the newer one wins; the
> changelog at the bottom dates every update.
> **Last updated: 2026-09-10 ~20:55 IST.** The code and tests have landed. The pilot has
> not been run yet.

## Goal

The graph-order campaign (`feature/graph-order`) read every chunk through `--mode
structured`. That mode is a fixed map→reduce: the sub-model reads every group, and one
root call answers from the notes. The root never decides what to read, so the pipeline
behaves like RAG. This work makes the **agentic RLM root (the REPL loop, `RLM.run`) the
orchestrator** instead. The root receives **every chunk of the document as a `chunks`
list, in a pre-computed order**, and is **told how that order was made**. It can then
read the most promising chunks first and answer with confidence once it has seen the
evidence.

| arm | flag | order of `chunks` | what the root is told |
|---|---|---|---|
| RLM + BM25 + graph (query-agnostic) | `--chunk-order bm25-graph` | The question-free organization (`organization.organize`). On LOFT this is the same cached artifact the structured campaign read. | It was organized before the question was known, and chunks that share distinctive words sit next to each other. Find a hit, then read it together with its neighbours. |
| RLM + BM25 (query-aware) | `--chunk-order bm25-query` | Every chunk, ranked by BM25 against the bare question (`ChunkIndex.rank_all`). Zero-score chunks come last, in source order. | It is sorted by relevance, and `chunks[0]` is the best match. Read the top of the list first, and FINAL with confidence once it answers the question. |

**Shared by both arms:**
- the same 2000-character, zero-overlap partition of the document (`organization.CONFIG`);
- the tools `context`, `chunks`, `llm_query` and the `FINAL*` family, with no `search()`;
- the base-RLM settings: `--max-steps 50`, 40 sub-calls, 900 s run timeout, 32000-character slice cap;
- chunk reads of at most 8192 tokens, priced exactly; an over-budget read is refused, never truncated.

**What differs:** the order, the paragraph describing it, and the worked session. The
prompts are rendered from `CHUNKS_HELP`, `CHUNK_ORDER_NOTES`, `SEARCH_TERMS_CHUNKS` and
`CHUNK_WORKED_SESSIONS` in `rlm.py`.

## Decisions, and why

**Made by the user on 2026-09-10:**
- **Two arms only, with no source-order control.** As a result, a gain over base RLM cannot be attributed to the ordering alone, and on qampari and quest it also carries the list-shape note. The clean pair is **`bm25-query` vs `bm25-graph`**.
- **The sorted list only, with no `search()`.** This keeps the graph arm query-agnostic in what it is given. Second hops go through keyword filters over `chunks`.
- **Chunks arrive as a REPL variable only**, with no preview in the first message. The worked example makes the first move.
- **RULER-128k: all 13 tasks × 55 rows**, after LOFT.

**Design choices, each resting on earlier evidence:**
- **Worked sessions demonstrate each arm's intended move, never `.find`.** The root copies demonstrations (commit ec2c8a2).
- **The BM25 query is the bare question.** LOFT's instruction wrapper dropped recall@2 on nq from 0.95 to 0.00. For RULER, the query is the text after the last `Question:`.
- **Chunks are a variable, not something returned by `search()`.** A frozen set behind `search()` trapped the root into 11 identical searches (commit a1ff98f).
- **qampari and quest get the bracketed-list FINAL note.** The structured arms have `listfmt`; without it, correct lists score 0.000. Legacy agentic prompts are unchanged.
- **One chunk read is at most 8192 tokens**, the same as one structured reader group. It is priced with the served model's tokenizer on the complete reader message, and the gate requires `reader_input_tokens == reader_served_prompt_tokens`.

## Where everything is

| what | where |
|---|---|
| branch | `feature/rlm-ordered-chunks`, cut from `feature/graph-order` @ f8f7551. Pushed to **`nas`** only (fox `/mnt/nas/gautammahale/retrieval_checkout`). |
| Mac checkout | `linear/benchmark`. Tests run with `.venv-mac`; black and isort are in `.venv`. |
| compute host | **cat**, GPU 0 (RTX A6000, UUID `GPU-de398869-9f4d-2f4e-e0ab-738deda8383e`). cat has 32 cores. |
| server | tmux `go_root`: vLLM 0.8.5 on **:8002**, `--max-model-len 65536`, util 0.5, pinned to cores 2–5, with weights on cat-local `/tmp/gautammahale/hf_cache`. It is **reused** from the graph-order campaign, so do not kill it. |
| campaign dir | cat `/mnt/home/gautammahale/rlmorder/{checkout,results,results_pilot,organization,logs,report}` |
| LOFT organization cache | `/mnt/home/gautammahale/graphorder/organization`, the structured campaign's own cache. Every graph-arm row should report `organization_cache_hit: true`. |
| venv on cat | `/mnt/nas/gautammahale/venvs/rlm/bin/python` |
| HF cache | `/mnt/nas/gautammahale/hf_cache` (datasets and tokenizer). Lanes run with `HF_HUB_OFFLINE=1`. |
| comparators | Structured arms: `/mnt/home/gautammahale/graphorder/results`, with report `graphorder/report/`. Base RLM and vanilla: `/mnt/nas/gautammahale/benchmark/evaluation/results/loft128k_guards`. |

## Status board

| stage | nq | hotpotqa | musique | qampari | quest |
|---|---|---|---|---|---|
| code + tests | done: 532 tests pass; flake8 and mypy clean | | | | |
| LOFT pilot (3 rows × 2 arms) | todo | todo | todo | todo | todo |
| LOFT full (55 rows × 2 arms) | todo | todo | todo | todo | todo |
| RULER-128k (13 tasks × 55 × 2) | not started | | | | |

**The comparator to beat.** These are the structured graph-order results on the same 55
test rows. That campaign's gate passed on 2026-09-10.

| subset | metric | structured source | structured bm25-graph |
|---|---|---:|---:|
| nq | subspan_em | 0.709 | 0.764 |
| hotpotqa | subspan_em | 0.491 | 0.527 |
| musique | subspan_em | 0.109 | 0.255 |
| qampari | coverage | 0.599 | 0.479 |
| quest | coverage | 0.467 | 0.364 |

Those runs cost about 126–128k input tokens and 35–56 s per question: every chunk was read.

## How to resume (exact commands)

```bash
# 0. Is the server alive? Test it by GENERATING: a wedged vLLM still answers /v1/models.
ssh cat.cse.iitb.ac.in 'curl -s --max-time 60 localhost:8002/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen/Qwen3-4B-Instruct-2507\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":4}"'
# If it is dead: restart with evaluation/rlm/serve_root.sh (it resolves the GPU UUID to an index and sets UV_USE_IO_URING=0).

# 1. Update the compute checkout. While lanes are running, pull ONLY doc-only commits.
ssh cat.cse.iitb.ac.in 'cd /mnt/home/gautammahale/rlmorder/checkout && git pull --ff-only'

# 2. Pilot: 3 rows, both arms, every LOFT subset, one lane per arm.
ssh cat.cse.iitb.ac.in 'cd /mnt/home/gautammahale/rlmorder && for a in bm25-graph:8 bm25-query:9; do arm=${a%%:*}; core=${a##*:};
  tmux new-session -d -s "ro_pilot_$arm" "LIMIT=3 RESULTS=/mnt/home/gautammahale/rlmorder/results_pilot checkout/evaluation/rlm/rlm_order_lane.sh $arm $core > logs/pilot_$arm.log 2>&1"; done'

# 3. Full LOFT run: one lane per (arm, subset), pinned to cores 6-15.
ssh cat.cse.iitb.ac.in 'cd /mnt/home/gautammahale/rlmorder && core=6; for arm in bm25-graph bm25-query; do for s in nq hotpotqa musique qampari quest; do
  tmux new-session -d -s "ro_${arm#bm25-}_$s" "checkout/evaluation/rlm/rlm_order_lane.sh $arm $core $s > logs/full_${arm}_$s.log 2>&1"; core=$((core+1)); done; done'

# 4. Progress, gate, report. Gate before quoting ANY number.
ssh cat.cse.iitb.ac.in 'ls /mnt/home/gautammahale/rlmorder/results/*/metrics.json | wc -l'
ssh cat.cse.iitb.ac.in 'cd /mnt/home/gautammahale/rlmorder/checkout && PYTHONPATH=. /mnt/nas/gautammahale/venvs/rlm/bin/python \
  -m evaluation.rlm.rlm_order_gate ../results --expect 55 --structured-results /mnt/home/gautammahale/graphorder/results'
ssh cat.cse.iitb.ac.in 'cd /mnt/home/gautammahale/rlmorder/checkout && PYTHONPATH=. HF_HOME=/mnt/nas/gautammahale/hf_cache \
  /mnt/nas/gautammahale/venvs/rlm/bin/python -m evaluation.rlm.rlm_order_report --results ../results --out ../report \
  --structured-results /mnt/home/gautammahale/graphorder/results \
  --guards /mnt/nas/gautammahale/benchmark/evaluation/results/loft128k_guards --repo .'
```

**RULER-128k:**
1. Stage the dataset once, with network access on and outside `HF_HUB_OFFLINE`, into the NAS HF cache. Load each of the 13 splits of `lighteval/RULER-131072-Qwen-3`.
2. Launch the lanes with `DATASET=ruler128k`, passing the task names as the subsets. RULER has no split, so the lane script passes none.

## Traps
- **vLLM 0.8.5 parses `CUDA_VISIBLE_DEVICES` as an integer**, so a GPU UUID crashes startup. Resolve the UUID to an index; `serve_root.sh` does this.
- **`taskset` does not cover libuv's io_uring polling thread.** Set `UV_USE_IO_URING=0`; `serve_root.sh` does this.
- **A wedged vLLM still answers `/v1/models`.** Always preflight by generating; the lane script does.
- **One example at n=55 is 0.018.** Do not report deltas smaller than that.
- **qampari and quest are scored on coverage, never on subspan_em.**
- **The guards runs have 110 rows: dev rows 0–9, then test.** Our test row i is guards position i+10. The report aligns them and checks the gold answers row by row.
- **`/mnt/nas` is 99% full.** Results go to `/mnt/home`.
- **Never run `make style` on the Mac.** Its uv sync corrupts `.venv-mac`. Run `.venv/bin/python -m black/isort` directly.
- **Only doc-only commits may be pulled into a checkout whose lanes are running.**
- **Show loft128k numbers only** (standing user rule). Ask before showing any RULER-128k numbers.

## Next steps (ranked)
1. Push to `nas`, clone the campaign checkout on cat, and run the LOFT pilot. Gate it, and read 2–3 transcripts per arm. Check three things: does query-aware start with `chunks[:5]`, does graph locate a chunk and then read its neighbourhood, and do the qampari lists parse. Prompts may change only before the full launch.
2. Launch the full LOFT run, then gate, report, and update this file.
3. RULER-128k: stage the data, pilot, full run, gate, report.

## Changelog (newest first)
- **2026-09-10 ~20:55 IST: code, tests and this file landed.**
  - Added the `--chunk-order {bm25-graph,bm25-query}` and `--chunk-call-tokens` flags, and `ChunkIndex.rank_all`.
  - `run()` gained `ordered_chunks`, and priced order-preserving chunk reads via `OrderedPayload`.
  - Added the `ruler128k` dataset.
  - Added `rlm_order_lane.sh`, `rlm_order_gate.py` and `rlm_order_report.py`.
  - Added `tests/evaluation/test_rlm_chunk_order.py` (58 tests). The full suite is 532 passed.
- **2026-09-10 20:25 IST: the graph-order campaign was gated (PASSED, 10/10 runs, 0 errors) and reported** to `graphorder/report/`.
