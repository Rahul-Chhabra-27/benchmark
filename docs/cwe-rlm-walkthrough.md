# CWE through revision 6 ordered-chunk RLM

This is the complete record of one cached `lighteval/RULER-131072-Qwen-3` CWE
example through both revision-6 ordered-chunk arms. The run is deliberately a
one-example diagnostic, not the 13-task campaign gate.

## What was fixed and what was measured

The active checkout is Skylight's `gautam-dev` at `0ecb0cdc`. `origin` is
`git@github.com:skylight-org/intel-longcontext-pc.git`; `master` was not changed.
The working copy uses its own repaired Python 3.12 environment and contains no
active `.pth` reference to the moved `benchmark/` checkout. The opt-in tracer is
[`evaluation/rlm/diagnostic.py`](../evaluation/rlm/diagnostic.py). It uses Python
call events and records project functions, generated REPL code, parent/child
edges, call counts, source locations, model messages/responses, and HTTP/client
boundaries. It never serializes arbitrary locals. The fake-client tests in
[`tests/evaluation/test_rlm_diagnostic.py`](../tests/evaluation/test_rlm_diagnostic.py)
prove identical answers, observations, requests, and metrics with tracing on and
off, including nested and generated functions.

## Model loading

On cat, GPU 0 was checked idle (1 MiB used). The server was started with
`evaluation/rlm/serve_root.sh`, vLLM `0.8.5.post1`, Qwen3-4B-Instruct-2507,
`--port 8011 --max-model-len 65536 --gpu-memory-utilization 0.5 --enforce-eager
-O0`, pinned to CPUs 2–5. Readiness was a real chat generation, not `/models`.
The server log shows `EngineArgs.create_engine_config`, tokenizer loading,
`AsyncLLM.from_engine_args`, `EngineCore.__init__`, `Worker.load_model`,
`DefaultModelLoader._prepare_weights`, `Qwen3ForCausalLM.load_weights`, KV-cache
initialization, and `AsyncLLM.generate`. Installed versions were Python 3.11.2,
torch 2.6.0, transformers 4.51.3, and vLLM 0.8.5.post1. The named entry points,
source hashes, startup log, and readiness response are in the archived
`evidence/server-sources.json` and `evidence/server.log`.

## Dataset preparation and pin

The cached dataset was loaded with `load_examples` → `load_benchmark_dataset` →
`_load_ruler128k` → `split_ruler_input` (500 rows processed while preparing the
cached frame). The loader applies the task's last question pattern, separates
context, question, answer prefix, and `outputs`, then `iter_benchmark_examples`
normalizes the example contract. Dataset generation happened previously; this run
only loaded and normalized the cached artifact. The first CWE row is
`ruler128k:cwe:0`, with document SHA-256
`c30bd0d8b8b15469d3db2c38b4f2dbfa6de74522fcbb0946b0bc4069ebcb3f07`, 242,886
characters, and gold words: `poignance, element, change, wonderful, loose,
godfather, huge, waist, cannon, silicon`. The pin is in `evidence/pin.json`.

## Common control flow

```text
cached row → split/normalize → 2,000-char, zero-overlap grid (122 chunks)
       ├─ bm25-query: bare question → BM25 rank_all
       └─ bm25-graph: question-free BM25 graph → forest/DFS order (fresh cache)
                         ↓
             validate permutation and source text
                         ↓
 RLM init: root/sub HTTP clients, revision-6 prompts, persistent REPL, 50 steps,
 40 sub-calls, 900 s example timeout, 8,192-token reader cap
                         ↓
 root chat → generated Python → REPL observation → reader HTTP request(s) → FINAL
                         ↓
 prediction CSV → canonical RULER scorer → string_match_all
```

The implementation points are `build_chunk_view` and `chunk_diagnostics` in
`evaluation/rlm/run_benchmark.py`, `organize`/`graph_order`/`OrderedPayload` in
`evaluation/rlm/organization.py`, and `RLM.run`, `_make_env`, `_exec`, and
`llm_query` in `evaluation/rlm/rlm.py`. Reader messages preserve each source
offset label. `ModelTokens.count_messages` prices the complete served-model chat
template; the server's reported prompt counts matched exactly.

## Actual executions

Both arms used the same ID, document hash, question, answer prefix, limits, and
gold list. Query-aware BM25 ranked 11 chunks with positive score. It made one
reader call over ranks 1–5 (5,127 input tokens), then finished in two root steps:

```python
ans = llm_query("What are the 10 most common words in the above list?", chunks[:5])
print(ans)
```

The reader returned `powder, concrete, sheath, center, polarisation, ejector,
wingtip, hourglass, psychoanalyst, ectodermal`; the root repeated that list in
`FINAL`. It consumed 8,238 total tokens (2 root calls, 1 sub-call), and scored
0.0%.

The graph arm built 379 graph edges and a 121-edge spanning forest in 4.09 CPU
seconds. Its first root step found candidate indices; its second read ranks 1–6
(6,218 input tokens), and its third called `FINAL`. It consumed 11,388 total
tokens (3 root calls, 1 sub-call), and also scored 0.0%.

The saved transcripts are reproduced here (whitespace shortened only):

**bm25-query**

```text
step 1 code: ans = llm_query("What are the 10 most common words in the above list?", chunks[:5]); print(ans)
reader: The 10 most common words in the list are: powder, concrete, sheath, center,
        polarisation, ejector, wingtip, hourglass, psychoanalyst, ectodermal
step 2 code: FINAL("Answer: The top 10 words that appear most often in the list are: 1. powder 2. concrete 3. sheath 4. center 5. polarisation 6. ejector 7. wingtip 8. hourglass 9. psychoanalyst 10. ectodermal")
end: final_called
```

**bm25-graph**

```text
step 1 code: hits = [i for i, c in enumerate(chunks) if "common" in c.text.lower() or "words" in c.text.lower() or "list" in c.text.lower()]; print(hits)
repl: [0, 1, 7, 9, 13, 17, 18, 23, 35, 40, 48, 58, 60, 61, 62, 65, 66, 67, 76, 84, 86, 90, 94, 95, 98, 101, 102, 105, 107, 110, 112, 115, 119, 121]
step 2 code: i = hits[0]; ans = llm_query("Extract the 10 most common words from the list in the chunk around index i. List them in descending order of frequency.", chunks[max(0, i-2):i+6]); print(ans)
reader: The 10 most common words in the list are: ejector, ectodermal, powder, sheath,
        concrete, center, polarisation, wingtip, hourglass, psychoanalyst
step 3 code: FINAL("Answer: The top 10 words that appear most often in the list are: ejector, ectodermal, powder, sheath, concrete, center, polarisation, wingtip, hourglass, psychoanalyst")
end: final_called
```

The complete transcripts, generated cells, observations, HTTP messages, function
traces, and summaries are in the local `benchmark_artifacts/cwe-diagnostic-20260911/raw/`
copy of the NAS archive. The tracer observed 175 project function identities and
131,513 call events for query, and 186 project identities and 218,473 events for
graph (third-party internals are represented only at named client/HTTP boundaries).

## Output, scoring, and failure

Both rows ended `final_called`, with no guard termination, timeout, overflow, or
integrity error. The graph run's existing checker notes that its organization was
not a cache hit: this is intentional because the graph cache was fresh. Every arm
covered all 122 chunks in its view, but the root read only 5 or 6. Explicit
word-by-word inspection found 4/10 gold words in the query reader payload and 8/10
in graph; neither returned prediction contained any gold word.

Scoring dispatches through `score_prediction_frame` to the RULER `string_match_all`
implementation. It strips control characters, checks each gold word by
case-insensitive substring, averages the ten matches, and returns 0.0 for each
arm. Recomputing from saved `predictions.csv` produced the same percentages and
was recorded in `evidence/validation.json`. The existing per-run integrity checks
returned no problems for query; graph's only note is the expected fresh-cache
condition. No full 13-task campaign gate was claimed.

## Recovery and provenance

The complete evidence archive is `cwe-diagnostic.tar.gz`, SHA-256
`bc963387be3a6744e442f73f0d99fedd16419df908a4b3b6e1ece15c3f0facda`, verified over
542 extracted files and copied from cat's NAS archive. The consolidation archive,
inventory, moved originals, and restoration instructions are in
[`cleanup-review/REVIEW.md`](../../cleanup-review/REVIEW.md). No original material
was deleted. The server started for this run was stopped at the end.
