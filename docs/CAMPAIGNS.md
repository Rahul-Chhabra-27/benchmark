# RLM campaigns

Every finished RLM campaign in this repository, oldest first. Each entry gives the question,
the setup, the headline numbers, the lessons that still apply, where the raw results are
archived, and where the full write-up lives. How to run any of it is in
[`RUNNING.md`](RUNNING.md). The design is in [`RLM.md`](RLM.md).

This file replaces the per-campaign handoffs, state files and results notes. Their full text
is still in git history, and each entry ends with the `git show` that prints it.

**Reading the numbers.**
- **What they cover.** LOFT-128k, **55 test rows per subset** unless an entry says otherwise.
  The ordered-chunk campaign's RULER-128k scores are also shown. Results on other datasets
  (longbench128k) are described, and their archive named, but not tabulated.
- **Metrics.** nq, hotpotqa and musique use `subspan_em`. qampari and quest use `coverage`,
  because the list `subspan_em` demands every gold and pins them near zero.
- **Noise.** One row at n=55 is **0.018**. vLLM's batching numerics move a configuration by
  about ±1 row even at temperature 0, so deltas under ~0.02 are noise.
- **Which rows.** The vanilla and base-RLM references are the `loft128k_guards` runs
  (2026-08-30). Those ran 110 rows, with dev rows 0–9 first, so test row *i* is their
  position *i*+10. They are realigned to the same 55 rows here.
- **Rescoring.** Score from `metrics.json` or `checkpoint.jsonl`, never from
  `predictions.csv`. The CSV turns `answers` into a string, and rescoring from it collapses
  multi-gold rows: nq 0.800 reads as 0.6545.

| # | campaign | when | branch |
|---|---|---|---|
| 1 | [LOFT-128k: vanilla vs RLM vs scratchpad vs KV arms](#1-loft-128k-vanilla-vs-rlm-vs-scratchpad-vs-kv-arms) | to 2026-08-27 | `rlm-scratchpad-loft128k` |
| 2 | [The RLM audit: generation-2 fixes](#2-the-rlm-audit-generation-2-fixes) | 2026-08-27 | `rlm-budget-derived-chunk-size` |
| 3 | [Fixed-chunk × KV-retention grid](#3-fixed-chunk--kv-retention-grid) | 2026-09-01 – 09-04 | `feature/fixed-chunk-grid` |
| 4 | [Structured chunks vs the agentic loop](#4-structured-chunks-vs-the-agentic-loop) | 2026-09-08 | `feature/structured-chunks` |
| 5 | [Chunk geometry and the retrieval-query bug](#5-chunk-geometry-and-the-retrieval-query-bug) | 2026-09-09 | `feature/structured-chunks` |
| 6 | [Query-agnostic selection](#6-query-agnostic-selection) | 2026-09-09 – 09-11 | `feature/query-agnostic` |
| 7 | [Ordered-chunk arms: bm25-query vs bm25-graph](#7-ordered-chunk-arms-bm25-query-vs-bm25-graph) | 2026-09-10 – 09-12 | `feature/rlm-ordered-chunks`, `feature/ordered-bxf` |
| 8 | [Qwen3.5-4B on LOFT-128k](#8-qwen35-4b-on-loft-128k) | 2026-09-14 | master `c92f4e2` |

---

## 1. LOFT-128k: vanilla vs RLM vs scratchpad vs KV arms

**Question.** At 128k tokens, does an RLM beat reading the whole document? The RLM holds the
document in a REPL variable, and its root reads slices through `llm_query`. The campaign also
asked whether a scratchpad or KV-compressed sub-calls help.

**Setup.** `Qwen/Qwen3-4B-Instruct-2507` on vLLM 0.8.5. The model holds 262k tokens natively,
so the 139264-token server window needed no YaRN. It is a non-thinking model, so the runs
passed no `--no-think`. All five LOFT RAG subsets.

| arm | what it tests |
|---|---|
| `vanilla` | the whole ~119k-token document in the window |
| `rlm` | the document as a REPL variable; the root chunks and recurses |
| `rlm+scratchpad` | the same, plus a `note()` buffer that survives eviction |
| 4a / 4b | sub-calls read their slice through a KVzip-compressed cache, at 32k-char chunks (4a) or ~32k-token chunks (4b) |
| cell 5 | the press alone, through kvpress's `evaluate.py` |

**Results** (test rows 0–54, from `loft128k_guards`):

| subset | metric | vanilla | base RLM |
|---|---|---:|---:|
| nq | subspan_em | 0.473 | 0.545 |
| hotpotqa | subspan_em | 0.491 | 0.382 |
| musique | subspan_em | 0.236 | 0.109 |
| qampari | coverage | 0.396 | 0.058 |
| quest | coverage | 0.145 | 0.006 |

- **Over all 110 rows, vanilla beat RLM on every subset** (nq 0.500, hotpotqa 0.455). On the
  55 test rows, RLM leads only on nq.
- **There is no context wall at 128k.** So RLM's result here is cost, not accuracy: about
  **2,000 tokens** of root context against vanilla's **~120,000**, for 71–82% of vanilla's
  score (on the 110 rows).

**Lessons that still apply.**
- **Check `truncated_fraction` is 0 in every vanilla run.** Every nq context is 475,516
  characters. The default `--vanilla-char-limit` of 400,000 drops about 16% off the end of
  every document, which hands RLM a win it did not earn. Use `--vanilla-char-limit 700000
  --vanilla-max-prompt-tokens 134000`.
- **`KVzipPress` masks evicted KV, and never frees it.** A slice costs its full uncompressed
  KV whatever the budget, so budget-derived sizing buys comparability, not GPU headroom.
- **Two scoring quirks are faithful to LOFT and left alone.** On qampari and quest a list repr
  scores and prose does not: `Final Answer: France, Spain` extracts as one answer. And a
  bracketed citation such as `Paris [1]` extracts as `["1"]`.
- **Correct abstentions were being scored as guard rejections.** 7–39% of rows per subset
  ended `ungrounded_final`, and musique was worst. This led to generation 2's `FINAL_NONE`
  (§2).

**Archive:** `/mnt/nas/gautammahale/benchmark/evaluation/results/loft128k_guards` (base RLM
and vanilla). **Full record:** `git show 4f7396f:evaluation/rlm/loft128k/README.md`, which
has the arms, the colocation budget and the verification steps. To run it, see RUNNING.md §8.

---

## 2. The RLM audit: generation-2 fixes

**What.** An audit on 2026-08-27 of `rlm-budget-derived-chunk-size` (`16e191d`), checked
against the standalone `rlm-prajna2-git` fixes and campaign 1's data. Every item was applied
the same day, and 115 tests passed. **Runs after this are generation 2, so never compare them
row for row with generation-1 results.**

**What changed.**
- **The loop.**
  - Grounding uses word boundaries (`_mentions`).
  - A code fence cut off at `max_tokens` is recovered.
  - `FINAL_VAR` on a missing variable raises, where it used to score the text `<missing var>`.
  - The benchmark's answer format reaches both arms.
  - `FINAL_NONE` abstains cleanly: `end_reason=abstained`, counted in `runtime.abstained`.
  - Answers are grounded against untruncated output.
  - A root-context overflow evicts the oldest turns and retries (`overflow_evictions`),
    where it used to be an exception.
- **The harness.**
  - Checkpoints are de-duplicated by id.
  - `config.yaml` is written before the first example. A resume then refuses a changed
    `--limit`, `--split`, `--max-steps`, timeout or sub-call cap.
  - `--split dev|test|all` was added.
  - There is one results-path constant (`evaluation/results_layout.py`).
  - `compare.py` labels scratchpad runs, and uses a headline key per dataset (LOFT →
    `subspan_em`).
  - `progress_match` was renamed `progress_match_loose`.
  - The cost axis is `max(root_peak, sub_peak)`.
- **kvzip and tooling.**
  - The sub-call system prompt stays off the compressed side.
  - `run_infolab.sh` traps `HUP`, and the dead standalone-KVzip setup is gone.
  - The RLM tests moved under `tests/evaluation/`, where `make test` collects them.

**Left alone on purpose.**
- KVzip compression is logical.
- Decimal GB and binary GiB coexist.
- Auto sizing is not bit-reproducible across machines.
- The two-venv conflict is structural.
- `sub_model` is enforced on resume, not added to the run-directory slug, since adding it
  would rename every existing run directory.
- `_mentions` still grounds an enumerated multiple-choice label that appears in the task.

**Full record:** `git show 616ec42:fix_rlm.md`. It has every item's priority, the file and
line, and why.

---

## 3. Fixed-chunk × KV-retention grid

**Question.** Hold the logical KV budget B fixed. Does reading a bigger chunk, N = B × F
tokens, through KVzip at ratio 1 − 1/F, beat reading a smaller uncompressed chunk? The F = 1
column is the `no_press` control.

**Setup.** `longbench128k`: six LongBench subsets rebuilt to exactly 131,072 tokens, 60 rows
each. The v2 grid ran hotpotqa only, 20 rows, with 40 feasible cells, on dog and elk,
finishing 2026-09-03. A second grid, `mb_v1`, used budgets in MB so they match the plain
KVzip matrix.

**Finding.** It is described here, not tabulated (longbench128k):
- **Quality tracked the admitted context N, not the budget, peaking near N ≈ 8k tokens.**
  Along that anti-diagonal every compressed cell beat the uncompressed control.
- **F = 16 collapsed in every row.**
- **The evidence is thin:** one subset, n=20.
- **Some cells are not measurements.** The cells at N = 131,072, and some at 65,536, had
  their sub-model refuse slices for GPU memory. The headline cells were clean.
- **Run `evaluation/rlm/fixed_chunk_grid/audit_cells.py` on any tree before trusting a cell.**
- **The recommended next step was not another grid.** It was the N ≈ 8k cells (F = 1, 2, 4,
  8) at 100+ rows on all six subsets.

**Lessons.**
- KVzip retention is simulated.
- Use a fresh `RESULTS` directory for every `LIMIT`. `LIMIT` is resume-critical, but it is not
  in the directory name.
- `flock` on the NAS does not lock across hosts, so `run_grid.sh` locks with `mkdir`.
- `MIN_FREE_GIB` overrides the flat 12 GiB headroom.
- ant's Blackwell (sm_120) cards need a cu128 torch.

**Archive:** `/mnt/nas/gautammahale/fixedgrid/{v2,mb_v1,pilot}`. **Full record:**
`git show 5bbd479:evaluation/rlm/fixedgrid/README.md` and
`git show 5bbd479:evaluation/rlm/fixedgrid/RUNBOOK.md`. To run a cell, see RUNNING.md §8.

---

## 4. Structured chunks vs the agentic loop

**Question.** Is it better to let the model wander the document, or to hand it a fixed stack
of pages? Every arm label combines three choices: who reads, who picked the text, and which
prompt.

| arm | who picked the text | who reads |
|---|---|---|
| A | the root itself, with `str.find` | agentic (the REPL loop) |
| B | the root itself, with `search()`: BM25, top 5 | agentic |
| B′ | the harness: BM25 top 5, frozen | structured (map → reduce) |
| C | a larger model (Claude Code), offline, ahead of time | structured |

Revision 2 of the structured arm read one merged passage per call, and *required* the answer
shape. Revision 1 read one window per call, and offered the shape.

**Results** (nq and hotpotqa, 55 rows, `Qwen3-4B-Instruct-2507`):

| subset | A | B | B′ rev1 | B′ rev2 |
|---|---|---|---|---|
| nq | 0.509 | **0.600** | 0.473 | 0.527 |
| hotpotqa | 0.382 | 0.400 | 0.400 | 0.345 |

- **Structure buys no accuracy.** It ties A and loses to B. Pooled, B′ vs B was 16–23 in
  rev1 and 15–22 in rev2, neither significant.
- **It buys cost.** Peak root context was 315–456 tokens for the structured arms, against
  2,170–2,619 for the agentic ones: **6–8× less**.
- **Better chunks help, but only under the fixed pipeline.** On the *hard slice* (51 rows
  where B retrieved the gold and still answered wrong), C beat B′ 15–3 (p=0.0075). The agentic
  root given C's own chunks lost to C 4–16 (p=0.012), and did no better than frozen BM25
  chunks (5–11).
- **The rev2 bundle, separated.** Merging windows was worth +0.036 subspan. Requiring the
  answer shape cost 0.036 subspan and bought exact match (+0.19 with merging on).
- **Stripping LOFT's instructions from the map prompt made it worse** (10–20). The
  instructions were acting as a filter.

**Caveat, found in campaign 5.** The structured arms queried BM25 with LOFT's whole task
string, while the agentic arm B wrote its own clean query. "Structure loses to B" is partly
that confound. Re-run with `--chunks-query bare` before quoting it.

**Full record:** `git show 24f54e4:evaluation/rlm/STRUCTURED_RESULTS.md` (every table and
p-value) and `git show 6bbac5d:evaluation/rlm/ARMS_EXPLAINED.md` (the arms in plain words).

---

## 5. Chunk geometry and the retrieval-query bug

**Setup.** 117 cells and 6,435 example-runs, with **0 errors**. nq, hotpotqa, musique and
qampari, 55 rows each. Qwen3-4B-Instruct-2507 as root and sub, on one vLLM (fox GPU 4).

**The bug that mattered most.** `--mode structured` gave BM25 the whole LOFT task string: a
page of "print the TITLE and ID…", with the question on its last line. Only 5 of 25 query
terms were the question. Recall of the gold answer, before → after:

| subset | r@2 | r@5 | r@10 |
|---|---|---|---|
| nq | **0.00 → 0.95** | 0.65 → 0.96 | 0.80 → 0.96 |
| qampari | 0.00 → 0.95 | 0.53 → 0.98 | 0.95 → 1.00 |
| hotpotqa | 0.05 → 0.71 | 0.64 → 0.89 | 0.82 → 0.91 |
| musique | 0.11 → 0.38 | 0.29 → 0.58 | 0.53 → 0.78 |

The fix is `--chunks-query bare`. It is opt-in, so existing runs reproduce byte for byte, and
it adds `qbare` to the run-directory name. **Pass it for any new measurement.**

| subset | published (w2000 k5) | best task-query cell | best bare-query cell | oracle |
|---|---|---|---|---|
| nq | 0.545 | 0.709 | **0.800** (w2000 k2) | 0.800 |
| hotpotqa | 0.345 | 0.509 | 0.545 | 0.491 |
| musique | 0.109 | 0.218 | 0.182 | 0.218 |
| qampari | 0.079 | 0.368 | 0.539 | 0.588 |

- **nq reaches its oracle ceiling** at 310 peak root tokens, reading 4,000 characters. Its
  retrieval is solved, so the reader is the limit.
- **A second retrieval round** (`--structured-hops 2`), queried from what the first round
  *found*, took hotpotqa from 0.345 to 0.582.
- **musique responds to nothing.** Retrieval doubles, the score stays within noise, and its
  oracle is only 0.218.
- **Multi-value scoring fix.** `--answer-list-format auto` (the default) shows qampari and
  quest the bracketed list their scorer parses: 0.000 → 0.800 on the oracle.
  **Any qampari or quest number from before this fix is wrong.**
- **The geometry is three knobs now.**
  - `--search-window` / `--search-overlap` set the retrieval unit.
  - `--structured-map-chars` sets how much the reader sees.
  - `--reduce-finding-chars` must be raised with the window, or the sweep measures the cap.

**Still open.**
- The corrected-query grid is partial for hotpotqa (8 of 12 cells) and qampari (5 of 12).
- The oracle, exposure and hops runs used the old query.

**Operational lessons.**
- **A wedged vLLM still answers `/v1/models`.** Health-check by generating.
- **Key any analysis on the axes that separate arms**, not just the ones swept. Three arms sit
  at the same (window, k) coordinates.
- **Never reuse a chunk file across window sizes.**
- **An NFS weight load stalled for 13 minutes**, so stage weights on local disk.

**Archive:** `/mnt/nas/gautammahale/structured_grid`, and the chunk sets in
`/mnt/nas/gautammahale/chunksets/`. **Full record:**
`git show d8c6c64:STRUCTURED_CHUNKS_CHANGES.md` and
`git show 348b891:evaluation/rlm/HANDOFF.md`.

---

## 6. Query-agnostic selection

**Question.** Campaign 5 left every number a function of what BM25 was asked. How much of
nq's 0.800 is the question *buying*? The campaign built context selectors that never see the
question:
- `lead`, `stride`, `random` and `all` (`query_free_chunks.py`);
- a query-free notes index (`window_notes.py`);
- held-out rows 55–99, which no configuration choice had seen;
- paired statistics (`evaluation/rlm/paired.py`).

**Result at k = 2** (a 4,000-character read; 16 of about 85 ladder cells were in when the
record stops):

| subset | bm25 task | bm25 bare | lead | stride |
|---|---|---|---|---|
| nq | 0.000 | **0.782** | 0.000 | — |
| hotpotqa | 0.000 | **0.345** | 0.000 | 0.000 |
| musique | 0.000 | **0.109** | 0.000 | — |
| qampari | 0.000 | **0.220** | 0.004 | 0.004 |
| quest (held out) | 0.000 | **0.258** | — | — |

- **At this budget the query-free floor is zero on every subset.** The method has no
  performance that is not the query's.
- **The published default query (`bm25 task`) scores 0 at k = 2 too.** That is campaign 5's
  recall@2 = 0.00, now seen end to end.
- **quest's 0.258 is the cleanest generalisation number,** since quest was never in a sweep.
- **The record stops mid-campaign.** The k = 5 and 10 ladder, the exhaustive and notes arms,
  and the holdout were still running. Their outputs are in the archive, unsummarised.

**Traps each produced a plausible number, not an error.** All are now guarded:
- **Colliding run directories:** every file-sourced arm at one k got the same name, so the
  harness now stamps `sel-<tag>`.
- **Invented findings:** past `--max-sub-calls`, the `[SUB-CALL LIMIT REACHED]` string was
  counted as a finding. A budget that doesn't cover the chunk set is now refused.
- **A one-call sweep:** passage granularity collapsed an exhaustive sweep into one call, so
  the ladder runs with `window`.
- **`--chunks-k` on a file source is a payload cap** of k × window.

**Archive:** `/mnt/nas/gautammahale/archive/qagnostic/`. The lane scripts are in
`evaluation/rlm/qagnostic/`. Their paths write into `$HOME`, which predates the storage
policy, so repoint `RESULTS`, `QCHUNKS` and `--out-dir` before reusing them.
**Full record:** `git show 00b01a6:evaluation/rlm/QAGNOSTIC_HANDOFF.md` and
`git show 04c271a:evaluation/rlm/qagnostic/README.md`.

---

## 7. Ordered-chunk arms: bm25-query vs bm25-graph

**Question.** Make the agentic RLM root the orchestrator. It is handed **every** 2000-character
chunk as a `chunks` list, in a pre-computed order, and told how the order was made.

| arm | flag | order |
|---|---|---|
| query-aware | `--chunk-order bm25-query` | every chunk, ranked by BM25 against the bare question |
| query-agnostic | `--chunk-order bm25-graph` | a BM25-graph organization built before the question is known |

**Shared by both arms.**
- There is no `search()`.
- A chunk read is capped at 8,192 tokens, priced exactly, and refused if over budget, never
  truncated.
- The base-RLM limits apply: 50 steps, 40 sub-calls, 900 s.
- Revisions 1–6 changed only the prompts and guards, never the order: the `Final Answer:`
  cell steer, a 3-read abstention bar, the lower-case note, the empty-read refusal, and
  composed-answer grounding (r6).
- Rows 0–2 of every subset tuned the revisions, so the reports repeat every pair on rows
  3–54.

**LOFT-128k** (revision 5, `2b62ef2`, 550 rows, gate PASSED):

| subset | metric | agentic query | agentic graph | Δ [95% CI] | structured source | structured graph | base RLM | vanilla |
|---|---|---:|---:|---|---:|---:|---:|---:|
| nq | subspan_em | 0.818 | 0.709 | +0.109 [0.000, +0.236] | 0.709 | 0.764 | 0.545 | 0.473 |
| hotpotqa | subspan_em | 0.709 | 0.600 | +0.109 [−0.036, +0.255] | 0.491 | 0.527 | 0.382 | 0.491 |
| musique | subspan_em | 0.291 | 0.273 | +0.018 [−0.109, +0.145] | 0.109 | 0.255 | 0.109 | 0.236 |
| qampari | coverage | 0.365 | 0.167 | +0.198 [+0.124, +0.276] | 0.599 | 0.479 | 0.058 | 0.396 |
| quest | coverage | 0.212 | 0.294 | −0.082 [−0.176, +0.012] | 0.467 | 0.364 | 0.006 | 0.145 |

- **Query order vs graph order.**
  - The query order wins clearly on qampari (sign p = 7e-5).
  - It wins less surely on nq and hotpotqa: both CIs reach 0.
  - musique is a tie, and quest leans towards graph.
  - Rows 3–54 give the same picture.
- **Agentic vs map → reduce.** On single-answer subsets the agentic query arm matches or beats
  map → reduce. It does so on **7–16k input tokens per question, against about 127k**. It
  loses clearly on the list subsets, where it reads 5–12 of about 230 chunks.
- **Where the gold sits.** The query order puts the gold in the top 5 for 62–96% of rows. The
  graph order's median gold position is 42–106.

**RULER-128k** (revision 6, `c466401`, 13 tasks × 55 rows × 2 arms, `string_match` 0–100):

| task | query | graph | Δ | note |
|---|---:|---:|---:|---|
| cwe | 0.0 | 0.0 | 0.0 | a few chunks cannot count words across the document |
| fwe | 95.2 | 92.1 | +3.0 | |
| niah_multikey_1 | 96.4 | 89.1 | +7.3 | |
| niah_multikey_2 | 98.2 | 96.4 | +1.8 | |
| niah_multikey_3 | 90.9 | 47.3 | +43.6 | graph gate FAIL: 25 rows looped on over-budget reads and score 0 |
| niah_multiquery | 96.8 | 81.8 | +15.0 | |
| niah_multivalue | 95.5 | 88.6 | +6.8 | |
| niah_single_1 | 100.0 | 100.0 | 0.0 | |
| niah_single_2 | 100.0 | 98.2 | +1.8 | |
| niah_single_3 | 78.2 | 87.3 | −9.1 | 12 gold UUIDs cross a chunk boundary; graph's 8-chunk reads rejoined 5 of them |
| qa_1 | 72.7 | 72.7 | 0.0 | |
| qa_2 | 56.4 | 45.5 | +10.9 | |
| vt | 21.1 | 22.5 | −1.5 | both find about 1 of 5 chained variables; r6 rejects invented counts |
| **mean** | **77.0** | **70.9** | | without niah_multikey_3: 75.9 vs 72.9 |

**Audit findings that still apply.**
- **LOFT's list `coverage` is strict.** Decorated items score 0, as in "Red Dawn (1984)". A
  lenient match lifts every arm by 0.006–0.089 and changes no ranking.
- **The case trap caused most no-read rows:** a capitalised term tested against
  `c.text.lower()`.
- **Revision 7 is open.** It would name a concrete slice in the over-budget refusal, which
  would fix niah_multikey_3 graph. Quote that cell only with the gate caveat.

**Follow-ons.**
- **B × F cells** (`feature/ordered-bxf`): the ordered arms read through an in-process KVzip
  model. Their 4-cell smoke test passed on 2026-09-12; RUNNING.md §7 has the output.
- **The CWE diagnostic:** a traced run of one RULER CWE row through both revision-6 arms.
  Both scored 0.0. Their readers saw 4 and 8 of the 10 gold words, and neither prediction
  held any. The tracer is `evaluation/rlm/diagnostic.py`. The evidence tarball is
  `cwe-diagnostic.tar.gz`, SHA-256 `bc963387…0facda`.

**Archive:** `/mnt/nas/gautammahale/archive/rlmorder/`, with `full_{results,report,logs}`,
`ruler_full_{results,report,organization,logs}` and every pilot. **Reproducing:** RUNNING.md
§6. **Full record:** `git show 1aa27ea:evaluation/rlm/RLM_ORDER_STATE.md`, which has the
decisions, every revision, gate policy, audit and changelog. The CWE trace is at
`git show 4ac060a:docs/cwe-rlm-walkthrough.md`.

---

## 8. Qwen3.5-4B on LOFT-128k

**Setup.** `Qwen/Qwen3.5-4B`, a hybrid linear/full-attention model with arch `qwen3_5`.
- vLLM 0.8.5 cannot serve it, so it ran on vLLM 0.19 (cu128) with `--no-think`.
- One card at utilisation 0.5. Its KV is about 32 KiB per token.
- The rlm arm only, on nq and hotpotqa, 55 test rows.

| subset | code | subspan_em | errors | abstained | s/row | peak ctx | sub-calls/q |
|---|---|---:|---:|---:|---:|---:|---:|
| nq | `6b18aef` (public benchmark repo) | 0.509 | 0 | 7 | 41 | 5.4k | – |
| nq | master `c92f4e2` | **0.527** | 0 | 9 | 36 | 5.5k | 0.15 |
| hotpotqa | `6b18aef` | 0.473 | 0 | 7 | 25 | 5.0k | – |
| hotpotqa | master `c92f4e2` | **0.491** | 0 | 5 | 21 | 4.3k | 0.25 |

- **Against Qwen3-4B-2507's base RLM** (0.545 / 0.382): about 0.02–0.04 lower on nq, and about
  0.09–0.11 higher on hotpotqa.
- **Master vs `6b18aef`.** Master is about 0.018 higher and a little faster, but only 4–5
  rows change per subset, which is noise.
- **The root barely uses `llm_query`** (0.15–0.25 calls per question). It searches the
  document with code and reads snippets itself, so these scores measure a code-search agent,
  not recursive delegation.
- **No Qwen3.5 vanilla run exists yet.** RUNNING.md §5 runs one in about 35 minutes.

**Archive:** `/mnt/nas/gautammahale/archive/loft128k_qwen35/`; the master run is in
`master_c92f4e2/`. There is no separate write-up: this entry is the record.
