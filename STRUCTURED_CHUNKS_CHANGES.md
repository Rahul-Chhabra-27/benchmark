# Chunk geometry, and the retrieval bug underneath it

What landed on `feature/structured-chunks` on 2026-09-09, why, and what to adopt in
what order. Companion to [`evaluation/rlm/README.md`](evaluation/rlm/README.md) (the
flags) and [`evaluation/rlm/ARMS_EXPLAINED.md`](evaluation/rlm/ARMS_EXPLAINED.md)
(the same thing in plain words).

Campaign: 117 cells, 6,435 example-runs, **0 errors**, LOFT-128K
nq / hotpotqa / musique / qampari, 55 test rows each, Qwen3-4B-Instruct-2507 as both
root and sub, one served vLLM.

---

## 1. The retrieval query bug — adopt this first, it is nearly free

**What was wrong.** `--mode structured` handed BM25 the benchmark's *whole task
string*. On LOFT that string is a page of instructions with the question on a final
`query:` line:

```
====== Now let's start! ======
Based on the documents above, can you answer the following query? Print out
the TITLE and ID of the documents you use to answer. Then format the answers
into a list.
query: when does monday night raw come on hulu
```

BM25 does not read sentences. It scored windows on all 25 surviving terms —
*documents, answer, format, print, title, list, query* — of which **only 5 are the
question**. Those 20 boilerplate terms appear in essentially every window of a LOFT
corpus, because the corpus IS a list of documents with titles and IDs.

**Cost, measured offline over 55 rows per subset** (recall of the gold answer string):

| subset | r@2 | r@5 | r@10 |
|---|---|---|---|
| nq | **0.00 → 0.95** | 0.65 → 0.96 | 0.80 → 0.96 |
| qampari | 0.00 → 0.95 | 0.53 → 0.98 | 0.95 → 1.00 |
| hotpotqa | 0.05 → 0.71 | 0.64 → 0.89 | 0.82 → 0.91 |
| musique | 0.11 → 0.38 | 0.29 → 0.58 | 0.53 → 0.78 |

At depth 2 it is total: the top two windows never contained the answer.

**The fix.** `--chunks-query bare` routes the query through `bare_query()`, which was
already in `rlm.py` for the map step and had never been carried across to retrieval.

**End-to-end effect on nq:** `0.545 → 0.800` subspan_em, at **310 peak root tokens**,
reading 4,000 characters instead of 10,000 — and that lands exactly on the oracle
ceiling, i.e. retrieval on that subset is now solved.

> **It also confounds the published §1 result.** The agentic `--search-k` arm lets the
> MODEL write its query and gets a clean one; the structured arm did not. "Structure
> loses to the BM25 agentic arm" was partly measuring which arm was allowed to ask a
> sensible question. That comparison needs re-running before it is quoted again.

**Default is `task`, on purpose.** Every `--structured-*` flag keeps the invariant
that a no-flag run reproduces the arms already on disk byte for byte. `bare` is
opt-in and stamps `qbare` into the run directory, so the two can never merge into one
score. **Pass it for any new measurement.**

**Generalising beyond LOFT:** `bare_query()` is conservative — a task with no `query:`
line comes back untouched. Any benchmark that wraps its question in an instruction
blob has this bug; check by tokenizing the query and counting how many terms are
actually the question.

---

## 2. Multi-value scoring — a correctness fix, not a tuning knob

`loft/calculate_metrics.py` parses qampari and quest through a splitter that only
accepts a **bracketed** list, but LOFT ships them the same bare `Final Answer: `
prefix as every other subset. Nothing ever told the model the shape.

Measured on the oracle arm at n=3: three replies naming **every** gold answer, scored
`coverage` **0.000**. One was `Rival Schools, Gorilla Biscuits, Walking Concert,
Quicksand, Dead Heavens` against exactly those five golds — extracted as ONE string,
intersecting nothing.

`--answer-list-format auto` (the default) shows the shape, and only where the scorer
parses it that way: structured mode, LOFT, qampari/quest. **0.000 → 0.800.**
Single-value subsets are untouched; the vanilla and agentic arms are deliberately
left alone because they are the reference line.

Also: `evaluation/compare.py` now picks `coverage` for qampari/quest instead of the
pinned `subspan_em`. Pass the task to `headline_score(metrics, dataset, task)`;
omitting it reproduces the old behaviour exactly.

---

## 3. The geometry axes — one knob became three

`--search-window` used to set the retrieval unit AND the reader's payload together.

| flag | default | what it controls |
|---|---|---|
| `--search-window` / `--search-overlap` | `2000` / `400` | the RETRIEVAL unit. 512 tokens at 20% overlap — the literature optimum, and NQ's own measured best. **Probably already right.** |
| `--structured-map-chars` | `0` (all) | the READ size: characters of each passage the map call sees |
| `--reduce-finding-chars` | `2000` | how much of each finding reaches the reduce step. **Raise it with the window** or a wide window is truncated back here and the sweep measures the cap |

**Exposure result** (retrieval frozen at w=8000 k=5, only the read size moves):
monotonic on every subset, no turnover — hotpotqa 0.000 at 400 chars → **0.382** at
8000. Findings per example go 0.02 → 1.33. That confirms the window effect is a real
read-budget effect, not an artifact of how BM25 ranks wider windows.

**The blocker this had to fix.** `win<W>o<O>` was stamped in the run-dir name only
when `--search-k` was set, and `--mode structured` *rejects* `--search-k`. So every
window cell named the same directory: the first ran, the rest aborted on a resume
conflict. **The sweep could not run at all** — not a wrong answer, no answer.

---

## 4. A second retrieval round

`--structured-hops 2` runs one follow-up round whose query is built from what the
first round's map step **found** — never from the retrieved window text. That
distinction is load-bearing: blind pseudo-relevance feedback over raw windows was
measured on this corpus and made recall *worse* everywhere (musique 47.5% → 37.5%).

hotpotqa **0.345 → 0.582**, beating even its own single-hop oracle (0.491).

There is deliberately no hop 3; a second routing level is a documented
non-improvement. **Caveat: this ran with the OLD query**, so some of the gain may be
hop 2 rescuing a bad first search. The clean version has not been run.

---

## 5. New tooling

| file | what it does |
|---|---|
| `evaluation/rlm/oracle_chunks.py` | builds a chunk file of windows that provably carry a gold answer, greedy set-cover over DISTINCT golds so multi-answer subsets get a real oracle. Retrieval perfect by construction ⇒ what is missing is the reader. |
| `evaluation/rlm/structured_decompose.py` | score by window × k, then splits each step up in k into gold vs non-gold windows added |
| `evaluation/rlm/structured_grid.sh` | the campaign driver: `geometry \| exposure \| oracle \| hops \| qbare \| all` |
| `evaluation/rlm/serve_root.sh` | one vLLM on one named GPU, refusing a card someone else is on and a port already listening |

`oracle_chunks.py` always writes `spans`. The two hand-made chunk files carry none,
which leaves `ChunkSet.hits_for`'s grid check unable to fire: window ids re-resolved
at a different `--search-window` stay in range and resolve to *real* text, so the run
reports a plausible score for windows it never chose. **Do not reuse a chunk file
across window sizes.**

---

## 6. Two operational lessons worth stealing

**A wedged vLLM still answers `/v1/models`.** fox's 5-day-old server accepted chat
requests, logged "Added request", and completed none; the GPU it was bound to showed
1 MiB. A campaign pointed at it looks exactly like one that is running — processes
alive, no errors, no rows. `structured_grid.sh` now asks for eight tokens as preflight
and refuses to launch otherwise. **Health-check by GENERATING, never by a static
route.**

**Key the analysis on the axes that separate arms, not only the ones being swept.**
A grid keyed on `(window, k)` silently rendered whichever cell it read last, and three
different arms sit at identical coordinates: the oracle `file` cell, the BM25 control,
and the corrected-query cell. nq at 2000/5 displayed 0.800 (the oracle) where the
control is 0.545. Same rule the run-directory names already follow.

---

## Integrating: suggested order

1. **`--chunks-query bare`** — biggest effect, no cost, and it changes what earlier
   results mean. Re-run the §1 structured-vs-agentic comparison with it before
   quoting that result again.
2. **`--answer-list-format`** (already on by default) and the `compare.py` change —
   correctness, not tuning. Any qampari/quest number produced before this is wrong.
3. **`--reduce-finding-chars`** — set it to `max(2000, window)` whenever the window
   exceeds 2000, or the geometry axis measures the cap.
4. `--structured-map-chars` and `--structured-hops` — new experiments, opt-in.
5. `serve_root.sh`'s generate-preflight — cheap insurance on any long campaign.

**Reproducing the campaign** (one 48 GB GPU is enough; the floor is ~14–16 GB, and
the full 80-cell run took 77 minutes with four lanes against one card):

```bash
GPU=4 evaluation/rlm/serve_root.sh                        # wait for a real generation
evaluation/rlm/structured_grid.sh all   nq hotpotqa musique qampari
evaluation/rlm/structured_grid.sh qbare nq hotpotqa musique qampari
python -m evaluation.rlm.structured_decompose --results <tree> --csv cells.csv
```

## Caveats to carry

- **n=55.** One example is 0.018, and the reproducibility floor is about ±1 example
  under concurrent load (the same nq configuration re-ran 0.545 against a published
  0.527, almost certainly vLLM batching numerics). Deltas under ~0.02 are noise.
- The corrected-query grid is **complete for nq and musique, partial for hotpotqa and
  qampari**.
- Oracle, exposure and hops all ran with the **old** query.
- `tests/test_pipeline.py`'s 42 failures are **pre-existing** — identical at
  `24f54e4`, a stale kvpress log-message assertion, unrelated to this work.
