# Structured chunks vs the agentic loop — LOFT-128k

Four arms, one model (`Qwen/Qwen3-4B-Instruct-2507`), one served vLLM, one scorer.
Headline key is `subspan_em`, LOFT's own (`compare.py:44`). Arms A and B are the
finished `loft128k_main` campaign, reused rather than re-run.

| Arm | chunks from | run shape | flags |
|---|---|---|---|
| A | the root's own `str.find` | agentic REPL | `--mode rlm --search-k 0` |
| B | `search()` in the REPL | agentic REPL | `--mode rlm --search-k 5` |
| B′ | BM25 top-5, harness-supplied | map→reduce | `--mode structured --chunks-source bm25 --chunks-k 5` |
| C | a Claude Code instance, offline | map→reduce | `--mode structured --chunks-source file` |

Revision 1 read one window per map call and offered the answer format. Revision 2
reads one **contiguous passage** per call (`_cluster_hits`) and **requires** the
answer shape. Both are reported: the two changes landed together and are therefore
confounded with each other, so neither can be credited alone.

> **Scoring note.** Score every frame with `answers` as a real **list**. Read back
> from `predictions.csv` it is a string, and `loft/calculate_metrics.py:227-233`
> wraps a non-list gold in `[gold]` — so a multi-gold row (12 of 55 on nq) collapses
> into one bogus answer and every arm is understated. `metrics.json`, written from
> `checkpoint.jsonl`, is unaffected and authoritative. An earlier draft of this file
> carried numbers from the buggy path; the tables below are re-derived and match
> `metrics.json` exactly.

## 1. Structure alone does not improve accuracy

The clean test: B′ runs the same 55 rows B did, and nothing is selected on.

| subset | A | B | B′ rev1 | B′ rev2 |
|---|---|---|---|---|
| nq_128k | 0.5091 | **0.6000** | 0.4727 | 0.5273 |
| hotpotqa_128k | 0.3818 | 0.4000 | 0.4000 | 0.3455 |

Paired, pooled (n=110): **B′rev1 vs B 16–23** (p=0.34), **B′rev2 vs B 15–22**
(p=0.32), **B′rev2 vs A 18–19** (p=1.00).

Replacing the REPL with a fixed pipeline over the *same* chunks buys no accuracy in
either revision. It ties base RLM and loses to the BM25 agentic arm.

## 2. It ties base RLM at 6–8× less root context

`average_peak_context_tokens` — the axis `compare.py` puts against KVPress's
retained tokens.

| arm | nq | hotpotqa |
|---|---|---|
| A agentic | 2255 | 2170 |
| B agentic | 2562 | 2619 |
| B′ structured | **315** | **386** |
| C structured | **325** | **456** |

That is the arm's solid result, and it comes from removing the loop rather than from
improving the chunks. The loop was not buying accuracy over a fixed pipeline; it was
spending context. Nothing here needed a bigger model.

## 3. Better chunks beat BM25 chunks under the same pipeline

The **hard slice** is the 51 rows where arm B retrieved the gold answer string and
answered wrongly anyway (nq 22 of 55, hotpotqa 29 of 55).

| subset | A | B | B′ rev2 | C rev1 | C rev2 |
|---|---|---|---|---|---|
| nq (n=22) | 0.2727 | 0.0455 | 0.3636 | **0.6818** | 0.6364 |
| hotpotqa (n=29) | 0.4138 | 0.0345 | 0.3103 | **0.4828** | 0.3448 |

Paired, pooled (n=51):

| comparison | result | p |
|---|---|---|
| C rev1 vs B′ rev1 — same pipeline, same cap, only the chunks differ | 15 – 3 | **0.0075** |
| C rev2 vs B′ rev2 — same, after the fixes | 8 – 1 | **0.0391** |
| C rev1 vs A base RLM — the unselected comparator | 17 – 6 | **0.0347** |
| C rev2 vs A base RLM | 12 – 6 | 0.238 |

C vs B′ holds the pipeline, the cap and the revision fixed and varies only who chose
the chunks. It is significant in both revisions. C also reads **less than half** the
text: 1.5 windows per example on nq and 2.1 on hotpotqa against BM25's 5.

**What still limits this.** The slice is selected on B being wrong, and B′ shares B's
chunks — so B′ is disadvantaged by construction and C is not. That is why the A
comparison is reported beside it, and A survives significance only in revision 1.
n=51. Selecting on "the arm was wrong" is legitimate only because temperature is 0.0
and vLLM decodes greedily, so those rows reproduce rather than regressing to the mean.

## 4. What the revision-2 fixes actually did

They moved `em` up and `subspan_em` down, which is the trade the answer-shape change
was always going to make:

| run | subspan rev1 → rev2 | em rev1 → rev2 |
|---|---|---|
| C nq | 0.682 → 0.636 | 0.091 → **0.409** |
| C hotpotqa | 0.483 → 0.345 | 0.069 → **0.172** |
| B′ nq | 0.473 → 0.527 | 0.400 → 0.364 |
| B′ hotpotqa | 0.400 → 0.345 | 0.127 → **0.236** |

**Verbosity was hedging `subspan_em`.** A long answer contains more strings, so
substring containment hits more often. On `hotpotqa:33` revision 1 answered "The Hong
Kong actor, martial artist, and film director is Donnie Yen, ..." and scored a
subspan hit; revision 2 answered "Donnie Yen" and did not. Revision 1's advantage on
this metric was partly an artifact, and revision 2 is the more honest measurement.

**Clustering cost the map step its yield.** On nq's BM25 arm, merging 5 windows into
2.98 passages dropped findings per example from 1.73 to 0.69 and took abstentions
from 0 to 17 of 55. A 4B sub-model finds less in a bigger passage, and merging also
removes the second independent attempt at the same fact. The defect the merge fixed
was real; the cure appears to cost more than the disease on this model.

Because both changes shipped together, neither can be credited or blamed alone. The
ablation that would separate them — clustering on with the old prompt, and the new
prompt without clustering — has not been run.

## 5. The four runs that would settle it

Reading the transcripts behind §1 showed that **B′ differs from B in three ways, not
one**, so §1 is a bundle rather than a result about structure:

- the agentic arm reads all five windows in **one** call (45 of 50 nq examples, 40 of
  49 hotpotqa), while B′ splits the same text across calls;
- the map prompt interpolates LOFT's `task` verbatim — *"Print out the TITLE and ID of
  the documents you use to answer…"* — which derails a sub-model looking at one window
  (`nq:12` answers `TITLE: Philadelphia 76ers | ID: 732` and then `NOTHING RELEVANT`);
- the reduce step can throw a correct finding away (`nq:4`), which B structurally cannot.

Two of those are defects in this implementation, not properties of a fixed pipeline.
Each run below removes exactly one difference; the harness flags for all four are in
[`README.md`](README.md).

| run | flag | separates | comparator |
|---|---|---|---|
| 1 | `--structured-map-question query` | the derailed extraction from "structure" | B′ rev2 |
| 2 | `--structured-map-granularity all` | splitting the text from the pipeline | B′ rev2 |
| 3 | `--structured-answer-shape offer` | the two revision-2 changes | B′ rev2, B′ rev1 |
| 4 | `--mode rlm --chunks-source file` | chunk quality from the pipeline entirely | 4c, A |

Runs 1–3 are the structured arm on the full 55 rows of each subset, so they land
against §1's unselected comparison. Run 4 is the **agentic** root reading C's chunks
on the hard slice, delivered through `search()` so the system prompt stays
byte-identical to B's. It is run beside **4c** — the same agentic arm with BM25
chunks, same slice, same commit — because the historical arm B ran at an older commit
and scores ≈0 on this slice *by construction*, so it is not a control.

Results: _pending_.

## Reproducing

```bash
python -m evaluation.rlm.run_benchmark --dataset loft --data-dir nq_128k --split test --limit 55 \
  --mode structured --chunks-source bm25 --chunks-k 5 --sub-backend http --out $RESULTS
python -m evaluation.rlm.run_benchmark --dataset loft --data-dir nq_128k --split test \
  --ids-file <slice>.ids --mode structured --chunks-source file \
  --chunks-file evaluation/rlm/chunksets/nq_128k.hardslice.jsonl --sub-backend http --out $RESULTS
```

The slice comes from an existing `--search-k` checkpoint via
`python -m evaluation.rlm.hard_slice`. Chunk sets are committed under
`evaluation/rlm/chunksets/`, each record carrying its windows and how they were found.
`structured_revision` in `config.yaml` separates the two pipelines; a missing revision
is a hard resume conflict.
