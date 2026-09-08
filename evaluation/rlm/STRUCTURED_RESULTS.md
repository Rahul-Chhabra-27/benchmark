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

Both changes shipped together, so neither could be credited or blamed alone. **That
ablation has now been run** — §5 — and it reverses half of this section: clustering
was worth `+0.036` subspan with the prompt held constant, and the whole drop came
from the answer shape.

## 5. The four runs, run

Each removes exactly one of the differences above; the flags are documented in
[`README.md`](README.md). Runs 1–3 are the structured arm on the full 55 rows of each
subset, so they land against §1's unselected comparison. Run 4 is the **agentic** root
reading C's chunks on the hard slice.

### 5.1 The revision-2 changes, separated — clustering was not the problem

`--structured-map-granularity window` and `--structured-answer-shape offer` restore one
revision-1 component each, so the 2×2 closes. Pooled, n=110:

| clustering | answer shape | run | subspan | em | abstained |
|---|---|---|---|---|---|
| off | offer | B′ rev1 | 0.4364 | 0.2636 | 2 |
| off | require | **5** | 0.4000 | 0.3091 | 2 |
| on | offer | **3** | **0.4727** | 0.1091 | 22 |
| on | require | B′ rev2 | 0.4364 | 0.3000 | 21 |

The two effects are equal and opposite, which is why rev1 and rev2 both scored 0.4364
pooled by two different routes:

- **Clustering is worth `+0.036` subspan**, at both prompt settings — paired 7–3 with
  the prompt held at `offer`, 5–1 held at `require`. So §4's "the cure appears to cost
  more than the disease" **is not supported**. Clustering does cost the map step its
  yield and does create the abstentions, but those abstentions replaced *wrong answers*.
- **Requiring the shape costs `-0.036` subspan and buys em** — `+0.19` with clustering
  on (0.1091 → 0.3000), a smaller `+0.046` with it off. §4's reading of that trade is
  confirmed and now quantified.

Neither is significant alone at n=110, but each holds its direction in all four cells.

### 5.2 The map prompt — stripping LOFT's blob makes it worse

| subset | B′ rev2 | 1 bare query | abstained |
|---|---|---|---|
| nq_128k | 0.5273 | **0.4364** | 17 → **0** |
| hotpotqa_128k | 0.3455 | **0.2545** | 4 → **0** |

Paired, pooled: **10–20 (p=0.099)** — the opposite direction from the hypothesis, and
the same direction on both subsets. The mechanism is real but backwards: the citation
instructions were acting as a **filter**. Without them the sub-model finds something in
almost every passage (abstentions to zero on both subsets) and the reduce step then
answers wrongly from marginal findings. A derailed extraction was costing less than an
undisciplined one.

### 5.3 One merged read — splitting was not the gap either

| subset | B′ rev2 | 2 one map call | B agentic |
|---|---|---|---|
| nq_128k | 0.5273 | 0.5455 | 0.6000 |
| hotpotqa_128k | 0.3455 | **0.4000** | 0.4000 |
| pooled | 0.4364 | 0.4727 | 0.5000 |

Reading every window in one call — exactly the payload arm B assembles — is paired
**13–9 (p=0.52)** against B′. It closes a little over half the distance to B (0.4364 →
0.4727 against 0.5000) and matches B outright on hotpotqa, but it is not significant:
the single largest confound in §1 accounts for at most half the gap, and possibly none.

### 5.4 C's chunks read by the agentic root — the advantage does not transfer

The hard slice, n=51. `--max-search-calls 1` is the capped variant; see the trap below.
A's row is §3's numbers pooled over the same rows; peak context was not measured on
the slice for it, and the peaks below are the agentic arms' own.

| arm | pipeline | chunks | subspan | em | peak ctx | searches/ex |
|---|---|---|---|---|---|---|
| C | structured | selector | **0.4706** | 0.2745 | 400 | — |
| A | agentic | none | 0.3529 | — | — | 0 |
| 4c | agentic | frozen BM25 | 0.2745 | 0.2353 | 1971 | 1.7 |
| 4 | agentic | selector | 0.1569 | 0.0588 | 5166 | **21.4** |
| 4c capped | agentic | frozen BM25 | 0.3529 | 0.2941 | 2184 | 1.0 |
| **4 capped** | agentic | selector | **0.2353** | 0.1176 | 5284 | 1.0 |

| comparison | result | p |
|---|---|---|
| 4 capped vs **C** — same chunks, same rows, only the pipeline differs | 4 – 16 | **0.0118** |
| 4 capped vs 4c capped — same pipeline, only the chunks differ | 5 – 11 | 0.21 |
| 4 capped vs A base RLM | 2 – 8 | 0.109 |
| 4c capped vs A base RLM | 8 – 8 | 1.00 |

**The selector's advantage is a property of the pipeline that reads the chunks, not of
the chunks.** The same 51 rows and the same windows are worth 0.4706 to the fixed
pipeline and 0.2353 to the loop. Inside the agentic arm, C's chunks beat BM25's on
nothing at all (5–11) — where in the structured arm the same swap was 15–3, p=0.0075.
This is the reverse of what the run was expected to show.

**The trap, which is why 4c had to be frozen too.** A fixed chunk set returns the same
windows for every query and the root cannot tell. On the uncapped arm it rephrased and
re-searched **21.4 times per example** against the frozen-BM25 control's 1.7, burning
peak context to ~5,200 tokens, and at `--max-steps 12` runs ended before reading at all.
Capping searches at one recovers `+0.08` subspan (6–2 paired) and is the honest version
of the arm; the conclusion is unchanged either way. Comparing run 4 against the
*adaptive* `--search-k` arm would have measured the trap, which is why the control is a
frozen BM25 arm reading the very windows B′ reads.

### What this leaves

§1's headline survives, and its explanation changes. Structure still buys no accuracy —
but not because the pipeline splits the text (§5.3 removes that and gains a third of the
gap), and not because the map prompt is derailed (§5.2 removes that and *loses* score).
What §5.4 adds is sharper: the fixed pipeline is what makes good chunks pay, and the
agentic loop cannot use them. §2's context result is untouched.

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
