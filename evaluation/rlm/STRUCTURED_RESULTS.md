# Structured chunks vs the agentic loop — LOFT-128k, 2026-09-08

Four arms, one model (`Qwen/Qwen3-4B-Instruct-2507`), one served vLLM, one scorer
(`subspan_em`, LOFT's headline key). Arms A and B are the finished
`loft128k_main` campaign, reused rather than re-run; B′ and C were run for this.

| Arm | chunks from | run shape | flags |
|---|---|---|---|
| A | the root's own `str.find` | agentic REPL | `--mode rlm --search-k 0` |
| B | `search()` in the REPL | agentic REPL | `--mode rlm --search-k 5` |
| B′ | BM25 top-5, harness-supplied | map→reduce | `--mode structured --chunks-source bm25 --chunks-k 5` |
| C | a Claude Code instance, offline | map→reduce | `--mode structured --chunks-source file` |

## 1. Structure alone does not improve accuracy

The clean test, because B′ runs the same 55 rows B did and nothing is selected on:

| subset | A | B | B′ |
|---|---|---|---|
| nq_128k | 0.4182 | **0.5273** | 0.4000 |
| hotpotqa_128k | 0.3818 | 0.4000 | 0.4000 |

Paired, pooled over both subsets (n=110): **B′ vs B is 14-21 (p=0.31)** and
**B′ vs A is 19-19 (p=1.00)**. Per subset, nq is 5-12 against B′ and hotpotqa is
9-9.

So replacing the REPL with a fixed map→reduce over the *same* chunks buys nothing
in accuracy, and on nq it directionally loses. The original hypothesis — that a
structured run over the chunks beats what base RLM does — **is not supported on
accuracy**.

## 2. It is much cheaper, at the same accuracy as base RLM

`average_peak_context_tokens` is the axis `compare.py` puts against KVPress's
retained tokens:

| arm | nq peak ctx | hotpotqa peak ctx | steps | latency |
|---|---|---|---|---|
| A agentic | 2255 | 2170 | 3.9 / 3.1 | 20.3s / 19.3s |
| B agentic | 2562 | 2619 | 5.6 / 5.5 | 5.6s / 6.1s |
| B′ structured | **259** | **387** | 6.0 | 5.1s / 10.1s |
| C structured | **327** | **414** | 2.4 / 3.1 | 7.1s / 10.2s |

The structured arms hold **6-8x less root context** than the agentic ones, and
tie base RLM on accuracy (19-19 pooled). That is the real finding here: the loop
is not buying accuracy over a fixed pipeline, only spending context. Nothing in
this comparison needed a bigger model.

One caveat on the metric pair: on hotpotqa B′ matches B on `subspan_em` (0.400)
while `em` drops (0.127 vs 0.364). The reduce step is verbose, so the gold answer
is present as a substring more often than as the whole string. Read the two
together; `em` understates this arm.

## 3. Better chunks are where the accuracy signal is

The **hard slice** is the 51 rows where arm B retrieved the gold answer string and
answered wrongly anyway (nq 22 of 55, hotpotqa 29 of 55) — the rows where the
evidence was found and then wasted.

| subset | A | B | B′ | C |
|---|---|---|---|---|
| nq (n=22) | 0.1818 | 0.0455 | 0.2273 | **0.4091** |
| hotpotqa (n=29) | 0.4138 | 0.0345 | 0.3448 | **0.4828** |

Paired, pooled (n=51): **C vs B′ is 11-3 (p=0.057)**, C vs A is 13-6 (p=0.167).
Per subset C vs B′ is 5-1 on nq and 6-2 on hotpotqa — the same direction twice,
but underpowered alone.

C achieves this reading **less than half the text**: 1.5 windows per example on nq
and 2.1 on hotpotqa against BM25's 5, i.e. 3000 and 4276 characters against the
same 10000-character cap.

**Three reasons not to read this as established.**

1. The slice is selected on B being wrong. B′ shares B's chunks, so it is
   correlated with that selection and disadvantaged by construction; C is not.
   The honest bar is A, which is unselected — and C vs A is 13-6, p=0.167.
2. n = 51, and nothing here clears p < 0.05.
3. Selecting on "the arm was wrong" is only legitimate because temperature is 0.0
   and vLLM decodes greedily, so those rows reproduce rather than regressing to
   the mean.

## 4. What the selector found that BM25 could not

The selection pass ran offline over the same 2000/400 window grid, with the
question and never the answers. Two things a one-shot lexical ranker cannot do
accounted for most of the wins: trying several phrasings when the obvious keyword
is absent from the corpus, and following a second hop from an entity discovered in
the first.

It also surfaced a corpus property worth recording. `gold_in_retrieved` was True
for every row in the slice, but on several the *answer string* was present while
the *evidence* was not: "Arundel" occurs nowhere in hotpotqa's 270 windows,
"ribosome" nowhere in nq's 298, and hotpotqa:44's second hop has no
India/Gujarat/Lok Sabha article to find. Those examples are unanswerable from the
context, and no selector fixes them. This is the loose-diagnostic caveat in
`evaluation/rlm/README.md` showing up in practice — `gold_in_retrieved_fraction`
is not a score ceiling.

## Reproducing

```bash
# arms B' and C, against a vLLM server on :8000
python -m evaluation.rlm.run_benchmark --dataset loft --data-dir nq_128k --split test --limit 55 \
  --mode structured --chunks-source bm25 --chunks-k 5 --sub-backend http --out $RESULTS
python -m evaluation.rlm.run_benchmark --dataset loft --data-dir nq_128k --split test \
  --ids-file <slice>.ids --mode structured --chunks-source file \
  --chunks-file evaluation/rlm/chunksets/nq_128k.hardslice.jsonl --sub-backend http --out $RESULTS
```

The slice is derived from an existing `--search-k` checkpoint by
`python -m evaluation.rlm.hard_slice`. Chunk sets are committed under
`evaluation/rlm/chunksets/`, each record carrying the windows chosen and a note on
how they were found.
