# The arms, in plain words

A cheat sheet for [`STRUCTURED_RESULTS.md`](STRUCTURED_RESULTS.md) and the
**Structured RLM** dashboard tab. Every label is just three choices stuck
together: **who reads**, **who picked the text**, and **which prompt**.

---

## The setup

The document is too big to fit in the model's window. Think of it as a library
the model can't walk into. Something has to pick a few pages and hand them over.

The question the campaign asks: **is it better to let the model wander the
library, or to hand it a fixed stack of pages?**

---

## 1. Who reads — the two pipelines

| | what it is |
|---|---|
| **agentic** | The model gets a Python prompt. It writes code to hunt for text, reads a slice, thinks, hunts again. It decides when to stop. |
| **structured** | No hunting. The harness hands over N windows. One model call reads each one and reports what it found. One final call writes the answer. Done. |

Agentic = a student roaming the stacks.
Structured = an assembly line. Same pages every time, no choices.

---

## 2. Who picked the text — the letters

| arm | who picked the pages | who reads |
|---|---|---|
| **vanilla** | nobody — the *whole* document goes in | one call, no RLM |
| **A** | the model itself, with `str.find` (finds the *first* match only) | agentic |
| **B** | the model itself, with `search()` — real BM25, top 5 | agentic |
| **B′** | the harness, BM25 top 5, **frozen** | structured |
| **C** | a bigger model (Claude Code), offline, ahead of time | structured |

**vanilla is the control.** No chunking, no recursion — the entire 128k document
straight into the model's window. It fits: Qwen3-4B-2507 holds 262k natively and
was served at 139k, so nothing is cut. It also **wins** (0.500 on nq, 0.455 on
hotpotqa). At 128k there is no wall, so the RLM is paying overhead for nothing.
The RLM's real result is the price tag next to it: **~2,000 tokens of context
against vanilla's ~120,000**, for 71–82% of the score.

⚠️ It ran on a **different set of questions** (110, not 55). Treat it as a
reference line drawn across the chart, not a head-to-head.

Each step changes exactly one thing:

- **A → B** gives the model a better way to find pages.
- **B → B′** keeps the pages, swaps the reader for the assembly line.
- **B′ → C** keeps the assembly line, swaps in better-chosen pages.

**Frozen** means the same pages come back no matter what the model asks. B is
*not* frozen — it can rephrase and search again.

---

## 3. Which prompt — rev1 vs rev2

Two changes shipped at once, so both versions are reported.

| | rev1 | rev2 |
|---|---|---|
| how much text per read | one window | one merged passage (neighbours glued together) |
| answer format | *offered* | *required* |

**Why the scores move in opposite directions.** `subspan_em` gives a point if the
right answer appears *anywhere* in the reply. Rambling wins. rev1 said "The Hong
Kong actor, martial artist, and film director is Donnie Yen, ..." and scored.
rev2 said "Donnie Yen" and didn't. So rev2 looks worse on `subspan_em` and much
better on `em` (exact match). rev2 is the more honest number.

---

## 4. The ablations — pulling the bundle apart

rev2 changed two things at once, so nobody could tell which one mattered. Four
runs each undo one thing.

| label | what it undoes | result |
|---|---|---|
| **abl 1** — bare query prompt | strips the long instruction blob out of the reading prompt | **backfired.** The blob was acting as a filter. Without it the reader "finds" something in every passage, and the final answer gets built from junk. |
| **abl 2** — one merged map call | reads all 5 windows in a single call instead of one at a time | small gain, not significant |
| **abl 3** | merging on, format merely offered | see grid |
| **abl 5** | merging off, format required | see grid |

**The grid** (abl 3 and 5 fill the empty corners):

| merging | format | which arm | subspan | em |
|---|---|---|---|---|
| off | offered | B′ rev1 | 0.4364 | 0.2636 |
| off | required | **abl 5** | 0.4000 | 0.3091 |
| on | offered | **abl 3** | 0.4727 | 0.1091 |
| on | required | B′ rev2 | 0.4364 | 0.3000 |

Read it as two separate effects that cancel out:

- **Merging helps** (+0.036 subspan), in both rows.
- **Requiring the format** costs a little subspan, buys a lot of `em`.

That's why rev1 and rev2 both landed on 0.4364 — same total, different routes.
It also **overturns** an earlier guess that merging was the problem.

---

## 5. Run 4 and 4c — does good text help the wanderer?

C's pages beat BM25's pages on the assembly line. So: hand those same pages to
the agentic model. Does it win too?

| label | who reads | which pages |
|---|---|---|
| **4** | agentic | C's pages |
| **4c** | agentic | frozen BM25 pages (*exactly* what B′ reads) |
| **capped** | same, but allowed only **one** search | |

**Why the cap.** Frozen pages come back identical for every query, and the model
can't tell. So it kept rephrasing and searching — **21.4 times per example**
against the control's 1.7 — burning context and sometimes running out of steps
before reading anything. Capping searches at one is the fair version.

**Why 4c has to exist.** Comparing 4 against normal adaptive B would measure the
frozen-ness, not the pages. 4c is frozen too, so both sides pay that cost.

**The answer is no.** Same 51 rows, same pages:

| | structured | agentic |
|---|---|---|
| **subspan** | 0.4706 (C) | 0.2353 (4 capped) |

And inside the agentic pipeline, swapping BM25 pages for C's pages wins nothing
(5–11) — where the *identical* swap on the assembly line was 15–3, p=0.0075.

> **The good pages aren't what helps. The pipeline that reads them is.**

---

## 6. The two row sets

| name | what it is |
|---|---|
| **full 55** | all 55 questions per subset. Nothing cherry-picked. |
| **hard slice (n=51)** | only the questions where arm B *did* pull the right answer into view and *still* got it wrong (22 on nq, 29 on hotpotqa). |

The hard slice separates "couldn't find it" from "found it and fumbled it."

**Its catch:** it's chosen by B being wrong, and B′ reads B's pages — so B′ walks
in with a handicap and C doesn't. That's why arm A is always shown next to it as
the neutral comparison.

`pooled_n110` / `pooled_n51` on the dashboard = those same row sets with both
subsets added together.

---

## 7. How BM25 picks pages, briefly

1. Chop the document into 2000-character windows, overlapping by 400 (so an
   answer sitting on a boundary isn't sliced in half).
2. Score each window against the question:
   - **rare words count more** — "1983" is everywhere and worth little; an odd
     surname is worth a lot
   - **repeats help, with diminishing returns**
   - **long windows don't get an unfair edge**
3. Take the top 5.

It's word matching. No embeddings, no model. Great when the answer sits in its
own passage and the question shares words with it (LOFT nq/hotpotqa: 83–95%
recall@10). Weak on questions phrased nothing like the text, and useless when the
task needs the *whole* document.

---

## The one-line summary

| | |
|---|---|
| Plain vanilla beats every RLM arm | no context wall at 128k — but it reads ~120,000 tokens |
| Structure beats the loop on **cost** | 6–8× less context for the same score |
| Structure doesn't beat it on **accuracy** | ties A, loses to B |
| Better pages help — **but only the assembly line** | 15–3 structured, 5–11 agentic |

Read the caveats in [`STRUCTURED_RESULTS.md`](STRUCTURED_RESULTS.md) before
quoting any of these numbers.
