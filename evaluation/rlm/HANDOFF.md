# Handoff — chunk-geometry campaign, 2026-09-09

For whoever picks this up next. Everything here is measured; where something is a
guess it says so.

**Read first:** [`../../STRUCTURED_CHUNKS_CHANGES.md`](../../STRUCTURED_CHUNKS_CHANGES.md)
(what changed and what to adopt), [`ARMS_EXPLAINED.md`](ARMS_EXPLAINED.md) (the arms in
plain words), [`STRUCTURED_RESULTS.md`](STRUCTURED_RESULTS.md) (the *previous*
campaign, whose §1 this one partly undermines — see Trap 1).

---

## State right now

**All GPUs are stopped, deliberately** — the user asked. Nothing is running on fox:
no tmux sessions, no vLLM, ports 8000 and 8001 free. Do not restart anything without
being asked.

**Constraint the user set: use fox GPU 4 only.** `evaluation/rlm/serve_root.sh`
defaults to it and refuses a card someone else is on.

| where | what |
|---|---|
| branch | `feature/structured-chunks` @ `d8c6c64` |
| remotes | `fork` = ikindacodeabit · `origin` = Rahul-Chhabra-27 · `skylight` = skylight-org/intel-longcontext-pc · `nas` = fox:/mnt/nas/gautammahale/retrieval_checkout |
| pushed to | skylight `gautam-dev` (FF) and `feature/structured-chunks`; skylight `master` untouched, 36 behind |
| compute checkout | fox `/mnt/nas/gautammahale/structured_checkout` (its only git remote is the local path `retrieval_checkout`; the Mac pushes to `nas`, fox fast-forwards from it) |
| results | fox `/mnt/nas/gautammahale/structured_grid` — 117 cells, 6,435 example-runs, **0 errors** |
| chunk sets | fox `/mnt/nas/gautammahale/chunksets/*.oracle.w2000o400k5.jsonl` (4 subsets) |
| venv | fox `/mnt/nas/gautammahale/venvs/rlm/bin/python` (no pytest; tests run on the Mac's `.venv-mac`) |

**NAS is 99% full (776 GB).** Check `df -h /mnt/nas` before a campaign.

---

## What is complete and what is not

All 55 test rows per subset (`--split test --limit 55`), identical rows every cell, so
comparisons are paired.

| subset | geometry (task query) | geometry (**bare** query) | oracle | hops2 | exposure |
|---|---|---|---|---|---|
| nq | 12/12 | **12/12** | 1 | 1 | 6/6 |
| musique | 12/12 | **12/12** | 1 | 1 | 6/6 |
| hotpotqa | 12/12 | 8/12 | 1 | 1 | 6/6 |
| qampari | 12/12 | 5/12 | 1 | 1 | 6/6 |

Missing: hotpotqa `(4000,10)` + all `w8000`; qampari `(2000,10)`, all `w4000`, all
`w8000`. **Oracle / hops / exposure all ran with the OLD (task) query.**

---

## The results, compressed

`subspan_em`, except qampari which is `coverage`.

**The headline is not chunk size — it is what BM25 was asked.** The structured arm was
handing the retriever LOFT's whole task string (a page of "print the TITLE and ID…"),
so windows were ranked on boilerplate present in every one of them. Recall of the gold
at depth 2 on nq: **0.00 → 0.95** once the wrapper is stripped.

| | published w2000 k5 | best task-query cell | best **bare**-query cell | oracle ceiling |
|---|---|---|---|---|
| nq | 0.545 | 0.709 (w8000 k10) | **0.800** (w2000 **k2**) | 0.800 |
| hotpotqa | 0.345 | 0.509 (w4000 k10) | 0.545 (w4000 k5) | 0.491 |
| musique | 0.109 | 0.218 (w4000 k10) | 0.182 | 0.218 |
| qampari | 0.079 | 0.368 (w8000 k10) | 0.539 (w1000 k10) | 0.588 |

Five things worth carrying:

1. **The best cell in the campaign reads LESS text than the published arm** — nq
   w2000 **k=2**, 4,000 chars, at **310 peak root tokens** (vanilla: ~119,000 at
   0.500). Grid 1's "bigger and deeper is better" was retrieval compensating for its
   own bad ranking; with a clean query the k-curve flattens or turns down.
2. **nq now sits exactly on its oracle ceiling** — retrieval there is solved and the
   reader is the only thing left.
3. **hops=2 beats even the oracle on hotpotqa** (0.582 vs 0.491).
4. **musique responds to nothing.** Retrieval doubles (gold reaching the model
   0.29→0.58), score moves +3 examples = noise. Oracle only 0.218. Not a retrieval
   problem, not a chunking problem.
5. **Exposure is monotonic to the largest value tested** (hotpotqa 0.000 at 400 chars
   → 0.382 at 8000; findings/example 0.02 → 1.33). The window effect is a real
   read-budget effect, not a BM25-ranking artifact.

---

## Traps that cost time today

1. **The published §1 comparison is confounded.** Arm B (agentic) lets the MODEL write
   its search query; arm B′ (structured) got the boilerplate. "Structure loses to the
   BM25 agentic arm" partly measured which arm could ask a sensible question. **Re-run
   before quoting it.**
2. **A wedged vLLM still answers `/v1/models`.** fox's 5-day-old server accepted chat
   requests, logged "Added request", completed none, and showed 1 MiB on its GPU. The
   campaign looked like it was running for 45 minutes. **Health-check by GENERATING.**
   `structured_grid.sh` and `serve_root.sh` both do now.
3. **Key any analysis on the axes that SEPARATE arms**, not only the swept ones. I made
   this mistake three times: `chunks_source`, `chunks_query` and `hops` all sit at
   identical (window, k) coordinates, and the grid silently rendered whichever it read
   last — nq 2000/5 showed 0.800 (the oracle) where the control is 0.545.
4. **±1 example reproducibility.** The same nq config re-ran 0.545 against a published
   0.527 — vLLM batching numerics under 4 concurrent lanes at temperature 0.0. **One
   example = 0.018, so deltas under ~0.02 are noise.**
5. **Never reuse a chunk file across window sizes.** `oracle_chunks.py` writes `spans`
   so the grid check fires; the two hand-made `*.hardslice.jsonl` files carry none, and
   stale window ids resolve to real-but-wrong text.
6. **`/mnt/nas` NFS stalled a vLLM weight load for ~13 minutes.** fox has a local
   7.3 T disk at `/mnt/fox/data` (3.7 T free) — stage weights there. The
   `infolab-fox-host` memory's "no scratch partition" line is wrong and now corrected.
7. **The public dashboard is NOT tracking git.** See below.

---

## What I would do next, in order

1. **Finish the corrected-query grid** — 11 cells, ~35 min:
   ```bash
   GPU=4 evaluation/rlm/serve_root.sh          # then wait for a REAL generation
   evaluation/rlm/structured_grid.sh qbare hotpotqa qampari
   ```
2. **Re-run oracle / hops / exposure with `--chunks-query bare`.** They all ran on the
   broken query, so hotpotqa's +0.237 from hop 2 may partly be hop 2 rescuing a bad
   first search. This is the biggest open question in the campaign.
3. **Re-run the §1 structured-vs-agentic comparison with the fixed query** (Trap 1).
4. **Bigger reader.** nq is now reader-bound (sitting on its oracle). Qwen3-8B is
   ~16 GB, 14B ~28 GB — both fit the same 48 GB card. This is the experiment the
   oracle gaps point at.
5. **The exhaustive arm the user proposed:** let `--chunks-k` mean "every window",
   sweeping the whole document instead of retrieving. ~68 chunks/example at 8000 chars,
   ~0.47 s/map call ⇒ ~30 min for a 55-row subset. **Prediction: exhaustive ≤ oracle**,
   since the oracle already gives perfect recall; the gap measures pure distractor cost
   at k≈68, which the k≤10 sweep could not reach.
6. Consider n=100 (the full LOFT test split) for the handful of cells worth publishing
   — ~1.8× runtime, halves the error bar. musique needs it; everything there is inside
   the noise.

---

## Reporting rules in force

- **Show loft128k only** (standing user preference).
- **qampari and quest are scored on `coverage`, never `subspan_em`** — the multi-value
  subspan metric demands every gold be matched and is pinned near zero.
- vanilla is n=110 and these are n=55: a reference line, never a paired comparison.

---

## Dashboard — broken, and not by this work

`Rahul-Chhabra-27/benchmark-dashboard` main has the new tabs (`694b04d`,
`structured-geometry-*`). **The live Netlify site is not serving them, and has not
tracked `main` since ~2026-09-04** — it shows 18 sources where the repo has 26, and is
missing the 2026-09-08 `structured-rlm` tab too. The live HTML is *larger* (1.41 MB)
than any repo commit (~1.25 MB) with FEWER sources, so it is built from a different
template or source, not a stale commit. One branch, no `netlify.toml`. Someone with
Netlify access on Rahul's account has to look; it is not fixable from the repo.

Known label bug to fix when it does deploy: the chip reading `published · w2000 k5 ·
task query` is a **re-run** of that configuration, not the published figures
(nq 0.545 here vs 0.527 published). Rename to `baseline · …`.

---

## Cost model, measured

80 cells / 4,400 examples / **77 minutes** on ONE RTX A6000, four lanes against one
vLLM at 3.28× effective concurrency. ~1.05 s/example; slowest cell 12.1 s/example
(qampari w8000 k10). Real memory floor ~14–16 GB (8 GB weights + ~6 GB KV);
`--gpu-memory-utilization` pre-allocates and is not a requirement. **More lanes are
close to free; more GPUs mostly are not.**
