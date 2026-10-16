# hard_questions — what does an RLM need beyond ordering?

Ordering gave the biggest jump. This finds the questions that are still wrong
**after** ordering, so the next ideas (compression, offloading, falling back to
vanilla) can be developed and checked quickly on a small set:

| category | meaning |
|---|---|
| `vanilla_only` | vanilla (whole document in the window) solves it; no RLM arm does |
| `compression_rescues` | RLM + ordering fails; an RLM with compressed (wider) reads solves it |
| `both_solve` | vanilla and RLM + ordering both solve it |
| `rlm_only` | RLM + ordering solves it; vanilla does not |
| `nobody` | nothing solves it |

**`validation_set` = `vanilla_only` + `compression_rescues`**: questions an RLM can
only get right by doing something beyond ordering. *Solved* means the official score
(F1; coverage for qampari/quest) is at least `--solved` (default 0.5).

## 1. Run the arms on the same questions

Any finished runs work; the tool labels each from its `config.yaml`. For LOFT-128k,
55 test rows per task:

```bash
# vanilla: the whole document in the window. Raise the char limit -- the default (400,000)
# cuts LOFT-128k documents (~430,000 chars), and the server needs a window that holds them.
for t in nq hotpotqa musique qampari quest; do
  python -m evaluation.rlm.run_benchmark --dataset loft --data-dir ${t}_128k --split test --limit 55 \
    --mode vanilla --vanilla-char-limit 600000 --out $RESULTS/vanilla
done

# RLM + bm25-graph ordering, uncompressed (F = 1) and compressed (F = 2, 4), at B = 1024 MB
LIMIT=55 SUBSETS="nq hotpotqa musique qampari quest" BUDGETS_MB=1024 FACTORS="1 2 4" \
  ARMS=bm25-graph RESULTS=$RESULTS/bxf bash scripts/rlm/run_bxf.sh
```

`minimal_rlm` runs count too (`order_graph.yaml` is an ordered arm, `graph_x2.yaml` a
compressed one).

## 2. Find the hard questions

```bash
python -m evaluation.rlm.hard_questions --results $RESULTS --out hard_questions_loft128k
```

It prints every arm it found and its kind, then a table of questions per task and
category, and the mean score of every arm on the compared questions. It writes:

| file | contents |
|---|---|
| `hard_questions.csv` | one row per question: every arm's score, `best_*` per kind, the category |
| `summary.md` | the two tables above |
| `ids/<task>__validation_set.txt` | the validation set of that task (also `__vanilla_only`, `__compression_rescues`) |

Only questions answered by a vanilla arm **and** an uncompressed RLM arm are compared.
"Uncompressed RLM" means the ordered arms when there are any, else plain RLM.

## 3. Use the validation set

```bash
python -m evaluation.rlm.minimal_rlm run evaluation/rlm/minimal_rlm/configs/graph_x2.yaml \
  --set data.ids_file=hard_questions_loft128k/ids/hotpotqa_128k__validation_set.txt --set name=hard_graph_x2
# or with the older harness
python -m evaluation.rlm.run_benchmark ... --ids-file hard_questions_loft128k/ids/hotpotqa_128k__validation_set.txt
```

A method that is worth keeping should raise the score on this set without lowering it
on `both_solve`.
