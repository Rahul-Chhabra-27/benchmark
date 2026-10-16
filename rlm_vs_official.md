# Our RLM (`none`) vs official recursive-llm

This is a comparison of the two LOFT-128k n=20 arms that share a REPL-over-`context` loop and **no chunk ordering**:

| | Official | Ours (`none`) |
|---|---|---|
| Code | `baselines/official_rlm/recursive-llm` (`src/rlm/core.py`) | `evaluation/rlm/rlm.py` |
| Runner | `baselines/official_rlm/run_benchmark.py` | `python -m evaluation.rlm.run_benchmark --mode rlm` |
| Results | `evaluation/results/rlm/recursive-llm/loft128k-n20-think-split/` | `evaluation/results/rlm/loft128k-qwen35-n20-order-grid/` cells **without** `__order-*` |

Both campaigns used the same vLLM servers (Qwen3.5-4B/9B/27B), thinking ON, 30 root turns, test split, first 20 rows.

`--mode original` (`evaluation/rlm/original_rlm.py`) is a closer clone of the paper loop. It was **not** the `none` arm of the order grid.

---

## What is the same

The long document is **not** in the root prompt. It lives as `context` in a persistent Python REPL. The root writes code to look at slices. At the default depth boundary (`max_depth=1` official / a single `llm_query` hop ours) a sub-LM reads only the text the root passes it. Both loops cap the root at 30 turns. Both keep `re` and `context` in the environment.

That is the shared RLM paradigm (Zhang, Kraska & Khattab, 2025). The loops diverge after that.

---

## Protocol: how a turn is supposed to look

**Official.** Each root reply must be *exactly one* of:

1. executable Python (no `FINAL` inside it), or
2. a standalone protocol directive `FINAL("answer")` or `FINAL_VAR(name)` as the **entire** message.

`FINAL` is parsed with `re.fullmatch` in `baselines/official_rlm/recursive-llm/src/rlm/parser.py`. If the model fences the code, writes `FINAL(...)` inside Python, or mixes thought + code, the turn is not an answer. `FINAL` is not a Python function; calling it in the REPL does nothing.

**Ours.** Each root reply must be a fenced ` ```python ``` ` block. The harness executes only the first block. `FINAL(...)` and `FINAL_NONE(...)` are **REPL builtins**. A guessed string literal (`FINAL("42")`) is rejected unless that string already appeared in real REPL output (`seen_output`). Abstention is a first-class ending (`FINAL_NONE`), not a missing `FINAL`.

This is the largest behavioral gap on small Qwen sizes. The official parser is strict and silent; ours is strict in a different way and **nudges** (no code block, ungrounded `FINAL`, premature `FINAL_NONE`) up to a small budget.

---

## Tools in the REPL

**Official** (`_build_repl_env`):

- `context`, `query`, `re`
- `recursive_llm(sub_query, sub_context)` — paper name; at `max_depth=1` this is one plain LM completion
- aliases `rlm_query`, `llm_query` with the same two-arg shape `(sub_query, sub_context="")`
- batched variants

Locate is ordinary Python: `context.find`, `re.findall`, slicing. There is no BM25 `search()`. Observation from a REPL step is truncated at `--max-output-chars` (campaign default **2000**).

**Ours `none`** (grid did not pass `--search-k` or `--chunk-order`, so `search_k=0`):

- `context`, `print`, `FINAL` / `FINAL_NONE`
- `llm_query(prompt)` — **one argument**. The sub-LM sees only that string. The root must embed the slice: `llm_query("…?\n" + context[i:j])`. Each call is truncated at **32 000** characters (`--max-subcall-chars`).
- **No** `search()`, **no** `chunks`, **no** `recursive_llm`

Locate is still `find` / regex; the prompt tells the root to search **distinctive terms**, not the whole question, then hand a region to `llm_query`. The worked example is a 3-turn `.find` → `llm_query` → `FINAL(ans)` session.

`search()` and ordered `chunks` exist only on other arms (`--search-k > 0`, `--chunk-order`). They are not part of `none`.

---

## Prompt

**Official** (`baselines/official_rlm/recursive-llm/src/rlm/prompts.py`): short paper prompt. “You cannot see the context. Write Python. `FINAL` is a protocol directive, not a function.” Examples are `print(context[:500])`, `re.findall`, `context.find`. Depth / `max_depth` are stated explicitly.

**Ours:** a long strategy prompt (`ROOT_SYSTEM_PROMPT` in `evaluation/rlm/rlm.py`). It forbids simulating REPL output, forbids guessing, requires `FINAL` on a computed value, allows `FINAL_NONE` after a real read, and injects the LOFT task into the system message. On `none`, `answer_list_format` is **off** (the bm25 arms of the same grid used `auto` on qampari/quest — a confound, not a `none` vs official difference).

Sub-LM prompt official: leaf “answer the subproblem from supplied context, no REPL/FINAL”. Ours: `SUB_SYSTEM_PROMPT` — “answer using ONLY the text provided, concise and factual.”

---

## REPL execution

| | Official | Ours `none` |
|---|---|---|
| Executor | RestrictedPython in a worker process | in-process `exec` with a timeout |
| Imports | blocked; `re` pre-imported | unrestricted Python (cluster-trust model) |
| REPL timeout | `--repl-timeout` 60 s | `--exec-timeout` 60 s (excludes `llm_query` time) |
| Per-example wall clock | `--max-elapsed-seconds` 900 s | `--run-timeout` 900 s |
| Observation shown to root | 2000 chars | truncated for display; **ungtruncated** copy used for grounding |

Official `MaxIterationsError` / `REPLTimeoutError` / `BudgetExceededError` become checkpoint `error` fields. Ours maps timeouts and loop breakers to `end_reason` and still writes a prediction.

---

## Ending a row

**Official.** A row finishes only when `FINAL` / `FINAL_VAR` parses. After 30 iterations: `MaxIterationsError`, `pred=null`, scorer **drops** the row (`runtime.scored` < 20). Context overflow is the same: error, not a score of 0 inside the average.

**Ours.** After 30 steps the row is still scored (`unfinished` in `runtime`). Other endings: `abstained`, `ungrounded_final`, `unproductive_broken`, `error_loop_broken`, `run_timeout`. Empty or truncated answers count as wrong, but they stay in the **denominator of 20**.

That is why 4B nq is 0.050 official vs 0.750 ours: official 18/20 never emitted a legal `FINAL`; ours scored all 20.

---

## Guards ours has that official does not

Defaults from the order-grid runner (`evaluation.rlm.run_benchmark`):

- `--max-sub-calls 40` — further `llm_query` calls return a notice
- `--min-sub-calls-before-abstain 1` — `FINAL_NONE` without a read is nudged away
- `--max-unproductive-steps 3` — consecutive empty prints → forced grounded answer
- `--max-error-steps 3` — consecutive exceptions → break out
- ungrounded `FINAL` literal rejected (must appear in `seen_output`)
- optional `MemoryBudget` / `Scratchpad` — **off** on this campaign

Official has LiteLLM retries (`--max-retries 2`), a 900 s elapsed budget, and an optional `final_answer_validator`. It does not nudge protocol mistakes and does not force an answer on empty/error loops.

---

## Thinking / client stack

**Official.** LiteLLM against the vLLM OpenAI endpoint. `--enable-thinking` sets `extra_body.chat_template_kwargs.enable_thinking=True`. vLLM often concatenates `<think>…</think>` into `message.content`; `install_vllm_reasoning_split()` peels that **before** the FINAL/REPL parsers.

**Ours.** Direct OpenAI SDK (`evaluation/rlm/client.py`). Campaign did **not** pass `--no-think`. `enable_thinking=False` is only forced when `--no-think` is set; thinking-on is the server/template default rather than an explicit extra_body on this path. There is no equivalent peel in `RLM` (Skylight has `_strip_thinking`; this loop does not).

Same GPUs and `max-model-len` 32768; different clients and parse-time handling of traces.

---

## Query construction

Official `build_query` can prepend the LOFT `answer_prefix` (e.g. `Final Answer: `) unless `--no-answer-prefix`. Ours puts the task in the system prompt (`{task}`) rather than that prefix convention.

---

## What this does **not** compare

- **bm25-graph / bm25-query** — those replace locate with a frozen `chunks` list and change `llm_query` to take slices of chunks. Official recursive-llm has no analogue.
- **`--search-k`** — BM25 `search()` in our REPL; off on `none`.
- **`--mode original`** — paper-style `FINAL` / `recursive_llm` inside our harness; unused here.
- **Skylight** — different agent entirely.

---

## Practical reading of the LOFT numbers

On Qwen3.5-4B/9B the official loop’s headline scores are mostly a **completion** story: the model explores with Python but does not emit a message that is *only* `FINAL(...)`, so the row is dropped. Rescaling official `metrics.json` to a denominator of 20 (`metric × scored / 20`) is required before putting it next to ours.

On 27B the official loop finishes most nq/hotpotqa rows, and `subspan_em` lands in the same band as ours `none`. The remaining gap is then about answers (EM/F1 near zero official because generations are long) and list `coverage` (near zero official vs modest ours), not about whether `FINAL` happened.

Ours `none` still leaves headroom on musique/quest via `unfinished` (hit 30 steps with a scored but truncated trajectory). That is a different failure mode from official `MaxIterationsError`.
