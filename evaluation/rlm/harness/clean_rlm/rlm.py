# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible vLLM recursive language model."""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

from evaluation.rlm.shared import retrieval
from evaluation.rlm.shared.retrieval import Hit

from .backend import VLLMClient, is_context_overflow
from .grounding import _mentions, composed_answer_grounded
from .prompts import (
    CHARS_PER_TOKEN_ESTIMATE,
    CHUNK_ORDERS,
    CHUNK_READS_BEFORE_ABSTAIN,
    CODE_RE,
    DEFAULT_CHUNK_CALL_TOKENS,
    DEFAULT_REDUCE_FINDING_CHARS,
    FOLD_MARKER,
    FOLD_SCRATCHPAD_TEMPLATE,
    NOTE_TOOL_HELP,
    PREMATURE_ABSTAIN_NUDGE,
    PREMATURE_ABSTAIN_NUDGE_CHUNK_READS,
    PREMATURE_ABSTAIN_NUDGE_CHUNKS,
    PROSE_ANSWER_CELL_ERROR,
    RLM_ANSWER_LIST_NOTE,
    ROOT_SYSTEM_PROMPT,
    SCRATCHPAD_TEMPLATE,
    SEARCH_ABLATABLE,
    STRUCTURED_ANSWER_SHAPES,
    STRUCTURED_MAP_GRANULARITIES,
    STRUCTURED_MAP_QUESTIONS,
    SUB_SYSTEM_PROMPT,
    TEXT_FINAL_NONE_RE,
    TEXT_FINAL_RE,
    TEXT_FINAL_VAR_RE,
    UNGROUNDED_FINAL_CHUNKS,
    UNTERMINATED_CODE_RE,
)
from .repl import ReplMixin
from .spans import _validated_view
from .structured import StructuredMixin
from .types import MemoryBudget, RLMResult, Scratchpad, TokenCounter


class RLM(ReplMixin, StructuredMixin):
    def __init__(
        self,
        root_client: VLLMClient,
        sub_client: VLLMClient | None = None,
        max_steps: int = 12,
        obs_limit: int = 6000,
        max_subcall_chars: int = 32000,
        min_subcall_chars: int = 0,
        budget: Optional[MemoryBudget] = None,
        scratchpad: Optional[Scratchpad] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        cache_subcalls: bool = True,
        exec_timeout: Optional[float] = 60.0,
        run_timeout: Optional[float] = 900.0,
        max_sub_calls: Optional[int] = 40,
        subcall_deadline_check: bool = False,
        min_sub_calls_before_abstain: int = 1,
        max_unproductive_steps: int = 3,
        max_error_steps: int = 3,
        search_k: int = 0,
        search_window_chars: int = retrieval.DEFAULT_CHUNK_CHARS,
        search_overlap_chars: int = retrieval.DEFAULT_OVERLAP,
        max_search_calls: int = 200,
        search_ablate: Sequence[str] = (),
        structured_map_question: str = "task",
        structured_map_granularity: str = "passage",
        structured_answer_shape: str = "require",
        structured_map_chars: int = 0,
        reduce_finding_chars: int = DEFAULT_REDUCE_FINDING_CHARS,
        structured_hops: int = 1,
        structured_answer_list: str = "",
        ordered_token_counter=None,
        structured_group_input_tokens: int = 8192,
        structured_context_tokens: int = 65536,
        chunk_order: Optional[str] = None,
        chunk_call_tokens: int = DEFAULT_CHUNK_CALL_TOKENS,
    ):
        self.ordered_token_counter = ordered_token_counter
        self.structured_group_input_tokens = structured_group_input_tokens
        self.structured_context_tokens = structured_context_tokens
        self.root = root_client
        self.sub = sub_client or root_client
        self.max_steps = max_steps
        self.obs_limit = obs_limit
        self.max_subcall_chars = max_subcall_chars
        # Optional floor on a document slice before it is folded into the HTTP prompt.
        self.min_subcall_chars = min_subcall_chars
        self.budget = budget
        self.scratchpad = scratchpad
        self.cache_subcalls = cache_subcalls
        # Each guard below is independently disabled by passing None (or 0 from the
        # CLI). They are on by default because an unguarded run can consume a SLURM
        # allocation on one pathological example -- but a debugging session that
        # wants to watch a single example run to completion can switch any of them
        # off without affecting the others.
        self.exec_timeout = exec_timeout
        # Wall-clock ceiling for ONE example. exec_timeout deliberately excludes
        # llm_query time (a slow sub-call is legitimate), and VLLMClient retries with
        # backoff -- so without a separate deadline a single pathological example can
        # stall a job for hours. None disables.
        self.run_timeout = run_timeout
        self.max_sub_calls = max_sub_calls
        # Optionally check the run deadline inside llm_query as well as between turns.
        self.subcall_deadline_check = subcall_deadline_check
        # How many llm_query calls an abstention must be backed by. FINAL has to be
        # earned (the grounding guard rejects a literal never seen in REPL output),
        # but FINAL_NONE was free -- and "the answer is absent" is the one claim no
        # output can ground, which made abstaining the cheapest way out of a hard
        # example. On the LOFT-1m smoke, 5 of 6 abstentions came after a SINGLE
        # failed substring search with zero sub-calls: the root had searched for the
        # whole natural-language question as a literal string, found nothing (as it
        # never could), and declared the document silent on the matter. 0 disables.
        self.min_sub_calls_before_abstain = min_sub_calls_before_abstain
        # Consecutive cells printing nothing before the run is broken out of. The
        # repetition breaker only catches IDENTICAL code, which a model rambling in
        # comments slips past while learning nothing (see the unproductive-step
        # breaker in run). 0 disables.
        self.max_unproductive_steps = max_unproductive_steps
        # Consecutive cells raising before the run is broken out of. Neither of the
        # other two breakers sees this: an exception counts as output, and broken
        # code that changes between attempts is never an exact repeat. 0 disables.
        self.max_error_steps = max_error_steps
        # BM25 retrieval over the document. 0 = off, which is every arm run so far,
        # and off means the `search` name is not bound in the REPL at all. Above 0
        # the root gets a real locate primitive instead of `str.find`, which returns
        # only the FIRST literal occurrence: across the 879 finished LOFT-1m
        # transcripts 80.5% of examples never surfaced the gold answer into any REPL
        # output, against 5.2% that saw it and still answered wrong.
        self.search_k = search_k
        self.search_window_chars = search_window_chars
        self.search_overlap_chars = search_overlap_chars
        # Runaway guard in the shape of --max-sub-calls, not a target. Searching is
        # pure Python over a prebuilt index, so this should never bind; it exists so
        # a model that loops on search() cannot spin out the per-example deadline.
        self.max_search_calls = max_search_calls
        # Which parts of the search treatment to withhold, so the bundle can be taken
        # apart. Turning on retrieval changes FOUR things at once -- the tool, the
        # prose that tells the root to use it, the worked example that demonstrates
        # it, and the abstention effort gate -- and a combined result cannot say
        # which did the work. Empty (the default) is the full treatment.
        #   tool     always on when search_k > 0; withhold it with --search-k 0
        #   locate   the FINDING THINGS bullet and the read strategy (prose)
        #   example  the worked session (demonstration)
        #   gate     an abstention must be backed by at least one search
        # `locate` vs `example` is the prose-vs-demonstration split that commit
        # ec2c8a2 already showed matters: the demonstration outvoted the prose.
        unknown = set(search_ablate) - SEARCH_ABLATABLE
        if unknown:
            raise ValueError(
                f"unknown search_ablate components: {sorted(unknown)}; " f"choose from {sorted(SEARCH_ABLATABLE)}"
            )
        self.search_ablate = frozenset(search_ablate)
        # The structured arm's three axes. Validated here rather than trusted from the
        # CLI, because run_structured is also called directly by tests and by anything
        # that imports this class -- and an unknown value would otherwise fall through
        # to the revision-2 default and be reported as the arm that was asked for.
        for name, value, allowed in (
            ("structured_map_question", structured_map_question, STRUCTURED_MAP_QUESTIONS),
            ("structured_map_granularity", structured_map_granularity, STRUCTURED_MAP_GRANULARITIES),
            ("structured_answer_shape", structured_answer_shape, STRUCTURED_ANSWER_SHAPES),
        ):
            if value not in allowed:
                raise ValueError(f"unknown {name} {value!r}; choose from {list(allowed)}")
        self.structured_map_question = structured_map_question
        self.structured_map_granularity = structured_map_granularity
        self.structured_answer_shape = structured_answer_shape
        # The read side of the geometry, separated from the retrieval side. 0 = off,
        # which is every arm run so far: the map call sees the whole passage. Above 0
        # it sees only this many characters of it, so "how wide is a window" and "how
        # much of one does the reader get" stop being the same number.
        if structured_map_chars < 0:
            raise ValueError(f"structured_map_chars must be >= 0, got {structured_map_chars}")
        self.structured_map_chars = structured_map_chars
        if reduce_finding_chars <= 0:
            raise ValueError(f"reduce_finding_chars must be positive, got {reduce_finding_chars}")
        self.reduce_finding_chars = reduce_finding_chars
        # 1 = the shipped arm: one retrieval round, no follow-up. 2 adds a round
        # seeded by what the first round FOUND. Capped at 2 deliberately -- see
        # run_structured; a third level is a documented non-improvement, not an
        # untried idea.
        if structured_hops not in (1, 2):
            raise ValueError(f"structured_hops must be 1 or 2, got {structured_hops}")
        self.structured_hops = structured_hops
        # A worked example of the bracketed list a multi-value benchmark is parsed
        # with, or "" for the single-value default. See STRUCTURED_ANSWER_LIST_NOTE.
        self.structured_answer_list = structured_answer_list
        # The ordered-chunk arms (see CHUNK_ORDERS). search() is refused beside them: the
        # root is handed every chunk already ordered, so a second locate primitive would
        # make the arm a different treatment -- and a query-aware one on the
        # question-free order.
        if chunk_order is not None:
            if chunk_order not in CHUNK_ORDERS:
                raise ValueError(f"unknown chunk_order {chunk_order!r}; choose from {list(CHUNK_ORDERS)}")
            if search_k > 0:
                raise ValueError("chunk_order hands the root every chunk in a fixed order; it has no search()")
            if chunk_call_tokens <= 0:
                raise ValueError(f"chunk_call_tokens must be positive, got {chunk_call_tokens}")
        self.chunk_order = chunk_order
        self.chunk_call_tokens = chunk_call_tokens
        # (document identity, chunk count) -> (base, worst) token costs. Size 1, like
        # retrieval's index cache: a LOFT subset is one document, so it hits every row.
        self._chunk_cost_cache: dict = {}
        self._sub_call_budget: Optional[int] = None
        self._deadline: Optional[float] = None
        self.tok = TokenCounter(token_counter)
        # True only while _exec's SIGALRM code-timeout is armed; lets llm_query
        # pause that watchdog around its (legit, possibly slow) sub-LLM call.
        self._alarm_active = False

    def run(
        self,
        context: str,
        task: str,
        answer_format: str | None = None,
        chunks: Sequence[Hit] | None = None,
        ordered_chunks: Sequence[Hit] | None = None,
    ) -> RLMResult:
        """The agentic arm. `chunks`, when given, replaces what `search` returns with a
        set chosen outside the run -- same REPL, same system prompt, different windows.

        `ordered_chunks` is the ordered-chunk arms' view: EVERY chunk of the document, in
        the order `chunk_order` names, bound in the REPL as `chunks` and described to the
        root (CHUNKS_HELP, CHUNK_ORDER_NOTES). It comes with `chunk_order` or not at all.
        """
        if (ordered_chunks is None) != (self.chunk_order is None):
            raise ValueError("ordered_chunks and chunk_order come together, or neither is given")
        chunk_view = None
        if ordered_chunks is not None:
            if chunks is not None:
                raise ValueError("an ordered-chunk run has no search() for a provided chunk set to arrive through")
            if self.ordered_token_counter is None:
                raise ValueError("ordered chunk reads are priced with the served model's tokenizer; none was given")
            ordered_chunks = _validated_view(context, ordered_chunks)
            chunk_view = (
                len(ordered_chunks),
                max(hit.end - hit.start for hit in ordered_chunks),
                self._chunks_per_call(context, ordered_chunks),
            )
        metrics = self._base_metrics()
        if chunk_view is not None:
            metrics.update(
                chunks_total=chunk_view[0],
                chunks_per_call_advertised=chunk_view[2],
                chunk_payload_calls=0,
                chunk_budget_refusals=0,
                premature_abstentions=0,
                prose_answer_cells=0,
                empty_chunk_reads=0,
                chunk_ranks_read=[],
                reader_input_tokens=[],
                reader_served_prompt_tokens=[],
            )
        cache: dict = {}
        notes: list = []
        self._sub_call_budget = self.max_sub_calls
        self._deadline = time.monotonic() + self.run_timeout if self.run_timeout else None
        env = self._make_env(context, metrics, cache, notes, provided_chunks=chunks, ordered_chunks=ordered_chunks)

        def abstain_nudge() -> str:
            """The rejection of an unearned FINAL_NONE (see _abstention_earned)."""
            if self.chunk_order is None:
                return PREMATURE_ABSTAIN_NUDGE
            if not metrics["sub_calls"]:
                return PREMATURE_ABSTAIN_NUDGE_CHUNKS
            read = len(set(metrics.get("chunk_ranks_read", [])))
            return PREMATURE_ABSTAIN_NUDGE_CHUNK_READS.format(read=read, total=metrics["chunks_total"])

        system_msg = {
            "role": "system",
            "content": ROOT_SYSTEM_PROMPT.format(
                ctx_len=len(context),
                obs_limit=self.obs_limit,
                max_steps=self.max_steps,
                task=task,
                # The same output contract vanilla_answer receives, phrased for the
                # FINAL() protocol, so neither arm is scored on an answer shape the
                # other was never asked for. Without it the RLM answered via
                # str(FINAL(x)) while vanilla was told "answer concisely", and
                # string-matching metrics measured that style difference as accuracy.
                answer_format_note=(
                    f"\nThe value you pass to FINAL(...) must be formatted as: {answer_format}\n"
                    if answer_format
                    else ""
                )
                # The multi-value list shape, on the ordered-chunk arms only: every other
                # agentic run on disk was scored without it, so they keep their prompt.
                + (
                    RLM_ANSWER_LIST_NOTE.format(example=self.structured_answer_list)
                    if self.chunk_order is not None and self.structured_answer_list
                    else ""
                ),
                budget_note=self._budget_note(),
                note_tool=(NOTE_TOOL_HELP if self.scratchpad is not None else ""),
                **self._read_instructions(chunk_view),
            ),
        }
        begin_msg = {"role": "user", "content": "Begin. Write your first code block."}

        full_history: list = []  # all (assistant, user) messages, server-side
        evicted_count = 0  # number of leading pairs already dropped (evicted)
        forced_drop_pairs = 0  # leading pairs the SERVER rejected us for (see root_reply)
        transcript: list = []
        nudges = 0
        last_code_norm: str | None = None  # repetition-breaker state (see below)
        repeat_count = 0
        no_output_count = 0  # unproductive-step state (see below)
        error_count = 0  # repeated-exception state (see below)
        seen_output = ""  # grounding accumulator — NEVER compacted, never truncated

        def _grounded(val: str) -> bool:
            """Did this answer come from something the run actually observed?

            The task side uses word boundaries: plain containment whitelisted almost
            every short answer, since any question mentioning "12" licensed the
            ungrounded answer 12, and every multiple-choice task enumerates its
            labels in the prompt -- so that whole class of tasks had no guard at all.
            An answer composed from several reads is also grounded
            (composed_answer_grounded): the word-for-word test alone discarded answers
            whose every content word had been observed, because the root joined a title
            and an id it read on two calls into the requested answer template.
            """
            if val in seen_output or _mentions(task, val):
                return True
            return composed_answer_grounded(val, seen_output, f"{task}\n{answer_format or ''}")

        def _abstention_earned() -> bool:
            """Has this run read enough of the document to claim the answer is absent?

            Absence is the one claim REPL output cannot ground, so it gets an effort
            test rather than a grounding test: a run that never called llm_query has
            not read the document at all, whatever its searches returned.

            With a search arm the bar also includes having actually run the
            retrieval primitive. Abstaining without ever calling search() is the
            2026 analogue of searching for the whole natural-language question as a
            literal string: it costs a compliant model nothing, and 77% of the 331
            LOFT-1m abstentions were reached with one sub-call or none.
            """
            if self.chunk_order is not None:
                # Reading, here, is passing chunks to llm_query -- and one read of the
                # top of the list is too little to claim absence (CHUNK_READS_BEFORE_ABSTAIN).
                needed = max(self.min_sub_calls_before_abstain, CHUNK_READS_BEFORE_ABSTAIN)
                return metrics.get("chunk_payload_calls", 0) >= needed
            if metrics["sub_calls"] < self.min_sub_calls_before_abstain:
                return False
            if self.search_k > 0 and "gate" not in self.search_ablate and metrics.get("search_calls", 0) < 1:
                return False
            return True

        def build_sent() -> tuple[list, int]:
            """Construct the bounded view actually sent to the root model."""
            nonlocal evicted_count
            base = [system_msg, begin_msg]
            n_pairs = len(full_history) // 2
            pairs = [full_history[i * 2 : i * 2 + 2] for i in range(n_pairs)]
            # Pairs the SERVER made us drop (see root_reply). Applied whether or not
            # a MemoryBudget is configured -- an unbudgeted run still has a ceiling,
            # it just isn't one the operator picked.
            pairs = pairs[forced_drop_pairs:] if forced_drop_pairs else pairs

            if self.budget is None:
                fold_n = n_pairs - len(pairs)
                flat = [m for p in pairs for m in p]
                # Scratchpad notes are re-injected every turn even without a
                # budget — that visibility is the whole point of the tool.
                if self.scratchpad is not None and notes:
                    notes_msg = {
                        "role": "user",
                        "content": (
                            FOLD_SCRATCHPAD_TEMPLATE.format(n=fold_n, notes=self._notes_block(notes))
                            if fold_n
                            else SCRATCHPAD_TEMPLATE.format(notes=self._notes_block(notes))
                        ),
                    }
                    return base + [notes_msg] + flat, fold_n
                if fold_n:
                    return base + [{"role": "user", "content": FOLD_MARKER.format(n=fold_n)}] + flat, fold_n
                return base + flat, 0

            keep = self.budget.keep_recent_turns
            kept = pairs[-keep:] if keep > 0 else []

            def assemble(kept_pairs, fold_n):
                msgs = list(base)
                if self.scratchpad is not None:
                    if fold_n > 0:
                        msgs.append(
                            {
                                "role": "user",
                                "content": FOLD_SCRATCHPAD_TEMPLATE.format(n=fold_n, notes=self._notes_block(notes)),
                            }
                        )
                    elif notes:
                        msgs.append(
                            {
                                "role": "user",
                                "content": SCRATCHPAD_TEMPLATE.format(notes=self._notes_block(notes)),
                            }
                        )
                elif fold_n > 0:
                    msgs.append({"role": "user", "content": FOLD_MARKER.format(n=fold_n)})
                for p in kept_pairs:
                    msgs.extend(p)
                return msgs

            fold_n = n_pairs - len(kept)
            sent = assemble(kept, fold_n)
            # squeeze: drop more recent pairs until within budget
            while self.tok.count_messages(sent) > self.budget.max_context_tokens and kept:
                kept = kept[1:]
                fold_n = n_pairs - len(kept)
                sent = assemble(kept, fold_n)

            # record newly-evicted pairs (dropped for good)
            if fold_n > evicted_count:
                evicted_count = fold_n
                metrics["evictions"] += 1
            return sent, fold_n

        def root_reply() -> str:
            """Ask the root, letting the SERVER decide whether the view fits.

            Without a --max-context-tokens budget the view grows monotonically, and
            LLMClient does not retry 4xx: one overflow ended the example as
            end_reason="exception" -- the mechanism that zeroed the whole RULER-32k
            RLM run. vanilla_answer already treats the server as the authority on
            what fits; do the same here rather than requiring the operator to have
            guessed a budget in advance. Evicting the oldest turns is the same
            remedy the budget path applies, just triggered by the server.
            """
            nonlocal forced_drop_pairs
            for _ in range(12):
                sent, _ = build_sent()
                ctx_tokens = self.tok.count_messages(sent)
                try:
                    reply = self.root.chat(sent)
                except Exception as exc:  # noqa: BLE001 - re-raised unless it is an overflow
                    if not is_context_overflow(exc) or forced_drop_pairs >= len(full_history) // 2:
                        # Nothing left to evict: the system prompt and the task alone
                        # exceed the window, which no amount of dropping can fix.
                        raise
                    forced_drop_pairs += 1
                    metrics["overflow_evictions"] += 1
                    continue
                metrics["root_prompt_tokens"] += ctx_tokens
                # Prefer the server's own count for the cost axis: TokenCounter is
                # tiktoken (a different tokenizer) or len//4 (a guess), and this
                # number is the one compared against KVPress's retained-token metric.
                served = getattr(getattr(self.root, "usage", None), "last_prompt_tokens", None)
                metrics["peak_context_tokens"] = max(metrics["peak_context_tokens"], served or ctx_tokens)
                return reply
            raise RuntimeError("root context overflow could not be resolved by eviction")

        for step in range(1, self.max_steps + 1):
            if self._deadline is not None and time.monotonic() > self._deadline:
                # Out of wall-clock. A distinct end_reason so a timeout stays
                # distinguishable from a genuine abstention when triaging.
                return RLMResult(None, step, False, transcript, "run_timeout", metrics)
            metrics["steps"] = step

            reply = root_reply()
            metrics["root_completion_tokens"] += self.tok.count(reply)
            blocks = CODE_RE.findall(reply)
            if not blocks:
                # A reply cut off at max_tokens has an opening fence and no closing
                # one. Read as "no code block" it burned a nudge, and after three
                # such replies the raw truncated text was returned as the answer.
                m_trunc = UNTERMINATED_CODE_RE.search(reply)
                if m_trunc and m_trunc.group(1).strip():
                    blocks = [m_trunc.group(1)]
                    metrics["truncated_code_blocks"] = metrics.get("truncated_code_blocks", 0) + 1

            # adaptive observation limit: never let one observation exceed the budget
            obs_limit = self.obs_limit
            if self.budget is not None:
                obs_limit = max(512, min(self.obs_limit, self.budget.max_context_tokens * CHARS_PER_TOKEN_ESTIMATE))

            # --- No code block in the reply ---
            if not blocks:
                mn = TEXT_FINAL_NONE_RE.search(reply)
                if mn:
                    # An abstention needs no grounding: "I found nothing" is exactly
                    # the claim that cannot appear in the output it is about. It does
                    # need EFFORT, though -- see _abstention_earned.
                    if not _abstention_earned() and nudges < 2:
                        nudges += 1
                        metrics["premature_abstentions"] = metrics.get("premature_abstentions", 0) + 1
                        transcript.append(
                            {
                                "step": step,
                                "reply": reply,
                                "code": None,
                                "observation": "[premature FINAL_NONE rejected]",
                            }
                        )
                        full_history.append({"role": "assistant", "content": reply})
                        full_history.append({"role": "user", "content": abstain_nudge()})
                        continue
                    transcript.append(
                        {
                            "step": step,
                            "reply": reply,
                            "code": None,
                            "observation": "[FINAL_NONE parsed from prose]",
                        }
                    )
                    metrics["abstain_reason"] = (mn.group(1) or "").strip()
                    return RLMResult(None, step, True, transcript, "abstained", metrics)
                m = TEXT_FINAL_RE.search(reply)
                if m:
                    val = m.group(1).strip()
                    if val and _grounded(val):
                        transcript.append(
                            {
                                "step": step,
                                "reply": reply,
                                "code": None,
                                "observation": "[FINAL parsed from prose]",
                            }
                        )
                        return RLMResult(val, step, True, transcript, "final_in_prose", metrics)
                    nudges += 1
                    transcript.append(
                        {
                            "step": step,
                            "reply": reply,
                            "code": None,
                            "observation": "[ungrounded prose FINAL rejected]",
                        }
                    )
                    if nudges > 2:
                        return RLMResult(None, step, False, transcript, "ungrounded_final", metrics)
                    full_history.append({"role": "assistant", "content": reply})
                    full_history.append(
                        {
                            "role": "user",
                            "content": (
                                UNGROUNDED_FINAL_CHUNKS.format(val=val)
                                if self.chunk_order is not None
                                else f"REJECTED: your answer {val!r} never appeared in any actual REPL "
                                "output, so it looks like a guess. Do NOT invent answers. Write a "
                                "```python code block that finds the answer in `context` (string "
                                "search / regex / llm_query with the snippet pasted in), look at "
                                "the real output, and only then FINAL it. If you have genuinely "
                                "searched and the document does not contain the answer, say so with "
                                'FINAL_NONE("reason") instead.'
                            ),
                        }
                    )
                    continue
                mv = TEXT_FINAL_VAR_RE.search(reply)
                if mv and mv.group(1) in env:
                    # Only when the variable actually exists: naming a variable that
                    # was never assigned used to answer the literal string
                    # "<missing var x>" and record it as a finished, successful
                    # prediction. A miss falls through to the no-code-block nudge.
                    val = str(env[mv.group(1)])
                    transcript.append(
                        {
                            "step": step,
                            "reply": reply,
                            "code": None,
                            "observation": "[FINAL_VAR parsed from prose]",
                        }
                    )
                    return RLMResult(val, step, True, transcript, "final_var_in_prose", metrics)
                nudges += 1
                transcript.append(
                    {
                        "step": step,
                        "reply": reply,
                        "code": None,
                        "observation": "[no code block - nudged]",
                    }
                )
                if nudges > 2:
                    return RLMResult(
                        reply.strip(),
                        step,
                        False,
                        transcript,
                        "gave_up_no_code",
                        metrics,
                    )
                full_history.append({"role": "assistant", "content": reply})
                full_history.append(
                    {
                        "role": "user",
                        "content": "Your reply contained no code block, so NOTHING was executed and no "
                        "answer was recorded. Reply with exactly one ```python code block. "
                        "If you already know the answer from a previous REPL output, reply with "
                        'a code block containing only: FINAL("your answer")',
                    }
                )
                continue

            # --- Execute ONLY the first block ---
            code = blocks[0]
            obs, obs_full = self._exec(code, env, obs_limit)
            if obs == PROSE_ANSWER_CELL_ERROR:
                metrics["prose_answer_cells"] = metrics.get("prose_answer_cells", 0) + 1
            if len(blocks) > 1:
                obs += (
                    "\n[WARNING: you wrote multiple code blocks; ONLY the FIRST was "
                    "executed. Anything you wrote after it (including any 'output' you "
                    "predicted) did NOT happen.)"
                )
            transcript.append({"step": step, "reply": reply, "code": code, "observation": obs})
            # The UNtruncated output: what the model was shown is a display concern,
            # what actually printed is what the answer has to be grounded in.
            seen_output += "\n" + obs_full

            if env["_final_box"]["done"]:
                val = env["_final_box"]["value"]
                if env["_final_box"].get("abstained"):
                    if not _abstention_earned() and nudges < 2:
                        nudges += 1
                        metrics["premature_abstentions"] = metrics.get("premature_abstentions", 0) + 1
                        env["_final_box"]["done"] = False
                        env["_final_box"]["abstained"] = False
                        env["_final_box"]["reason"] = ""
                        full_history.append({"role": "assistant", "content": reply})
                        full_history.append({"role": "user", "content": abstain_nudge()})
                        continue
                    metrics["abstain_reason"] = env["_final_box"].get("reason", "")
                    return RLMResult(None, step, True, transcript, "abstained", metrics)
                if val in (env.get("_rlm_final_literals") or []) and not _grounded(val):
                    env["_final_box"]["done"] = False
                    env["_final_box"]["value"] = None
                    nudges += 1
                    if nudges > 2:
                        return RLMResult(None, step, False, transcript, "ungrounded_final", metrics)
                    full_history.append({"role": "assistant", "content": reply})
                    full_history.append(
                        {
                            "role": "user",
                            "content": (
                                UNGROUNDED_FINAL_CHUNKS.format(val=val)
                                if self.chunk_order is not None
                                else f"REJECTED: FINAL({val!r}) is a literal constant that never appeared "
                                "in any REPL output — it looks like a guess. Find the real answer in "
                                "`context` first (e.g. re.search / context.find / llm_query with the "
                                "snippet pasted in), print it, then call FINAL on the variable "
                                "holding it. If you have genuinely searched and the document does not "
                                'contain the answer, say so with FINAL_NONE("reason") instead.'
                            ),
                        }
                    )
                    continue
                return RLMResult(val, step, True, transcript, "final_called", metrics)

            # --- Repetition breaker ---
            # A stuck small model re-emits the SAME code cell every turn — identical
            # code -> identical observation — and burns all max_steps (and the whole
            # run_timeout) without ever calling FINAL. Detect an exact consecutive
            # repeat and escalate: first a hard nudge to stop and answer or change
            # tactic, then a forced GROUNDED fallback that answers from what was
            # actually seen, so the example still yields a real answer instead of
            # grinding to max_steps with score 0.
            code_norm = " ".join(code.split())
            if code_norm == last_code_norm:
                repeat_count += 1
            else:
                repeat_count = 1
                last_code_norm = code_norm

            # --- Unproductive-step breaker ---
            # The repetition breaker keys on IDENTICAL code, which a model that
            # rambles in comments defeats without ever learning anything: on the
            # LOFT-1m smoke one example spent 11 of 13 steps emitting comment-only
            # cells ("the correct dwarf is X -- no, actually Y -- no...") that printed
            # nothing, each textually different, until run_timeout killed it 32
            # minutes in. A cell that produces no output teaches the model nothing,
            # so N of them in a row is the same dead end by another route.
            # startswith, not ==: _exec appends note markers and the multi-block
            # warning to the observation, and a cell that printed nothing is still
            # unproductive with one of those stuck on the end.
            if self.max_unproductive_steps and obs.strip().startswith("[no output]"):
                no_output_count += 1
            else:
                no_output_count = 0

            # --- Repeated-exception breaker ---
            # The third route to the same dead end. An exception IS output, so the
            # no-output counter resets on every one; and a model retrying slightly
            # different broken code never repeats a cell exactly, so the repetition
            # breaker does not fire either. On the LOFT-1m smoke one example threw on
            # 34 of its 38 steps -- SyntaxError, then NameError on a mis-cased FINAL --
            # and was only caught at step 37, by which point it had spent 73% of the
            # entire run's tokens.
            if self.max_error_steps and "[EXCEPTION]" in obs:
                error_count += 1
            else:
                error_count = 0

            unproductive = self.max_unproductive_steps and no_output_count >= self.max_unproductive_steps
            erroring = self.max_error_steps and error_count >= self.max_error_steps
            if repeat_count >= 3 or unproductive or erroring:
                # Still looping after the nudge: answer from the real REPL output
                # accumulated so far (grounded), plus the task. Bypasses FINAL()'s
                # literal-grounding guard deliberately — this is the escape hatch.
                material = seen_output.strip()[-self.max_subcall_chars :]
                fb_prompt = (
                    "Answer the task using ONLY the material below, which was extracted "
                    "from a long document by prior code. Be concise and factual.\n\n"
                    f"Task: {task}\n\nExtracted material:\n{material}"
                )
                ans = self.sub.chat(
                    [
                        {"role": "system", "content": SUB_SYSTEM_PROMPT},
                        {"role": "user", "content": fb_prompt},
                    ]
                )
                fb_tokens = self.tok.count(fb_prompt)
                metrics["sub_calls"] += 1
                metrics["sub_call_tokens"] += fb_tokens + self.tok.count(ans)
                if erroring:
                    why = f"{error_count} consecutive cells that raised"
                elif unproductive:
                    why = f"{no_output_count} consecutive cells that printed nothing"
                else:
                    why = "3 identical code cells"
                transcript.append(
                    {
                        "step": step,
                        "reply": "[repetition-breaker fallback]",
                        "code": None,
                        "observation": f"[forced grounded answer after {why}]",
                        # The answer was extracted BY the harness, not chosen by the
                        # model, and it skipped the grounding guard. Triage must be
                        # able to tell these apart from answers the model committed to.
                        "forced": True,
                    }
                )
                # Distinct end_reasons: all three are forced answers, but "went quiet",
                # "kept crashing" and "got stuck on one cell" call for different fixes,
                # and collapsing them would hide whichever dominates a campaign.
                if erroring:
                    end_reason = "error_loop_broken"
                elif unproductive:
                    end_reason = "unproductive_broken"
                else:
                    end_reason = "repetition_broken"
                return RLMResult(ans, step, True, transcript, end_reason, metrics)

            full_history.append({"role": "assistant", "content": reply})
            if self.max_error_steps and error_count == self.max_error_steps - 1:
                full_history.append(
                    {
                        "role": "user",
                        "content": f"STOP: your last {error_count} code cells all raised an exception, so "
                        "none of them ran. Read the traceback in the output above and fix the "
                        "actual error before doing anything else. Write the SIMPLEST possible "
                        "cell that works — one statement, e.g. "
                        "print(context.lower().count('term')) — and build up from there once it "
                        "runs. The terminal calls are FINAL(value), FINAL_NONE('reason') and "
                        "FINAL_VAR('name'); case variants are accepted, but nothing else is.",
                    }
                )
            elif self.max_unproductive_steps and no_output_count == self.max_unproductive_steps - 1:
                full_history.append(
                    {
                        "role": "user",
                        "content": f"STOP: your last {no_output_count} code cells printed NOTHING, so you "
                        "have learned nothing from them. Commented-out reasoning and recalling "
                        "facts from memory do not count — you cannot answer from memory here, "
                        "only from text you have actually SEEN in a REPL output. Your next cell "
                        "MUST call print() on something real: print(len(context)), "
                        "print(context[:2000]) to see the format, or "
                        "idx = context.lower().find(term.lower()); print(context[idx-200:idx+500]) "
                        "for a distinctive term. If a term is genuinely absent, print that you "
                        "checked it and try a DIFFERENT term.",
                    }
                )
            elif repeat_count == 2:
                full_history.append(
                    {
                        "role": "user",
                        "content": "STOP: you just ran this EXACT same code twice and got the same "
                        "result — repeating it will not help. Do ONE of these, and do NOT "
                        "repeat the previous code:\n"
                        "1) If you already have enough to answer, reply with a code block "
                        "containing only FINAL(<expr>) (a variable/expression you computed, "
                        "not a guessed literal).\n"
                        "2) If your search found nothing, your approach is WRONG — the answer "
                        "may be a category/label, or phrased differently. Try a COMPLETELY "
                        "different tactic: inspect context[:2000], search a different keyword, "
                        "or llm_query a relevant snippet.",
                    }
                )
            else:
                full_history.append(
                    {
                        "role": "user",
                        "content": f"ACTUAL REPL output:\n```\n{obs}\n```\n"
                        f"Continue. ({self.max_steps - step} turns left) "
                        "Remember: one code block only; finish with FINAL(...) once you have "
                        "seen the answer in a real output.",
                    }
                )

        return RLMResult(None, self.max_steps, False, transcript, "max_steps", metrics)
