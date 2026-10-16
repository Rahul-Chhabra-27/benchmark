# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent REPL, retrieval tools, and context-budget plumbing."""

from __future__ import annotations

import ast
import contextlib
import io
import re
import signal
import threading
import time
import traceback
from typing import Optional, Sequence

from .. import retrieval
from ..organization import OrderedPayload, reader_messages
from ..retrieval import Hit
from .grounding import _static_string
from .prompts import (
    ALARM_REARM_FLOOR_S,
    CHUNK_ORDER_NOTES,
    CHUNK_QUESTION_ALLOWANCE,
    CHUNK_WORKED_SESSIONS,
    CHUNKS_HELP,
    EMPTY_CHUNK_READ_ERROR,
    EXAMPLE_LLM_QUERY_DENSE,
    EXAMPLE_LLM_QUERY_SEARCH,
    FINAL_NAMES,
    FINAL_NONE_NAMES,
    FINAL_VAR_NAMES,
    LLM_QUERY_HELP_DENSE,
    LLM_QUERY_HELP_SEARCH,
    ONELINE_FIX_RE,
    PROSE_ANSWER_CELL_ERROR,
    PROSE_ANSWER_LINE_RE,
    READ_STRATEGY_DEFAULT,
    READ_STRATEGY_SEARCH,
    SEARCH_TERMS_BM25,
    SEARCH_TERMS_CHUNKS,
    SEARCH_TERMS_FIND,
    STRUCTURED_MAP_QUESTION,
    STRUCTURED_NOTHING,
    SUB_SYSTEM_PROMPT,
    WORKED_SESSION_FIND,
    WORKED_SESSION_SEARCH,
    _ExecTimeout,
    _raise_exec_timeout,
    bare_query,
)
from .spans import HIT_SEPARATOR, _clip_spans, _merge_spans, _widen_spans


class ReplMixin:
    # ---------------- REPL plumbing ----------------
    def _cap_subcall(self, question: str, chunk: Optional[str]) -> tuple[str, Optional[str], bool]:
        """Fit (question, slice) into ONE max_subcall_chars budget.

        Capping each side independently let a split call carry ~2x the size the
        prompt advertises, which is what the sub model's KV fit check then rejects
        -- with a string the root is free to ignore. The question is an instruction
        rather than the payload, so it gets a quarter of the budget and the slice
        takes the rest; on the dense path (no slice) the question gets all of it.
        """
        budget = self.max_subcall_chars
        truncated = False
        if chunk is None:
            if len(question) > budget:
                question, truncated = question[:budget], True
            return question, None, truncated
        question_cap = max(1, budget // 4)
        if len(question) > question_cap:
            question, truncated = question[:question_cap], True
        room = max(0, budget - len(question))
        if len(chunk) > room:
            chunk, truncated = chunk[:room], True
        return question, chunk, truncated

    def _read_instructions(self, chunk_view: Optional[tuple[int, int, int]] = None) -> dict:
        """Return prompt slots for the single OpenAI-compatible HTTP path."""
        if self.chunk_order is not None:
            n_chunks, chunk_chars, per_call = chunk_view or (0, 0, 0)
            return {
                "llm_query_help": CHUNKS_HELP.format(
                    n_chunks=n_chunks,
                    chunk_chars=chunk_chars,
                    per_call=per_call,
                    call_tokens=self.chunk_call_tokens,
                ),
                "read_strategy": CHUNK_ORDER_NOTES[self.chunk_order],
                "search_terms": SEARCH_TERMS_CHUNKS,
                "worked_session": CHUNK_WORKED_SESSIONS[self.chunk_order],
            }
        if self.search_k > 0:
            teach_locate = "locate" not in self.search_ablate
            teach_example = "example" not in self.search_ablate
            return {
                "llm_query_help": LLM_QUERY_HELP_SEARCH.format(k=self.search_k),
                "example_llm_query": EXAMPLE_LLM_QUERY_SEARCH if teach_example else EXAMPLE_LLM_QUERY_DENSE,
                "read_strategy": (
                    READ_STRATEGY_SEARCH.format(k=self.search_k) if teach_locate else READ_STRATEGY_DEFAULT
                ),
                "search_terms": SEARCH_TERMS_BM25.format(k=self.search_k) if teach_locate else SEARCH_TERMS_FIND,
                "worked_session": (
                    WORKED_SESSION_SEARCH
                    if teach_example
                    else WORKED_SESSION_FIND.format(example_llm_query=EXAMPLE_LLM_QUERY_DENSE)
                ),
            }
        return {
            "search_terms": SEARCH_TERMS_FIND,
            "worked_session": WORKED_SESSION_FIND.format(example_llm_query=EXAMPLE_LLM_QUERY_DENSE),
            "llm_query_help": LLM_QUERY_HELP_DENSE.format(max_chars=self.max_subcall_chars),
            "example_llm_query": EXAMPLE_LLM_QUERY_DENSE,
            "read_strategy": READ_STRATEGY_DEFAULT,
        }

    def _expand_subcall(self, chunk: str, document: str) -> tuple[str, bool]:
        """Widen an undersized slice back out to ``min_subcall_chars``.

        The root hands `llm_query` a STRING, not the indices it sliced with, so the
        window is recovered by locating that string in the source document and
        growing it symmetrically. A payload the root built rather than sliced
        (concatenated fragments, an f-string) will not be found verbatim and is
        left exactly as it was -- widening something whose position is unknown
        would be inventing context, not restoring it.

        Symmetric growth is re-anchored at both edges so a hit near the start or
        the end of the document still yields a full-width window rather than a
        truncated one.
        """
        floor = self.min_subcall_chars
        if floor <= 0 or not chunk or len(chunk) >= floor:
            return chunk, False
        want = min(floor, len(document))
        if want <= len(chunk):
            return chunk, False
        start = document.find(chunk)
        if start < 0:
            return chunk, False
        # Centre the original slice inside the wider window, then clamp. The second
        # max() re-anchors when the right edge hit the end of the document, so the
        # window keeps its full width instead of shrinking.
        grow = want - len(chunk)
        new_start = max(0, start - grow // 2)
        new_end = min(len(document), new_start + want)
        new_start = max(0, new_end - want)
        return document[new_start:new_end], True

    def _chunks_per_call(self, document: str, view: Sequence[Hit]) -> int:
        """How many chunks one llm_query carries, stated conservatively in the prompt.

        Priced the way the real request is -- the same renderer and tokenizer -- on the
        view's most expensive chunk and a long question (CHUNK_QUESTION_ALLOWANCE), so a
        call of this many chunks with an ordinary question is not refused. Every call is
        still checked exactly; this is only the number the root is told.
        """
        key = (hash(document), len(document), len(view))
        if key not in self._chunk_cost_cache:
            counter = self.ordered_token_counter

            def cost(hits: tuple) -> int:
                messages = reader_messages(SUB_SYSTEM_PROMPT, CHUNK_QUESTION_ALLOWANCE, OrderedPayload(hits), document)
                return counter.count_messages(messages)

            base = cost(())
            worst = max(cost((hit,)) for hit in view) - base
            self._chunk_cost_cache = {key: (base, worst)}
        base, worst = self._chunk_cost_cache[key]
        return max(1, (self.chunk_call_tokens - base) // max(worst, 1))

    def _chunk_refusal(self, question: str, hits: Sequence[Hit], document: str, count: int, output_cap: int) -> str:
        """What the root is told when a chunk read is over budget: how many WOULD fit.

        Refused rather than truncated, so the root is never answered from a read it
        believes covered chunks that were cut off -- and told the number, because a 4B
        root told only "too long" retries with a list that is still too long.
        """
        fit = 0
        for n in range(1, len(hits)):
            messages = reader_messages(SUB_SYSTEM_PROMPT, question, OrderedPayload(tuple(hits[:n])), document)
            size = self.ordered_token_counter.count_messages(messages)
            if size > self.chunk_call_tokens or size + output_cap > self.structured_context_tokens:
                break
            fit = n
        if not fit:
            return (
                f"llm_query refused: your question and the first chunk alone exceed the {self.chunk_call_tokens}-token "
                "limit for one call. Shorten the question. Nothing was read."
            )
        return (
            f"llm_query refused: those {len(hits)} chunks and your question come to {count} tokens, over the "
            f"{self.chunk_call_tokens}-token limit for ONE call; only the first {fit} fit. Pass at most {fit} "
            f"chunks per call -- e.g. llm_query(question, chunks[a:a+{fit}]) -- and make another call for the "
            "rest. Nothing was read."
        )

    def _make_env(
        self,
        context: str,
        metrics: dict,
        cache: dict,
        notes: list,
        provided_chunks: Optional[Sequence[Hit]] = None,
        ordered_chunks: Optional[Sequence[Hit]] = None,
    ) -> dict:
        final_box: dict = {"value": None, "done": False}
        # `llm_query`'s second parameter is also called `context` and shadows the
        # document inside it, so bind the document under a name the closure keeps.
        document = context
        # The ordered-chunk view, addressed by span, so llm_query can tell this run's
        # chunks from any other Hit the root might build or keep from elsewhere.
        view_lookup = {(hit.start, hit.end): hit for hit in ordered_chunks} if ordered_chunks is not None else None
        coverage_spans: list[tuple[int, int]] = []
        # Every span search() returned, whether or not the root read it.
        # run_benchmark uses these for the answer-string presence diagnostic.
        retrieved_spans: list[tuple[int, int]] = metrics.setdefault("search_retrieved_spans", [])

        def record_coverage(start: int, end: int) -> None:
            """Merge one real document span and expose the union as a fraction."""
            nonlocal coverage_spans
            coverage_spans = _merge_spans([*coverage_spans, (start, end)])
            covered = sum(span_end - span_start for span_start, span_end in coverage_spans)
            metrics["document_coverage_fraction"] = covered / len(document) if document else 0.0

        def llm_query(prompt: str, context=None) -> str:
            # Key on the FULL strings. Keying on the TRUNCATED ones makes two
            # different calls that share a 32k-char prefix collide -- e.g.
            # context[0:100000] and context[0:200000] -- silently returning one
            # slice's answer for the other. The key is a (question, slice) pair so
            # the same question over two slices never collides either.
            question = str(prompt)
            # The second argument is a raw slice (the documented one- and two-arg
            # forms) OR the hit list `search` returned. Accepting hits is what lets
            # the harness attribute the payload to real document spans: a payload
            # the MODEL joins is not findable in the document, so it records no
            # coverage and reads downstream exactly like a cell whose sub-model
            # never ran (fixed_chunk_grid/audit_cells.py treats coverage 0 as that tell).
            hit_spans: Optional[list[tuple[int, int]]] = None
            # An ordered-chunk read: a slice of `chunks`, or one chunk, is read IN THE
            # ORDER GIVEN with each chunk labelled by its own offsets -- the structured
            # arm's OrderedPayload, reused. Only Hits that ARE this run's chunks qualify;
            # any other Hit list keeps the merge-into-document-order path below.
            chunk_call = False
            reader_count = 0
            if view_lookup is not None and context is not None and not isinstance(context, (str, OrderedPayload)):
                items = [context] if isinstance(context, Hit) else list(context)
                if not items:
                    # A filter that matched nothing (EMPTY_CHUNK_READ_ERROR): sent on, the
                    # reader would describe text it was never shown.
                    metrics["empty_chunk_reads"] = metrics.get("empty_chunk_reads", 0) + 1
                    raise ValueError(EMPTY_CHUNK_READ_ERROR)
                if all(isinstance(h, Hit) and view_lookup.get((h.start, h.end)) == h for h in items):
                    context = OrderedPayload(tuple(items))
                    chunk_call = True
                else:
                    # Materialized once, so a generator reaches the path below intact.
                    context = items
            ordered = isinstance(context, OrderedPayload)
            if ordered:
                hit_spans = [(h.start, h.end) for h in context.hits]
                chunk = context.render(document)
                messages = reader_messages(SUB_SYSTEM_PROMPT, question, context, document)
                count = self.ordered_token_counter.count_messages(messages)
                output_cap = getattr(self.sub, "max_tokens", 0) or 0
                if chunk_call:
                    if count > self.chunk_call_tokens or count + output_cap > self.structured_context_tokens:
                        metrics["chunk_budget_refusals"] = metrics.get("chunk_budget_refusals", 0) + 1
                        raise ValueError(self._chunk_refusal(question, context.hits, document, count, output_cap))
                    # Recorded beside the server's count once a call is actually made: a
                    # cache hit or a budget notice sends nothing, and would leave the two
                    # lists different lengths.
                    reader_count = count
                elif count > self.structured_group_input_tokens or count + output_cap > self.structured_context_tokens:
                    raise ValueError(
                        f"ordered reader exceeds token budget: {count} input + {output_cap} output against "
                        f"{self.structured_group_input_tokens} group / {self.structured_context_tokens} context"
                    )
                else:
                    metrics.setdefault("reader_input_tokens", []).append(count)
            elif context is None or isinstance(context, str):
                chunk = None if context is None else str(context)
            else:
                items = list(context)
                if items and all(isinstance(h, Hit) for h in items):
                    metrics["search_hits_read"] = metrics.get("search_hits_read", 0) + len(items)
                    spans = _merge_spans([(h.start, h.end) for h in items])
                    if self.min_subcall_chars > 0:
                        spans = _widen_spans(spans, self.min_subcall_chars, len(document))
                    hit_spans = spans
                    chunk = HIT_SEPARATOR.join(document[start:end] for start, end in spans)
                elif not items:
                    # search() found nothing. Send the question alone rather than the
                    # string "[]", which is what str() of an empty list would paste in.
                    chunk = None
                else:
                    chunk = str(context)
            # Widen BEFORE keying, so the cache is keyed on what the sub model
            # actually receives. (Only an exact repeat collapses: windows are centred
            # on the slice, so two NEARBY probes widen to overlapping-but-distinct
            # windows. Snapping them to a shared grid would collapse those too, but
            # it would also let a hit land at the very edge of its window with its
            # following context cut off -- the opposite of what a bigger read is for.)
            # Counted only once the call actually happens, below: every early return
            # past this point (cache hit, budget, deadline, fit failures) leaves
            # sub_calls alone, so counting here would let the expanded FRACTION
            # exceed 1.0.
            expanded = False
            needs_expansion = bool(
                chunk
                and hit_spans is None
                and self.min_subcall_chars > 0
                and len(chunk) < min(self.min_subcall_chars, len(document))
            )
            # A hit payload is assembled from spans the harness already knows, so
            # `document.find` cannot locate it and `_expand_subcall` has nothing to
            # anchor on -- the spans were widened above instead.
            if chunk is not None and hit_spans is None:
                chunk, expanded = self._expand_subcall(chunk, document)
            unlocatable = needs_expansion and not expanded
            key = (question, chunk)
            if self.cache_subcalls and key in cache:
                metrics["sub_cache_hits"] += 1
                return cache[key]
            if self._sub_call_budget is not None and metrics["sub_calls"] >= self._sub_call_budget:
                return (
                    "[SUB-CALL LIMIT REACHED] No further llm_query calls are "
                    "available for this example. Answer from what you have already "
                    "seen, or use plain string/regex operations on `context`."
                )
            if self.subcall_deadline_check and self._deadline is not None and time.monotonic() > self._deadline:
                return (
                    "[TIME LIMIT REACHED] The per-example time budget is exhausted; "
                    "no further llm_query calls are available. Answer NOW from what "
                    "you have already seen, or use plain string/regex operations on "
                    "`context`."
                )
            # Resolve where this payload sits before folding it into the HTTP
            # request, so coverage still describes the document text actually sent.
            spans_sent: list[tuple[int, int]] = []
            if hit_spans is not None:
                spans_sent = list(hit_spans)
            elif chunk is not None:
                start = document.find(chunk)
                if start >= 0:
                    spans_sent = [(start, start + len(chunk))]
            payload_chars = len(chunk) if chunk is not None else 0
            if chunk is not None:
                # The OpenAI chat endpoint receives one user message, so fold an
                # optional document slice into that message before applying the cap.
                question = f"{question}\n\n{chunk}"
                chunk = None
            question_chars = len(question)
            if ordered:
                truncated = False
            else:
                question, chunk, truncated = self._cap_subcall(question, chunk)
            if spans_sent and truncated:
                # Keep only what survived the cap. _cap_subcall truncates from the
                # right, so a clipped payload must stop claiming coverage for text
                # the sub model never received. On the folded path the payload rides
                # inside the question, so the surviving payload is whatever is left
                # of it after the question's own prefix.
                kept = len(chunk) if chunk is not None else max(0, len(question) - (question_chars - payload_chars))
                spans_sent = _clip_spans(spans_sent, kept, len(HIT_SEPARATOR))
                metrics["sub_payload_truncated_calls"] = metrics.get("sub_payload_truncated_calls", 0) + 1
            if truncated:
                # Always on the QUESTION side: on the split path the context slice
                # is what gets compressed, and a notice buried there can be evicted.
                question += (
                    f"\n[NOTE: your prompt was truncated at {self.max_subcall_chars} "
                    "chars TOTAL; pass a smaller snippet]"
                )
            # Pause the code-exec watchdog (armed in _exec) around this blocking
            # sub-LLM call: a slow-but-legit network call / rate-limit sleep must
            # NOT be mistaken for a runaway loop. Only pure-Python time between
            # sub-calls counts toward exec_timeout.
            remaining = None
            if self._alarm_active:
                remaining, _ = signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                ans = self.sub.chat(
                    [
                        {"role": "system", "content": SUB_SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ]
                )
            finally:
                # `remaining is not None`, not `remaining` -- setitimer returns 0.0
                # for an already-expired timer, and a truthiness test then skips
                # re-arming, silently disabling the exec watchdog for the REST of the
                # cell while _alarm_active still reads True.
                if self._alarm_active and remaining is not None:
                    # Grant a floor rather than re-arming at the literal remainder:
                    # a timer that expired DURING the sub-call would otherwise fire
                    # ~50ms after llm_query returns, billing sub-call latency to the
                    # pure-Python code watchdog and aborting a cell that never looped.
                    signal.setitimer(signal.ITIMER_REAL, max(remaining, ALARM_REARM_FLOOR_S))
            metrics["sub_calls"] += 1
            if ordered:
                # The server's own count, beside the local one above: equal lists are
                # the proof that the packing budget was priced on what vLLM received.
                served = getattr(getattr(self.sub, "usage", None), "last_prompt_tokens", None)
                metrics.setdefault("reader_served_prompt_tokens", []).append(served)
                if chunk_call:
                    metrics.setdefault("reader_input_tokens", []).append(reader_count)
                    metrics["chunk_payload_calls"] = metrics.get("chunk_payload_calls", 0) + 1
                    metrics.setdefault("chunk_ranks_read", []).extend(h.rank for h in context.hits)
            if expanded:
                metrics["sub_slices_expanded"] = metrics.get("sub_slices_expanded", 0) + 1
            if unlocatable:
                metrics["sub_slice_unlocatable_calls"] = metrics.get("sub_slice_unlocatable_calls", 0) + 1
            metrics["sub_call_tokens"] += (
                self.tok.count(question) + (self.tok.count(chunk) if chunk else 0) + self.tok.count(ans)
            )
            # Size of the PAYLOAD actually sent, separate from sub_call_tokens (which
            # folds in the question and the decoded answer and so can only bound it).
            # Recorded for every HTTP sub-call: without it "how big are the
            # root's slices" was only answerable by re-parsing saved transcripts, and
            # the http arms had no chunk-size column at all.
            payload = len(chunk) if chunk is not None else len(question)
            metrics["sub_payload_chars"] = metrics.get("sub_payload_chars", 0) + payload
            metrics["sub_payload_chars_max"] = max(metrics.get("sub_payload_chars_max", 0), payload)
            for span_start, span_end in spans_sent:
                record_coverage(span_start, span_end)
            if self.cache_subcalls:
                cache[key] = ans
            return ans

        def FINAL(answer) -> None:
            final_box["value"] = str(answer)
            final_box["done"] = True

        def FINAL_NONE(reason: str = "") -> None:
            """Abstain: the document does not contain the answer.

            Distinct from FINAL so the grounding guard cannot reject it. A model
            that searched honestly and found nothing used to have no way to say so
            -- FINAL("unknown") is by construction absent from the REPL output, so
            the guard read the one truthful ending as a hallucinated guess.
            """
            final_box["value"] = None
            final_box["reason"] = str(reason).strip()
            final_box["done"] = True
            final_box["abstained"] = True

        env = {
            "context": context,
            "llm_query": llm_query,
            "re": re,
            # Not for the REPL -- the sandbox only exposes what the root is told
            # about, and the root is never told about this. It is here so the notes
            # path in run_structured can record what it read without reimplementing
            # the span merge that backs document_coverage_fraction.
            "record_coverage": record_coverage,
        }
        if ordered_chunks is not None:
            # A fresh list: the root may sort or slice it as it likes, and the view the
            # harness checks reads against (view_lookup) is unaffected.
            env["chunks"] = list(ordered_chunks)
        # Case variants bound to the same callables (see FINAL_NAMES).
        env.update({name: FINAL for name in FINAL_NAMES})
        env.update({name: FINAL_NONE for name in FINAL_NONE_NAMES})

        if self.scratchpad is not None:

            def note(text) -> str:
                text = str(text).strip()
                if text:
                    notes.append(text)
                    metrics["notes_saved"] += 1
                return f"[saved note #{len(notes)}]"

            env["note"] = note

        if self.search_k > 0:
            # A chunk set handed to the AGENTIC root. `search` is the delivery channel
            # rather than a new primitive precisely so the system prompt stays
            # BYTE-IDENTICAL to the BM25 arm's: the only difference between the two
            # runs is which windows come back, which is what makes "who chose the text"
            # a clean single variable. The cost is that rephrasing is neutralised --
            # every query returns the same set -- so this arm answers "does the loop
            # help GIVEN good chunks", not "does a better retriever help the loop".
            fixed_hits = list(provided_chunks) if provided_chunks is not None else None
            if fixed_hits is not None:
                metrics["chunks_provided"] = len(fixed_hits)
                metrics["search_fixed"] = True

            def _search_limit(k) -> int:
                # Clamp DOWNWARD only. `k` is the grid's swept axis, and the worked
                # example in the system prompt is copied verbatim by the root -- if
                # the example showed `k=5`, every cell would read 5 windows whatever
                # the flag said, and the axis would measure nothing. Asking for fewer
                # is harmless; asking for more cannot be allowed to leak the axis.
                limit = self.search_k
                if k is not None:
                    requested = max(1, int(k))
                    if requested > self.search_k:
                        metrics["search_k_clamped_calls"] = metrics.get("search_k_clamped_calls", 0) + 1
                    limit = min(requested, self.search_k)
                return limit

            def _record_search(hits: list) -> list:
                """One accounting path for both retrievers, so the counters cannot drift."""
                metrics["search_calls"] = metrics.get("search_calls", 0) + 1
                metrics["search_hits_returned"] = metrics.get("search_hits_returned", 0) + len(hits)
                for hit in hits:
                    retrieved_spans.append((hit.start, hit.end))
                return hits

            def search(query, k=None) -> list:
                """The document's best-matching windows for `query` (BM25) -- or, when this
                run was handed a chunk set, that set: the same windows for any query."""
                if metrics.get("search_calls", 0) >= self.max_search_calls:
                    return []
                if fixed_hits is not None:
                    return _record_search(fixed_hits[: _search_limit(k)])
                index = retrieval.get_index(document, self.search_window_chars, self.search_overlap_chars)
                return _record_search(index.search(str(query), _search_limit(k)))

            env["search"] = search

        def FINAL_VAR(name) -> None:
            # Raise rather than answering "<missing var X>", which used to be recorded
            # as a FINISHED, successful prediction. The NameError lands in the REPL
            # observation, so the model sees and can correct it.
            key = str(name)
            if key not in env:
                raise NameError(
                    f"FINAL_VAR({key!r}): no variable named {key!r} exists in the "
                    "REPL. Assign it first, or call FINAL(<expression>) directly."
                )
            final_box["value"] = str(env[key])
            final_box["done"] = True

        env.update({name: FINAL_VAR for name in FINAL_VAR_NAMES})
        env["_final_box"] = final_box
        return env

    def _exec(self, code: str, env: dict, obs_limit: int) -> tuple[str, str]:
        """Run one code cell. Returns (what the model sees, what actually printed).

        The two differ once output exceeds obs_limit. Only the first is sent back
        to the model; the second feeds the grounding accumulator, which must not
        be truncated -- an answer that appeared in the elided middle of a long
        output is grounded, and reading only the display copy called it a guess.
        """
        code = code.strip()
        note = ""
        try:
            compile(code, "<rlm>", "exec")
        except SyntaxError:
            # Ordered-chunk runs only: every other agentic run keeps the diagnosis it
            # was scored with.
            if self.chunk_order is not None and PROSE_ANSWER_LINE_RE.search(code):
                return PROSE_ANSWER_CELL_ERROR, PROSE_ANSWER_CELL_ERROR
            if "\n" not in code:
                fixed = ONELINE_FIX_RE.sub("\n", code)
                try:
                    compile(fixed, "<rlm>", "exec")
                    code = fixed
                    note = (
                        "\n[note: your code block was written on a single line; it was "
                        "auto-reformatted. Please use real newlines inside code blocks.]"
                    )
                except SyntaxError:
                    unparsable = (
                        "[SYNTAX ERROR] Your code block was written on a single line and "
                        "could not be parsed. Rewrite it as a properly formatted multi-line "
                        "```python block with real newlines."
                    )
                    return unparsable, unparsable
        buf = io.StringIO()
        # Bound model-generated code with a wall-clock timeout so an infinite or
        # runaway loop can't hang the whole benchmark. SIGALRM interrupts a stuck
        # pure-Python loop between bytecodes (main thread only); it does NOT
        # interrupt a C-level regex — a known CPython limitation.
        use_alarm = (
            bool(self.exec_timeout)
            and hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread()
        )
        old_handler = None
        try:
            with contextlib.redirect_stdout(buf):
                try:
                    tree = ast.parse(code)
                except SyntaxError:
                    tree = None
                literals = []
                if tree:
                    for node in ast.walk(tree):
                        if (
                            isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name)
                            # Every accepted spelling, or an aliased call would
                            # carry a guessed literal past the grounding guard.
                            and node.func.id in FINAL_NAMES
                            and node.args
                        ):
                            static = _static_string(node.args[0])
                            if static is not None:
                                literals.append(static)
                env["_rlm_final_literals"] = literals
                if use_alarm:
                    old_handler = signal.signal(signal.SIGALRM, _raise_exec_timeout)
                    signal.setitimer(signal.ITIMER_REAL, self.exec_timeout)
                    self._alarm_active = True
                if tree and tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = tree.body[-1]
                    assign = ast.Assign(
                        targets=[ast.Name(id="_rlm_last", ctx=ast.Store())],
                        value=last.value,
                    )
                    tree.body[-1] = ast.copy_location(assign, last)
                    ast.fix_missing_locations(tree)
                    sentinel = object()
                    env["_rlm_last"] = sentinel
                    exec(compile(tree, "<rlm>", "exec"), env)  # noqa: S102
                    val = env.get("_rlm_last")
                    if val is not sentinel and val is not None:
                        env["_"] = val
                        print(val if isinstance(val, str) else repr(val))
                else:
                    exec(code, env)  # noqa: S102
        except _ExecTimeout:
            buf.write(
                f"\n[TIMEOUT] your code ran longer than {self.exec_timeout:.0f}s and was "
                "aborted — almost certainly an infinite or runaway loop. Rewrite it to "
                "terminate: avoid unbounded while-loops, bound every iteration, and operate "
                "on slices of `context` instead of rescanning it repeatedly."
            )
        except KeyboardInterrupt:
            raise  # operator Ctrl-C must still stop the run
        except BaseException:
            # BaseException, not Exception: model code calling exit()/quit()/sys.exit()
            # raises SystemExit, which would otherwise escape _exec, escape run(), and
            # escape run_benchmark's `except Exception` -- terminating the whole sweep
            # mid-campaign. Treat it as an ordinary cell failure.
            buf.write("\n[EXCEPTION]\n" + traceback.format_exc(limit=3))
        finally:
            self._alarm_active = False
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
        full = buf.getvalue()
        out = full
        if len(out) > obs_limit:
            half = obs_limit // 2
            out = out[:half] + f"\n...[truncated {len(out) - obs_limit} chars]...\n" + out[-half:]
        return (out if out.strip() else "[no output]") + note, full

    # ---------------- budget / context view ----------------
    def _budget_note(self) -> str:
        if self.budget is None:
            return ""
        if self.scratchpad is not None:
            return (
                f"\n- MEMORY BUDGET: your working context is capped at ~{self.budget.max_context_tokens} "
                "tokens. Once you exceed it, your OLDEST turns are DROPPED automatically. Raw REPL "
                "output that scrolls out vanishes, but Python VARIABLES persist across turns and "
                "your note() SCRATCHPAD is always re-shown. So save anything you will need for "
                "FINAL with note('...') (or keep it in a variable); never rely on old output "
                "staying visible."
            )
        return (
            f"\n- MEMORY BUDGET: your working context is capped at ~{self.budget.max_context_tokens} "
            "tokens. Once you exceed it, your OLDEST turns are DROPPED automatically and are gone "
            "for good — they are not saved or summarized anywhere. Raw REPL output that scrolls out "
            "vanishes, but Python VARIABLES in the REPL persist across turns. So store anything you "
            "will need for FINAL in a variable (or be ready to recompute it); never rely on old "
            "output staying visible."
        )

    def _notes_block(self, notes: list) -> str:
        text = "\n".join(f"- {n}" for n in notes) if notes else "(nothing saved yet)"
        cap = self.scratchpad.max_notes_tokens if self.scratchpad else 1024
        # Keep the scratchpad within its sub-budget by dropping the OLDEST text.
        # The marker goes on once, at the end: prepending it every pass stacked a
        # copy per iteration and then counted each copy against the same cap. The
        # old loop also stopped at 200 chars, so a small cap was silently ignored.
        marker = "- ...[oldest notes truncated]\n"
        truncated = False
        while text and self.tok.count(text) + (self.tok.count(marker) if truncated else 0) > cap:
            text = text[max(1, int(len(text) * 0.2)) :]
            truncated = True
        return marker + text if truncated else text

    # ---------------- main loop ----------------
    def structured_question(self, task: str) -> str:
        """The question every structured map/reader call is asked.

        One function for both callers, so the harness packs reader groups against
        exactly the question `run_structured` later sends -- a budget priced on a
        different string would not bound the real request.
        """
        map_task = bare_query(task) if self.structured_map_question == "query" else task
        return STRUCTURED_MAP_QUESTION.format(task=map_task, nothing=STRUCTURED_NOTHING)

    def _base_metrics(self) -> dict:
        """The per-example counters every arm reports.

        Shared by `run` and `run_structured` so the two cannot drift: a key one
        of them forgot would read downstream as a zero rather than as missing,
        and `average_*` aggregates would quietly average over the wrong arms.
        """
        metrics: dict = {
            "steps": 0,
            "root_prompt_tokens": 0,
            "root_completion_tokens": 0,
            "peak_context_tokens": 0,
            "sub_calls": 0,
            "sub_call_tokens": 0,
            "sub_cache_hits": 0,
            # Kept for metric compatibility with the parent's HTTP path. The clean
            # client cannot produce an in-process backend fit failure, so it stays 0.
            "sub_fit_failures": 0,
            # Pre-initialized like every other counter: incremented only via
            # .get(..., 0) + 1, it was absent from a healthy run's metrics and
            # downstream aggregation had to special-case its absence.
            "sub_slice_unlocatable_calls": 0,
            "document_coverage_fraction": 0.0,
            "evictions": 0,
            "overflow_evictions": 0,
            "budget": (self.budget.max_context_tokens if self.budget else None),
        }
        if self.scratchpad is not None:
            metrics["scratchpad"] = True
            metrics["notes_saved"] = 0
        # Search counters are present only when retrieval is enabled.
        if self.search_k > 0:
            metrics["search_calls"] = 0
            metrics["search_hits_returned"] = 0
            metrics["search_hits_read"] = 0
            metrics["search_k_clamped_calls"] = 0
        return metrics
