# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured map/reduce and ordered-reader execution."""

from __future__ import annotations

import time
from typing import Callable, Mapping, Sequence

from evaluation.rlm.shared.organization import OrderedPayload, reader_messages
from evaluation.rlm.shared.retrieval import Hit

from .grounding import _mentions
from .prompts import (
    ORDERED_READER_FAILURES,
    STRUCTURED_ANSWER_LIST_NOTE,
    STRUCTURED_ANSWER_SHAPE,
    STRUCTURED_ANSWER_SHAPE_OFFER,
    STRUCTURED_NOTHING,
    STRUCTURED_REDUCE_SYSTEM,
    STRUCTURED_REDUCE_TEMPLATE,
    STRUCTURED_REDUCE_TEMPLATE_GROUPS,
    STRUCTURED_REDUCE_TEMPLATE_OFFER,
    SUB_SYSTEM_PROMPT,
    bare_query,
)
from .spans import _hop_query, _map_groups, _truncate_hits, _window_of
from .types import RLMResult


class StructuredMixin:
    def run_structured(
        self,
        context: str,
        task: str,
        chunks: Sequence[Hit],
        answer_format: str | None = None,
        retrieve: Callable[[str, int], Sequence[Hit]] | None = None,
        window_notes: Mapping[int, str] | None = None,
        ordered_groups: Sequence[OrderedPayload] | None = None,
    ) -> RLMResult:
        """Read a FIXED set of chunks with a fixed pipeline: map, then reduce.

        The agentic loop is removed, not assisted. `run` gives the root a REPL and
        lets it decide what to read and when to stop; this reads exactly the chunks
        it was handed, once each, and then answers from what came back. That is the
        whole treatment, so a win over `run` is attributable to the pipeline and the
        chunks rather than to a luckier search.

        The chunks come from outside: BM25 at a fixed k, or a chunk file written by
        a larger model (see chunkset.py). Either way they arrive as `Hit`s, which is
        what lets `llm_query` attribute the payload to real document spans -- the
        same path the search arm uses, reused rather than reimplemented, so
        coverage, truncation, caching and every sub_* counter mean the same thing
        here as they do there.

        `window_notes` replaces the map step rather than assisting it. The notes were
        written by a pass over the corpus that never saw the question (see
        window_notes.py), so this arm reads a representation built without the query
        and the question enters exactly once, at the reduce. The map step is SKIPPED,
        not cached: `map_calls` is 0 because no model call is made here, and the
        build cost lives in the notes file's header where it can be amortised over
        the queries that share the corpus.
        """
        if ordered_groups is not None:
            if (
                self.ordered_token_counter is None
                or window_notes is not None
                or retrieve is not None
                or self.structured_hops != 1
            ):
                raise ValueError("ordered groups require tokenizer and retrieval-free single pass")
            if self.structured_map_chars or self.structured_answer_shape != "require":
                raise ValueError("ordered groups read whole chunks under the required answer shape")
            flattened = [h for group in ordered_groups for h in group.hits]
            spans = sorted((h.start, h.end) for h in flattened)
            contiguous = all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
            if not spans or spans[0][0] != 0 or spans[-1][1] != len(context) or not contiguous:
                raise ValueError("ordered groups must cover every source character exactly once")
            for group in ordered_groups:
                group.render(context)
            chunks = flattened
        metrics = self._base_metrics()
        metrics["chunks_provided"] = len(chunks)
        metrics["map_calls"] = 0
        metrics["chunks_with_findings"] = 0
        if ordered_groups is not None:
            metrics["group_count"] = len(ordered_groups)
            metrics["group_chunk_counts"] = [len(group.hits) for group in ordered_groups]
        cache: dict = {}
        notes: list = []
        self._sub_call_budget = self.max_sub_calls
        self._deadline = time.monotonic() + self.run_timeout if self.run_timeout else None
        env = self._make_env(context, metrics, cache, notes)
        llm_query = env["llm_query"]
        # Record the provided spans where a search arm records its retrieved ones, so
        # run_benchmark's gold-in-retrieved diagnostic reads this arm with no new code.
        # This is what the selector OFFERED; document_coverage_fraction, maintained by
        # llm_query itself, is what was actually sent.
        retrieved: list = metrics.setdefault("search_retrieved_spans", [])
        retrieved.extend((hit.start, hit.end) for hit in chunks)

        transcript: list = []
        if not chunks:
            # Distinct from an abstention: nothing was read because nothing was
            # offered. Scoring it as "the model found nothing" would blame the model
            # for an empty chunk file.
            return RLMResult(None, 0, False, transcript, "structured_no_chunks", metrics)

        # The map step sees the QUESTION; the reduce step sees the benchmark's whole
        # task string. LOFT wraps its query in citation and formatting instructions
        # that mean something to a model which has read the corpus and derail one
        # looking at a single window -- and the same string carries the output
        # contract the scorer matches on, so stripping it at the reduce would confound
        # accuracy with scoring.
        map_question = self.structured_question(task)
        findings: list[tuple[int, str]] = []
        timed_out = False

        if window_notes is not None:
            # The map step is replaced by a lookup. Coverage is still recorded: the
            # notes ARE derived from that text, so an arm reading notes for the whole
            # grid has read the whole document in the only sense this metric tracks.
            record_coverage = env["record_coverage"]
            metrics["notes_read"] = 0
            for hit in sorted(chunks, key=lambda h: (h.start, h.end)):
                note = window_notes.get(_window_of(hit, self.search_window_chars, self.search_overlap_chars), "")
                record_coverage(hit.start, hit.end)
                if not note:
                    continue
                transcript.append(
                    {
                        "step": len(transcript) + 1,
                        "reply": f"[note] window at [{hit.start}:{hit.end}]",
                        "code": "window_notes[...]",
                        "observation": note,
                    }
                )
                findings.append((len(transcript), note))
                metrics["notes_read"] += 1
            metrics["chunks_with_findings"] = metrics["notes_read"]
            metrics["map_spans"] = len(chunks)
            metrics["steps"] = len(transcript)

        def map_round(round_hits: Sequence[Hit], hop: int) -> bool:
            """Read one retrieval round's windows, one call per group. True if it ran out of time.

            A closure rather than a second copy of the loop, so hop 2 cannot drift
            from hop 1: the granularity, the sentinel test, the transcript shape and
            every counter are the same code on both rounds. A comparison between
            `--structured-hops 1` and `2` then differs in the retrieval and nothing
            else.
            """
            # One call per CONTIGUOUS PASSAGE by default: see _cluster_hits and
            # _map_groups. Every granularity puts the passages in document order,
            # which is the order the agentic path's payload is assembled in and the
            # order the reduce announces.
            # Ordered groups arrive already packed, and are read exactly as given.
            if ordered_groups is not None:
                clusters = [list(g.hits) for g in ordered_groups]
            else:
                clusters = _map_groups(round_hits, self.structured_map_granularity)
            if self.structured_map_chars and ordered_groups is None:
                clusters = [_truncate_hits(c, self.structured_map_chars, context) for c in clusters]
                clusters = [c for c in clusters if c]
            metrics["map_spans"] = metrics.get("map_spans", 0) + len(clusters)
            for index, cluster in enumerate(clusters, start=1):
                if self._deadline is not None and time.monotonic() > self._deadline:
                    if ordered_groups is not None:
                        raise ValueError(
                            f"incomplete ordered read: run timeout after {index - 1}/{len(clusters)} groups"
                        )
                    return True
                if ordered_groups is not None:
                    # A notice string past a budget would read as a FINDING, so an
                    # ordered read refuses to reach one rather than paste it in.
                    if self._sub_call_budget is not None and metrics["sub_calls"] >= self._sub_call_budget:
                        raise ValueError("incomplete ordered read: map-call limit")
                    answer = (llm_query(map_question, OrderedPayload(tuple(cluster))) or "").strip()
                    if not answer or answer.startswith(ORDERED_READER_FAILURES):
                        raise ValueError(f"incomplete ordered read: reader failed on group {index}: {answer[:80]!r}")
                else:
                    answer = (llm_query(map_question, cluster) or "").strip()
                metrics["map_calls"] += 1
                span = ", ".join(f"[{h.start}:{h.end}]" for h in cluster)
                step = len(transcript) + 1
                transcript.append(
                    {
                        "step": step,
                        "reply": f"[map hop {hop}] passage {index}/{len(clusters)} at {span} "
                        f"({len(cluster)} window(s))",
                        "code": f"llm_query(MAP, hop{hop}.cluster[{index - 1}])",
                        "observation": answer,
                        # The exact reader request, so a transcript shows what was read.
                        **(
                            {
                                "source_spans": [(h.start, h.end) for h in cluster],
                                "messages": reader_messages(
                                    SUB_SYSTEM_PROMPT, map_question, OrderedPayload(tuple(cluster)), context
                                ),
                            }
                            if ordered_groups is not None
                            else {}
                        ),
                    }
                )
                # Prefix match, case-insensitive: asked for an exact sentinel a 4B
                # model returns it with punctuation or a short apology attached often
                # enough that equality would count those chunks as real findings and
                # paste "NOTHING RELEVANT." into the reduce step as evidence.
                if answer and not answer.upper().lstrip().startswith(STRUCTURED_NOTHING):
                    findings.append((step, answer))
                    metrics["chunks_with_findings"] += 1
            return False

        timed_out = map_round(chunks, 1) if window_notes is None else False

        # The second hop, and there is no third. A fixed pipeline can only follow a
        # bridge entity it has already READ, so the round-2 query is built from the
        # findings rather than from the windows -- see _hop_query for the measurement
        # that forces that choice. Windows round 1 already read are excluded: sending
        # them again would spend the budget re-reading text whose findings are
        # already in hand.
        if self.structured_hops >= 2 and findings and retrieve is not None and not timed_out:
            seen = {(hit.start, hit.end) for hit in chunks}
            wanted = len(chunks) or 1
            try:
                candidates = list(retrieve(_hop_query(bare_query(task), [text for _, text in findings]), wanted * 2))
            except Exception as exc:  # pragma: no cover - a retriever fault must not void the round
                metrics["hop2_error"] = f"{type(exc).__name__}: {exc}"
                candidates = []
            fresh = [hit for hit in candidates if (hit.start, hit.end) not in seen][:wanted]
            metrics["hop2_chunks"] = len(fresh)
            metrics["chunks_provided"] += len(fresh)
            retrieved.extend((hit.start, hit.end) for hit in fresh)
            if fresh:
                timed_out = map_round(fresh, 2)
        metrics["steps"] = len(transcript)

        if ordered_groups is not None and (timed_out or metrics.get("document_coverage_fraction") != 1.0):
            # Raised, not returned: run_benchmark records an exception as an ERROR,
            # whereas an end_reason would be scored as an ordinary wrong answer.
            raise ValueError(
                f"incomplete ordered read: coverage {metrics.get('document_coverage_fraction')}, "
                f"{metrics['map_calls']}/{len(ordered_groups)} groups read"
            )

        if not findings:
            if timed_out:
                return RLMResult(None, len(transcript), False, transcript, "run_timeout", metrics)
            # Every chunk was read and none of them spoke to the question. That is an
            # abstention in the same sense FINAL_NONE is, and it is earned by
            # construction: the map step read everything it was given. The REPL's
            # effort gate does not apply -- it exists to stop a root claiming absence
            # without calling llm_query, which cannot happen here.
            metrics["abstain_reason"] = "no chunk contained relevant information"
            return RLMResult(None, len(transcript), True, transcript, "abstained", metrics)

        findings_block = "\n\n".join(f"[{i}] {text[:self.reduce_finding_chars]}" for i, text in findings)
        # `offer` is revision 1's treatment restored whole: its reduce line ("Be
        # concise.") and its note ("Format your answer as: ..."), so the answer-shape
        # change can be scored apart from the clustering change it shipped with.
        offered = self.structured_answer_shape == "offer"
        if ordered_groups is not None:
            reduce_template = STRUCTURED_REDUCE_TEMPLATE_GROUPS
        else:
            reduce_template = STRUCTURED_REDUCE_TEMPLATE_OFFER if offered else STRUCTURED_REDUCE_TEMPLATE
        shape_note = STRUCTURED_ANSWER_SHAPE_OFFER if offered else STRUCTURED_ANSWER_SHAPE
        answer_format_note = (
            shape_note.format(answer_format=answer_format.strip())
            if answer_format
            else ("" if offered else "\nReply with the answer itself and nothing else.")
        )
        if self.structured_answer_list:
            answer_format_note += STRUCTURED_ANSWER_LIST_NOTE.format(example=self.structured_answer_list)
        reduce_prompt = reduce_template.format(
            task=task,
            n_chunks=len(findings),
            findings=findings_block,
            answer_format_note=answer_format_note,
        )
        messages = [
            {"role": "system", "content": STRUCTURED_REDUCE_SYSTEM},
            {"role": "user", "content": reduce_prompt},
        ]
        ctx_tokens = self.tok.count_messages(messages)
        if ordered_groups is not None:
            ctx_tokens = self.ordered_token_counter.count_messages(messages)
            metrics["reducer_input_tokens"] = ctx_tokens
            metrics["reducer_finding_chars"] = len(findings_block)
            metrics["retained_finding_count"] = len(findings)
            output_cap = getattr(self.root, "max_tokens", 0) or 0
            if ctx_tokens + output_cap > self.structured_context_tokens:
                raise ValueError(
                    f"ordered reducer overflow: {ctx_tokens} input + {output_cap} output "
                    f"> {self.structured_context_tokens} context"
                )
        reply = self.root.chat(messages)
        metrics["root_prompt_tokens"] += ctx_tokens
        metrics["root_completion_tokens"] += self.tok.count(reply or "")
        # The server's own count, for the same reason root_reply prefers it: this is
        # the number compared against KVPress's retained-token metric.
        served = getattr(getattr(self.root, "usage", None), "last_prompt_tokens", None)
        metrics["peak_context_tokens"] = max(metrics["peak_context_tokens"], served or ctx_tokens)
        transcript.append(
            {
                "step": len(transcript) + 1,
                "reply": reply,
                "code": "[reduce]",
                **({"messages": messages} if ordered_groups is not None else {}),
                "observation": "",
            }
        )
        metrics["steps"] = len(transcript)

        answer = (reply or "").strip()
        # Recorded, not rejected. `run`'s grounding guard exists to catch a root that
        # answers from parametric memory without reading the document -- a failure
        # this arm cannot have, since the root is shown nothing but findings quoted
        # out of real chunks. Rejecting here would also have nowhere to go: there is
        # no loop to nudge the model back into. So the flag stays as a diagnostic.
        if answer and not any(_mentions(text, answer) for _, text in findings) and not _mentions(task, answer):
            metrics["reduce_answer_not_quoted"] = True
        return RLMResult(answer, len(transcript), True, transcript, "structured_complete", metrics)
