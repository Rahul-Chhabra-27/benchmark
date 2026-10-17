# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal long-context harness with a memory budget (Skylight RLM).

The long document never enters the root model's context. The root sees only
the system prompt, the task, and the action / observation history.

Each root turn must end with a stop form:
  1. SUB_CALL — read chunk(s) through a sub-model (SCAN feeds one chunk at a
     time so the sub-call stays small; control returns only when the scan
     finishes; each chunk reply is appended to history).
  2. final_answer={...} — history already has enough evidence; that string is
     the answer used for evaluation.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from evaluation.rlm.shared.client import LLMClient
from evaluation.rlm.shared.organization import CONFIG as ORGANIZATION_CONFIG
from evaluation.rlm.shared.organization import ordered_hits, organize
from evaluation.rlm.shared.retrieval import Hit, get_index

from .rlm import MemoryBudget, RLMResult, TokenCounter, _is_context_overflow

# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #

ROOT_SYSTEM_PROMPT = """\
You are solving a task over a LONG document that does NOT fit in your context.

CHUNKS: there are {n_chunks} chunks in the document (valid indices: 0..{last_idx}).
Each chunk is about {chunk_chars} characters. You never see the raw document —
only your action history and the observations returned for those actions.

There IS always a solution in the document. Do not give up and do not invent
facts. Escalate through the tactics below until history actually contains the
answer, then stop with final_answer={{...}}.

WHAT TO TRY (in this order):
  1. FIND a chunk, then read it. Use RFIND on a distinctive string (a name,
     title, number, rare word — never the whole question). INDEX(0) reads the
     best hit. Multi-hop questions need several finds: read one fact, RFIND
     what you just learned, read again, until every piece is in history.
  2. SCAN a prefix. Two useful orders:
       STATE=BM25  — query BM25: chunk 0 is the best match to QUERY; SCAN(i)
                     reads the top i+1 chunks one-by-one (e.g. SCAN(4)).
       STATE=ORDERED — BM25-graph: related chunks sit next to each other;
                     INDEX a hit then SCAN(i,j) its neighbours.
     Use this when you are not sure which exact string to RFIND.
  3. If that is not enough, FULLSCAN the chosen order. Each chunk is still
     read one-by-one; every reply is written into history. Use this to locate
     the relevant passage and keep its extract in history.
  4. Last resort: LOOK=VANILLA. That sends the ENTIRE document plus QUERY to
     the sub-model in one call — the same as answering in vanilla mode. Use
     this only after 1–3 have failed.

MEMORY: older history may be dropped to stay within a memory budget. Re-read
anything you still need. Do not invent chunk contents.

Each turn, emit EXACTLY ONE stop form, then STOP. Write nothing after it. The two
stop forms are either 1) Ask for more information or 2) Finish with final answer.

────────────────────────────────────────
1) Ask for more information
────────────────────────────────────────
Parameters:
  st    = ORDERED | UNORDERED | BM25 | RFIND(text)
  i     = integer (0-based index in the chosen STATE order)
  span  = INDEX(i) | SCAN(i) | FULLSCAN | VANILLA | SCAN(i,j)

Then stop with:
  QUERY: <question for the sub-model>
  SUB_CALL(STATE=st, LOOK=span)

STATE:
  UNORDERED — chunks in document (source) order.
  ORDERED   — BM25-graph order (default). Built once BEFORE your question:
              each chunk is scored as a BM25 query against the others, then
              a similarity graph is walked so related chunks sit NEXT TO
              EACH OTHER. This is NOT ranked by your QUERY.
  BM25      — query BM25: every chunk ranked against this QUERY (best first).
              SCAN(i) then reads the i+1 most relevant chunks.
  RFIND(text) — substring search (case-insensitive). Matching chunks move to
              the FRONT in ORDERED relative order; the rest follow, still in
              ORDERED. Example: SUB_CALL(STATE=RFIND("spielberg"), LOOK=INDEX(0))

LOOK:
  INDEX(i)  — ask QUERY about chunk i only. Reply comes back immediately.
  SCAN(i)   — linear scan of chunks 0..i inclusive, one chunk at a time.
              Control returns only after the whole prefix scan finishes.
  SCAN(i,j) — linear scan of chunks i..j inclusive (a neighbourhood around a
              hit). Example: after INDEX finds chunk 7, SCAN(5,10) reads its
              neighbours. Bounds are clamped; if j<i they are swapped.
  FULLSCAN  — same one-by-one scan over every chunk. Control returns only
              when the entire scan finishes. Each reply is stored in history.
  VANILLA   — one sub-call over the FULL document + QUERY (vanilla mode).
              STATE is ignored. Last resort.

Child replies are stored in your history as ANSWER + SUMMARY. Read those
summaries before you SUB_CALL again — do not re-FIND or re-SCAN a chunk
whose summary you already have.

────────────────────────────────────────
2) Finish when history is enough
────────────────────────────────────────
Stop with:
  final_answer={{<your answer>}}

Use this only after observations support the answer. The text inside the
braces is the answer used for evaluation.

────────────────────────────────────────
Worked example (shape only)
────────────────────────────────────────
  You:  QUERY: Who directed the 1975 shark film?
        SUB_CALL(STATE=RFIND("shark"), LOOK=INDEX(0))
  Obs:  [INDEX 0 | RFIND(shark)]
        ANSWER: Steven Spielberg
        SUMMARY: This passage names Spielberg as director of the 1975 shark film.
  You:  QUERY: Where was that director born?
        SUB_CALL(STATE=ORDERED, LOOK=SCAN(0,3))
  Obs:  [SCAN 0 (0..3) | ORDERED]
        ANSWER: Cincinnati, Ohio
        SUMMARY: Neighbour chunks give Spielberg's birthplace as Cincinnati, Ohio.
  You:  final_answer={{Cincinnati, Ohio}}

The user's task is:
{task}
{answer_format_note}"""

SUB_SYSTEM_PROMPT = """\
You are a reading assistant for a root model that cannot see the document.
Use ONLY the provided text.

Reply with EXACTLY this shape (two labels, nothing else):
ANSWER: <direct answer to the question, or NOTHING RELEVANT>
SUMMARY: <2-4 sentences the root should remember: names, numbers, relations,
where a next fact might live. If the question is unanswered, still note any
distinctive terms or structure in this text so the root does not re-read it
blindly.>

Do not discuss how you were invoked."""

FOLD_MARKER = (
    "[MEMORY NOTICE] {n} earlier message(s) were dropped to stay within your "
    "memory budget. They are gone from your context. Re-issue a SUB_CALL if "
    "you need anything from those turns. Continue from the recent history below."
)

BEGIN_MSG = (
    "The document is split into {n_chunks} chunks (indices 0..{last_idx}). "
    "You do not see those chunks — only later observations. "
    "Begin. End your first turn with QUERY + SUB_CALL, or final_answer={{...}}."
)

_STATE_ATOM = r"ORDERED|UNORDERED|BM25|st|" r"RFIND\s*\(\s*(?:\"[^\"]*\"|'[^']*'|[^,)]+?)\s*\)"
_FINAL_RE = re.compile(r"final_answer\s*=\s*\{(?P<body>.*)\}\s*$", re.IGNORECASE | re.DOTALL)
_SUBCALL_RE = re.compile(
    r"SUB_CALL\s*\(\s*STATE\s*=\s*(?P<state>" + _STATE_ATOM + r")\s*,\s*"
    r"LOOK\s*=\s*(?P<look>INDEX\s*\(\s*(?P<index>i|\d+)\s*\)|"
    r"SCAN\s*\(\s*(?P<scan>i|\d+)\s*(?:,\s*(?P<scan_end>j|\d+)\s*)?\)|"
    r"FULLSCAN|VANILLA|span)\s*"
    r"(?:,\s*QUERY\s*=\s*(?P<qstr>\"[^\"]*\"|'[^']*'))?\s*\)",
    re.IGNORECASE,
)
_QUERY_LINE_RE = re.compile(r"(?im)^\s*QUERY\s*:\s*(?P<query>.+?)\s*$")
_ST_BIND_RE = re.compile(
    r"(?im)^\s*st\s*=\s*(?P<v>ORDERED|UNORDERED|BM25|" r"RFIND\s*\(\s*(?:\"[^\"]*\"|'[^']*'|[^)\n]+)\s*\))\s*$"
)
_I_BIND_RE = re.compile(r"(?im)^\s*i\s*=\s*(?P<v>-?\d+)\s*$")
_J_BIND_RE = re.compile(r"(?im)^\s*j\s*=\s*(?P<v>-?\d+)\s*$")
_SPAN_BIND_RE = re.compile(
    r"(?im)^\s*span\s*=\s*(?P<v>INDEX\s*\(\s*(?:i|\d+)\s*\)|"
    r"SCAN\s*\(\s*(?:i|\d+)\s*(?:,\s*(?:j|\d+)\s*)?\)|FULLSCAN|VANILLA)\s*$"
)
_RFIND_RE = re.compile(r"^RFIND\s*\(\s*(?P<inner>.*)\s*\)\s*$", re.IGNORECASE | re.DOTALL)


@dataclass
class SubCall:
    state: str  # ORDERED | UNORDERED | BM25 | RFIND
    look: str  # INDEX | SCAN | FULLSCAN | VANILLA
    index: Optional[int]  # INDEX(i); SCAN(i) end; SCAN(i,j) start
    query: str
    find: Optional[str] = None
    end: Optional[int] = None  # SCAN(i,j) end; None means SCAN(i) is 0..index

    @property
    def state_label(self) -> str:
        if self.state == "RFIND" and self.find:
            return f"RFIND({self.find})"
        return self.state


@dataclass
class SkylightResult:
    answer: str | None
    steps: int
    finished: bool
    transcript: list = field(default_factory=list)
    end_reason: str = ""
    metrics: dict = field(default_factory=dict)

    def to_rlm_result(self) -> RLMResult:
        return RLMResult(
            answer=self.answer,
            steps=self.steps,
            finished=self.finished,
            transcript=self.transcript,
            end_reason=self.end_reason,
            metrics=self.metrics,
        )


def _parse_state_token(token: str) -> tuple[str, Optional[str]]:
    raw = (token or "").strip()
    upper = raw.upper()
    if upper in ("ORDERED", "UNORDERED", "BM25"):
        return upper, None
    found = _RFIND_RE.match(raw)
    if not found:
        raise ValueError(f"unrecognized STATE {token!r}; use ORDERED, UNORDERED, BM25, or RFIND(text)")
    inner = found.group("inner").strip()
    if len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in "\"'":
        inner = inner[1:-1]
    inner = inner.strip()
    if not inner:
        raise ValueError("RFIND() requires a non-empty search string")
    return "RFIND", inner


def _resolve_look(
    look_raw: str, i_value: Optional[int], j_value: Optional[int] = None
) -> tuple[str, Optional[int], Optional[int]]:
    compact = re.sub(r"\s+", "", look_raw.upper())
    if compact in ("FULLSCAN", "VANILLA"):
        return compact, None, None
    m = re.fullmatch(r"INDEX\((I|\d+)\)", compact)
    if m:
        inner = m.group(1)
        if inner == "I":
            if i_value is None:
                raise ValueError("LOOK uses i but no i=<integer> binding was given")
            return "INDEX", i_value, None
        return "INDEX", int(inner), None
    m = re.fullmatch(r"SCAN\((I|\d+)(?:,(J|\d+))?\)", compact)
    if not m:
        raise ValueError(f"unrecognized LOOK {look_raw!r}")
    a, b = m.group(1), m.group(2)
    if a == "I":
        if i_value is None:
            raise ValueError("LOOK uses i but no i=<integer> binding was given")
        start = i_value
    else:
        start = int(a)
    if b is None:
        return "SCAN", start, None
    if b == "J":
        if j_value is None:
            raise ValueError("LOOK uses j but no j=<integer> binding was given")
        return "SCAN", start, j_value
    return "SCAN", start, int(b)


def parse_root_action(text: str) -> tuple[str, object]:
    """Return ('final', answer_str) or ('subcall', SubCall) or ('invalid', reason)."""
    stripped = (text or "").strip()
    if not stripped:
        return "invalid", "empty reply"

    final = _FINAL_RE.search(stripped)
    sub = _SUBCALL_RE.search(stripped)
    final_at = final.end() if final else -1
    sub_at = sub.end() if sub else -1

    if final_at < 0 and sub_at < 0:
        return "invalid", ("Reply must end with either QUERY + SUB_CALL(STATE=..., LOOK=...) " "or final_answer={...}.")

    if final_at > sub_at:
        return "final", final.group("body").strip()

    assert sub is not None
    st_bind = _ST_BIND_RE.search(stripped)
    i_bind = _I_BIND_RE.search(stripped)
    j_bind = _J_BIND_RE.search(stripped)
    span_bind = _SPAN_BIND_RE.search(stripped)
    i_value = int(i_bind.group("v")) if i_bind else None
    j_value = int(j_bind.group("v")) if j_bind else None

    state_tok = sub.group("state")
    if state_tok.strip().upper() == "ST":
        if not st_bind:
            return "invalid", "SUB_CALL STATE=st requires a line st=ORDERED|UNORDERED|BM25|RFIND(text)."
        state_tok = st_bind.group("v")
    try:
        state, find = _parse_state_token(state_tok)
    except ValueError as exc:
        return "invalid", str(exc)

    look_tok = sub.group("look")
    if look_tok.upper().strip() == "SPAN":
        if not span_bind:
            return "invalid", "SUB_CALL LOOK=span requires a line span=INDEX(i)|SCAN(i)|SCAN(i,j)|FULLSCAN|VANILLA."
        look_tok = span_bind.group("v")
    try:
        look, index, end = _resolve_look(look_tok, i_value, j_value)
    except ValueError as exc:
        return "invalid", str(exc)

    qstr = sub.group("qstr")
    if qstr:
        query = qstr[1:-1].strip()
    else:
        qline = _QUERY_LINE_RE.search(stripped)
        query = qline.group("query").strip() if qline else ""
    if not query:
        return "invalid", "SUB_CALL requires QUERY: <question> (or QUERY= inside the call)."

    return "subcall", SubCall(state=state, look=look, index=index, query=query, find=find, end=end)


def _source_chunks(document: str, artifact: dict) -> list[Hit]:
    spans = artifact["spans"]
    return [
        Hit(rank=r, start=start, end=end, score=0.0, text=document[start:end])
        for r, (start, end) in enumerate(spans, start=1)
    ]


def _rfind_order(graph: Sequence[Hit], needle: str) -> list[Hit]:
    """Matches first (ORDERED relative order), then the rest (still ORDERED)."""
    key = needle.lower()
    hits = [h for h in graph if key in h.text.lower()]
    rest = [h for h in graph if key not in h.text.lower()]
    return [
        Hit(rank=r, start=h.start, end=h.end, score=h.score, text=h.text) for r, h in enumerate([*hits, *rest], start=1)
    ]


def _order_chunks(
    state: str,
    source: Sequence[Hit],
    graph: Sequence[Hit],
    find: Optional[str] = None,
    query: Optional[str] = None,
    bm25_index=None,
) -> list[Hit]:
    if state == "UNORDERED":
        return list(source)
    if state == "RFIND":
        return _rfind_order(graph, find or "")
    if state == "BM25" and bm25_index is not None:
        ranked = bm25_index.rank_all(query or "")
        return [
            Hit(rank=r, start=h.start, end=h.end, score=h.score, text=h.text) for r, h in enumerate(ranked, start=1)
        ]
    return list(graph)


def _split_turns(history: list[dict]) -> list[list[dict]]:
    """One turn = one assistant action plus the user observations that follow it."""
    turns: list[list[dict]] = []
    current: list[dict] = []
    for msg in history:
        if msg["role"] == "assistant" and current:
            turns.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append(current)
    return turns


def _ensure_summary(text: str) -> str:
    """Guarantee a SUMMARY block in what the root stores from a child call."""
    body = (text or "").strip()
    if re.search(r"(?im)^SUMMARY\s*:", body):
        return body
    return f"{body}\nSUMMARY: {body}" if body else "ANSWER: NOTHING RELEVANT\nSUMMARY: empty child reply"


def _flatten(turns: Sequence[Sequence[dict]]) -> list[dict]:
    return [m for turn in turns for m in turn]


_DEBUG_EQ = "=" * 88
_DEBUG_DASH = "-" * 88


def _debug_print(*parts: str) -> None:
    print(*parts, flush=True)


def format_chat_messages(messages: Sequence[dict]) -> str:
    blocks = []
    for i, msg in enumerate(messages, start=1):
        role = msg.get("role", "?")
        body = (msg.get("content") or "").rstrip()
        blocks.append(f"{_DEBUG_DASH}\n[{i}/{len(messages)} role={role}]\n{body}")
    return "\n".join(blocks)


def debug_call_block(
    who: str,
    history: Sequence[str],
    *,
    prompt: Sequence[dict] | None = None,
    generation: str | None = None,
) -> None:
    """Readable live dump: call history, then this call's prompt and output."""
    _debug_print(f"\n{_DEBUG_EQ}")
    _debug_print(who)
    _debug_print(_DEBUG_EQ)
    _debug_print("----- CALL HISTORY -----")
    for line in history:
        marker = "  <-- this call" if line == who else ""
        _debug_print(f"  {line}{marker}")
    if prompt is not None:
        _debug_print(f"----- {who} FULL PROMPT -----")
        _debug_print(format_chat_messages(prompt))
    if generation is not None:
        _debug_print(f"----- {who} GENERATION -----")
        _debug_print(generation.rstrip() if generation else "<empty>")
    _debug_print(_DEBUG_EQ)


class SkylightRLM:
    """Root loop + one-chunk-at-a-time sub-reads under an optional memory budget."""

    def __init__(
        self,
        root_client: LLMClient,
        sub_client: LLMClient | None = None,
        max_steps: int = 20,
        chunk_chars: int = ORGANIZATION_CONFIG["chunk_chars"],
        chunk_overlap: int = ORGANIZATION_CONFIG["overlap"],
        budget: Optional[MemoryBudget] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        obs_limit: int = 4000,
        run_timeout: Optional[float] = None,
        max_sub_calls: Optional[int] = 200,
        debug: bool = False,
        organization_dir: Optional[str] = None,
    ):
        self.root = root_client
        self.sub = sub_client or root_client
        self.max_steps = max_steps
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.budget = budget
        self.tok = TokenCounter(token_counter)
        self.obs_limit = obs_limit
        self.run_timeout = run_timeout
        self.max_sub_calls = max_sub_calls
        self.debug = debug
        self.organization_dir = organization_dir

    def run(self, context: str, task: str, answer_format: str | None = None) -> SkylightResult:
        artifact, _, cache_hit = organize(context, "bm25-graph", self.organization_dir)
        graph_chunks = ordered_hits(context, artifact)
        source_chunks = _source_chunks(context, artifact)
        bm25_index = get_index(context, chunk_chars=self.chunk_chars, overlap=self.chunk_overlap)
        n_chunks = len(source_chunks)
        deadline = (time.monotonic() + self.run_timeout) if self.run_timeout else None

        metrics: dict = {
            "n_chunks": n_chunks,
            "chunk_order": "bm25-graph",
            "organization_cache_hit": cache_hit,
            "organization_build_cpu_seconds": artifact.get("build_cpu_seconds"),
            "sub_calls": 0,
            "chunks_read": 0,
            "evictions": 0,
            "overflow_evictions": 0,
            "peak_context_tokens": 0,
            "root_prompt_tokens": 0,
            "invalid_actions": 0,
        }

        system_msg = {
            "role": "system",
            "content": ROOT_SYSTEM_PROMPT.format(
                n_chunks=n_chunks,
                last_idx=max(0, n_chunks - 1),
                chunk_chars=self.chunk_chars,
                task=task,
                answer_format_note=(
                    f"\nFormat the value inside final_answer={{...}} as: {answer_format}\n" if answer_format else ""
                ),
            ),
        }
        begin_msg = {
            "role": "user",
            "content": BEGIN_MSG.format(n_chunks=n_chunks, last_idx=max(0, n_chunks - 1)),
        }

        full_history: list[dict] = []
        forced_drop_turns = 0
        evicted_count = 0
        transcript: list[dict] = []
        call_log: list[str] = []

        def timed_out() -> bool:
            return deadline is not None and time.monotonic() > deadline

        def clip(text: str) -> str:
            if len(text) <= self.obs_limit:
                return text
            return text[: self.obs_limit] + "\n...[truncated]"

        def squeeze_turns(turns: list[list[dict]]) -> list[list[dict]]:
            """Drop oldest observations, then oldest turns, until the view fits."""
            if self.budget is None:
                return turns
            kept = [list(t) for t in turns]
            cap = self.budget.max_context_tokens

            def over(candidate: list[list[dict]]) -> bool:
                fold = sum(len(t) for t in turns) - sum(len(t) for t in candidate)
                msgs = [system_msg, begin_msg]
                if fold:
                    msgs.append({"role": "user", "content": FOLD_MARKER.format(n=fold)})
                msgs.extend(_flatten(candidate))
                return self.tok.count_messages(msgs) > cap

            while kept and over(kept):
                head = kept[0]
                user_idxs = [i for i, m in enumerate(head) if m["role"] == "user"]
                if len(user_idxs) > 1:
                    del head[user_idxs[0]]
                elif len(head) > 1 and head[0]["role"] == "assistant" and user_idxs:
                    del head[user_idxs[0]]
                else:
                    kept.pop(0)
            return kept

        def build_sent() -> tuple[list[dict], int]:
            nonlocal evicted_count
            turns = _split_turns(full_history)
            if forced_drop_turns:
                turns = turns[forced_drop_turns:]

            if self.budget is not None:
                keep = self.budget.keep_recent_turns
                if keep > 0 and len(turns) > keep:
                    turns = turns[-keep:]
                turns = squeeze_turns(turns)

            kept_n = sum(len(t) for t in turns)
            fold_n = len(full_history) - kept_n
            sent = [system_msg, begin_msg]
            if fold_n > 0:
                sent.append({"role": "user", "content": FOLD_MARKER.format(n=fold_n)})
            sent.extend(_flatten(turns))
            if fold_n > evicted_count:
                evicted_count = fold_n
                metrics["evictions"] += 1
            return sent, fold_n

        def emit_debug(who: str, prompt: Sequence[dict] | None, generation: str | None) -> None:
            if not self.debug:
                return
            debug_call_block(who, call_log, prompt=prompt, generation=generation)

        def root_reply() -> tuple[str, list[dict]]:
            nonlocal forced_drop_turns
            n_turns = len(_split_turns(full_history))
            sent: list[dict] = []
            for _ in range(12):
                sent, _ = build_sent()
                ctx_tokens = self.tok.count_messages(sent)
                try:
                    reply = self.root.chat(sent)
                except Exception as exc:  # noqa: BLE001
                    if not _is_context_overflow(exc) or forced_drop_turns >= n_turns:
                        raise
                    forced_drop_turns += 1
                    metrics["overflow_evictions"] += 1
                    continue
                metrics["root_prompt_tokens"] += ctx_tokens
                served = getattr(getattr(self.root, "usage", None), "last_prompt_tokens", None)
                metrics["peak_context_tokens"] = max(metrics["peak_context_tokens"], served or ctx_tokens)
                return reply, sent
            raise RuntimeError("root context overflow could not be resolved by eviction")

        def ask_chunk(query: str, chunk: Hit, who: str) -> str:
            """One sub-model call: question + this chunk only. No SCAN/LOOK protocol."""
            if self.max_sub_calls is not None and metrics["sub_calls"] >= self.max_sub_calls:
                raise RuntimeError(f"max_sub_calls ({self.max_sub_calls}) exceeded")
            metrics["sub_calls"] += 1
            metrics["chunks_read"] += 1
            messages = [
                {"role": "system", "content": SUB_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n" "Write ANSWER and SUMMARY as instructed.\n\n" f"Passage:\n{chunk.text}"
                    ),
                },
            ]
            ans = clip((self.sub.chat(messages) or "").strip())
            call_log.append(who)
            emit_debug(who, messages, ans)
            return _ensure_summary(ans)

        def ask_vanilla(query: str, who: str) -> str:
            """One sub-model call over the full document (vanilla last resort)."""
            if self.max_sub_calls is not None and metrics["sub_calls"] >= self.max_sub_calls:
                raise RuntimeError(f"max_sub_calls ({self.max_sub_calls}) exceeded")
            metrics["sub_calls"] += 1
            metrics["chunks_read"] += n_chunks
            messages = [
                {"role": "system", "content": SUB_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n" "Write ANSWER and SUMMARY as instructed.\n\n" f"Document:\n{context}"
                    ),
                },
            ]
            ans = clip((self.sub.chat(messages) or "").strip())
            call_log.append(who)
            emit_debug(who, messages, ans)
            return _ensure_summary(ans)

        def observations_for(call: SubCall, root_step: int) -> list[str]:
            child_n = 0

            def child_who() -> str:
                nonlocal child_n
                child_n += 1
                return f"ROOT({root_step}) -> CHILD({child_n})"

            if call.look == "VANILLA":
                ans = ask_vanilla(call.query, child_who())
                return [f"[VANILLA | full document]\n{ans}"]

            ordered = _order_chunks(
                call.state,
                source_chunks,
                graph_chunks,
                call.find,
                query=call.query,
                bm25_index=bm25_index,
            )
            n = len(ordered)
            if n == 0:
                return ["[empty document]"]

            if call.look == "INDEX":
                assert call.index is not None
                if not 0 <= call.index < n:
                    return [f"[error] INDEX({call.index}) out of range; valid indices are 0..{n - 1}"]
                ans = ask_chunk(call.query, ordered[call.index], child_who())
                return [f"[INDEX {call.index} | {call.state_label}]\n{ans}"]

            if call.look == "SCAN":
                assert call.index is not None
                if call.end is None:
                    start, finish = 0, call.index
                else:
                    start, finish = call.index, call.end
                if start > finish:
                    start, finish = finish, start
                if start < 0 or finish < 0:
                    return [f"[error] SCAN requires non-negative indices, got {start},{finish}"]
                start = min(start, n - 1)
                finish = min(finish, n - 1)
            else:
                start, finish = 0, n - 1

            tag = "SCAN" if call.look == "SCAN" else "FULLSCAN"
            out: list[str] = []
            for i in range(start, finish + 1):
                if timed_out():
                    raise TimeoutError("run_timeout exceeded during scan")
                ans = ask_chunk(call.query, ordered[i], child_who())
                out.append(f"[{tag} {i} ({start}..{finish}) | {call.state_label}]\n{ans}")
            return out

        for step in range(1, self.max_steps + 1):
            if timed_out():
                return SkylightResult(
                    answer=None,
                    steps=step - 1,
                    finished=False,
                    transcript=transcript,
                    end_reason="run_timeout",
                    metrics=metrics,
                )
            try:
                reply, sent = root_reply()
            except Exception as exc:  # noqa: BLE001
                return SkylightResult(
                    answer=None,
                    steps=step,
                    finished=False,
                    transcript=transcript,
                    end_reason="exception",
                    metrics={**metrics, "exception": str(exc)},
                )

            who = f"ROOT({step})"
            call_log.append(who)
            emit_debug(who, sent, reply)

            transcript.append({"step": step, "role": "assistant", "content": reply})
            kind, payload = parse_root_action(reply)

            if kind == "final":
                answer = str(payload)
                full_history.append({"role": "assistant", "content": reply})
                transcript.append({"step": step, "role": "final", "content": answer})
                return SkylightResult(
                    answer=answer,
                    steps=step,
                    finished=True,
                    transcript=transcript,
                    end_reason="final_answer",
                    metrics=metrics,
                )

            if kind == "invalid":
                metrics["invalid_actions"] += 1
                nudge = (
                    f"REJECTED: {payload}\n"
                    "End the turn with either:\n"
                    "  QUERY: <question>\n"
                    "  SUB_CALL(STATE=ORDERED|UNORDERED|BM25|RFIND(text),\n"
                    "           LOOK=INDEX(i)|SCAN(i)|SCAN(i,j)|FULLSCAN|VANILLA)\n"
                    "or:\n"
                    "  final_answer={{...}}"
                )
                full_history.append({"role": "assistant", "content": reply})
                full_history.append({"role": "user", "content": nudge})
                transcript.append({"step": step, "role": "nudge", "content": nudge})
                continue

            assert isinstance(payload, SubCall)
            try:
                obs_list = observations_for(payload, step)
            except TimeoutError:
                return SkylightResult(
                    answer=None,
                    steps=step,
                    finished=False,
                    transcript=transcript,
                    end_reason="run_timeout",
                    metrics=metrics,
                )
            except Exception as exc:  # noqa: BLE001
                return SkylightResult(
                    answer=None,
                    steps=step,
                    finished=False,
                    transcript=transcript,
                    end_reason="exception",
                    metrics={**metrics, "exception": str(exc)},
                )

            full_history.append({"role": "assistant", "content": reply})
            for obs in obs_list:
                full_history.append({"role": "user", "content": obs})
                transcript.append({"step": step, "role": "observation", "content": obs})

        return SkylightResult(
            answer=None,
            steps=self.max_steps,
            finished=False,
            transcript=transcript,
            end_reason="max_steps",
            metrics=metrics,
        )


def _demo() -> None:
    """Tiny smoke path: parse protocol without a live server."""
    samples = [
        "QUERY: Who directed Jaws?\nSUB_CALL(STATE=ORDERED, LOOK=INDEX(0))",
        "QUERY: find birthplace\nSUB_CALL(STATE=UNORDERED, LOOK=SCAN(3))",
        "QUERY: ranked\nSUB_CALL(STATE=BM25, LOOK=SCAN(4))",
        "QUERY: whole doc\nSUB_CALL(STATE=UNORDERED, LOOK=VANILLA)",
        'QUERY: hop\nst=RFIND("spielberg")\ni=0\nspan=INDEX(i)\nSUB_CALL(STATE=st, LOOK=span)',
        'QUERY: find beta\nSUB_CALL(STATE=RFIND("beta-only"), LOOK=INDEX(0))',
        "final_answer={Cincinnati, Ohio}",
        "I think the answer is 42",
    ]
    for s in samples:
        print(parse_root_action(s))


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Skylight RLM: minimal long-context harness")
    ap.add_argument("--demo-parse", action="store_true", help="print parse_root_action smoke checks")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--context-file", type=str, default="")
    ap.add_argument("--task", type=str, default="")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--chunk-chars", type=int, default=2000)
    ap.add_argument("--max-context-tokens", type=int, default=None)
    ap.add_argument("--keep-recent-turns", type=int, default=3)
    ap.add_argument("--organization-dir", type=str, default=None, help="cache dir for the bm25-graph order")
    ap.add_argument("--debug", action="store_true", help="print ROOT/CHILD prompts and generations live")
    args = ap.parse_args(argv)

    if args.demo_parse or not args.context_file:
        _demo()
        return

    context = open(args.context_file, encoding="utf-8").read()
    if not args.task:
        ap.error("--task is required when running against a live model")

    budget = None
    if args.max_context_tokens is not None:
        budget = MemoryBudget(
            max_context_tokens=args.max_context_tokens,
            keep_recent_turns=args.keep_recent_turns,
        )
    client = LLMClient(model=args.model, base_url=args.base_url)
    harness = SkylightRLM(
        root_client=client,
        max_steps=args.max_steps,
        chunk_chars=args.chunk_chars,
        budget=budget,
        debug=args.debug,
        organization_dir=args.organization_dir,
    )
    result = harness.run(context, args.task)
    print(f"end_reason={result.end_reason} steps={result.steps} answer={result.answer!r}")
    print(f"metrics={result.metrics}")


if __name__ == "__main__":
    main()
