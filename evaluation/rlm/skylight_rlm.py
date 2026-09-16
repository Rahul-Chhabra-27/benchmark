# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal long-context harness with a memory budget (Skylight RLM).

The long document never enters the root model's context. The root sees only
the system prompt, the task, and the action / observation history.

Each root turn must end with a stop form:
  1. SUB_CALL — INDEX is one sub-read; SCAN/FULLSCAN is one batched vLLM
     request over the selected chunks (control returns when the batch
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

from .client import LLMClient
from .organization import CONFIG as ORGANIZATION_CONFIG
from .organization import ordered_hits, organize
from .retrieval import Hit, get_index
from .rlm import MemoryBudget, RLMResult, TokenCounter, _is_context_overflow

# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #

ROOT_SYSTEM_PROMPT = """\
CHUNKS: there are {n_chunks} chunks (valid indices: 0..{last_idx}).
You control retrieval over this document.
The answer exists in the document. You see only retrieval summaries.

OUTPUT ONLY ONE OF THESE FORMS. DO NOT WRITE ANALYSIS OR A PREAMBLE.

QUERY: {retrieval_query}
SUB_CALL(STATE=BM25, LOOK=SCAN(4))

or:

final_answer={{<answer in the required format>}}

For every retrieval, copy the QUERY line shown above exactly. Never output
placeholder text or angle brackets.

States:
- STATE=RFIND("text"): chunks containing this exact text first. Use a
  distinctive phrase copied from the task or a previous summary. Its complete
  call is: SUB_CALL(STATE=RFIND("exact phrase"), LOOK=INDEX(0))
- STATE=BM25: chunks ranked for QUERY. This is the best general search.
- STATE=ORDERED: related chunks adjacent in BM25-graph order.
- STATE=UNORDERED: original document order.

Looks. Every call needs BOTH a STATE and a LOOK:
- LOOK=INDEX(i): read one chunk.
- LOOK=SCAN(i): read chunks 0 through i in one batched sub-call.
- LOOK=SCAN(i,j): read chunks i through j in one batched sub-call.
- LOOK=FULLSCAN: read every chunk in one batched sub-call,
  as SUB_CALL(STATE=ORDERED, LOOK=FULLSCAN)
- LOOK=VANILLA: whole document in one call, as
  SUB_CALL(STATE=ORDERED, LOOK=VANILLA)

Search policy:
1. Start with RFIND on a phrase from the task, or BM25 with SCAN(4).
2. If ANY observation contains "SUMMARY: FOUND", the search succeeded. Ignore
   NOT_FOUND observations from other chunks and output final_answer immediately,
   copying the answer= text from that FOUND observation verbatim.
3. Only when every observation says NOT_FOUND, try BM25 SCAN(4), then BM25
   SCAN(10). Do not search invented answer candidates or repeat a search.
4. Use FULLSCAN or VANILLA only as a last resort.

Only the text inside final_answer={{...}} is graded, so it must contain the
answer itself and nothing else. {answer_format_note}
If the task also asks you to print document titles or IDs, that part is not
graded: leave titles and IDs out of the braces entirely.

Example. Given the observation
  SUMMARY: FOUND | answer=Melvil Dewey | evidence=Dewey invented the system |
  documents=Dewey Decimal Classification (199)
the correct reply is exactly:
  final_answer={{Melvil Dewey}}

TASK:
{task}"""

SUB_SYSTEM_PROMPT = """\
Answer the question using only the passage. Output exactly one line.

If the passage answers the question:
SUMMARY: FOUND | answer=<direct answer> | evidence=<short supporting fact> | documents=<TITLE (ID)>

Otherwise:
SUMMARY: NOT_FOUND | documents=<titles and IDs examined>

The answer= value must copy the fullest exact answer span from the passage.
Do not shorten a person's full name to the document title or a common name.
Copy titles and IDs exactly. No analysis, preamble, markdown, or extra lines."""

FOLD_MARKER = (
    "[MEMORY NOTICE] {n} earlier message(s) were dropped to stay within your "
    "memory budget. They are gone from your context. Re-issue a SUB_CALL if "
    "you need anything from those turns. Continue from the recent history below."
)

BUDGET_SPENT_MSG = (
    "[BUDGET SPENT] All {spent} retrieval calls for this task are used up. No "
    "further SUB_CALL will return anything. Answer now with final_answer={{...}} "
    "using the best candidate in your history, even if you are not certain."
)

ROOT_SAMPLING = {
    "repetition_penalty": 1.05,
    "frequency_penalty": 0.4,
    "presence_penalty": 0.2,
}

BEGIN_MSG = (
    "There are {n_chunks} chunks (indices 0..{last_idx}). Begin retrieval. "
    "Output only QUERY + SUB_CALL, or final_answer={{...}}."
)

_STATE_ATOM = r"ORDERED|UNORDERED|BM25|st|" r"RFIND\s*\(\s*(?:\"[^\"]*\"|'[^']*'|[^,)]+?)\s*\)"
# Matched anywhere, not just at end of reply: a root that second-guesses itself
# after writing the stop form still committed to an answer.
_FINAL_RE = re.compile(r"final_answer\s*=\s*\{(?P<body>[^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.IGNORECASE | re.DOTALL)
# Thinking dumps often include template finals; never treat these as a stop.
_PLACEHOLDER_FINAL_RE = re.compile(
    r"(?is)^\s*(?:"
    r"\.\.\.|…|"
    r"value|answer|result|output|text|query|"
    r"your\s+answer|<value>|<answer>|"
    r"\{+\s*(?:\.\.\.|…|value|answer)\s*\}+"
    r")\s*$"
)


def _is_placeholder_final(body: str) -> bool:
    """True for CoT/template shells like ``{value}`` / ``{...}``, not real answers."""
    text = (body or "").strip()
    if not text:
        return True
    return bool(_PLACEHOLDER_FINAL_RE.match(text))
_SUBCALL_RE = re.compile(
    r"SUB_CALL\s*\(\s*STATE\s*=\s*(?P<state>" + _STATE_ATOM + r")\s*,\s*"
    r"LOOK\s*=\s*(?P<look>INDEX\s*\(\s*(?P<index>i|\d+)\s*\)|"
    r"SCAN\s*\(\s*(?P<scan>i|\d+)\s*(?:,\s*(?P<scan_end>j|\d+)\s*)?\)|"
    r"FULLSCAN|VANILLA|span)\s*"
    r"(?:,\s*QUERY\s*=\s*(?P<qstr>\"[^\"]*\"|'[^']*'))?\s*\)",
    re.IGNORECASE,
)
# FULLSCAN and VANILLA need no index, so the root often drops LOOK= and writes
# them as the STATE. The intent is unambiguous, and rejecting it just burns a
# step on a mistake the root tends to repeat.
_WHOLE_DOC_LOOK_RE = re.compile(
    r"SUB_CALL\s*\(\s*(?:STATE|LOOK)\s*=\s*(?P<look>FULLSCAN|VANILLA)\s*\)",
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


def parse_root_action(text: str, default_query: str = "") -> tuple[str, object]:
    """Return ('final', answer_str) or ('subcall', SubCall) or ('invalid', reason).

    ``default_query`` fills a missing QUERY line. Placeholder finals such as
    ``final_answer={...}`` / ``{value}`` that thinking dumps into the CoT are ignored.
    """
    stripped = _strip_thinking(text or "")
    if not stripped:
        return "invalid", "empty reply"

    finals = [m for m in _FINAL_RE.finditer(stripped) if not _is_placeholder_final(m.group("body"))]
    final = finals[-1] if finals else None
    sub = _SUBCALL_RE.search(stripped) or _WHOLE_DOC_LOOK_RE.search(stripped)
    final_at = final.end() if final else -1
    sub_at = sub.end() if sub else -1

    if final_at < 0 and sub_at < 0:
        return "invalid", ("Reply must end with either QUERY + SUB_CALL(STATE=..., LOOK=...) " "or final_answer={...}.")

    if final_at > sub_at:
        return "final", final.group("body").strip()

    assert sub is not None
    if sub.re is _WHOLE_DOC_LOOK_RE:
        qline = _QUERY_LINE_RE.search(stripped)
        query = qline.group("query").strip() if qline else ""
        if not query:
            query = (default_query or "").strip()
        if not query:
            return "invalid", "SUB_CALL requires QUERY: <question> (or QUERY= inside the call)."
        look = sub.group("look").upper()
        return "subcall", SubCall(state="ORDERED", look=look, index=None, query=query)

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
        query = (default_query or "").strip()
    if not query:
        return "invalid", "SUB_CALL requires QUERY: <question> (or QUERY= inside the call)."

    return "subcall", SubCall(state=state, look=look, index=index, query=query, find=find, end=end)


_FOUND_RE = re.compile(r"(?i)SUMMARY:\s*FOUND\b")
_NO_ANSWER_RE = re.compile(
    r"(?is)^(not[\s_-]*found|nothing(?:\s+relevant)?|n/?a|none|"
    r"no(?:t)?(?:\s+an?)?\s*answer|unknown|unanswerable|"
    r"(?:i\s+)?(?:do\s+not|don't)\s+know|cannot\s+(?:find|determine)|"
    r"no\s+information)\s*[.!]?\s*$"
)


def _unwrap_answer_shell(text: str) -> str:
    body = (text or "").strip()
    pairs = {"'": "'", '"': '"', "[": "]", "(": ")", "{": "}"}
    while len(body) >= 2 and pairs.get(body[0]) == body[-1]:
        body = body[1:-1].strip()
    return body


def _is_no_answer(text: str) -> bool:
    """True when the root abstained rather than naming a fact."""
    body = _unwrap_answer_shell(_strip_thinking(text or ""))
    if not body:
        return True
    return bool(_NO_ANSWER_RE.match(re.sub(r"\s+", " ", body)))


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


_THINK_RE = re.compile(r"<think(ing)?>.*?</think(ing)?>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"(?i)SUMMARY\s*:")

# A child that ignores the SUMMARY protocol would otherwise dump its whole
# deliberation into the root's history, which is what the protocol exists to
# prevent. Keep a bounded tail instead of the entire generation.
UNLABELLED_SUMMARY_CHARS = 600


def _strip_thinking(text: str) -> str:
    """Drop think traces so FINAL/SUB_CALL parsers see only protocol text.

    Handles paired ``<think>...</think>`` and the vLLM Qwen3.5 case where the
    opening tag is omitted and only a trailing ``</think>`` remains. When the
    model never closes the think block, leave the text intact: parse_root_action
    ignores placeholder finals and still finds real SUB_CALLs inside the CoT.
    """
    clean = _THINK_RE.sub("", text or "")
    parts = _THINK_CLOSE_RE.split(clean, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return clean.strip()


def _answer_format_note(answer_format: str | None) -> str:
    """The graded value is what follows the dataset's answer cue, not the cue."""
    cue = (answer_format or "").strip()
    if not cue:
        return "Write only the direct answer."
    return f'The task asks for "{cue} <value>", so write only <value>, in exactly that form.'


def _retrieval_query(task: str) -> str:
    """Use the benchmark's final `query:` line, or the whole task as fallback."""
    matches = re.findall(r"(?im)^\s*query\s*:\s*(.+?)\s*$", task)
    return matches[-1].strip() if matches else task.strip()


def _child_summary_for_root(text: str, limit: int = UNLABELLED_SUMMARY_CHARS) -> str:
    """Only what the child wrote after its last SUMMARY: reaches the root."""
    clean = _strip_thinking(text)
    matches = list(_SUMMARY_RE.finditer(clean))
    if matches:
        body = clean[matches[-1].end() :].strip()
        if body:
            return "SUMMARY: " + body
    if not clean:
        return "SUMMARY: NOTHING RELEVANT"
    tail = clean[-limit:].strip()
    return "SUMMARY: " + tail


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
        root_sampling: Optional[dict] = None,
        vanilla_max_prompt_tokens: Optional[int] = None,
        enable_thinking: bool = True,
    ):
        self.root = root_client
        self.root_sampling = ROOT_SAMPLING if root_sampling is None else root_sampling
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
        self.vanilla_max_prompt_tokens = vanilla_max_prompt_tokens
        self.enable_thinking = enable_thinking

    def _thinking_template(self, extra: dict) -> dict:
        template = dict(extra.get("chat_template_kwargs") or {})
        template["enable_thinking"] = bool(self.enable_thinking)
        return template

    def _root_chat_kwargs(self) -> dict:
        """Root action turn: keep sampling penalties; thinking follows ``enable_thinking``."""
        extra = dict(getattr(self.root, "extra_body", None) or {})
        extra.update(self.root_sampling)
        extra["chat_template_kwargs"] = self._thinking_template(extra)
        # Thinking traces need headroom; the protocol answer still comes after </think>.
        max_tokens = 2048 if self.enable_thinking else 256
        return {"extra_body": extra, "max_tokens": max_tokens}

    def _sub_chat_kwargs(self) -> dict:
        """Chunk extractor; thinking follows ``enable_thinking``."""
        extra = dict(getattr(self.sub, "extra_body", None) or {})
        extra["chat_template_kwargs"] = self._thinking_template(extra)
        max_tokens = 1024 if self.enable_thinking else 192
        return {"extra_body": extra, "max_tokens": max_tokens}

    def _vanilla_chat_kwargs(self) -> dict:
        """Full-document last step; thinking follows ``enable_thinking``."""
        extra = dict(getattr(self.sub, "extra_body", None) or {})
        extra["chat_template_kwargs"] = self._thinking_template(extra)
        max_tokens = 2048 if self.enable_thinking else 1024
        return {"extra_body": extra, "max_tokens": max_tokens}

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
                retrieval_query=_retrieval_query(task),
                answer_format_note=_answer_format_note(answer_format),
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
            root_kwargs = self._root_chat_kwargs()
            sent: list[dict] = []
            for _ in range(12):
                sent, _ = build_sent()
                ctx_tokens = self.tok.count_messages(sent)
                try:
                    reply = self.root.chat(sent, **root_kwargs)
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

        def remaining_sub_calls() -> Optional[int]:
            if self.max_sub_calls is None:
                return None
            return max(0, self.max_sub_calls - metrics["sub_calls"])

        def chunk_messages(query: str, chunk: Hit) -> list[dict]:
            return [
                {"role": "system", "content": SUB_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}\n\nPassage:\n{chunk.text}"},
            ]

        def ask_chunk(query: str, chunk: Hit, who: str) -> str:
            """One INDEX-style sub-read. Counts as one SUB_CALL."""
            metrics["sub_calls"] += 1
            metrics["chunks_read"] += 1
            messages = chunk_messages(query, chunk)
            ans = (self.sub.chat(messages, **self._sub_chat_kwargs()) or "").strip()
            call_log.append(who)
            emit_debug(who, messages, ans)
            return clip(_child_summary_for_root(ans))

        def ask_scan(query: str, chunks: Sequence[Hit], whos: Sequence[str]) -> list[str]:
            """All SCAN/FULLSCAN chunks in one vLLM batch. Counts as one SUB_CALL."""
            if not chunks:
                return []
            metrics["sub_calls"] += 1
            metrics["chunks_read"] += len(chunks)
            jobs = [chunk_messages(query, chunk) for chunk in chunks]
            kwargs = self._sub_chat_kwargs()
            batch = getattr(self.sub, "chat_batch", None)
            if batch is not None:
                raw = batch(jobs, **kwargs)
            else:
                raw = [self.sub.chat(j, **kwargs) for j in jobs]
            out: list[str] = []
            for who, messages, ans in zip(whos, jobs, raw):
                text = clip(_child_summary_for_root((ans or "").strip()))
                call_log.append(who)
                emit_debug(who, messages, ans or "")
                out.append(text)
            return out

        def ask_vanilla(query: str, who: str) -> str:
            """One sub-model call over the full document (vanilla last resort)."""
            metrics["sub_calls"] += 1
            metrics["chunks_read"] += n_chunks
            messages = [
                {"role": "system", "content": SUB_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nDocument:\n{context}",
                },
            ]
            ans = (self.sub.chat(messages, **self._sub_chat_kwargs()) or "").strip()
            call_log.append(who)
            emit_debug(who, messages, ans)
            return clip(_child_summary_for_root(ans))

        # "The document was exhausted" is about coverage, not about the FULLSCAN
        # token: SCAN(0,n-1) reads every chunk too. Chunks are identified by
        # source offset because each STATE reorders the positions.
        scanned_starts: set[int] = set()
        any_found = False

        def observations_for(call: SubCall, root_step: int) -> list[str]:
            nonlocal any_found
            child_n = 0

            def child_who() -> str:
                nonlocal child_n
                child_n += 1
                return f"ROOT({root_step}) -> CHILD({child_n})"

            budget = remaining_sub_calls()
            if budget == 0:
                return [BUDGET_SPENT_MSG.format(spent=metrics["sub_calls"])]

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
                if _FOUND_RE.search(ans or ""):
                    any_found = True
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
            if timed_out():
                raise TimeoutError("run_timeout exceeded during scan")
            indices = list(range(start, finish + 1))
            chunk_list = [ordered[i] for i in indices]
            whos = [child_who() for _ in indices]
            answers = ask_scan(call.query, chunk_list, whos)
            scanned_starts.update(c.start for c in chunk_list)
            if any(_FOUND_RE.search(a or "") for a in answers):
                any_found = True
            return [f"[{tag} {i} ({start}..{finish}) | {call.state_label}]\n{ans}" for i, ans in zip(indices, answers)]

        def vanilla_last_step(step: int) -> SkylightResult | None:
            """One full-document call, only if the untruncated prompt fits."""
            fmt = f"\nFormat your answer as: {answer_format}" if answer_format else ""
            prompt = f"Document:\n{context}\n\nTask: {task}\nAnswer concisely.{fmt}"
            messages = [{"role": "user", "content": prompt}]
            n_tok = self.tok.count_messages(messages)
            metrics["vanilla_prompt_tokens"] = n_tok
            cap = self.vanilla_max_prompt_tokens
            if cap is not None and n_tok > cap:
                metrics["vanilla_last_step"] = "skipped_too_long"
                return None
            who = f"VANILLA({step})"
            try:
                raw = (self.sub.chat(messages, **self._vanilla_chat_kwargs()) or "").strip()
            except Exception as exc:  # noqa: BLE001
                if _is_context_overflow(exc):
                    metrics["vanilla_last_step"] = "skipped_overflow"
                    return None
                raise
            metrics["vanilla_last_step"] = "ok"
            metrics["sub_calls"] += 1
            call_log.append(who)
            emit_debug(who, messages, raw)
            kind, payload = parse_root_action(raw, default_query=_retrieval_query(task))
            answer = str(payload) if kind == "final" else _strip_thinking(raw)
            transcript.append({"step": step, "role": "vanilla", "content": raw})
            transcript.append({"step": step, "role": "final", "content": answer})
            return SkylightResult(
                answer=answer or None,
                steps=step,
                finished=True,
                transcript=transcript,
                end_reason="vanilla",
                metrics=metrics,
            )

        def finish_without_answer(step: int, end_reason: str) -> SkylightResult:
            # Last-resort vanilla is only for "we read the whole document and
            # still found nothing". Partial coverage, or a chunk that did report
            # FOUND, is not that case.
            metrics["chunks_scanned"] = len(scanned_starts)
            if len(scanned_starts) < n_chunks:
                metrics["vanilla_last_step"] = "skipped_partial_scan"
            elif any_found:
                metrics["vanilla_last_step"] = "skipped_found"
            else:
                vanilla = vanilla_last_step(step)
                if vanilla is not None:
                    return vanilla
            return SkylightResult(
                answer=None,
                steps=step,
                finished=False,
                transcript=transcript,
                end_reason=end_reason,
                metrics=metrics,
            )

        budget_notified = False
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
            if step == self.max_steps:
                return finish_without_answer(step, "max_steps")
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
            kind, payload = parse_root_action(reply, default_query=_retrieval_query(task))

            if kind == "final":
                answer = str(payload)
                full_history.append({"role": "assistant", "content": reply})
                transcript.append({"step": step, "role": "final", "content": answer})
                if _is_no_answer(answer):
                    metrics["abstained"] = True
                    return finish_without_answer(step, "no_answer")
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
                    "  final_answer={...}\n"
                    "FULLSCAN and VANILLA are LOOK values, not STATE values: write\n"
                    "SUB_CALL(STATE=ORDERED, LOOK=FULLSCAN)."
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

            # Once retrieval is spent, further SUB_CALLs cannot make progress.
            # Give the root one turn to answer from history, then stop.
            if remaining_sub_calls() == 0:
                if budget_notified:
                    return finish_without_answer(step, "budget_exhausted")
                budget_notified = True

        return finish_without_answer(self.max_steps, "max_steps")


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
