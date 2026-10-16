# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The whole minimal RLM in one file, in the order a reader needs it:

    prompts      every word the models see: a short base prompt + one block per option
    ordering     none | bm25-query | bm25-graph -> the `chunks` list
    records      one CallRecord per sub-call: tokens read, GPU / offloaded split, cost
    readers      how llm_query reads: plain http, or one kvpress model at factor F
    repl         the persistent Python REPL the root writes into
    loop         the RLM loop and its llm_query / FINAL tools

The configuration (config.py) says which options are on; run.py runs it over a dataset.
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
import re
import signal
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol, Sequence

from ..client import LLMClient
from ..organization import CONFIG as GRID
from ..organization import ordered_hits, organize
from ..retrieval import Hit, get_index
from .config import ExperimentConfig

# ======================================================================================
# Prompts: every word the models are shown
# ======================================================================================
# Every word the models are shown. A short base prompt, plus one block per option.
#
# The base is the RLM idea and nothing more: the document lives in a REPL, the root
# writes code, `llm_query` hands text to a sub-model, `FINAL` ends the run. Each
# option (ordering) adds only the lines that describe what it adds, so the
# difference between two configs is visible in the prompt diff.


BASE_PROMPT = """\
You answer a question about a long document that you cannot read directly. The document
is stored in a Python REPL. Each turn, reply with ONE ```python code block; its printed
output is shown to you in the next message (at most {output_chars} characters).

In the REPL:
  context                       the whole document, a string of {context_chars:,} characters
{tools}  llm_query(question, text)     a sub-model reads `text` and answers `question`. It sees
                                nothing else. One call reads at most {read_tokens:,} tokens
                                (about {read_chars:,} characters); a longer text is refused.
  FINAL(answer)                 end with your answer. Call it only on something you read.

Search `context` with Python (find, regular expressions, slicing), read the relevant parts
with llm_query, then call FINAL. Variables persist between turns. You have {max_steps} turns.
{ordering}{memory}
Question:
{question}
{answer_format}"""

CHUNKS_TOOL = """\
  chunks                        the document cut into {n_chunks} pieces of ~{chunk_chars} characters,
                                in the order described below. Each has .text, .start, .end;
                                llm_query(question, chunks[i:j]) reads a list of chunks.
"""

ORDERING_NOTES = {
    "bm25-query": (
        "\n`chunks` is sorted by BM25 keyword relevance to the question: chunks[0] matches best."
        "\nThe answer is usually near the top.\n"
    ),
    "bm25-graph": (
        "\n`chunks` is grouped by topic, before the question was known: chunks sharing names and"
        "\nrare words sit next to each other. Find a chunk mentioning the question's distinctive"
        "\nterms, then read it together with its neighbours.\n"
    ),
}

# The memory / cost block: only present when a config turns on factor choice, offloading
# or cost reporting, so every other config renders exactly the prompt it always did.
FACTOR_NOTE = (
    "\nMemory: one read keeps at most {budget:,} tokens of KV memory (B). llm_query(question, text, factor=F)"
    "\nwith F in {factors} reads up to B x F tokens, compressed F times: F=1 is exact, a higher F reads"
    "\nwider but keeps less detail. Without factor= a read uses F={default}.\n"
)
OFFLOAD_NOTE = (
    "\nA read may be larger than B x F (up to {cap:,} tokens): the memory beyond B is offloaded to CPU"
    "\nand fetched back when the read is answered.\n"
)
COST_NOTE = (
    "\nCost: every read reports what it cost: {gpu:g} per memory token kept on the GPU, {offloaded:g} per"
    "\ntoken offloaded to CPU. Answer correctly at the lowest total cost.\n"
)
COST_LINE = (
    "[cost: read {read:,} tokens at x{factor} -> {gpu:,} on GPU, {offloaded:,} offloaded; {cost:g} (total {total:g})]"
)

ANSWER_FORMAT = "Answer format: {answer_format}"
LIST_ANSWER_FORMAT = 'Answer format: {answer_format} followed by a list, e.g. ["first", "second", "third"]'

NO_CODE_BLOCK = "Your reply had no ```python code block. Reply with exactly one code block."
REPL_OUTPUT = "REPL output:\n{output}"
BEGIN = "Begin."

SUB_PROMPT = "Answer the question using ONLY the text below. Be concise and factual.\n\nQuestion: {question}"

CHARS_PER_TOKEN = 4  # only to tell the root roughly how many characters fit in one read


def system_prompt(
    *,
    question: str,
    context_chars: int,
    read_tokens: int,
    max_steps: int,
    output_chars: int,
    ordering: str = "none",
    n_chunks: int = 0,
    chunk_chars: int = 0,
    answer_format: Optional[str] = None,
    list_answer: bool = False,
    budget_tokens: Optional[int] = None,
    factors: tuple[int, ...] = (1,),
    default_factor: int = 1,
    offload_cap: Optional[int] = None,
    cost: Optional[tuple[float, float]] = None,
) -> str:
    """The root's system prompt for one question under one config.

    `offload_cap` is set when offloading is on; `cost` (gpu, offloaded) when costs are shown.
    """
    ordered = ordering != "none"
    memory = ""
    if budget_tokens is not None and len(factors) > 1:
        memory += FACTOR_NOTE.format(budget=budget_tokens, factors=list(factors), default=default_factor)
    if offload_cap is not None:
        memory += OFFLOAD_NOTE.format(cap=offload_cap)
    if cost is not None:
        memory += COST_NOTE.format(gpu=cost[0], offloaded=cost[1])
    if answer_format:
        fmt = (LIST_ANSWER_FORMAT if list_answer else ANSWER_FORMAT).format(answer_format=answer_format.strip())
    else:
        fmt = ""
    return BASE_PROMPT.format(
        output_chars=output_chars,
        context_chars=context_chars,
        tools=CHUNKS_TOOL.format(n_chunks=n_chunks, chunk_chars=chunk_chars) if ordered else "",
        read_tokens=read_tokens,
        read_chars=read_tokens * CHARS_PER_TOKEN,
        max_steps=max_steps,
        ordering=ORDERING_NOTES.get(ordering, ""),
        memory=memory,
        question=question.strip(),
        answer_format=fmt,
    )


# ======================================================================================
# Ordering: the `chunks` list
# ======================================================================================
# The `ordering` option: hand the root the document as an ordered list of chunks.
#
#     none        no `chunks`; the root searches `context` itself
#     bm25-query  chunks ranked by BM25 against the question (query-aware)
#     bm25-graph  chunks grouped by a similarity graph built once per document,
#                 before any question is seen (query-agnostic; cached on disk)
#
# Both orders use the same grid: 2000-character chunks, no overlap.


def query_text(question: str) -> str:
    """The question without LOFT's citation boilerplate (the text after a `query:` line).

    Left in, words like "documents" and "answer" match every chunk equally.
    """
    lines = question.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].lstrip()
        if stripped[:6].lower() == "query:":
            return "\n".join([stripped[6:], *lines[index + 1 :]]).strip()
    return question.strip()


def build_chunks(document: str, question: str, ordering: str, organization_dir: Optional[str]) -> list[Hit]:
    """Every chunk of the document, in the given order. Empty for `none`."""
    if ordering == "none":
        return []
    if ordering == "bm25-graph":
        artifact, _, _ = organize(document, "bm25-graph", organization_dir)
        return ordered_hits(document, artifact)
    if ordering == "bm25-query":
        index = get_index(document, GRID["chunk_chars"], GRID["overlap"])
        return index.rank_all(query_text(question))
    raise ValueError(f"unknown ordering {ordering!r}")


def chunk_chars() -> int:
    return int(GRID["chunk_chars"])


# ======================================================================================
# Per-call records, placement (GPU / offloaded) and cost
# ======================================================================================
# What every sub-call did and cost, so a run can be inspected decision by decision.
#
# One CallRecord per llm_query: which turn, how much it read, at which factor, how its
# KV split between GPU and CPU (offloaded), what it cost, whether it was refused, and
# -- once the question ends -- whether compression was needed and whether the read fed
# the final answer. `summarize` folds them into per-question numbers for metrics.json.


PREVIEW_CHARS = 200


@dataclass(frozen=True)
class Placement:
    """Where one read's KV lives: kept = read / F; up to B on the GPU, the rest offloaded."""

    kept: int
    gpu: int
    offloaded: int

    @staticmethod
    def of(read_tokens: int, factor: int, budget_tokens: Optional[int]) -> "Placement":
        kept = math.ceil(read_tokens / factor)
        if budget_tokens is None:  # no budget: nothing is ever offloaded
            return Placement(kept, kept, 0)
        return Placement(kept, min(kept, budget_tokens), max(0, kept - budget_tokens))

    def cost(self, gpu_token: float, offloaded_token: float) -> float:
        return self.gpu * gpu_token + self.offloaded * offloaded_token


@dataclass(frozen=True)
class CallRecord:
    step: int
    question: str
    chunks: Optional[int]  # chunks passed, when the root read a chunk list
    read_tokens: int
    kept_tokens: int  # KV kept after compression (= read_tokens at F = 1)
    limit_tokens: int  # the most a read at this factor may carry
    budget_tokens: Optional[int]  # B, the KV one read may keep on the GPU; None without a budget
    factor: int
    seconds: float
    gpu_tokens: int = 0
    offloaded_tokens: int = 0
    cost: float = 0.0
    refused: str = ""  # why the read was not made, else ""
    answer: str = ""
    # Filled when the question ends:
    compression_needed: Optional[bool] = None  # F > 1 only: would the read NOT have fit in B uncompressed?
    contributed: Optional[bool] = None  # does the read's answer share a content word with FINAL's?

    def to_dict(self) -> dict:
        return asdict(self)


def preview(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1] + "…"


_WORD = re.compile(r"[a-z0-9]+")
_FILLER = frozenset("the a an of in on at to and or is was are were for by with from as that this it its".split())


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(str(text).lower()) if w not in _FILLER}


def contributed(read_answer: str, final_answer: Optional[str]) -> bool:
    """A read fed the answer if they share a content word. Crude, but it separates reads
    that found something from the "nothing relevant here" reads, which is the question."""
    return bool(final_answer) and bool(content_words(read_answer) & content_words(final_answer or ""))


def summarize(calls: Iterable[CallRecord]) -> dict:
    """Per-question numbers: how many reads, how big, at which factors, what they cost."""
    calls = list(calls)
    made = [c for c in calls if not c.refused]
    out = {
        "sub_calls": len(made),
        "refused_calls": len(calls) - len(made),
        "read_tokens": sum(c.read_tokens for c in made),
        "kept_tokens_max": max((c.kept_tokens for c in made), default=0),
        "offloaded_tokens": sum(c.offloaded_tokens for c in made),
        "offloaded_calls": sum(1 for c in made if c.offloaded_tokens),
        "cost": sum(c.cost for c in made),
        "factors": dict(sorted(Counter(f"x{c.factor}" for c in made).items())),
        # Compressed reads that would have fit uncompressed: compression bought nothing there.
        "compressed_unneeded": sum(1 for c in made if c.compression_needed is False),
        "contributing_calls": sum(1 for c in made if c.contributed),
        "read_seconds": round(sum(c.seconds for c in made), 3),
    }
    if made:
        out["read_tokens_mean"] = out["read_tokens"] / len(made)
        # How much of what one read may carry the root actually used: near 0 means it
        # reads tiny slices and the compression budget buys nothing.
        out["read_fill_mean"] = sum(c.read_tokens / c.limit_tokens for c in made) / len(made)
    return out


# ======================================================================================
# Readers: how one llm_query reads
# ======================================================================================
# How one llm_query reads its text: the `compression` option.
#
#     HttpReader  plain reads through the vLLM server (no budget set)
#     KVReader    reads through ONE in-process kvpress model, whose press is set per read:
#                 F = 1 is that model with no press (the control), F > 1 presses by 1 - 1/F.
#                 So the root can choose a different factor on every call, with no extra
#                 servers and one copy of the weights on the GPU.
#
# Both count the tokens they read; the loop turns that into kept / GPU / offloaded
# tokens and a cost (stats.Placement).


@dataclass(frozen=True)
class Read:
    answer: str
    read_tokens: int  # tokens of document text the sub-model was given
    seconds: float


class Reader(Protocol):
    def count_tokens(self, text: str) -> int: ...

    def read(self, question: str, text: str, factor: int = 1) -> Read: ...


class HttpReader:
    """Dense sub-calls to the OpenAI-compatible server (factor 1 only)."""

    def __init__(self, client: Any):
        self.client = client

    def count_tokens(self, text: str) -> int:
        return math.ceil(len(text) / CHARS_PER_TOKEN)

    def read(self, question: str, text: str, factor: int = 1) -> Read:
        if factor != 1:
            raise ValueError("http reads are uncompressed; a factor > 1 needs compression.budget_mb")
        started = time.monotonic()
        answer = self.client.chat([{"role": "user", "content": f"{SUB_PROMPT.format(question=question)}\n\n{text}"}])
        return Read(str(answer or ""), self.count_tokens(text), time.monotonic() - started)


class KVReader:
    """Sub-calls through kvpress: the text is the compressed side, the question is not."""

    def __init__(self, sub: Any):
        self.sub = sub
        self.press = sub.press  # the KVzipPress, or None when no allowed factor compresses

    def count_tokens(self, text: str) -> int:
        return int(self.sub.token_len(text))

    def _use(self, factor: int) -> None:
        """Press the next read by exactly 1 - 1/F (no press at all for F = 1)."""
        if factor == 1:
            self.sub.press = None
        else:
            if self.press is None:
                raise ValueError(f"factor {factor} needs a press; the reader was built without one")
            self.press.compression_ratio = 1 - 1 / factor
            self.sub.press = self.press
            self.sub.press_min_tokens = 1  # every read at F > 1 is pressed, so F is exact
        self.sub.compression_factor = float(factor)

    def read(self, question: str, text: str, factor: int = 1) -> Read:
        self._use(factor)
        started = time.monotonic()
        answer = self.sub.chat_split(question=SUB_PROMPT.format(question=question), context=text)
        return Read(str(answer or ""), self.count_tokens(text), time.monotonic() - started)


def make_reader(config: ExperimentConfig) -> Reader:
    """The reader a config asks for. Loads a model on the GPU when a budget is set."""
    compression = config.compression
    if not compression.in_process:
        client = LLMClient(model=config.models.sub_model, base_url=config.models.base_url)
        if not config.models.thinking:
            client.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        return HttpReader(client)
    from ..kvzip_backend import KVzipSubClient  # torch and kvpress, only when needed

    largest = max(compression.factors)
    sub = KVzipSubClient(
        model=config.models.sub_model,
        press_name="kvzip" if largest > 1 else "no_press",
        memory_budget=compression.budget_mb,
        memory_budget_unit="MB",
        max_context_tokens=config.largest_read_tokens,
        compression_factor=largest,
    )
    return KVReader(sub)


# ======================================================================================
# The REPL the root writes code into
# ======================================================================================
# The persistent Python REPL the root writes code into.
#
# Runs one cell at a time in a namespace that survives between turns, echoes a bare
# final expression like the interactive shell, turns exceptions into output instead
# of crashes, and stops a runaway cell after `timeout` seconds.
#
# A sub-call runs inside `outside()`: the cell timer is paused, so a slow read is never
# mistaken for a loop, and anything the sub-model's libraries print goes to the real
# terminal instead of into the output the root reads.


CODE_RE = re.compile(r"```(?:python|py|repl)?[ \t]*\n(.*?)```", re.DOTALL)
MIN_REARM_S = 1.0  # a resumed timer never fires sooner than this


class CellTimeout(Exception):
    pass


def _on_alarm(signum, frame):  # pragma: no cover - signal handler
    raise CellTimeout


def extract_code(reply: str) -> Optional[str]:
    """The first fenced code block of a reply, or None."""
    match = CODE_RE.search(reply or "")
    return match.group(1).strip() if match else None


class Repl:
    def __init__(self, namespace: dict[str, Any], timeout: Optional[float] = 60.0):
        self.namespace = namespace
        self.timeout = timeout
        self._timed = False
        self._stdout = sys.stdout  # the terminal, while a cell's prints are being captured

    def _can_time(self) -> bool:
        return (
            bool(self.timeout) and hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
        )

    @contextlib.contextmanager
    def outside(self) -> Iterator[None]:
        """For a sub-call: pause the cell timer and print to the terminal, then restore both."""
        remaining = signal.setitimer(signal.ITIMER_REAL, 0)[0] if self._timed else 0.0
        try:
            with contextlib.redirect_stdout(self._stdout):
                yield
        finally:
            if self._timed:
                signal.setitimer(signal.ITIMER_REAL, max(remaining, MIN_REARM_S))

    def run(self, code: str, output_chars: int) -> str:
        """Execute one cell; return what it printed (truncated in the middle if long)."""
        buffer = io.StringIO()
        self._stdout = sys.stdout
        timed = self._can_time()
        old_handler = signal.signal(signal.SIGALRM, _on_alarm) if timed else None
        try:
            with contextlib.redirect_stdout(buffer):
                tree = ast.parse(code)
                # Like the interactive shell: the value of a final bare expression is shown.
                last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                if timed:
                    self._timed = True
                    signal.setitimer(signal.ITIMER_REAL, self.timeout)
                exec(compile(tree, "<cell>", "exec"), self.namespace)  # noqa: S102
                if isinstance(last, ast.Expr):
                    value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), self.namespace)  # noqa: S307
                    if value is not None:
                        print(value if isinstance(value, str) else repr(value))
        except CellTimeout:
            buffer.write(f"\n[TIMEOUT] the cell ran longer than {self.timeout:.0f}s and was stopped.")
        except KeyboardInterrupt:
            raise
        except BaseException:  # SystemExit from exit() included: a cell failure, not the run's
            buffer.write("\n[ERROR]\n" + traceback.format_exc(limit=2))
        finally:
            if timed:
                self._timed = False
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
        output = buffer.getvalue()
        if len(output) > output_chars:
            half = output_chars // 2
            output = f"{output[:half]}\n...[{len(output) - output_chars} characters cut]...\n{output[-half:]}"
        return output if output.strip() else "[no output]"


# ======================================================================================
# The RLM loop
# ======================================================================================
# The RLM loop: the root writes code, the REPL runs it, until FINAL or a limit.
#
#     turn:  root reply -> first ```python block -> run it -> show its output
#     tools: context, [chunks], llm_query(question, text_or_chunks[, factor=F]), FINAL(answer)
#     ends:  final | max_steps | run_timeout | root_error
#
# Nothing here knows which options are on: ordering arrives as a chunk list (or
# none), compression as a Reader plus the factors and limits in `Limits`. Every
# sub-call leaves a CallRecord with its GPU / offloaded split and its cost.


@dataclass
class Result:
    answer: Optional[str]
    steps: int
    end_reason: str  # final | max_steps | run_timeout | root_error
    calls: list[CallRecord] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    # Each call's full sub-model answer (None if refused), for the end-of-question marks.
    read_answers: list[Optional[str]] = field(default_factory=list, repr=False)


@dataclass(frozen=True)
class Limits:
    max_steps: int = 30
    max_sub_calls: int = 40
    exec_timeout_s: float = 60.0
    run_timeout_s: float = 900.0
    output_chars: int = 4000
    read_tokens: int = 8192  # the most one read may carry, unless read_limits says otherwise
    budget_tokens: Optional[int] = None  # B: KV one read may keep on the GPU
    factors: tuple[int, ...] = (1,)  # factors llm_query accepts
    default_factor: int = 1
    read_limits: Optional[Mapping[int, int]] = None  # factor -> most tokens one read may carry
    offload: bool = False  # KV beyond B is offloaded (and costed) instead of refused
    gpu_token_cost: float = 0.0
    offloaded_token_cost: float = 100.0
    show_cost: bool = False

    def read_limit(self, factor: int) -> int:
        return (self.read_limits or {}).get(factor, self.read_tokens)


def is_context_overflow(exc: Exception) -> bool:
    """An OpenAI-compatible 400 saying the prompt is longer than the model's window."""
    message = str(exc).lower()
    return getattr(exc, "status_code", None) == 400 and ("context length" in message or "too long" in message)


class RLM:
    def __init__(self, root: Any, reader: Reader, limits: Limits, ordering: str = "none"):
        self.root = root
        self.reader = reader
        self.limits = limits
        self.ordering = ordering

    def run(
        self,
        document: str,
        question: str,
        chunks: Sequence[Hit] = (),
        answer_format: Optional[str] = None,
        list_answer: bool = False,
        chunk_chars: int = 0,
    ) -> Result:
        limits = self.limits
        started = time.monotonic()
        result = Result(answer=None, steps=0, end_reason="max_steps")
        repl = Repl({}, limits.exec_timeout_s)
        repl.namespace.update(self._tools(document, chunks, repl, result))
        system = self.system_prompt(question, len(document), len(chunks), chunk_chars, answer_format, list_answer)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": BEGIN}]
        for step in range(1, limits.max_steps + 1):
            if time.monotonic() - started > limits.run_timeout_s:
                result.end_reason = "run_timeout"
                break
            result.steps = step
            reply = self._root_reply(messages)
            if reply is None:
                result.end_reason = "root_error"
                break
            code = extract_code(reply)
            output = NO_CODE_BLOCK if code is None else repl.run(code, limits.output_chars)
            result.transcript.append({"step": step, "reply": reply, "output": output})
            if result.answer is not None:
                result.end_reason = "final"
                break
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": REPL_OUTPUT.format(output=output)},
            ]
        result.seconds = time.monotonic() - started
        result.calls = [
            self._mark(call, answer, result.answer) for call, answer in zip(result.calls, result.read_answers)
        ]
        return result

    def system_prompt(
        self,
        question: str,
        context_chars: int,
        n_chunks: int = 0,
        chunk_chars: int = 0,
        answer_format: Optional[str] = None,
        list_answer: bool = False,
    ) -> str:
        """The root's prompt under these limits (also what `check --show-prompt` prints)."""
        limits = self.limits
        return system_prompt(
            question=question,
            context_chars=context_chars,
            read_tokens=limits.read_limit(limits.default_factor),
            max_steps=limits.max_steps,
            output_chars=limits.output_chars,
            ordering=self.ordering,
            n_chunks=n_chunks,
            chunk_chars=chunk_chars,
            answer_format=answer_format,
            list_answer=list_answer,
            budget_tokens=limits.budget_tokens,
            factors=limits.factors,
            default_factor=limits.default_factor,
            offload_cap=limits.read_limit(max(limits.factors)) if limits.offload else None,
            cost=(limits.gpu_token_cost, limits.offloaded_token_cost) if limits.show_cost else None,
        )

    @staticmethod
    def _mark(call: CallRecord, read_answer: Optional[str], final: Optional[str]) -> CallRecord:
        """Once the question ends: was compression needed, and did the read feed the answer?"""
        if call.refused or read_answer is None:
            return call
        needed = None
        if call.factor > 1 and call.budget_tokens is not None:
            needed = call.read_tokens > call.budget_tokens  # would it NOT have fit uncompressed?
        return replace(call, compression_needed=needed, contributed=contributed(read_answer, final))

    def _root_reply(self, messages: list[dict]) -> Optional[str]:
        """One root completion. On a window overflow, drop the oldest turn and retry."""
        while True:
            try:
                return self.root.chat(messages)
            except Exception as exc:  # noqa: BLE001 - the client raises the SDK's errors
                # messages = [system, begin, (assistant, output)*]: keep the first two.
                if not is_context_overflow(exc) or len(messages) <= 3:
                    return None
                del messages[2:4]

    def _refusal(self, result: Result, factor: Any, tokens: int) -> str:
        limits = self.limits
        if sum(1 for c in result.calls if not c.refused) >= limits.max_sub_calls:
            return f"[REFUSED] the limit of {limits.max_sub_calls} llm_query calls is reached."
        if factor not in limits.factors:
            return f"[REFUSED] factor={factor!r} is not available; use one of {list(limits.factors)}."
        limit = limits.read_limit(factor)
        if tokens > limit:
            return (
                f"[REFUSED] that text is {tokens:,} tokens; one llm_query at factor {factor} reads at most "
                f"{limit:,}. Split it into smaller pieces."
            )
        return ""

    def _tools(self, document: str, chunks: Sequence[Hit], repl: Repl, result: Result) -> dict:
        limits, reader = self.limits, self.reader

        def llm_query(question: str, text: Any = "", factor: Any = None) -> str:
            factor = limits.default_factor if factor is None else factor
            n_chunks = None
            if isinstance(text, (list, tuple)):
                n_chunks = len(text)
                text = "\n\n".join(getattr(c, "text", str(c)) for c in text)
            text = str(text)
            tokens = reader.count_tokens(text)
            record = dict(
                step=result.steps,
                question=preview(question),
                chunks=n_chunks,
                read_tokens=tokens,
                budget_tokens=limits.budget_tokens,
            )
            refusal = self._refusal(result, factor, tokens)
            if refusal:
                limit = limits.read_limit(factor) if factor in limits.factors else 0
                factor_seen = factor if isinstance(factor, int) else 0
                result.calls.append(
                    CallRecord(
                        kept_tokens=0, limit_tokens=limit, factor=factor_seen, seconds=0.0, refused=refusal, **record
                    )
                )
                result.read_answers.append(None)
                return refusal
            placement = Placement.of(tokens, factor, limits.budget_tokens)
            cost = placement.cost(limits.gpu_token_cost, limits.offloaded_token_cost)
            with repl.outside():
                read = reader.read(str(question), text, factor)
            result.calls.append(
                CallRecord(
                    kept_tokens=placement.kept,
                    limit_tokens=limits.read_limit(factor),
                    factor=factor,
                    seconds=read.seconds,
                    gpu_tokens=placement.gpu,
                    offloaded_tokens=placement.offloaded,
                    cost=cost,
                    answer=preview(read.answer),
                    **record,
                )
            )
            result.read_answers.append(read.answer)
            if not limits.show_cost:
                return read.answer
            total = sum(c.cost for c in result.calls)
            line = COST_LINE.format(
                read=tokens, factor=factor, gpu=placement.gpu, offloaded=placement.offloaded, cost=cost, total=total
            )
            return f"{read.answer}\n{line}"

        def FINAL(answer: Any) -> str:  # noqa: N802 - the name the prompt uses
            result.answer = str(answer)
            return result.answer

        tools: dict[str, Any] = {"context": document, "llm_query": llm_query, "FINAL": FINAL}
        if chunks:
            tools["chunks"] = list(chunks)
        return tools
