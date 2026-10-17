# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The base RLM: the RLM idea and nothing more, in the order a reader needs it.

    prompts   every word the root and sub model see
    records   one CallRecord per sub-call
    reading   the Reader interface, and PlainReader (uncompressed reads over http)
    features  the Feature hooks other files plug into
    repl      the persistent Python REPL the root writes into
    loop      the RLM loop and its llm_query / FINAL tools

Everything else is added ON TOP of this base, never inside it, through two seams:
  a Reader decides how llm_query reads -- compression.py is one (KV budget, factor
    choice, offloading, cost);
  Features add tools, prompt lines and checks -- ordering.py (the `chunks` list) is one.
Our original, fully tuned RLM is not built from this base: `loop: tuned` runs it as it
is (tuned.py).
This file imports none of them.
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
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Iterator, Optional, Protocol, Sequence

from evaluation.rlm.shared.client import LLMClient

# ======================================================================================
# Prompts: every word the models are shown
# ======================================================================================
# A short base prompt. A feature adds only its own lines (tools in the tool list, notes
# after the instructions); the reader supplies the sentence on how much one read may
# carry and any notes of its own.


BASE_PROMPT = """\
You answer a question about a long document that you cannot read directly. The document
is stored in a Python REPL. Each turn, reply with ONE ```python code block; its printed
output is shown to you in the next message (at most {output_chars} characters).

In the REPL:
  context                       the whole document, a string of {context_chars:,} characters
{feature_tools}  llm_query(question, text)     a sub-model reads `text` and answers `question`. It sees
                                nothing else. {read_help}
  FINAL(answer)                 end with your answer. Call it only on something you read.

Search `context` with Python (find, regular expressions, slicing), read the relevant parts
with llm_query, then call FINAL. Variables persist between turns. You have {max_steps} turns.
{feature_notes}{reader_notes}
Question:
{question}
{answer_format}"""

READ_LIMIT_HELP = (
    "One call reads at most {tokens:,} tokens\n"
    "                                (about {chars:,} characters); a longer text is refused."
)

ANSWER_FORMAT = "Answer format: {answer_format}"
LIST_ANSWER_FORMAT = 'Answer format: {answer_format} followed by a list, e.g. ["first", "second", "third"]'

NO_CODE_BLOCK = "Your reply had no ```python code block. Reply with exactly one code block."
REPL_OUTPUT = "REPL output:\n{output}"
BEGIN = "Begin."

SUB_PROMPT = "Answer the question using ONLY the text below. Be concise and factual.\n\nQuestion: {question}"

CHARS_PER_TOKEN = 4  # only to tell the root roughly how many characters fit in one read


def read_limit_help(tokens: int) -> str:
    """The sentence in the llm_query line that says how much one read may carry."""
    return READ_LIMIT_HELP.format(tokens=tokens, chars=tokens * CHARS_PER_TOKEN)


def system_prompt(
    *,
    question: str,
    context_chars: int,
    read_help: str,
    max_steps: int,
    output_chars: int,
    answer_format: Optional[str] = None,
    list_answer: bool = False,
    feature_tools: str = "",
    feature_notes: str = "",
    reader_notes: str = "",
) -> str:
    """The root's system prompt for one question."""
    if answer_format:
        fmt = (LIST_ANSWER_FORMAT if list_answer else ANSWER_FORMAT).format(answer_format=answer_format.strip())
    else:
        fmt = ""
    return BASE_PROMPT.format(
        output_chars=output_chars,
        context_chars=context_chars,
        feature_tools=feature_tools,
        read_help=read_help,
        max_steps=max_steps,
        feature_notes=feature_notes,
        reader_notes=reader_notes,
        question=question.strip(),
        answer_format=fmt,
    )


# ======================================================================================
# Records: one CallRecord per sub-call
# ======================================================================================
# What the loop knows about a read: which turn, the question, how many chunks, whether
# it was refused, how long it took, what came back, and -- once the question ends --
# whether it fed the final answer. Everything a reader adds (tokens, factor, cost, ...)
# rides along in `details`.


PREVIEW_CHARS = 200


@dataclass(frozen=True)
class CallRecord:
    step: int
    question: str
    chunks: Optional[int]  # chunks passed, when the root read a chunk list
    seconds: float
    refused: str = ""  # why the read was not made, else ""
    answer: str = ""
    contributed: Optional[bool] = None  # filled when the question ends
    details: dict = field(default_factory=dict)  # what the reader reports about this read

    def to_dict(self) -> dict:
        out = asdict(self)
        return {**{k: v for k, v in out.items() if k != "details"}, **out["details"]}


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


def summarize_calls(calls: Iterable[CallRecord]) -> dict:
    """What the loop can say about a question's reads, whatever the reader."""
    calls = list(calls)
    made = [c for c in calls if not c.refused]
    return {
        "sub_calls": len(made),
        "refused_calls": len(calls) - len(made),
        "contributing_calls": sum(1 for c in made if c.contributed),
        "read_seconds": round(sum(c.seconds for c in made), 3),
    }


# ======================================================================================
# Reading: the Reader interface, and plain reads
# ======================================================================================
# A Reader is everything about HOW llm_query reads: how much one read may carry, what
# the root is told about it, the model call, and what it reports. The base RLM ships one,
# PlainReader. compression.py adds another; the loop cannot tell them apart.


@dataclass(frozen=True)
class Read:
    answer: str  # the sub-model's answer
    details: dict = field(default_factory=dict)  # what this read did, for its CallRecord
    note: str = ""  # appended for the root after the answer (e.g. a cost line), never scored
    refused: str = ""  # non-empty: the read was not made, and this is what the root is told
    seconds: float = 0.0


class Reader(Protocol):
    read_help: str  # the sentence on how much one read may carry (in the llm_query line)
    notes: str  # further prompt lines, or ""

    def begin(self) -> None: ...  # a new question starts

    def read(self, question: str, text: str, **options: Any) -> Read: ...

    def summarize(self, details: list[dict]) -> dict: ...  # one question's reads -> numbers


def summarize_sizes(details: list[dict]) -> dict:
    """Read sizes against their limit: shared by every reader that reports them."""
    sized = [d for d in details if "read_tokens" in d]
    out: dict = {"read_tokens": sum(d["read_tokens"] for d in sized)}
    if sized:
        out["read_tokens_mean"] = out["read_tokens"] / len(sized)
        # How much of what one read may carry the root actually used: near 0 means it
        # reads tiny slices and a bigger read limit buys it nothing.
        out["read_fill_mean"] = sum(d["read_tokens"] / d["limit_tokens"] for d in sized) / len(sized)
    return out


class PlainReader:
    """Uncompressed reads through the OpenAI-compatible server, up to `limit_tokens` each."""

    notes = ""

    def __init__(self, client: Any, limit_tokens: int):
        self.client = client
        self.limit_tokens = limit_tokens
        self.read_help = read_limit_help(limit_tokens)

    def begin(self) -> None:
        pass

    def count_tokens(self, text: str) -> int:
        return math.ceil(len(text) / CHARS_PER_TOKEN)

    def read(self, question: str, text: str, **options: Any) -> Read:
        if options:
            refusal = f"[REFUSED] llm_query takes no {sorted(options)} here: call llm_query(question, text)."
            return Read("", refused=refusal)
        tokens = self.count_tokens(text)
        details = {"read_tokens": tokens, "limit_tokens": self.limit_tokens}
        if tokens > self.limit_tokens:
            refusal = (
                f"[REFUSED] that text is {tokens:,} tokens; one llm_query reads at most "
                f"{self.limit_tokens:,}. Split it into smaller pieces."
            )
            return Read("", details, refused=refusal)
        started = time.monotonic()
        answer = self.client.chat([{"role": "user", "content": f"{SUB_PROMPT.format(question=question)}\n\n{text}"}])
        return Read(str(answer or ""), details, seconds=time.monotonic() - started)

    def summarize(self, details: list[dict]) -> dict:
        return summarize_sizes(details)


def plain_reader(model: str, base_url: str, limit_tokens: int, thinking: bool = False) -> PlainReader:
    client = LLMClient(model=model, base_url=base_url)
    if not thinking:
        client.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    return PlainReader(client, limit_tokens)


# ======================================================================================
# Features: the hooks other files plug into
# ======================================================================================
# A Feature is something added on top of the base RLM. Every hook is optional and the
# loop calls them in order, knowing nothing about what a feature does:
#
#     prepare(document, question)   a new question starts
#     tools(result)                 extra names in the REPL (e.g. `chunks`, FINAL_NONE)
#     tool_help() / notes()         lines added to the prompt's tool list / after it
#     check_final(answer, code, result)   "" to accept a FINAL, or a rejection message
#     after_turn(output, result)    a note for the root's next message ("" for none)
#     summary(result)               numbers for the question's summary
#
# A feature may end the question itself by setting result.done and result.ending.


class Feature:
    def prepare(self, document: str, question: str) -> None:
        pass

    def tools(self, result: "Result") -> dict:
        return {}

    def tool_help(self) -> str:
        return ""

    def notes(self) -> str:
        return ""

    def check_final(self, answer: str, code: str, result: "Result") -> str:
        return ""

    def after_turn(self, output: str, result: "Result") -> str:
        return ""

    def summary(self, result: "Result") -> dict:
        return {}


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
# The root writes code, the REPL runs it, until FINAL or a limit.
#
#     turn:  root reply -> first ```python block -> run it -> show its output
#     tools: context, llm_query(question, text_or_chunks, **options), FINAL(answer), + features'
#     ends:  final | max_steps | run_timeout | root_error | whatever a feature ends with


@dataclass
class Result:
    answer: Optional[str]
    steps: int
    end_reason: str  # final | max_steps | run_timeout | root_error | (a feature's ending)
    calls: list[CallRecord] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    summary: dict = field(default_factory=dict)  # the question's reads (and features), summarised
    done: bool = False  # set by FINAL, or by a feature that ends the question
    ending: str = ""  # why it is done
    # Each call's full sub-model answer (None if refused), for the end-of-question marks.
    read_answers: list[Optional[str]] = field(default_factory=list, repr=False)


@dataclass(frozen=True)
class Limits:
    max_steps: int = 30
    max_sub_calls: int = 40
    exec_timeout_s: float = 60.0
    run_timeout_s: float = 900.0
    output_chars: int = 4000


def is_context_overflow(exc: Exception) -> bool:
    """An OpenAI-compatible 400 saying the prompt is longer than the model's window."""
    message = str(exc).lower()
    return getattr(exc, "status_code", None) == 400 and ("context length" in message or "too long" in message)


class RLM:
    def __init__(self, root: Any, reader: Reader, limits: Limits, features: Sequence[Feature] = ()):
        self.root = root
        self.reader = reader
        self.limits = limits
        self.features = list(features)

    def system_prompt(
        self,
        question: str,
        context_chars: int,
        answer_format: Optional[str] = None,
        list_answer: bool = False,
    ) -> str:
        """The root's prompt for one question (also what `check` prints)."""
        return system_prompt(
            question=question,
            context_chars=context_chars,
            read_help=self.reader.read_help,
            max_steps=self.limits.max_steps,
            output_chars=self.limits.output_chars,
            answer_format=answer_format,
            list_answer=list_answer,
            feature_tools="".join(f.tool_help() for f in self.features),
            feature_notes="".join(f.notes() for f in self.features),
            reader_notes=self.reader.notes,
        )

    def run(
        self,
        document: str,
        question: str,
        answer_format: Optional[str] = None,
        list_answer: bool = False,
    ) -> Result:
        limits = self.limits
        started = time.monotonic()
        result = Result(answer=None, steps=0, end_reason="max_steps")
        self.reader.begin()
        for feature in self.features:
            feature.prepare(document, question)
        repl = Repl({}, limits.exec_timeout_s)
        repl.namespace.update(self._tools(document, repl, result))
        for feature in self.features:
            repl.namespace.update(feature.tools(result))
        system = self.system_prompt(question, len(document), answer_format, list_answer)
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
            # Recorded before the features look: `printed` is what the code itself printed,
            # `output` what the root is shown (with any rejection or note a feature adds).
            turn = {"step": step, "reply": reply, "output": output, "printed": output}
            result.transcript.append(turn)
            if result.done and result.ending == "final":
                rejections = [m for f in self.features if (m := f.check_final(result.answer or "", code or "", result))]
                if rejections:
                    result.answer, result.done, result.ending = None, False, ""
                    turn["output"] += "\n" + "\n".join(rejections)
            notes = [n for f in self.features if (n := f.after_turn(turn["output"], result))]
            turn["output"] += "".join("\n" + n for n in notes)
            output = turn["output"]
            if result.done:
                result.end_reason = result.ending
                break
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": REPL_OUTPUT.format(output=output)},
            ]
        result.seconds = time.monotonic() - started
        result.calls = [
            c if answer is None else replace(c, contributed=contributed(answer, result.answer))
            for c, answer in zip(result.calls, result.read_answers)
        ]
        made = [c.details for c in result.calls if not c.refused]
        result.summary = {**summarize_calls(result.calls), **self.reader.summarize(made)}
        for feature in self.features:
            result.summary.update(feature.summary(result))
        return result

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

    def _tools(self, document: str, repl: Repl, result: Result) -> dict:
        limits, reader = self.limits, self.reader

        def llm_query(question: str, text: Any = "", **options: Any) -> str:
            n_chunks = None
            if isinstance(text, (list, tuple)):
                n_chunks = len(text)
                text = "\n\n".join(getattr(c, "text", str(c)) for c in text)
            if sum(1 for c in result.calls if not c.refused) >= limits.max_sub_calls:
                read = Read("", refused=f"[REFUSED] the limit of {limits.max_sub_calls} llm_query calls is reached.")
            else:
                with repl.outside():
                    read = reader.read(str(question), str(text), **options)
            result.calls.append(
                CallRecord(
                    step=result.steps,
                    question=preview(question),
                    chunks=n_chunks,
                    seconds=read.seconds,
                    refused=read.refused,
                    answer=preview(read.answer),
                    details=read.details,
                )
            )
            result.read_answers.append(None if read.refused else read.answer)
            if read.refused:
                return read.refused
            return f"{read.answer}\n{read.note}" if read.note else read.answer

        def FINAL(answer: Any) -> None:  # noqa: N802 - the name the prompt uses
            # Returns nothing, so the REPL never echoes the answer back as if it were output.
            result.answer, result.done, result.ending = str(answer), True, "final"

        return {"context": document, "llm_query": llm_query, "FINAL": FINAL}
