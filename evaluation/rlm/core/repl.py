# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The persistent Python REPL the root writes code into.

Runs one cell at a time in a namespace that survives between turns, echoes a bare
final expression like the interactive shell, turns exceptions into output instead
of crashes, and stops a runaway cell after `timeout` seconds.

A sub-call runs inside `outside()`: the cell timer is paused, so a slow read is never
mistaken for a loop, and anything the sub-model's libraries print goes to the real
terminal instead of into the output the root reads.
"""

from __future__ import annotations

import ast
import contextlib
import io
import re
import signal
import sys
import threading
import traceback
from typing import Any, Iterator, Optional

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
