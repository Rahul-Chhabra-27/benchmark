# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in call tracing: python -P evaluation/rlm/diagnostic.py --trace-dir DIR -- <benchmark args>.

Only explicit model messages/responses and REPL source are captured, never arbitrary
locals. Profiling adds elapsed-time overhead; it does not alter requests or guards.
Generator resumes count as entries, as defined by Python's profiling protocol.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import marshal
import runpy
import sys
import threading
import time
from pathlib import Path


class DiagnosticTrace:
    def __init__(self, directory, project_root=None):
        self.directory = Path(directory)
        self.root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.active = {}
        self.functions = {}
        self.counts: collections.Counter[str] = collections.Counter()
        self.edges: collections.Counter[tuple[str, str]] = collections.Counter()
        self.classes = {}
        self.code_keys = {}
        self.serial = 0

    def _kind(self, frame):
        code = frame.f_code
        if code not in self.classes:
            filename = code.co_filename
            module = frame.f_globals.get("__name__", "")
            if filename == __file__:
                kind = "tracer"
            elif filename.startswith(str(self.root) + "/") and "/." not in filename[len(str(self.root)) :]:
                kind = "project"
            elif filename.startswith("<") and (filename in ("<rlm>", "<string>") or module.startswith("evaluation.")):
                kind = "generated"
            else:
                kind = "library"
            self.classes[code] = kind
        return self.classes[code]

    def _emit(self, event):
        self.stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _profile(self, frame, event, arg):
        if event not in ("call", "return"):
            return
        key = id(frame)
        if event == "return":
            active = self.active.pop(key, None)
            if active:
                record = dict(event="return", call=active[0], elapsed_seconds=time.perf_counter() - active[2])
                if active[3] == "chat" and isinstance(arg, str):
                    record["response"] = arg
                if active[3] == "exec" and isinstance(arg, tuple):
                    record["observation"], record["full_output"] = arg
                self._emit(record)
            return
        kind = self._kind(frame)
        if kind == "tracer":
            return
        caller = frame.f_back
        boundary = caller is not None and self._kind(caller) in ("project", "generated")
        module = frame.f_globals.get("__name__", "")
        name = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        http = (module.startswith("httpx.") and name in ("Client.send", "Client.request")) or (
            module.startswith("openai.") and name.endswith("Completions.create")
        )
        if kind == "library" and not (boundary or http):
            return
        parent = caller
        while parent is not None and id(parent) not in self.active:
            parent = parent.f_back
        parent_entry = self.active.get(id(parent)) if parent is not None else None
        code = frame.f_code
        if code not in self.code_keys:
            digest = hashlib.sha256(marshal.dumps(code)).hexdigest()[:16]
            self.code_keys[code] = f"{module}:{name}@{code.co_filename}:{code.co_firstlineno}#{digest}"
        function = self.code_keys[code]
        self.functions.setdefault(
            function, dict(module=module, name=name, file=code.co_filename, line=code.co_firstlineno, kind=kind)
        )
        self.counts[function] += 1
        if parent_entry:
            self.edges[(parent_entry[1], function)] += 1
        self.serial += 1
        chat = module == "evaluation.rlm.client" and name == "LLMClient.chat"
        record = dict(
            event="call",
            call=self.serial,
            parent=parent_entry[0] if parent_entry else None,
            function=function,
            thread=threading.get_ident(),
            caller_line=caller.f_lineno if caller else None,
        )
        if chat:
            record["messages"] = frame.f_locals["messages"]
            record["model"] = frame.f_locals["self"].model
        is_exec = module == "evaluation.rlm.rlm" and name == "RLM._exec"
        if is_exec:
            record["code"] = frame.f_locals["code"]
        self.active[key] = (self.serial, function, time.perf_counter(), "chat" if chat else "exec" if is_exec else None)
        self._emit(record)

    def __enter__(self):
        if sys.getprofile() is not None or threading.getprofile() is not None:
            raise RuntimeError("Refusing to replace an existing profiler")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / "calls.jsonl").open("w", buffering=1)
        self.started = time.perf_counter()
        sys.setprofile(self._profile)
        threading.setprofile(self._profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)
        threading.setprofile(None)
        self.stream.close()
        summary = dict(
            elapsed_seconds=time.perf_counter() - self.started,
            entry_semantics="Python call events, including generator resumes; elapsed time includes profiler overhead",
            functions=[dict(key=key, calls=self.counts[key], **value) for key, value in self.functions.items()],
            edges=[dict(parent=a, child=b, calls=n) for (a, b), n in self.edges.items()],
            open_calls=[entry[0] for entry in self.active.values()],
        )
        (self.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    benchmark_args = args.benchmark_args
    if benchmark_args[:1] == ["--"]:
        benchmark_args = benchmark_args[1:]
    old_argv = sys.argv
    try:
        sys.argv = ["evaluation.rlm.run_benchmark", *benchmark_args]
        with DiagnosticTrace(args.trace_dir):
            runpy.run_module("evaluation.rlm.run_benchmark", run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
