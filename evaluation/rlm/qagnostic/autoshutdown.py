#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retire only the captured elk RLM server after its campaign exits.

A query-agnostic campaign host script (2026-09-09), kept as it ran; see README.md.
"""
import argparse
import datetime
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

BASE = Path("/mnt/home/gautammahale/qagnostic")
STATE = BASE / "logs/autoshutdown-state.json"
MANIFEST = BASE / "autoshutdown-manifest.json"


def process(pid):
    try:
        p = Path("/proc") / str(pid)
        stat = (p / "stat").read_text().rsplit(")", 1)[1].split()
        if p.stat().st_uid != os.getuid() or stat[0] == "Z":
            return None
        return dict(
            pid=pid,
            start=stat[19],
            ppid=int(stat[1]),
            pgid=int(stat[2]),
            cmd=(p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"),
        )
    except (FileNotFoundError, ProcessLookupError):
        return None


def snapshot():
    return {int(p.name): row for p in Path("/proc").iterdir() if p.name.isdigit() and (row := process(int(p.name)))}


def matches(row, identity):
    return row is not None and row["start"] == identity["start"]


def descendants(table, roots):
    selected = set(roots) & table.keys()
    while True:
        new = {pid for pid, row in table.items() if row["ppid"] in selected} - selected
        if not new:
            return selected
        selected.update(new)


def busy(table, manifest):
    roots = [r["pid"] for r in manifest["lanes"] if matches(table.get(r["pid"]), r)]
    selected = descendants(table, roots)
    # Also catch surviving workers and any newly launched campaign lanes.
    for pid, row in table.items():
        cmd = row["cmd"]
        if (
            "/qagnostic/lane.sh " in cmd
            or "/qagnostic/notes_lane.sh " in cmd
            or (" -m evaluation.rlm." in cmd and ("localhost:8001" in cmd or "127.0.0.1:8001" in cmd))
        ):
            selected.add(pid)
    selected.discard(os.getpid())
    return sorted(selected)


def status(phase, **kwargs):
    data = dict(time=datetime.datetime.now().astimezone().isoformat(), watcher_pid=os.getpid(), phase=phase, **kwargs)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(STATE)
    print(json.dumps(data), flush=True)


def signal_exact(identity, sig):
    # Opening the pidfd before rechecking start time prevents PID-reuse races.
    try:
        fd = os.pidfd_open(identity["pid"])
    except ProcessLookupError:
        return
    try:
        if matches(process(identity["pid"]), identity):
            signal.pidfd_send_signal(fd, sig)
    finally:
        os.close(fd)


def alive(identities):
    return [r for r in identities if matches(process(r["pid"]), r)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    assert os.uname().nodename == "elk"
    assert hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
    manifest = json.loads(MANIFEST.read_text())
    assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == manifest["boot_id"]
    server = manifest["server"]
    assert "vllm serve Qwen/Qwen3-4B-Instruct-2507" in server["cmd"]
    assert "--port 8001 " in server["cmd"]
    if args.check:
        table = snapshot()
        print(
            json.dumps(dict(server_alive=matches(table.get(server["pid"]), server), waiting_pids=busy(table, manifest)))
        )
        return
    lock = (BASE / "logs/autoshutdown.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tracked = {r["pid"]: r for r in manifest["server_tree"]}
    quiet_since = None
    previous = None
    while True:
        table = snapshot()
        roots = [pid for pid, row in tracked.items() if matches(table.get(pid), row)]
        for pid in descendants(table, roots):
            tracked[pid] = table[pid]
        pending = busy(table, manifest)
        if pending:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        if pending != previous:
            status("waiting", waiting_pids=pending, server_pid=server["pid"])
            previous = pending
        if quiet_since is not None and time.monotonic() - quiet_since >= 20:
            break
        time.sleep(10)
    # One final fresh check immediately before touching the server.
    if busy(snapshot(), manifest):
        raise RuntimeError("A campaign worker appeared during shutdown check")
    status("stopping_server", server_pid=server["pid"])
    signal_exact(server, signal.SIGTERM)
    for sig, timeout in [(None, 60), (signal.SIGTERM, 30), (signal.SIGKILL, 15)]:
        remaining = alive(list(tracked.values()))
        if sig is not None:
            for row in remaining:
                signal_exact(row, sig)
        deadline = time.monotonic() + timeout
        while alive(list(tracked.values())) and time.monotonic() < deadline:
            time.sleep(1)
        if not alive(list(tracked.values())):
            break
    if alive(list(tracked.values())):
        raise RuntimeError("Captured server processes did not exit")
    # GPU cleanup can lag process exit; do not claim release until verified.
    for attempt in range(30):
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        allocations = [line for line in output.splitlines() if int(line.split(",")[0].strip()) in tracked]
        if not allocations:
            status("complete", server_processes_exited=True, gpu_memory_released=True, server_pids=sorted(tracked))
            return
        time.sleep(2)
    raise RuntimeError("GPU release not confirmed within 60 seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("error", error=repr(exc))
        raise
