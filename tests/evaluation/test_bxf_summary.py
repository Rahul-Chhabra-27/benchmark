# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The B x F summary: one row per cell, and a cell that missed its B or F fails the command."""

import json

from evaluation.rlm import bxf_summary
from tests.evaluation.test_rlm_order_gate import KV_CONFIG, kv_record, write_run

CELL = dict(
    KV_CONFIG,
    chunk_order="bm25-query",
    dataset="loft",
    data_dir="nq_128k",
    sub_kv_memory_budget=512.0,
    sub_kv_memory_budget_unit="MB",
)
RUNTIME = dict(realized_compression_factor=4.0, max_sub_retained_context_tokens=22, sub_pressed_call_fraction=1.0)


def cell(path, row=None, **config):
    write_run(path, [row or kv_record()], dict(CELL, **config))
    (path / "metrics.json").write_text(json.dumps({"subspan_em": 0.5, "runtime": RUNTIME}))


def test_a_good_cell_is_one_passing_row(tmp_path):
    cell(tmp_path / "bxf_512MB_x4" / "run")
    (row,) = bxf_summary.collect(tmp_path)
    assert (row["B"], row["F"], row["arm"], row["N"], row["B_tok"]) == ("512MB", "4", "bm25-query", 13888, 3472)
    assert (row["score"], row["kept_max"], row["check"]) == (0.5, 22, "PASS")


def test_http_runs_are_not_cells(tmp_path):
    write_run(tmp_path / "http", [kv_record()], dict(CELL, sub_backend="http"))
    assert bxf_summary.collect(tmp_path) == []


def test_a_cell_that_missed_its_factor_fails_the_command(tmp_path, capsys, monkeypatch):
    cell(tmp_path / "run", row=kv_record(min_compression_ratio=0.5))
    monkeypatch.setattr("sys.argv", ["bxf_summary", str(tmp_path)])
    assert bxf_summary.main() == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "0 of 1 cells passed" in out


def test_an_unfinished_cell_fails(tmp_path):
    cell(tmp_path / "run")
    (tmp_path / "run" / "metrics.json").unlink()
    (row,) = bxf_summary.collect(tmp_path)
    assert row["check"] == "FAIL" and any("not finished" in p for p in row["problems"])
