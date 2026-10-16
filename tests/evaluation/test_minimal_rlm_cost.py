# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""minimal_rlm: per-read factor choice, offloading, cost, and the end-of-question marks."""

import json

import pytest

from evaluation.rlm.minimal_rlm.config import config_from_dict
from evaluation.rlm.minimal_rlm.loop import RLM, Limits
from evaluation.rlm.minimal_rlm.run import make_limits, run_experiment
from evaluation.rlm.minimal_rlm.stats import Placement, contributed, summarize
from tests.evaluation.test_minimal_rlm import FakeReader, ScriptedRoot, code, examples

B = 100  # KV tokens one read may keep; FakeReader counts len(text) // 4 tokens


def limits(**extra):
    base = dict(budget_tokens=B, factors=(1, 2, 4, 8), read_limits={1: B, 2: 2 * B, 4: 4 * B, 8: 8 * B})
    return Limits(**{**base, **extra})


def run(cells, final="FINAL(a)", reader=None, **extra):
    """Run one question whose root executes `cells` (each a code string), then `final`."""
    root = ScriptedRoot([code(c) for c in cells] + [code(final)])
    return RLM(root, reader or FakeReader("Paris"), limits(**extra)).run("x" * 40_000, "Capital of France?"), root


# --------------------------------------------------------------------------- placement and cost


def test_placement_splits_kept_kv_between_gpu_and_cpu():
    assert Placement.of(300, 2, B) == Placement(kept=150, gpu=100, offloaded=50)
    assert Placement.of(300, 4, B) == Placement(kept=75, gpu=75, offloaded=0)
    assert Placement.of(300, 1, None) == Placement(kept=300, gpu=300, offloaded=0)  # no budget, no offload
    assert Placement(150, 100, 50).cost(gpu_token=1, offloaded_token=100) == 100 + 5000


# --------------------------------------------------------------------------- config


def cost_config(**compression):
    return config_from_dict(
        {
            "name": "c",
            "data": {"tasks": ["hotpotqa"]},
            "compression": {"budget_mb": 1024, "factor": 1, "choices": [1, 2, 4, 8], **compression},
            "cost": {"show_root": True},
        }
    )


def test_read_limits_per_factor_with_and_without_offloading():
    plain = cost_config()
    assert plain.compression.factors == (1, 2, 4, 8)
    assert [plain.read_limit(f) for f in (1, 2, 8)] == [6944, 13888, 55552]
    offloading = cost_config(offload=True)
    assert {offloading.read_limit(f) for f in (1, 2, 8)} == {55552}  # as wide as the widest compressed read
    assert cost_config(offload=True, max_read_tokens=20000).read_limit(1) == 20000
    built = make_limits(offloading)
    assert built.offload and built.show_cost and built.read_limit(2) == 55552


@pytest.mark.parametrize(
    "compression, message",
    [
        ({"choices": [1, 3]}, "must be in"),
        ({"factor": 4, "choices": [1, 2]}, "must be one of compression.choices"),
        ({"budget_mb": None}, "needs compression.budget_mb"),
        ({"budget_mb": None, "choices": [1], "offload": True}, "offload needs compression.budget_mb"),
    ],
)
def test_bad_compression_settings_are_refused(compression, message):
    with pytest.raises(ValueError, match=message):
        cost_config(**compression)


# --------------------------------------------------------------------------- the root's choices


def test_the_root_picks_a_factor_per_read():
    result, _ = run(["a = llm_query('q', context[:1600], factor=4)", "b = llm_query('q', context[:400])"])
    assert [(c.factor, c.read_tokens, c.kept_tokens) for c in result.calls] == [(4, 400, 100), (1, 100, 100)]


def test_an_unavailable_factor_or_an_oversized_read_is_refused():
    result, _ = run(["print(llm_query('q', 'abcd', factor=3))", "print(llm_query('q', context[:2000], factor=2))"])
    assert "factor=3 is not available" in result.calls[0].refused
    assert "at most 200" in result.calls[1].refused  # 500 tokens > B x 2


def test_offloading_admits_a_wider_read_and_costs_it():
    result, _ = run(
        ["a = llm_query('q', context[:2000], factor=2)"], offload=True, read_limits={f: 800 for f in (1, 2, 4, 8)}
    )
    call = result.calls[0]
    assert not call.refused
    assert (call.read_tokens, call.kept_tokens, call.gpu_tokens, call.offloaded_tokens) == (500, 250, 100, 150)
    assert call.cost == 150 * 100  # gpu tokens cost 0 by default


def test_the_root_sees_each_reads_cost_only_when_asked():
    cells = ["print(llm_query('q', context[:2000], factor=2))", "print(llm_query('q', context[:2000], factor=2))"]
    caps = {f: 800 for f in (1, 2, 4, 8)}
    _, shown = run(cells, offload=True, read_limits=caps, show_cost=True)
    observation = shown.sent[2][-1]["content"]
    assert "[cost: read 500 tokens at x2 -> 100 on GPU, 150 offloaded; 15000 (total 30000)]" in observation
    _, hidden = run(cells, offload=True, read_limits=caps)
    assert "[cost:" not in hidden.sent[2][-1]["content"]


def test_prompt_mentions_memory_offloading_and_cost_only_when_on():
    plain = RLM(None, None, Limits()).system_prompt("q", 1000)
    assert "Memory:" not in plain and "offloaded" not in plain and "Cost:" not in plain
    full = RLM(None, None, limits(offload=True, show_cost=True)).system_prompt("q", 1000)
    assert "factor=F" in full and "offloaded to CPU" in full and "100 per" in full


# --------------------------------------------------------------------------- end-of-question marks


def test_reads_are_marked_needed_and_contributing():
    reader = FakeReader("The capital is Paris")
    cells = [
        "a = llm_query('q', context[:200], factor=2)",  # 50 tokens: fits B uncompressed -> compression unneeded
        "b = llm_query('q', context[:1600], factor=4)",  # 400 tokens > B -> compression needed
    ]
    result, _ = run(cells, final="FINAL('Paris')", reader=reader)
    assert [c.compression_needed for c in result.calls] == [False, True]
    assert all(c.contributed for c in result.calls)
    summary = summarize(result.calls)
    assert summary["compressed_unneeded"] == 1 and summary["factors"] == {"x2": 1, "x4": 1}
    assert contributed("nothing relevant here", "Paris") is False


def test_run_metrics_report_cost_and_factor_use(tmp_path):
    config = config_from_dict(
        {
            "name": "c",
            "data": {"tasks": ["hotpotqa"]},
            "output_dir": str(tmp_path),
            "compression": {"budget_mb": 1.5, "factor": 1, "choices": [1, 2], "offload": True, "max_read_tokens": 1000},
        }
    )
    script = [code("a = llm_query('q', context[:200], factor=2)"), code("FINAL(a)")] * 2
    metrics = run_experiment(
        config, root=ScriptedRoot(script), reader=FakeReader("Paris"), load=lambda *a: examples(), log=lambda _: None
    )["hotpotqa_128k"]["runtime"]
    assert metrics["factors"] == {"x2": 2} and metrics["cost_mean"] > 0 and metrics["offloaded_tokens_mean"] > 0
    saved = (tmp_path / "c" / "hotpotqa_128k" / "config.yaml").read_text()
    assert "choices:" in saved  # the resolved config (with its tuple of choices) round-trips through YAML
    record = json.loads((tmp_path / "c" / "hotpotqa_128k" / "checkpoint.jsonl").read_text().splitlines()[0])
    assert record["calls"][0]["offloaded_tokens"] > 0 and record["reads"]["cost"] > 0
