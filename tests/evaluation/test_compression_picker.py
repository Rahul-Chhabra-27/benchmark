# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""compression_picker: pick F per question up front, run the groups, merge and compare."""

import json

import pandas as pd
import pytest
import yaml

from evaluation.rlm.compression_picker import __main__ as cli
from evaluation.rlm.compression_picker.assemble import compare, picked_scores
from evaluation.rlm.compression_picker.budget import budget_tokens
from evaluation.rlm.compression_picker.calibrate import best_per_task, calibrate
from evaluation.rlm.compression_picker.features import QuestionFeatures, question_features
from evaluation.rlm.compression_picker.pickers import FixedPicker, PerTaskPicker, SpreadPicker, make_picker
from evaluation.rlm.compression_picker.plan import read_plan, write_plan
from evaluation.rlm.compression_picker.results import load_cell, score_records

LOFT_WRAPPER = (
    "====== Now let's start! ======\nBased on the documents above, can you answer the following "
    "query? Print out the TITLE and ID of the documents you use to answer.\nquery: {q}\n"
)


def chunk(i: int, extra: str = "") -> str:
    """One 2000-char grid chunk: a unique word, optional evidence, filler."""
    text = f"marker{i} {extra} "
    filler = " ".join(f"filler{i}w{k}" for k in range(400))
    return (text + filler)[:2000]


def document(evidence: dict[int, str], n: int = 20) -> str:
    return "".join(chunk(i, evidence.get(i, "")) for i in range(n))


def features(scores, n_chunks=100, chunk_tokens=500.0):
    return QuestionFeatures(query="q", n_chunks=n_chunks, chunk_tokens=chunk_tokens, scores=tuple(scores))


# --------------------------------------------------------------------------- budget


def test_budget_tokens_matches_the_lane_and_kvpress():
    assert budget_tokens(1024) == 6944  # int(1024e6 / 147456), as rlm_order_lane.sh computes
    assert budget_tokens(512) == 3472
    with pytest.raises(ValueError):
        budget_tokens(0)


# --------------------------------------------------------------------------- features


def test_features_strip_the_loft_wrapper_and_find_concentrated_evidence():
    doc = document({3: "quasar quasar"})
    f = question_features(LOFT_WRAPPER.format(q="Where is the quasar?"), doc)
    assert f.query == "Where is the quasar?"
    assert f.n_chunks == 20 and f.chunk_tokens == 500.0
    assert f.evidence_chunks(0.5) == 1


def test_features_count_evenly_spread_evidence():
    doc = document({1: "nebula", 4: "nebula", 7: "nebula", 9: "nebula"})
    assert question_features(LOFT_WRAPPER.format(q="nebula?"), doc).evidence_chunks(0.5) == 4


def test_a_question_matching_nothing_counts_as_spread_over_everything():
    assert features([], n_chunks=230).evidence_chunks(0.5) == 230


def test_ratio_selects_chunks_relative_to_the_best():
    f = features([10, 9, 6, 4, 1])
    assert [f.evidence_chunks(r) for r in (1.0, 0.9, 0.5, 0.1)] == [1, 2, 3, 5]
    with pytest.raises(ValueError):
        f.evidence_chunks(0)


# --------------------------------------------------------------------------- pickers


@pytest.mark.parametrize(
    "n_evidence, factor",
    [(1, 1), (2, 1), (3, 2), (4, 2), (8, 4), (16, 8), (60, 8)],  # B = 1000 tokens, 500 per chunk
)
def test_spread_picks_the_least_compression_that_holds_the_evidence(n_evidence, factor):
    picker = SpreadPicker(budget_tokens=1000, ratio=0.5)
    assert picker.pick(features([5.0] * n_evidence), "nq_128k").factor == factor


def test_spread_respects_a_restricted_factor_menu():
    picker = SpreadPicker(budget_tokens=1000, ratio=0.5, factors=(1, 4))
    assert picker.pick(features([5.0] * 3), "nq_128k").factor == 4


def test_per_task_accepts_bare_names_and_refuses_unknown_tasks():
    picker = PerTaskPicker({"hotpotqa": 1, "quest_128k": 8})
    assert picker.pick(features([1]), "hotpotqa_128k").factor == 1
    assert picker.pick(features([1]), "quest_128k").factor == 8
    with pytest.raises(KeyError):
        picker.pick(features([1]), "nq_128k")


def test_make_picker_parses_every_form(tmp_path):
    assert make_picker("fixed:4", 1000) == FixedPicker(4)
    assert make_picker("spread", 1000) == SpreadPicker(1000, 0.5)
    assert make_picker("spread:0.3", 1000).ratio == 0.3
    assert make_picker("per-task:nq=8,quest=1", 1000).table == {"nq": 8, "quest": 1}
    calibration = tmp_path / "cal.json"
    calibration.write_text(json.dumps({"chosen": {"picker": "spread", "ratio": 0.4}}))
    assert make_picker(f"calibrated:{calibration}", 1000) == SpreadPicker(1000, 0.4)
    with pytest.raises(ValueError):
        make_picker("fixed:3", 1000)  # not in the factor menu
    with pytest.raises(ValueError):
        make_picker("random", 1000)


# --------------------------------------------------------------------------- plan


def test_plan_writes_one_ids_file_per_task_and_factor_and_replans_cleanly(tmp_path):
    rows = [
        {"id": "a", "task": "nq_128k", "factor": 1, "reason": "", "query": ""},
        {"id": "b", "task": "nq_128k", "factor": 4, "reason": "", "query": ""},
        {"id": "c", "task": "nq_128k", "factor": 1, "reason": "", "query": ""},
    ]
    write_plan(rows, tmp_path, {"budget_mb": 1024})
    assert (tmp_path / "ids" / "nq_128k__x1.txt").read_text().split() == ["a", "c"]
    assert (tmp_path / "ids" / "nq_128k__x4.txt").read_text().split() == ["b"]
    meta, back = read_plan(tmp_path)
    assert back == rows and meta["groups"] == {"nq_128k__x1": 2, "nq_128k__x4": 1}
    write_plan(rows[:1], tmp_path, {"budget_mb": 1024})
    assert sorted(p.name for p in (tmp_path / "ids").iterdir()) == ["nq_128k__x1.txt"]


# --------------------------------------------------------------------------- results


CONFIG = dict(
    backend="rlm",
    mode="rlm",
    chunk_order="bm25-graph",
    sub_kv_memory_budget=1024.0,
    sub_kv_memory_budget_unit="MB",
)


def record(id_, pred, task="hotpotqa_128k", answers=("Paris",), **extra):
    return dict(id=id_, task=task, answers=list(answers), answer_prefix="Final Answer: ", pred=pred, **extra)


def write_cell(root, factor, records, name="run", **config):
    run = root / f"bxf_1024MB_x{factor}" / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(dict(CONFIG, compression_factor=float(factor), **config)))
    (run / "checkpoint.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def test_load_cell_finds_runs_by_config_and_skips_errors(tmp_path):
    write_cell(tmp_path, 2, [record("a", "Paris"), record("b", None, error="timeout")])
    write_cell(tmp_path, 2, [record("z", "Paris")], name="other_arm", chunk_order="bm25-query")
    write_cell(tmp_path, 4, [record("a", "Rome")])
    assert set(load_cell(tmp_path, 1024, 2)) == {"a"}
    assert load_cell(tmp_path, 1024, 4)["a"]["pred"] == "Rome"
    assert load_cell(tmp_path, 512, 2) == {}


def test_load_cell_refuses_one_id_answered_by_two_runs_of_a_cell(tmp_path):
    write_cell(tmp_path, 2, [record("a", "Paris")], name="one")
    write_cell(tmp_path, 2, [record("a", "Paris")], name="two")
    with pytest.raises(ValueError, match="answered by both"):
        load_cell(tmp_path, 1024, 2)


def test_score_records_uses_the_official_metric_per_task():
    scored = score_records(
        [
            record("a", "Paris"),
            record("b", "London"),
            record("c", "Final Answer: ['x', 'y']", task="qampari_128k", answers=("x", "y", "z", "w")),
        ]
    ).set_index("id")
    assert scored.loc["a", ["metric", "score"]].tolist() == ["f1", 1.0]
    assert scored.loc["b", "score"] == 0.0
    assert scored.loc["c", ["metric", "score"]].tolist() == ["coverage", 0.5]


# --------------------------------------------------------------------------- assemble


def cells(scores: dict[int, dict[str, float]], task="nq_128k"):
    return {
        f: pd.DataFrame(
            [{"id": i, "task": task, "metric": "f1", "score": s, "predicted_answer": ""} for i, s in by_id.items()]
        ).set_index("id")
        for f, by_id in scores.items()
    }


def test_compare_scores_each_question_at_its_own_factor_against_fixed_and_oracle():
    plan = [
        {"id": "a", "task": "nq_128k", "factor": 1, "reason": ""},
        {"id": "b", "task": "nq_128k", "factor": 4, "reason": ""},
    ]
    runs = cells({1: {"a": 1.0, "b": 0.0}, 4: {"a": 0.0, "b": 1.0}})
    table = compare(picked_scores(plan, runs), runs).iloc[0]
    assert (table["picked"], table["fixed x1"], table["fixed x4"], table["oracle"]) == (100.0, 50.0, 50.0, 100.0)
    assert table["picks"] == "x1:1 x4:1" and table["compared"] == 2


def test_a_question_without_an_answer_is_missing_not_zero():
    plan = [
        {"id": "a", "task": "nq_128k", "factor": 1, "reason": ""},
        {"id": "b", "task": "nq_128k", "factor": 4, "reason": ""},
    ]
    picked = picked_scores(plan, cells({1: {"a": 1.0}}))
    assert picked.set_index("id")["score"].isna().tolist() == [False, True]
    row = compare(picked).iloc[0]
    assert (row["planned"], row["compared"], row["picked"]) == (2, 1, 100.0)


# --------------------------------------------------------------------------- calibrate


def test_calibrate_prefers_spread_when_it_routes_questions_to_their_best_factor():
    # "narrow" questions are right only at x1, "wide" ones only at x8.
    runs = cells({1: {"n1": 1, "n2": 1, "w1": 0, "w2": 0}, 8: {"n1": 0, "n2": 0, "w1": 1, "w2": 1}})
    feats = {i: features([5.0] * (1 if i.startswith("n") else 12)) for i in ("n1", "n2", "w1", "w2")}
    result = calibrate(runs, feats, budget_tokens=1000, ratios=(0.5,))
    by_name = {c["name"]: c["score"] for c in result["candidates"]}
    assert by_name == {"fixed1": 50.0, "fixed8": 50.0, "spread0.5": 100.0, "per-task": 50.0}
    assert result["chosen"] == {"picker": "spread", "ratio": 0.5}


def test_best_per_task_breaks_ties_toward_less_compression():
    runs = cells({1: {"a": 1, "b": 0}, 4: {"a": 0, "b": 1}})
    from evaluation.rlm.compression_picker.calibrate import score_matrix

    assert best_per_task(score_matrix(runs)) == {"nq_128k": 1}


# --------------------------------------------------------------------------- end to end


def test_plan_then_assemble_through_the_command_line(tmp_path, monkeypatch, capsys):
    narrow = document({2: "quasar quasar"})
    wide = document({i: "nebula" for i in range(0, 20, 2)})
    examples = [
        {
            "id": "loft:hotpotqa_128k:0",
            "task": "hotpotqa_128k",
            "question": LOFT_WRAPPER.format(q="quasar?"),
            "context": narrow,
        },
        {
            "id": "loft:hotpotqa_128k:1",
            "task": "hotpotqa_128k",
            "question": LOFT_WRAPPER.format(q="nebula?"),
            "context": wide,
        },
    ]
    monkeypatch.setattr(
        "evaluation.rlm.compression_picker.plan.load_examples", lambda dataset, task, limit, split: iter(examples)
    )
    plan_dir = tmp_path / "plan"
    # B = 1000 tokens; 500 tokens per chunk: 1 evidence chunk -> x1, 10 -> x8.
    cli.main(["plan", "--tasks", "hotpotqa", "--budget-mb", "147.5", "--picker", "spread:0.5", "--out", str(plan_dir)])
    _, rows = read_plan(plan_dir)
    assert [r["factor"] for r in rows] == [1, 8]
    assert sorted(p.name for p in (plan_dir / "ids").iterdir()) == ["hotpotqa_128k__x1.txt", "hotpotqa_128k__x8.txt"]

    # Fixed-F baselines over both rows; the picked answers are the right ones by design.
    base = tmp_path / "baselines"
    config = dict(sub_kv_memory_budget=147.5)
    write_cell(base, 1, [record(examples[0]["id"], "Paris"), record(examples[1]["id"], "Rome")], **config)
    write_cell(base, 8, [record(examples[0]["id"], "Rome"), record(examples[1]["id"], "Paris")], **config)
    cli.main(
        ["assemble", "--plan", str(plan_dir), "--results", str(base), "--baselines", str(base), "--factors", "1", "8"]
    )
    table = pd.read_csv(plan_dir / "result" / "comparison.csv").iloc[0]
    assert (table["picked"], table["fixed x1"], table["fixed x8"], table["oracle"]) == (100.0, 50.0, 50.0, 100.0)
    assert "2/2 answered" in capsys.readouterr().out


def test_calibrate_through_the_command_line_writes_a_usable_picker(tmp_path, monkeypatch, capsys):
    narrow, wide = document({2: "quasar quasar"}), document({i: "nebula" for i in range(0, 20, 2)})
    dev = [
        {"id": "loft:nq_128k:0", "task": "nq_128k", "question": LOFT_WRAPPER.format(q="quasar?"), "context": narrow},
        {"id": "loft:nq_128k:1", "task": "nq_128k", "question": LOFT_WRAPPER.format(q="nebula?"), "context": wide},
    ]
    monkeypatch.setattr(
        "evaluation.rlm.compression_picker.__main__.load_examples",
        lambda dataset, task, limit, split: iter(dev) if split == "dev" else iter(()),
    )
    runs, config = tmp_path / "dev", dict(sub_kv_memory_budget=147.5)
    # The narrow question is right only uncompressed, the wide one only at x8.
    write_cell(
        runs, 1, [record(dev[0]["id"], "Paris", task="nq_128k"), record(dev[1]["id"], "Rome", task="nq_128k")], **config
    )
    write_cell(
        runs, 8, [record(dev[0]["id"], "Rome", task="nq_128k"), record(dev[1]["id"], "Paris", task="nq_128k")], **config
    )
    out = tmp_path / "cal.json"
    cli.main(
        [
            "calibrate",
            "--tasks",
            "nq",
            "--budget-mb",
            "147.5",
            "--factors",
            "1",
            "8",
            "--ratios",
            "0.5",
            "--dev-results",
            str(runs),
            "--out",
            str(out),
        ]
    )
    assert json.loads(out.read_text())["chosen"] == {"picker": "spread", "ratio": 0.5}
    assert make_picker(f"calibrated:{out}", 1000, (1, 8)) == SpreadPicker(1000, 0.5, (1, 8))
    assert "chosen: spread0.5" in capsys.readouterr().out
