# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""hard_questions: which questions does an RLM need more than ordering for?"""

import json

import pytest
import yaml

from evaluation.rlm.analysis import hard_questions as cli
from evaluation.rlm.analysis.hard_questions import classify, describe, discover, score_matrix, summarize
from evaluation.rlm.configurable_rlm.config import config_from_dict
from evaluation.rlm.configurable_rlm.run import run_experiment

TASK = "hotpotqa_128k"
VANILLA = dict(backend="rlm", mode="vanilla", dataset="loft")
ORDERED = dict(backend="rlm", mode="rlm", dataset="loft", chunk_order="bm25-graph", sub_backend="http")
COMPRESSED = dict(
    ORDERED, sub_backend="kvzip", compression_factor=2.0, sub_kv_memory_budget=1024.0, sub_kv_memory_budget_unit="MB"
)


def write_run(root, name, config, answers_right):
    """One finished run: question i is answered right when answers_right[i]."""
    run = root / name / TASK
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(dict(config, data_dir=TASK)))
    lines = []
    for i, right in enumerate(answers_right):
        lines.append(
            json.dumps(
                {
                    "id": f"q{i}",
                    "task": TASK,
                    "answers": ["Paris"],
                    "answer_prefix": "Final Answer: ",
                    "pred": "Paris" if right else "London",
                }
            )
        )
    (run / "checkpoint.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture
def runs(tmp_path):
    #                     q0     q1     q2     q3     q4
    write_run(tmp_path, "vanilla", VANILLA, [True, True, True, False, False])
    write_run(tmp_path, "ordered", ORDERED, [True, False, False, True, False])
    write_run(tmp_path, "x2", COMPRESSED, [False, False, True, False, False])
    return tmp_path


def test_arms_are_labelled_from_their_configs():
    assert describe(VANILLA) == ("vanilla", "vanilla", "loft")
    assert describe(ORDERED) == ("rlm+bm25-graph", "ordered", "loft")
    assert describe(COMPRESSED) == ("rlm+bm25-graph 1024MB x2", "compressed", "loft")
    assert describe(dict(ORDERED, chunk_order=None)) == ("rlm", "rlm", "loft")
    assert describe(dict(VANILLA, mode="structured"))[1] == "other"
    minimal = {"name": "graph_x4", "ordering": "bm25-graph", "compression": {"factor": 4}, "data": {"dataset": "loft"}}
    assert describe(minimal) == ("configurable_rlm:graph_x4", "compressed", "loft")
    assert describe({"backend": "kvpress"}) is None


def test_every_question_lands_in_one_category(runs):
    arms = discover([runs])
    table = classify(score_matrix(arms), arms)
    category = {i: c for (_, i), c in table["category"].items()}
    assert category == {
        "q0": "both_solve",  # vanilla and ordered both right
        "q1": "vanilla_only",  # only vanilla right
        "q2": "compression_rescues",  # ordered wrong, the compressed arm right
        "q3": "rlm_only",  # ordered right, vanilla wrong
        "q4": "nobody",
    }
    counts = summarize(table).loc[TASK]
    assert (counts["validation_set"], counts["compared"]) == (2, 5)


def test_only_questions_both_sides_answered_are_compared(tmp_path):
    write_run(tmp_path, "vanilla", VANILLA, [True, True])
    write_run(tmp_path, "ordered", ORDERED, [False])  # q1 never answered by the RLM
    arms = discover([tmp_path])
    assert [i for _, i in classify(score_matrix(arms), arms).index] == ["q0"]


def test_a_comparison_needs_vanilla_and_an_uncompressed_rlm(tmp_path):
    write_run(tmp_path, "ordered", ORDERED, [True])
    arms = discover([tmp_path])
    with pytest.raises(ValueError, match="need a vanilla arm"):
        classify(score_matrix(arms), arms)


def test_command_writes_the_validation_ids_and_a_report(runs, capsys):
    out = runs / "hard"
    cli.main(["--results", str(runs), "--out", str(out)])
    assert (out / "ids" / f"{TASK}__validation_set.txt").read_text().split() == ["q1", "q2"]
    assert (out / "ids" / f"{TASK}__vanilla_only.txt").read_text().split() == ["q1"]
    assert (out / "ids" / f"{TASK}__compression_rescues.txt").read_text().split() == ["q2"]
    assert "validation_set" in (out / "summary.md").read_text()
    assert "rlm+bm25-graph 1024MB x2" in capsys.readouterr().out


def test_configurable_rlm_runs_only_the_ids_in_a_file(tmp_path):
    from tests.evaluation.test_configurable_rlm import ScriptedRoot, code, examples, plain

    ids = tmp_path / "hard.txt"
    ids.write_text("loft:hotpotqa_128k:1\nloft:nq_128k:7\n")  # another task's id is ignored
    config = config_from_dict(
        {"name": "t", "data": {"tasks": ["hotpotqa"], "ids_file": str(ids)}, "output_dir": str(tmp_path)}
    )
    root = ScriptedRoot([code("FINAL('Paris')")])
    run_experiment(config, root=root, reader=plain(), load=lambda *a: examples(), log=lambda _: None)
    done = [json.loads(line)["id"] for line in (tmp_path / "t" / TASK / "checkpoint.jsonl").read_text().splitlines()]
    assert done == ["loft:hotpotqa_128k:1"]
    with pytest.raises(ValueError, match="set only one"):
        config_from_dict({"name": "t", "data": {"tasks": ["nq"], "ids_file": str(ids), "limit": 3}})
