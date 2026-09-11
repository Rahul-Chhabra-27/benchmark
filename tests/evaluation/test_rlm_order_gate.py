# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Campaign acceptance must reject the failures observed in the first live pilot."""

import copy
import hashlib
import json

import pytest
import yaml

from evaluation.rlm import rlm_order_gate as gate
from evaluation.rlm.rlm import CHUNK_ORDER_REVISION, WORKED_SESSION_RANKED


def record(i=0):
    return dict(
        id=f"loft:nq_128k:{i}",
        pred="answer",
        finished=True,
        end_reason="final_called",
        metrics=dict(
            chunks_total=2,
            document_sha256="doc",
            chunk_view_sha256="view",
            chunk_payload_calls=1,
            reader_input_tokens=[100],
            reader_served_prompt_tokens=[100],
            organization_cache_hit=True,
        ),
    )


def write_run(path, rows, config=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "transcripts").mkdir(exist_ok=True)
    (path / "checkpoint.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        t = dict(
            question=f"question {r['id']}",
            answers=["gold"],
            pred=r.get("pred"),
            end_reason=r["end_reason"],
            transcript=[dict(code="ans = llm_query(q, chunks[:5])", observation="answer")],
        )
        (path / "transcripts" / f"{r['id']}.json").write_text(json.dumps(t))
    if config:
        (path / "config.yaml").write_text(yaml.safe_dump(config))
    (path / "metrics.json").write_text("{}")


CONFIG = dict(chunk_call_tokens=8192, chunk_order_revision=CHUNK_ORDER_REVISION)


def test_ranked_example_begins_with_a_real_read():
    first_cell = WORKED_SESSION_RANKED.split("```python", 1)[1].split("```", 1)[0].strip()
    assert first_cell.startswith("ans = llm_query(")
    assert "chunks[:5]" in first_cell
    assert "print(c)" not in WORKED_SESSION_RANKED
    assert "Hit(" not in WORKED_SESSION_RANKED


def test_valid_reader_run_passes(tmp_path):
    write_run(tmp_path, [record()])
    assert gate.check_run(tmp_path, CONFIG, 1)[0] == []


@pytest.mark.parametrize(
    "failure",
    [
        "missing_counts",
        "mismatch",
        "over_budget",
        "missing_hash",
        "old_revision",
        "error",
        "corrupt_checkpoint",
        "missing_transcript",
        "missing_finished",
    ],
)
def test_live_failure_modes_are_rejected(tmp_path, failure):
    row, config = record(), dict(CONFIG)
    m = row["metrics"]
    if failure == "missing_counts":
        m.update(reader_input_tokens=[None], reader_served_prompt_tokens=[None])
    elif failure == "mismatch":
        m["reader_served_prompt_tokens"] = [101]
    elif failure == "over_budget":
        m.update(reader_input_tokens=[8193], reader_served_prompt_tokens=[8193])
    elif failure == "missing_hash":
        del m["chunk_view_sha256"]
    elif failure == "old_revision":
        config["chunk_order_revision"] -= 1
    elif failure == "error":
        row["error"] = "server failed"
    elif failure == "missing_finished":
        del row["finished"]
    write_run(tmp_path, [row])
    if failure == "corrupt_checkpoint":
        with (tmp_path / "checkpoint.jsonl").open("a") as f:
            f.write('{"id":')
    elif failure == "missing_transcript":
        (tmp_path / "transcripts" / f"{row['id']}.json").unlink()
    assert gate.check_run(tmp_path, config, 1)[0]


def test_an_earlier_campaign_is_gated_at_the_revision_it_ran(tmp_path):
    write_run(tmp_path, [record()])
    config = dict(CONFIG, chunk_order_revision=CHUNK_ORDER_REVISION - 1)
    assert gate.check_run(tmp_path, config, 1)[0]
    assert gate.check_run(tmp_path, config, 1, revision=CHUNK_ORDER_REVISION - 1)[0] == []


@pytest.mark.parametrize("notice", gate.NOTICES)
def test_limit_and_server_notices_fail(tmp_path, notice):
    row = record()
    write_run(tmp_path, [row])
    path = tmp_path / "transcripts" / f"{row['id']}.json"
    t = json.loads(path.read_text())
    t["transcript"][0]["observation"] = notice
    path.write_text(json.dumps(t))
    assert any(notice in p for p in gate.check_run(tmp_path, CONFIG, 1)[0])


def test_abstention_after_a_read_is_valid_and_counted(tmp_path):
    row = record()
    row.update(end_reason="abstained", pred="")
    write_run(tmp_path, [row])
    problems, info = gate.check_run(tmp_path, CONFIG, 1)
    assert not problems
    assert info["behaviour"]["abstentions"] == 1


def test_one_guard_ending_is_listed_for_review_not_failed(tmp_path):
    rows = [record(i) for i in range(3)]
    rows[0].update(finished=False, end_reason="max_steps", pred=None)
    write_run(tmp_path, rows)
    problems, info = gate.check_run(tmp_path, CONFIG, 3)
    assert not problems
    assert info["review"] == ["loft:nq_128k:0:max_steps"] and info["behaviour"]["guard endings"] == 1


def test_review_rows_warn_past_a_tenth_and_fail_past_a_quarter(tmp_path):
    rows = [record(i) for i in range(3)]
    rows[0].update(finished=False, end_reason="max_steps", pred=None)
    rows[1]["metrics"].update(chunk_payload_calls=0, reader_input_tokens=[], reader_served_prompt_tokens=[])
    write_run(tmp_path, rows)
    problems, info = gate.check_run(tmp_path, CONFIG, 3)
    assert any("need review" in p for p in problems)
    assert info["review"] == ["loft:nq_128k:0:max_steps", "loft:nq_128k:1:no_reads"]
    # Fifty-five rows: five pass quietly, six warn, fourteen fail.
    many = [record(i) for i in range(55)]
    for count, fails, warns in ((5, False, False), (6, False, True), (14, True, False)):
        for row in many[:count]:
            row.update(end_reason="repetition_broken")
        write_run(tmp_path / "many", many)
        problems, info = gate.check_run(tmp_path / "many", CONFIG, 55)
        assert bool(problems) is fails and bool(info["warnings"]) is warns


def test_pilot_ids_match_first_comparator_rows_not_all_55(tmp_path):
    ours, structured = tmp_path / "ours", tmp_path / "structured"
    rows = [record(i) for i in range(3)]
    s_rows = [record(i) for i in range(55)]
    for r in s_rows:
        r["metrics"]["organization_sha256"] = "doc"
    write_run(ours, rows)
    write_run(structured, s_rows)
    by_id = {r["id"]: r for r in rows}
    assert not gate.check_comparator(ours, by_id, structured, True, 3)
    assert gate.check_comparator(ours, by_id, structured, False, 3)
    assert gate.check_comparator(ours, {r["id"]: r for r in rows[1:]}, structured, True, 3)
    tpath = ours / "transcripts" / f"{rows[0]['id']}.json"
    t = json.loads(tpath.read_text())
    t["question"] = "different question, same gold"
    tpath.write_text(json.dumps(t))
    assert any("question" in p for p in gate.check_comparator(ours, by_id, structured, True, 3))


def test_graph_order_must_match_artifact_not_just_document():
    artifact = dict(
        document_sha256="doc", config=dict(order="bm25-graph"), spans=[[0, 2000], [2000, 4000]], order=[1, 0]
    )
    row = record()
    digest = hashlib.sha256(json.dumps([[2000, 4000], [0, 2000]]).encode()).hexdigest()
    row["metrics"]["chunk_view_sha256"] = digest
    rows = {row["id"]: row}
    assert not gate.check_graph_artifact(rows, [artifact])
    changed = copy.deepcopy(artifact)
    changed["order"] = [0, 1]
    assert gate.check_graph_artifact(rows, [changed])
    row["metrics"]["organization_cache_hit"] = False
    assert gate.check_graph_artifact(rows, [artifact])


def test_nq_alone_cannot_pass_complete_campaign(tmp_path):
    for arm in gate.ARMS:
        config = dict(CONFIG, dataset="loft", data_dir="nq_128k", mode="rlm", split="test", chunk_order=arm)
        write_run(tmp_path / arm, [record()], config)
    problems = gate.check_campaign(tmp_path, "loft", 1, None, True)
    assert sum("missing run" in p for p in problems) == 8
    assert any("--structured-results" in p for p in problems)


def test_missing_comparator_is_a_failure(tmp_path):
    for arm in gate.ARMS:
        config = dict(CONFIG, dataset="loft", data_dir="nq_128k", mode="rlm", split="test", chunk_order=arm)
        write_run(tmp_path / "results" / arm, [record()], config)
    problems = gate.check_campaign(tmp_path / "results", "loft", 1, tmp_path / "absent", True)
    assert sum("missing structured comparator" in p for p in problems) == 2


def complete_pilot(tmp_path, recorded_org=None):
    """A complete one-row LOFT pilot and its comparator: (results, structured, organization)."""
    results, structured, org = (tmp_path / name for name in ("results", "structured", "organization"))
    org.mkdir()
    for subset in gate.LOFT_SUBSETS:
        spans = [[0, 2000], [2000, 4000]]
        artifact = dict(document_sha256=subset, config=dict(order="bm25-graph"), spans=spans, order=[1, 0])
        (org / f"{subset}.json").write_text(json.dumps(artifact))
        digest = hashlib.sha256(json.dumps(spans[::-1]).encode()).hexdigest()
        for arm in gate.ARMS:
            row = record()
            row["id"] = f"loft:{subset}:0"
            row["metrics"].update(document_sha256=subset, chunk_view_sha256=digest)
            config = dict(CONFIG, dataset="loft", data_dir=subset, mode="rlm", split="test", chunk_order=arm)
            write_run(results / f"{subset}-{arm}", [row], config)
        for arm in ("source", "bm25-graph"):
            rows = [record(i) for i in range(2)]
            for i, row in enumerate(rows):
                row["id"] = f"loft:{subset}:{i}"
                row["metrics"]["organization_sha256"] = subset
            config = dict(
                dataset="loft", data_dir=subset, structured_order=arm, organization_dir=recorded_org or str(org)
            )
            write_run(structured / f"{subset}-{arm}", rows, config)
    return results, structured, org


def test_complete_pilot_passes_with_both_comparators_and_cached_orders(tmp_path):
    results, structured, _ = complete_pilot(tmp_path)
    assert not gate.check_campaign(results, "loft", 1, structured, True)


def test_archived_graph_artifacts_are_found_through_the_override(tmp_path):
    results, structured, org = complete_pilot(tmp_path, recorded_org=str(tmp_path / "moved-away"))
    assert any("structured graph artifact" in p for p in gate.check_campaign(results, "loft", 1, structured, True))
    assert not gate.check_campaign(results, "loft", 1, structured, True, org)


def test_a_reproduction_without_the_structured_campaign_can_opt_out_of_it(tmp_path):
    results, _, _ = complete_pilot(tmp_path)
    assert any("--structured-results" in p for p in gate.check_campaign(results, "loft", 1, None, True))
    assert not gate.check_campaign(results, "loft", 1, None, True, require_comparator=False)
