# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export graph-order arms + vanilla / KVzip / base-RLM references on the SAME 55
LOFT-128k test questions, with token usage, for the dashboard's Structured tab.

Row alignment: our runs id test rows `loft:<t>:<i>` (i = 0..54 within the test
split). The 110-row runs (guards vanilla/RLM, KVzip CSVs) are dev 10 + test 100,
so test row i sits at position i + 10. Every pairing is verified on gold answers.

This ran on cat for the graph-order campaign (2026-09-10), from a checkout with
PYTHONPATH=., as:
    python evaluation/rlm/graph_order_export_dashboard.py <kvzip csv dir> <out.json>
On 2026-09-11 (lab storage policy) its inputs moved off $HOME, so GO and QA below
point at the NAS archive.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from evaluation.benchmarks.loaders import _load_loft
from evaluation.rlm.paired import compare, load_rows, per_row_scores

TASKS = ["nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k"]
MODEL = "Qwen_Qwen3-4B-Instruct-2507"
GO = Path("/mnt/nas/gautammahale/archive/graphorder/results")
QA = Path("/mnt/nas/gautammahale/archive/qagnostic/results")
GUARDS = Path("/mnt/nas/gautammahale/benchmark/evaluation/results/loft128k_guards")
KVZIP = Path(sys.argv[1])  # dir holding the published loft128k-qwen3-4b-instruct-*.csv
OUT = Path(sys.argv[2])
TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)


def gold_key(answers):
    if isinstance(answers, str):
        # KVzip CSVs store numpy-array reprs (`['a' 'b']`, no commas), which
        # literal_eval would concatenate into one string. Take the quoted items.
        answers = [a or b for a, b in re.findall(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", answers)]

    # Sanity key only (alignment is by unique question text): the reprs escape
    # characters like \xa0 literally, so compare letters and digits alone.
    def norm(s):
        return re.sub(r"\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|[^0-9A-Za-z]", "", str(s)).lower()

    return json.dumps(sorted(norm(a) for a in answers))


def records(run_dir):
    rows = {}
    for line in (run_dir / "checkpoint.jsonl").read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[str(r["id"])] = r
    return rows


def summarize(label, kind, task, scores, recs, extra=None):
    ids = sorted(scores, key=lambda i: int(i.rsplit(":", 1)[1]))
    n = len(ids)
    tok = [recs[i]["tokens"] for i in ids]
    row = dict(
        label=label,
        kind=kind,
        task=task,
        n=n,
        score=100 * sum(scores[i] for i in ids) / n if n else None,
        tokens_per_query=sum(tok) / n if n else None,
        total_tokens=sum(tok),
        latency_s=(
            (sum(recs[i]["latency_s"] for i in ids) / n)
            if n and all(recs[i].get("latency_s") is not None for i in ids)
            else None
        ),
        errors=sum(1 for r in recs.values() if r.get("error")),
        preds=[
            dict(
                id=i,
                answers=json.dumps(recs[i].get("answers"), ensure_ascii=False),
                prediction=recs[i].get("pred", ""),
                tokens=recs[i]["tokens"],
                latency_s=recs[i].get("latency_s"),
                score=scores[i],
            )
            for i in ids
        ],
    )
    for key in ("prompt_tokens", "completion_tokens"):
        if n and all(recs[i].get(key) is not None for i in ids):
            row[key + "_per_query"] = sum(recs[i][key] for i in ids) / n
    row.update(extra or {})
    return row


out: dict = {"tasks": {}, "paired": {}}
for task in TASKS:
    # Gold for test rows 0..54, from a complete 55-row run on the same split.
    ref_dir = next(QA.glob(f"loft__{task}__{MODEL}__structured__split-test__*sel-all*"))
    ref = records(ref_dir)
    gold = {str(r["id"]): gold_key(r["answers"]) for r in ref.values()}
    gold_lists = {str(r["id"]): list(r["answers"]) for r in ref.values()}
    assert len(gold) == 55, (task, len(gold))
    # Question text for test rows 0..54, in the loader's order (== our ids).
    loft = _load_loft(task)
    test = loft[loft["split"] == "test"].reset_index(drop=True)
    questions = [str(q).strip() for q in test["question"].iloc[:55]]
    for i in range(55):
        assert gold_key(list(test["answers"].iloc[i])) == gold[f"loft:{task}:{i}"], (task, "loader", i)
    metric = "coverage" if task.startswith(("qampari", "quest")) else "subspan_em"
    rows = []

    # Graph-order arms (possibly still running: report what is finished).
    arm_dirs = {}
    for arm, label in (
        ("source", "Source order · every chunk, 8k reader groups"),
        ("bm25-graph", "BM25-graph order · every chunk, 8k reader groups"),
    ):
        # qampari/quest slugs carry a __listfmt suffix.
        found = sorted(GO.glob(f"loft__{task}__{MODEL}__structured__split-test__order-{arm}*/checkpoint.jsonl"))
        if not found:
            continue
        assert len(found) == 1, found
        d = found[0].parent
        arm_dirs[arm] = d
        frame, _ = load_rows(d)
        name, scores = per_row_scores(frame, "loft")
        recs = records(d)
        for i in scores:
            assert gold_key(recs[i]["answers"]) == gold[i], (task, arm, i)
        ms = [recs[i].get("metrics") or {} for i in scores]
        rows.append(
            summarize(
                label,
                arm,
                task,
                scores,
                recs,
                dict(
                    metric=name,
                    map_calls=sum(m.get("map_calls", 0) for m in ms) / len(ms) if ms else None,
                    group_count=sum(m.get("group_count", 0) for m in ms) / len(ms) if ms else None,
                    root_peak_tokens=sum(m.get("reducer_input_tokens", 0) for m in ms) / len(ms) if ms else None,
                    complete=len(scores) == 55,
                ),
            )
        )
    if set(arm_dirs) == {"source", "bm25-graph"}:
        r = compare(arm_dirs["source"], arm_dirs["bm25-graph"], resamples=10000, seed=0, allow_subset=True)
        out["paired"][task] = {
            k: r[k]
            for k in (
                "metric",
                "n",
                "mean_a",
                "mean_b",
                "delta",
                "ci_lo",
                "ci_hi",
                "wins",
                "losses",
                "ties",
                "sign_p",
                "errors_a",
                "errors_b",
            )
        }

    # Vanilla and base RLM: the Aug-30 guards rerun, 110 rows -> test rows via +10.
    for arm, label in (
        ("vanilla", "Vanilla · whole document in one call"),
        ("rlm", "Base RLM · agentic REPL (Aug 30 guards rerun)"),
    ):
        d = GUARDS / f"loft__{task}__{MODEL}__{arm}"
        frame, _ = load_rows(d)
        recs110 = records(d)
        mapped = {f"loft:{task}:{i}": recs110[f"loft:{task}:{i + 10}"] for i in range(55)}
        for i, r in mapped.items():
            assert gold_key(r["answers"]) == gold[i], (task, arm, i)
        frame = frame[frame["id"].isin([f"loft:{task}:{i + 10}" for i in range(55)])].copy()
        frame["id"] = frame["id"].map(lambda x: f"loft:{task}:{int(x.rsplit(':', 1)[1]) - 10}")
        name, scores = per_row_scores(frame, "loft")
        peaks = None
        rows.append(
            summarize(label, arm, task, scores, mapped, dict(metric=name, root_peak_tokens=peaks, complete=True))
        )

    # KVzip only (no RLM): published CSVs, rows in dev10+test100 order.
    for budget in ("no-compression", "256", "512", "1024", "2048", "4096"):
        f = KVZIP / f"loft128k-qwen3-4b-instruct-{task.replace('_128k', '')}-128k-{budget}.csv"
        if not f.exists():
            continue
        csv_rows = pd.read_csv(f)
        assert len(csv_rows) == 110, f
        # KVzip CSV row order is NOT the loader's: match each of our test rows by
        # its question text (unique among the CSV's test rows), then check gold.
        test_csv = csv_rows[csv_rows["split"] == "test"]
        recs, frame_rows, positions = {}, [], []
        for i in range(55):
            hits = [j for j, q in zip(test_csv.index, test_csv["question"]) if questions[i] in str(q)]
            assert len(hits) == 1, (task, budget, i, len(hits))
            positions.append(hits[0])
            src = csv_rows.loc[hits[0]]
            assert gold_key(src["answers"]) == gold[f"loft:{task}:{i}"], (task, budget, i)
            pred = "" if pd.isna(src["predicted_answer"]) else str(src["predicted_answer"])
            prompt = int(src["context_tokens"]) + len(TOK(str(src["question"]))["input_ids"])
            completion = len(TOK(pred)["input_ids"])
            rid = f"loft:{task}:{i}"
            recs[rid] = dict(
                answers=gold_lists[rid],
                pred=pred,
                tokens=prompt + completion,
                prompt_tokens=prompt,
                completion_tokens=completion,
                latency_s=None,
            )
            frame_rows.append(
                dict(
                    id=rid,
                    predicted_answer=pred,
                    answers=gold_lists[rid],
                    answer_prefix=src.get("answer_prefix"),
                    task=task,
                )
            )
        name, scores = per_row_scores(pd.DataFrame(frame_rows), "loft")
        retained = float(csv_rows.loc[positions, "retained_context_tokens"].mean())
        label = (
            "KVzip harness · no compression (no RLM)"
            if budget == "no-compression"
            else f"KVzip only · {budget}-token KV budget (no RLM)"
        )
        rows.append(
            summarize(
                label,
                "kvzip",
                task,
                scores,
                recs,
                dict(
                    metric=name,
                    root_peak_tokens=retained,
                    budget=None if budget == "no-compression" else int(budget),
                    complete=True,
                ),
            )
        )
    out["tasks"][task] = rows
    for r in rows:
        print(
            f"{task:14s} {r['label'][:48]:48s} n={r['n']:3d} {r['metric']:10s} {r['score']:6.2f}  "
            f"tok/q={r['tokens_per_query']:9.0f}"
        )

OUT.write_text(json.dumps(out, ensure_ascii=False, default=str))
print("wrote", OUT)
