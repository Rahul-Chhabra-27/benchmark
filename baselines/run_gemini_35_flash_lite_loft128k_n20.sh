#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Parallel recursive-llm LOFT-128k eval: 20 test rows x 5 RAG subsets
# using Gemini 3.5 Flash Lite (one process per subset).
set -euo pipefail

REPO="${REPO:-/data/aditya_desai/agentic-longcontext}"
if [ -f "$REPO/.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO/.env"
elif [ -f "$HOME/agentic-longcontext/.env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/agentic-longcontext/.env"
fi

PY="$REPO/baselines/recursive-llm/.venv/bin/python"
RUNNER="$REPO/baselines/run_benchmark.py"
MODEL="gemini/gemini-3.5-flash-lite"
OUT="$REPO/evaluation/results/rlm/recursive-llm/loft128k-gemini-3.5-flash-lite-n20"
LOGDIR="$OUT/logs"
TASKS=(nq_128k hotpotqa_128k musique_128k qampari_128k quest_128k)

mkdir -p "$LOGDIR"

cat > "$OUT/README.md" <<EOF
# recursive-llm LOFT-128k (n=20, Gemini 3.5 Flash Lite)

- Date: $(date -Iseconds)
- Split: test
- Limit: first 20 rows of each RAG subset
- Subsets: nq_128k, hotpotqa_128k, musique_128k, qampari_128k, quest_128k
- Model: ${MODEL}
- Backend: recursive-llm (RLM)
- Direct API: Google Gemini API
- Execution: 5 subset processes in parallel

Each subset is saved in a subfolder:
\`loft__<subset>__gemini-gemini-3.5-flash-lite__recursive-llm__split-test__limit-20__think/\`
containing \`checkpoint.jsonl\`, \`predictions.csv\`, \`metrics.json\`,
\`config.yaml\`, and \`transcripts/\`.

Summary and evaluation logs: \`README.md\`, \`summary.json\`, \`logs/\`.
EOF

summarize() {
  "$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for metrics in sorted(root.glob("loft__*/metrics.json")):
    data = json.loads(metrics.read_text())
    cfg = {}
    cfg_path = metrics.parent / "config.yaml"
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    runtime = data.get("runtime") or {}
    rows.append({
        "run_dir": metrics.parent.name,
        "model": cfg.get("model"),
        "task": cfg.get("data_dir"),
        "limit": cfg.get("limit"),
        "em": data.get("em"),
        "subspan_em": data.get("subspan_em"),
        "f1": data.get("f1"),
        "coverage": data.get("coverage"),
        "num_samples": data.get("num_samples"),
        "errors": runtime.get("errors"),
        "average_latency_s": runtime.get("average_latency_s"),
        "average_tokens": runtime.get("average_tokens"),
    })
(root / "summary.json").write_text(json.dumps({"runs": rows}, indent=2) + "\n")
print(json.dumps({"runs": rows}, indent=2))
PY
}

echo "[eval] Starting LOFT-128k (n=20) on 5 subsets in parallel with ${MODEL}"
pids=()
for TASK in "${TASKS[@]}"; do
  LOG="$LOGDIR/${TASK}.eval.log"
  echo "[eval] launching ${TASK} -> ${LOG}"
  "$PY" "$RUNNER" \
    --model "$MODEL" \
    --task "$TASK" \
    --split test \
    --limit 20 \
    --save-trajectory \
    --max-tokens 16384 \
    --timeout 600 \
    --max-elapsed-seconds 1800 \
    --out "$OUT" \
    >"$LOG" 2>&1 &
  pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  task="${TASKS[$i]}"
  if wait "$pid"; then
    echo "[eval] ${task} finished ok (pid ${pid})"
  else
    echo "[eval] ${task} failed (pid ${pid})" >&2
    fail=1
  fi
done

summarize
if (( fail )); then
  echo "[done] some subsets failed; partial results in $OUT" >&2
  exit 1
fi
echo "[done] results in $OUT"
