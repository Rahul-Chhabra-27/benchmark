#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Detached LOFT-128k campaign against three live vLLM servers (thinking ON,
# client-side think/content split). Re-runs 4B into a fresh results dir and
# evaluates 9B + 27B. After every endpoint is up, all 5 subsets × 3 models
# run in parallel. Does not stop the vLLM servers on exit.
set -euo pipefail

REPO="${REPO:-/data/aditya_desai/agentic-longcontext}"
VLLM="${VLLM:-$REPO/.venv-vllm/bin/vllm}"
export PATH="$(dirname "$VLLM"):$PATH"
PY="$REPO/baselines/recursive-llm/.venv/bin/python"
RUNNER="$REPO/baselines/run_benchmark.py"
OUT="$REPO/evaluation/results/rlm/recursive-llm/loft128k-n20-think-split"
LOGDIR="$OUT/logs"
LIMIT=20
TASKS=(nq_128k hotpotqa_128k musique_128k qampari_128k quest_128k)

# model|endpoint|gpu|port
CELLS=(
  "Qwen/Qwen3.5-4B|http://127.0.0.1:8012/v1|0|8012"
  "Qwen/Qwen3.5-9B|http://127.0.0.1:8014/v1|4|8014"
  "Qwen/Qwen3.5-27B|http://127.0.0.1:8013/v1|3|8013"
)

mkdir -p "$LOGDIR"

cat > "$OUT/README.md" <<EOF
# recursive-llm LOFT-128k (n=20, thinking ON, vLLM content split)

- Date: $(date -Iseconds)
- Split: test, first 20 rows of each RAG subset
- Subsets: nq_128k, hotpotqa_128k, musique_128k, qampari_128k, quest_128k
- Models (three concurrent servers, all 15 cells in parallel):
  - Qwen/Qwen3.5-4B GPU 0 port 8012
  - Qwen/Qwen3.5-9B GPU 4 port 8014
  - Qwen/Qwen3.5-27B GPU 3 port 8013
- Thinking ON; client peels \`</think>\` before FINAL/REPL
- Fresh dir so 4B is a full rerun (old loft128k-n20-think 4B rows were protocol failures)

Each cell: \`loft__<subset>__Qwen-Qwen3.5-<size>B__recursive-llm__split-test__limit-20__think/\`
EOF

model_slug() {
  echo "${1//\//-}"
}

run_dir_for() {
  local model="$1" task="$2"
  echo "$OUT/loft__${task}__$(model_slug "$model")__recursive-llm__split-test__limit-${LIMIT}__think"
}

task_complete() {
  local ckpt
  ckpt="$(run_dir_for "$1" "$2")/checkpoint.jsonl"
  if [[ ! -f "$ckpt" ]]; then
    return 1
  fi
  local n
  n="$(grep -c . "$ckpt" || true)"
  (( n >= LIMIT ))
}

served_ids() {
  local endpoint="$1"
  curl -sf "${endpoint}/models" | python -c 'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null || true
}

wait_for_endpoint() {
  local model="$1" endpoint="$2"
  local deadline=$((SECONDS + 1800))
  echo "[wait] ${endpoint} for ${model}"
  while (( SECONDS < deadline )); do
    local ids
    ids="$(served_ids "$endpoint")"
    if [[ " $ids " == *" $model "* ]]; then
      echo "[wait] up: ${ids}"
      return 0
    fi
    sleep 5
  done
  echo "[wait] timed out waiting for ${model} on ${endpoint}" >&2
  return 1
}

ensure_9b() {
  local model="Qwen/Qwen3.5-9B"
  local endpoint="http://127.0.0.1:8014/v1"
  local ids
  ids="$(served_ids "$endpoint")"
  if [[ " $ids " == *" $model "* ]]; then
    echo "[vllm] reusing 9B on ${endpoint}"
    return 0
  fi
  if pgrep -f "vllm serve ${model} .*--port 8014" >/dev/null 2>&1; then
    echo "[vllm] 9B serve already in flight"
    wait_for_endpoint "$model" "$endpoint"
    return 0
  fi
  local server_log="$LOGDIR/Qwen-Qwen3.5-9B.vllm.log"
  echo "[vllm] starting ${model} on GPU 4 port 8014"
  setsid -f bash -c "exec env CUDA_VISIBLE_DEVICES=4 PATH=$(dirname "$VLLM"):\$PATH nohup $VLLM serve $model --served-model-name $model --port 8014 --max-model-len 32768 --gpu-memory-utilization 0.8 --language-model-only >> $server_log 2>&1"
  wait_for_endpoint "$model" "$endpoint"
}

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
        "enable_thinking": cfg.get("enable_thinking"),
        "vllm_split_reasoning": cfg.get("vllm_split_reasoning"),
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

eval_one() {
  local model="$1" endpoint="$2" task="$3"
  local slug log
  slug="$(model_slug "$model")"
  log="$LOGDIR/${slug}.${task}.eval.log"
  echo "[eval] start ${model} ${task} ${endpoint}"
  "$PY" "$RUNNER" \
    --endpoint "$endpoint" \
    --model "$model" \
    --task "$task" \
    --split test \
    --limit "$LIMIT" \
    --enable-thinking \
    --save-trajectory \
    --max-tokens 16384 \
    --timeout 600 \
    --max-elapsed-seconds 1800 \
    --out "$OUT" \
    >"$log" 2>&1
  echo "[eval] done ${model} ${task}"
}

ensure_9b
for cell in "${CELLS[@]}"; do
  IFS='|' read -r model endpoint gpu port <<<"$cell"
  wait_for_endpoint "$model" "$endpoint"
done

pids=()
for cell in "${CELLS[@]}"; do
  IFS='|' read -r model endpoint gpu port <<<"$cell"
  for task in "${TASKS[@]}"; do
    if task_complete "$model" "$task"; then
      echo "[skip] ${model} ${task} already has ${LIMIT} checkpoint rows"
      continue
    fi
    eval_one "$model" "$endpoint" "$task" &
    pids+=("$!")
  done
done

if ((${#pids[@]})); then
  echo "[eval] waiting on ${#pids[@]} parallel tasks: ${pids[*]}"
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if (( fail )); then
    echo "[eval] one or more tasks failed" >&2
  fi
else
  echo "[eval] nothing left to run"
fi

echo "[done] results in $OUT"
summarize
