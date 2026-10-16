#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# recursive-llm LOFT-128k: 20 test rows × 5 RAG subsets in parallel per vLLM,
# thinking ON. Models: Qwen3.5-4B, 9B, then 27B on GPU 0 / port 8002.
set -euo pipefail

REPO="${REPO:-/data/aditya_desai/agentic-longcontext}"
VLLM="${VLLM:-$REPO/.venv-vllm/bin/vllm}"
export PATH="$(dirname "$VLLM"):$PATH"
PY="$REPO/baselines/recursive-llm/.venv/bin/python"
RUNNER="$REPO/baselines/run_benchmark.py"
PORT="${PORT:-8002}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
OUT="$REPO/evaluation/results/rlm/recursive-llm/loft128k-n20-think"
LOGDIR="$OUT/logs"
ENDPOINT="http://127.0.0.1:${PORT}/v1"
LIMIT=20
TASKS=(nq_128k hotpotqa_128k musique_128k qampari_128k quest_128k)
MODELS=(
  "Qwen/Qwen3.5-4B"
  "Qwen/Qwen3.5-9B"
  "Qwen/Qwen3.5-27B"
)
if [[ -n "${CAMPAIGN_MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS=($CAMPAIGN_MODELS)
fi

mkdir -p "$LOGDIR"

cat > "$OUT/README.md" <<EOF
# recursive-llm LOFT-128k (n=20, thinking ON)

- Date: $(date -Iseconds)
- Split: test
- Limit: first 20 rows of each RAG subset
- Subsets (run **in parallel** against one vLLM): nq_128k, hotpotqa_128k, musique_128k, qampari_128k, quest_128k
- Models (sequential servers): Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B, Qwen/Qwen3.5-27B
- Thinking: ON (\`chat_template_kwargs.enable_thinking=True\`)
- Server: vLLM on GPU ${GPU}, port ${PORT}, max-model-len 131072, --language-model-only
- Endpoint: ${ENDPOINT}

Each cell:

\`loft__<subset>__Qwen-Qwen3.5-<size>B__recursive-llm__split-test__limit-20__think/\`

\`checkpoint.jsonl\`, \`predictions.csv\`, \`metrics.json\`, \`config.yaml\`, \`transcripts/\`.

Campaign files: \`README.md\`, \`summary.json\`, \`logs/\`.
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
  curl -sf "${ENDPOINT}/models" | python -c 'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null || true
}

wait_for_server() {
  local model="$1"
  local deadline=$((SECONDS + 1800))
  echo "[wait] ${ENDPOINT} for ${model}"
  while (( SECONDS < deadline )); do
    local ids
    ids="$(served_ids)"
    if [[ -n "$ids" ]]; then
      echo "[wait] up: ${ids}"
      return 0
    fi
    sleep 5
  done
  echo "[wait] timed out waiting for ${model} on ${ENDPOINT}" >&2
  return 1
}

stop_server() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[vllm] stopping pid ${VLLM_PID}"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  VLLM_PID=""
  sleep 3
}

start_server() {
  local model="$1"
  local slug server_log
  slug="$(model_slug "$model")"
  server_log="$LOGDIR/${slug}.vllm.log"
  echo "[vllm] serving ${model} on GPU ${GPU} port ${PORT}"
  CUDA_VISIBLE_DEVICES="$GPU" "$VLLM" serve "$model" \
    --served-model-name "$model" \
    --port "$PORT" \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.8 \
    --language-model-only \
    >"$server_log" 2>&1 &
  VLLM_PID=$!
  echo "$VLLM_PID" > "$LOGDIR/${slug}.vllm.pid"
  wait_for_server "$model"
}

ensure_server() {
  local model="$1"
  local ids
  ids="$(served_ids)"
  if [[ " $ids " == *" $model "* ]]; then
    echo "[vllm] reusing existing server for ${model} (${ids})"
    if [[ -z "${VLLM_PID:-}" ]]; then
      VLLM_PID="$(pgrep -f "vllm serve ${model}" | head -1 || true)"
    fi
    return 0
  fi
  if pgrep -f "vllm serve ${model}" >/dev/null 2>&1; then
    VLLM_PID="$(pgrep -f "vllm serve ${model}" | head -1)"
    echo "[vllm] waiting for in-flight serve pid ${VLLM_PID} (${model})"
    wait_for_server "$model"
    return 0
  fi
  stop_server
  start_server "$model"
}

trap stop_server EXIT

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

eval_one_task() {
  local model="$1" task="$2"
  local slug log
  slug="$(model_slug "$model")"
  log="$LOGDIR/${slug}.${task}.eval.log"
  echo "[eval] start ${model} ${task}"
  "$PY" "$RUNNER" \
    --endpoint "$ENDPOINT" \
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

for MODEL in "${MODELS[@]}"; do
  ensure_server "$MODEL"
  pids=()
  for TASK in "${TASKS[@]}"; do
    if task_complete "$MODEL" "$TASK"; then
      echo "[skip] ${MODEL} ${TASK} already has ${LIMIT} checkpoint rows"
      continue
    fi
    eval_one_task "$MODEL" "$TASK" &
    pids+=("$!")
  done
  if ((${#pids[@]})); then
    echo "[eval] waiting on ${#pids[@]} parallel tasks for ${MODEL}: ${pids[*]}"
    fail=0
    for pid in "${pids[@]}"; do
      wait "$pid" || fail=1
    done
    if (( fail )); then
      echo "[eval] one or more tasks failed for ${MODEL}" >&2
    fi
  else
    echo "[eval] nothing left for ${MODEL}"
  fi
  summarize
  stop_server
done

echo "[done] results in $OUT"
summarize
