#!/bin/bash
# Run B x F cells of the ordered-chunk arms end to end, with one command.
#
#   bash evaluation/rlm/run_bxf.sh          # the smoke test: nq, 2 rows, both arms, 512 MB x F = 1 and 4
#
#   LIMIT=55 SUBSETS="nq hotpotqa musique qampari quest" BUDGETS_MB="512 1024" FACTORS="1 2 4 8" \
#     bash evaluation/rlm/run_bxf.sh        # a real grid
#
# What it does, in order:
#   1. starts the root vLLM server on GPU (unless one on PORT already generates) and waits for it;
#   2. runs every (budget, factor, arm) cell, one after another, through rlm_order_lane.sh;
#   3. prints one table of scores and per-cell checks (bxf_summary.py);
#   4. stops the server, if it was the one that started it.
# It exits 0 only if every cell ran and passed its checks. Re-running it resumes: rows
# already done are skipped. See RUNNING.md §7 at the repository root.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Lab storage policy: on a host with an infolab group, write as infolab. Re-run under sg
# so nobody has to remember to.
if getent group infolab >/dev/null 2>&1 && [ "$(id -gn)" != infolab ]; then
  exec sg infolab -c "$(printf '%q ' bash "$here/run_bxf.sh" "$@")"
fi

REPO=${REPO:-$(cd "$here/../.." && pwd)}
DATASET=${DATASET:-loft}
SUBSETS=${SUBSETS:-nq}
LIMIT=${LIMIT:-2}
BUDGETS_MB=${BUDGETS_MB:-512}
FACTORS=${FACTORS:-1 4}
ARMS=${ARMS:-bm25-query bm25-graph}
GPU=${GPU:-0}
PORT=${PORT:-8002}
GPU_UTIL=${GPU_UTIL:-0.5}
CPU=${CPU:-6}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
OUT=${OUT:-/tmp/$USER/bxf}
RESULTS=${RESULTS:-$OUT/results}
LOGS=${LOGS:-$OUT/logs}
export HF_HOME=${HF_HOME:-/mnt/nas/gautammahale/hf_cache}
export KVZIP_PYTHON=${KVZIP_PYTHON:-/mnt/nas/gautammahale/benchmark/.venv-kvpress/bin/python}
VLLM=${VLLM:-/mnt/nas/gautammahale/benchmark/.venv/bin/vllm}
mkdir -p "$RESULTS" "$LOGS"

base="http://localhost:$PORT/v1"
generates() {
  # A real generation, not /v1/models: a wedged vLLM still answers the static routes.
  curl -s --max-time 60 "$base/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":4}" |
    grep -q '"content"'
}

started=""
stop_server() {
  if [ -n "$started" ] && [ "${KEEP_SERVER:-0}" != 1 ]; then
    tmux kill-session -t "$started" 2>/dev/null && echo "stopped the server (tmux $started)"
  fi
}
trap stop_server EXIT

if generates; then
  echo "using the server already generating on :$PORT"
else
  started=${SESSION:-bxf_root}
  GPU=$GPU PORT=$PORT GPU_UTIL=$GPU_UTIL VLLM=$VLLM HF_HOME=$HF_HOME CPUS=${SERVER_CPUS:-} \
    LOG=$LOGS/server.log SESSION=$started bash "$here/serve_root.sh" >/dev/null || {
    started=""
    exit 1
  }
  echo "starting the root server on GPU $GPU (tmux $started); waiting for it to generate..."
  for _ in $(seq 120); do
    generates && break
    sleep 15
  done
  generates || {
    echo "the server did not come up in 30 minutes; see $LOGS/server.log" >&2
    exit 1
  }
  echo "server ready"
fi

failed=0
for budget in $BUDGETS_MB; do
  for factor in $FACTORS; do
    for arm in $ARMS; do
      log="$LOGS/bxf_${budget}MB_x${factor}_${arm}.log"
      echo "=== B=${budget} MB  F=${factor}  $arm  ($DATASET: $SUBSETS, $LIMIT rows)  $(date +%T)  log: $log"
      # shellcheck disable=SC2086  # SUBSETS is a word list on purpose.
      if ! BUDGET_MB=$budget FACTOR=$factor CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID \
        DATASET=$DATASET LIMIT=$LIMIT REPO=$REPO RESULTS=$RESULTS BASE_URL=$base MODEL=$MODEL \
        bash "$here/rlm_order_lane.sh" "$arm" "$CPU" $SUBSETS >"$log" 2>&1; then
        echo "!! this cell failed. The end of its log:"
        tail -n 8 "$log"
        failed=1
      fi
    done
  done
done

echo
echo "=== summary of $RESULTS"
cd "$REPO" && PYTHONPATH=. "$KVZIP_PYTHON" -m evaluation.rlm.bxf_summary "$RESULTS"
summary=$?
[ "$failed" = 0 ] && [ "$summary" = 0 ]
