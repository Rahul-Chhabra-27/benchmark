#!/bin/bash
# Pick ONE compression factor per question up front, run, and score -- one command.
#
#   PICKER=spread:0.3 SUBSETS="hotpotqa" BUDGET_MB=1024 bash evaluation/rlm/run_picked.sh
#
# What it does, in order (evaluation/rlm/factor_picker/README.md has the details):
#   1. plan      the picker chooses F in FACTORS for every question -- no model involved --
#                and writes PLAN/plan.jsonl and one ids file per (subset, F)
#   2. run       each ids file runs as an ordinary B x F cell through rlm_order_lane.sh
#                (F = 1 is the plain no_press control), after starting the root server
#   3. assemble  each question's answer at its own F is merged and scored; with
#                BASELINES=<root of full fixed-F cells> it is compared with every fixed F
#
# DRY_RUN=1 stops after step 1 and prints the cells step 2 would run.
# Pickers: fixed:F | spread[:RATIO] | per-task:nq=8,hotpotqa=1 | calibrated:FILE
# (calibrated:FILE comes from `python -m evaluation.rlm.factor_picker calibrate` on dev).
# Re-running resumes: rows already answered are skipped by run_benchmark.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if getent group infolab >/dev/null 2>&1 && [ "$(id -gn)" != infolab ]; then
  exec sg infolab -c "$(printf '%q ' bash "$here/run_picked.sh" "$@")"
fi

REPO=${REPO:-$(cd "$here/../.." && pwd)}
DATASET=${DATASET:-loft}
SUBSETS=${SUBSETS:-hotpotqa}
SPLIT=${SPLIT:-test}
LIMIT=${LIMIT:-}
BUDGET_MB=${BUDGET_MB:-1024}
FACTORS=${FACTORS:-1 2 4 8}
PICKER=${PICKER:-spread}
ARM=${ARM:-bm25-graph}
GPU=${GPU:-0}
PORT=${PORT:-8002}
GPU_UTIL=${GPU_UTIL:-0.5}
CPU=${CPU:-6}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
OUT=${OUT:-/tmp/$USER/picked}
NAME=${NAME:-$(echo "$PICKER" | tr ':/,=' '____')_${BUDGET_MB}MB}
PLAN=${PLAN:-$OUT/plans/$NAME}
RESULTS=${RESULTS:-$OUT/results/$NAME}
LOGS=${LOGS:-$OUT/logs/$NAME}
BASELINES=${BASELINES:-}
export HF_HOME=${HF_HOME:-/mnt/nas/gautammahale/hf_cache}
export KVZIP_PYTHON=${KVZIP_PYTHON:-/mnt/nas/gautammahale/benchmark/.venv-kvpress/bin/python}
VLLM=${VLLM:-/mnt/nas/gautammahale/benchmark/.venv/bin/vllm}
mkdir -p "$RESULTS" "$LOGS"
picker() { (cd "$REPO" && PYTHONPATH=. "$KVZIP_PYTHON" -m evaluation.rlm.factor_picker "$@"); }

echo "=== 1. plan: $PICKER over $DATASET $SUBSETS ($SPLIT${LIMIT:+, first $LIMIT}), B=${BUDGET_MB}MB, F in {$FACTORS}"
# shellcheck disable=SC2086  # SUBSETS and FACTORS are word lists on purpose.
picker plan --dataset "$DATASET" --tasks $SUBSETS --budget-mb "$BUDGET_MB" --factors $FACTORS \
  --picker "$PICKER" --split "$SPLIT" ${LIMIT:+--limit "$LIMIT"} --out "$PLAN" || exit 1

groups=("$PLAN"/ids/*.txt)
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "=== DRY_RUN: step 2 would run these cells into $RESULTS"
  for ids in "${groups[@]}"; do echo "  $(basename "$ids" .txt): $(wc -l <"$ids") questions"; done
  exit 0
fi

echo "=== 2. run ${#groups[@]} cells"
# shellcheck source=root_server.sh
source "$here/root_server.sh"
trap root_server_down EXIT
root_server_up || exit 1
failed=0
for ids in "${groups[@]}"; do
  group=$(basename "$ids" .txt)          # e.g. hotpotqa_128k__x2
  task=${group%__x*} factor=${group##*__x}
  subset=$task
  [ "$DATASET" = loft ] && subset=${task%_*}   # the lane adds _128k back
  log="$LOGS/$group.log"
  echo "--- $task at x$factor: $(wc -l <"$ids") questions  $(date +%T)  log: $log"
  if ! BUDGET_MB=$BUDGET_MB FACTOR=$factor IDS_FILE=$ids SPLIT=$SPLIT CUDA_VISIBLE_DEVICES=$GPU \
    CUDA_DEVICE_ORDER=PCI_BUS_ID DATASET=$DATASET REPO=$REPO RESULTS=$RESULTS \
    BASE_URL="http://localhost:$PORT/v1" MODEL=$MODEL \
    bash "$here/rlm_order_lane.sh" "$ARM" "$CPU" "$subset" >"$log" 2>&1; then
    echo "!! this cell failed. The end of its log:"
    tail -n 8 "$log"
    failed=1
  fi
done

echo "=== 3. assemble"
# shellcheck disable=SC2086
picker assemble --plan "$PLAN" --results "$RESULTS" --chunk-order "$ARM" \
  ${BASELINES:+--baselines "$BASELINES" --factors $FACTORS}
assembled=$?
[ "$failed" = 0 ] && [ "$assembled" = 0 ]
