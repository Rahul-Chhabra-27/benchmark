#!/bin/bash
# Find the hard questions -- vanilla right, RLM + ordering wrong -- in one command.
#
#   SUBSETS="nq hotpotqa musique qampari quest" LIMIT=55 bash scripts/rlm/run_hard_set.sh
#
# Needs the root server already running with a window that holds a whole 128k document,
# so one server answers both vanilla and the RLM (README §4):
#   GPU=0 PORT=8002 MAX_LEN=139264 GPU_UTIL=0.65 VLLM=$VLLM HF_HOME=$HF_HOME \
#     LOG=$S/logs/server.log SESSION=rlm_root bash scripts/rlm/serve_root.sh
#
# What it does, on the same questions (SPLIT, LIMIT) of every subset:
#   1. vanilla      the whole document in the window (--vanilla-char-limit 600000: the
#                   default 400,000 cuts LOFT-128k documents)
#   2. RLM          the configurable_rlm experiment(s) in RLM, default tuned_graph (our
#                   original rlm.py over bm25-graph chunks). Add a compressed one, e.g.
#                   RLM="tuned_graph tuned_graph_x2" (needs GPU room for the sub-model),
#                   to also find the questions only compression rescues
#   3. hard set     analysis/hard_questions.py over all of it -> $OUT/hard_set
# Re-running resumes every step. Results: $OUT/results, the hard set: $OUT/hard_set.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$here/../.." && pwd)}

SUBSETS=${SUBSETS:-nq hotpotqa}
LIMIT=${LIMIT:-55}
SPLIT=${SPLIT:-test}
RLM=${RLM:-tuned_graph}
PORT=${PORT:-8002}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
OUT=${OUT:-/tmp/$USER/hard_set_run}
PY=${PY:-/mnt/nas/gautammahale/benchmark/.venv-kvpress/bin/python}
export HF_HOME=${HF_HOME:-/mnt/nas/gautammahale/hf_cache}
base="http://localhost:$PORT/v1"
mkdir -p "$OUT/results" "$OUT/logs"
cd "$REPO" || exit 1
export PYTHONPATH=.

# A real generation, not /v1/models: a wedged vLLM still answers the static routes.
if ! curl -s --max-time 60 "$base/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":4}" |
  grep -q '"content"'; then
  echo "no server generating at $base -- start one first (see the top of this script)" >&2
  exit 1
fi

failed=0
echo "=== 1. vanilla over $SUBSETS ($SPLIT, $LIMIT rows each)"
for s in $SUBSETS; do
  log="$OUT/logs/vanilla_$s.log"
  if ! "$PY" -m evaluation.rlm.harness.run_benchmark --dataset loft --data-dir "${s}_128k" --split "$SPLIT" \
    --limit "$LIMIT" --mode vanilla --vanilla-char-limit 600000 --root-model "$MODEL" --sub-model "$MODEL" \
    --base-url "$base" --out "$OUT/results/vanilla" >"$log" 2>&1; then
    echo "!! vanilla failed on $s; the end of $log:"; tail -n 5 "$log"; failed=1
  fi
done

tasks=$(printf '%s,' $SUBSETS); tasks="[${tasks%,}]"
for experiment in $RLM; do
  echo "=== 2. $experiment over $SUBSETS"
  log="$OUT/logs/$experiment.log"
  if ! "$PY" -m evaluation.rlm.configurable_rlm run "$experiment" --set "data.tasks=$tasks" \
    --set "data.limit=$LIMIT" --set "data.split=$SPLIT" --set "models.base_url=$base" \
    --set "output_dir=$OUT/results" >"$log" 2>&1; then
    echo "!! $experiment failed; the end of $log:"; tail -n 5 "$log"; failed=1
  fi
done

echo "=== 3. hard questions"
"$PY" -m evaluation.rlm.analysis.hard_questions --results "$OUT/results" --out "$OUT/hard_set" || failed=1
[ "$failed" = 0 ]
