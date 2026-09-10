#!/bin/bash
# One lane of the agentic ordered-chunk campaign: one arm, the given subsets, one core.
#
# The AGENTIC root -- the REPL loop, not the structured map->reduce -- is handed EVERY
# 2000-char zero-overlap chunk of the document as a `chunks` list, in this arm's order,
# and told what that order is (--chunk-order; see CHUNK_ORDER_NOTES in rlm.py):
#   bm25-graph  the question-free organization. On LOFT, ORG defaults to the graph-order
#               campaign's own cache, so the order is the very artifact that campaign read
#   bm25-query  every chunk ranked by BM25 against the bare question
# Every other knob matches the base-RLM reference (loft128k_guards): --max-steps 50 and
# the defaults for sub-calls, timeouts and slice caps.
#
# Usage:  rlm_order_lane.sh <bm25-graph|bm25-query> <cpu> [subset ...]
#   DATASET=loft (default): subsets nq hotpotqa musique qampari quest, run as <s>_128k on
#     --split test.   DATASET=ruler128k: subsets are RULER task names (no split).
#   LIMIT=3 RESULTS=$CAMPAIGN/results_pilot  for the pilot; LIMIT=55 for the campaign.
set -euo pipefail
arm=${1:?arm: bm25-graph|bm25-query}; cpu=${2:?cpu core to pin to}; shift 2
DATASET=${DATASET:-loft}
CAMPAIGN=${CAMPAIGN:-/mnt/home/gautammahale/rlmorder}
REPO=${REPO:-$CAMPAIGN/checkout}
RESULTS=${RESULTS:?Set RESULTS to a scratch output directory, outside NFS home}
PY=${PY:-/mnt/nas/gautammahale/venvs/rlm/bin/python}
BASE_URL=${BASE_URL:-http://localhost:8002/v1}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
LIMIT=${LIMIT:-55}
MAX_STEPS=${MAX_STEPS:-50}
if [ "$DATASET" = loft ]; then
  SUBSETS=${*:-"nq hotpotqa musique qampari quest"}
  ORG=${ORG:-/mnt/home/gautammahale/graphorder/organization}
else
  SUBSETS=${*:-"niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2"}
  ORG=${ORG:-$CAMPAIGN/organization}
fi

if [ "$(id -gn)" != infolab ]; then
  echo "Launch this lane with primary group infolab (sg infolab -c ...)." >&2
  exit 1
fi
cd "$REPO" || exit 1
export HF_HOME=${HF_HOME:-/mnt/nas/gautammahale/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# GENERATE, do not just ask /v1/models: a wedged vLLM still answers the static routes.
body=$(curl -s --max-time 90 "$BASE_URL/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"reply with the word ok\"}],\"max_tokens\":8}")
case "$body" in
  *'"content"'*) echo "preflight OK: $BASE_URL generated a reply" ;;
  *) echo "PREFLIGHT FAILED: $BASE_URL did not complete a generation in 90s." >&2; exit 1 ;;
esac

for s in $SUBSETS; do
  if [ "$DATASET" = loft ]; then data_dir="${s}_128k"; split="--split test"; else data_dir="$s"; split=""; fi
  echo "################ $arm $DATASET/$data_dir limit=$LIMIT cpu=$cpu $(date +%T)"
  # $split is word-split on purpose: empty for RULER, which has no dev/test split.
  # shellcheck disable=SC2086
  taskset -c "$cpu" "$PY" -m evaluation.rlm.run_benchmark --dataset "$DATASET" --data-dir "$data_dir" \
      $split --limit "$LIMIT" --mode rlm --chunk-order "$arm" --organization-dir "$ORG" \
      --max-steps "$MAX_STEPS" \
      --root-model "$MODEL" --sub-model "$MODEL" --base-url "$BASE_URL" --sub-backend http \
      --out "$RESULTS"
done
echo "################ DONE $arm $DATASET $(date +%T)"
