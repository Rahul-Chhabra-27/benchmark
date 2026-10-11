#!/bin/bash
# One lane of the BM25-graph vs source-order campaign: one arm, every subset, one core.
#
# Both arms read EVERY 2000-char zero-overlap chunk of the document, packed into
# reader groups of the same token budget; they differ in chunk order alone. So the
# two lanes are the two arms, and each is a sequential stream of HTTP calls against
# the one server -- pinned to ONE physical core, because a benchmark worker that can
# spread over the host competes with everybody else's CPU jobs.
#
# Usage:  graph_order_lane.sh <source|bm25-graph> <cpu> [subset ...]
#   LIMIT=3 RESULTS=.../results_pilot  for the pilot; LIMIT=55 for the campaign.
set -u
arm=${1:?arm: source|bm25-graph}; cpu=${2:?cpu core to pin to}; shift 2
SUBSETS=${*:-"nq hotpotqa musique qampari quest"}
CAMPAIGN=${CAMPAIGN:-/mnt/home/gautammahale/graphorder}
REPO=${REPO:-$CAMPAIGN/checkout}
RESULTS=${RESULTS:-$CAMPAIGN/results}
ORG=${ORG:-$CAMPAIGN/organization}
PY=${PY:-/mnt/nas/gautammahale/venvs/rlm/bin/python}
BASE_URL=${BASE_URL:-http://localhost:8002/v1}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
LIMIT=${LIMIT:-55}
GROUP_TOKENS=${GROUP_TOKENS:-8192}

cd "$REPO" || exit 1
export HF_HOME=${HF_HOME:-/mnt/nas/gautammahale/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

# GENERATE, do not just ask /v1/models: a wedged vLLM still answers the static routes.
body=$(curl -s --max-time 90 "$BASE_URL/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"reply with the word ok\"}],\"max_tokens\":8}")
case "$body" in
  *'"content"'*) echo "preflight OK: $BASE_URL generated a reply" ;;
  *) echo "PREFLIGHT FAILED: $BASE_URL did not complete a generation in 90s." >&2; exit 1 ;;
esac

for s in $SUBSETS; do
  echo "################ $arm ${s}_128k limit=$LIMIT cpu=$cpu $(date +%T)"
  # --max-sub-calls 0: unlimited map calls (a cap would drop groups).
  # --run-timeout 0: a whole-document read must not be cut short; the ordered path
  #   raises on any incomplete read anyway.
  # --reduce-finding-chars stays at its 2000 default; the reader question is the
  #   default `task`, matching the historical exhaustive references.
  taskset -c "$cpu" "$PY" -m evaluation.rlm.run_benchmark --dataset loft --data-dir "${s}_128k" \
      --split test --limit "$LIMIT" --mode structured --structured-order "$arm" \
      --structured-group-input-tokens "$GROUP_TOKENS" --organization-dir "$ORG" \
      --max-sub-calls 0 --run-timeout 0 \
      --root-model "$MODEL" --sub-model "$MODEL" --base-url "$BASE_URL" --sub-backend http \
      --out "$RESULTS" 2>&1 | grep -vE "^\s*$"
done
echo "################ DONE $arm $(date +%T)"
