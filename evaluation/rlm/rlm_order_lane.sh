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
#   LIMIT=3 RESULTS=/tmp/gautammahale/rlmorder/results_pilot  for the pilot; LIMIT=55 for the
#   campaign. Raw output goes to cat-local scratch and is archived to NAS after each stage.
#   BUDGET_MB=512 FACTOR=4 CUDA_VISIBLE_DEVICES=0  runs the B x F cell instead (see below).
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
  # The structured campaign's cache, archived off $HOME under the lab storage policy.
  ORG=${ORG:-/mnt/nas/gautammahale/archive/graphorder/organization}
else
  SUBSETS=${*:-"niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2"}
  # Every RULER row brings its own haystack, so its organizations are new raw output: they
  # go on scratch next to RESULTS, never under $HOME (lab storage policy).
  ORG=${ORG:-$(dirname "$RESULTS")/organization_$DATASET}
fi

# A B x F cell: set BUDGET_MB (the KV one chunk read may keep, decimal MB) and FACTOR (the
# KVzip factor every read is pressed by; 1 is the uncompressed no_press control). Reads are
# then capped at N = B x F tokens and served by an in-process KVzip model on the GPU that
# CUDA_VISIBLE_DEVICES names (the root stays on BASE_URL). Each cell gets its own
# subdirectory, so rlm_order_gate.py can gate one cell at a time.
BUDGET_MB=${BUDGET_MB:-}
FACTOR=${FACTOR:-}
if [ -n "$BUDGET_MB$FACTOR" ]; then
  if [ -z "$BUDGET_MB" ] || [ -z "$FACTOR" ]; then
    echo "Set BUDGET_MB and FACTOR together." >&2
    exit 1
  fi
  PY=${KVZIP_PYTHON:-/mnt/nas/gautammahale/benchmark/.venv-kvpress/bin/python}
  press=kvzip
  if [ "$FACTOR" = 1 ]; then press=no_press; fi
  # ~8 GiB of weights + 1 GiB headroom + 2.0 x N x 147,456 B: the full KV of a read, which
  # KVzip masks but never frees (the rule kvzip_backend._fits_in_memory applies per call).
  min_free=$(awk -v b="$BUDGET_MB" -v f="$FACTOR" \
    'BEGIN{n=int(int(b*1e6/147456)*f); x=9+n*147456*2/2^30; m=int(x); if(x>m)m++; print m}')
  sub_flags=(--sub-backend kvzip --press "$press" --memory-budget "$BUDGET_MB" --memory-budget-unit MB
    --compression-factor "$FACTOR" --sub-min-free-gib "${MIN_FREE_GIB:-$min_free}")
  RESULTS="$RESULTS/bxf_${BUDGET_MB}MB_x${FACTOR}"
else
  sub_flags=(--sub-backend http)
fi

# Lab storage policy: on a host with an infolab group, write as infolab. A RunPod pod or
# another cloud VM has no such group, and the check does not apply there.
if getent group infolab >/dev/null 2>&1 && [ "$(id -gn)" != infolab ]; then
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
      --root-model "$MODEL" --sub-model "$MODEL" --base-url "$BASE_URL" "${sub_flags[@]}" \
      --out "$RESULTS"
done
echo "################ DONE $arm $DATASET $(date +%T)"
