#!/bin/bash
# The chunk-geometry campaign for the structured arm.
#
# One lane per SUBSET, because the per-cell work is a stream of short HTTP calls
# against one vLLM: lanes parallelise on the server's batching, not on GPUs, and
# partitioning by subset keeps each lane's results under one directory.
#
#   geometry  window x k, the headline. 2000/400/k=5 is the shipped arm and the
#             literature's retrieval optimum (512 tokens, 20% overlap); what has
#             never been swept is how much the READER is then given.
#   exposure  the read side alone, at a fixed window. Separates a genuine window
#             effect from "the map call simply saw more characters".
#   oracle    windows that provably carry the answer. Retrieval perfect by
#             construction, so what is missing afterwards is the reader's ceiling.
#   hops      one follow-up round seeded by what round 1 FOUND. musique is the
#             treatment; hotpotqa is the multi-hop control. There is no hop 3.
#
# --reduce-finding-chars tracks the window: left at 2000 a wide window is truncated
# back at the reduce step, so the top of the sweep would measure that cap instead.
#
# Usage:  structured_grid.sh <stage> <subset> [subset ...]
#   stage in {geometry, exposure, oracle, hops, all}
set -u
REPO=${REPO:-/mnt/nas/gautammahale/structured_checkout}
RESULTS=${RESULTS:-/mnt/nas/gautammahale/structured_grid}
PY=${PY:-/mnt/nas/gautammahale/venvs/rlm/bin/python}
BASE_URL=${BASE_URL:-http://localhost:8001/v1}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
LIMIT=${LIMIT:-55}
SPLIT=${SPLIT:-test}
WINDOWS=${WINDOWS:-"1000 2000 4000 8000"}
KS=${KS:-"2 5 10"}
EXPOSURES=${EXPOSURES:-"400 800 1500 2000 4000 8000"}
# The exposure curve runs at a WIDE fixed window on purpose. Exposure can only ever
# cut a passage down, so at the 2000-char default every setting above 2000 is a
# no-op and the curve would stop exactly where the interesting half begins. At 8000
# the retrieval is identical across all six cells and the only thing moving is how
# much of each passage the reader is shown -- which is the whole question.
EXPOSURE_WINDOW=${EXPOSURE_WINDOW:-8000}
CHUNKSETS=${CHUNKSETS:-/mnt/nas/gautammahale/chunksets}

cd "$REPO" || exit 1
export HF_HOME=${HF_HOME:-/mnt/nas/gautammahale/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
COMMON="--root-model $MODEL --sub-model $MODEL --base-url $BASE_URL --sub-backend http --out $RESULTS"

stage=${1:?stage: geometry|exposure|oracle|hops|all}; shift

preflight() {
  # GENERATE, do not just ask /v1/models. A vLLM whose engine core has wedged still
  # answers the static routes and still accepts chat requests -- it logs "Added
  # request" and then never finishes one. Every sub-call blocks, and a campaign
  # started against it looks exactly like a campaign that is running: processes
  # alive, no errors, no rows. That cost an evening once. Ninety seconds is far
  # longer than a 4B needs for eight tokens, so a timeout here means wedged.
  local body
  body=$(curl -s --max-time 90 "$BASE_URL/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"reply with the word ok\"}],\"max_tokens\":8}")
  case "$body" in
    *'"content"'*) echo "preflight OK: $BASE_URL generated a reply" ;;
    *) echo "PREFLIGHT FAILED: $BASE_URL did not complete a generation in 90s." >&2
       echo "  The engine is wedged or the port is serving something else. Restart it" >&2
       echo "  before launching -- do not run the campaign against it." >&2
       exit 1 ;;
  esac
}
preflight

cap_for() {  # cap_for <k> <window> -> --max-subcall-chars for that cell
  # Clustering can merge adjacent windows into one call, so the cap has to cover a
  # whole cluster or _cap_subcall truncates it and the cell measures the cap. The
  # ceiling is the SERVER's: 110000 chars is ~27.5k tokens, inside the 32768
  # max_model_len the root vLLM is serving, with room for the prompt.
  local cap=$(( ($1 * $2 * 4 / 3) + 1 ))
  [ "$cap" -lt 32000 ] && cap=32000
  [ "$cap" -gt 110000 ] && cap=110000
  echo "$cap"
}

run() {  # run <label> <extra args...>
  local label=$1; shift
  echo "################ $label $(date +%T)"
  # --max-subcall-chars is derived, not fixed: the harness refuses a cell whose
  # per-call payload the cap would shrink, and a 8000-char window needs more than
  # the 32000 default leaves once the question's quarter is reserved.
  $PY -m evaluation.rlm.run_benchmark --dataset loft --split "$SPLIT" --limit "$LIMIT" \
      --mode structured "$@" $COMMON 2>&1 | grep -vE "^\s*$"
}

for s in "$@"; do
  sub=${s}_128k

  if [ "$stage" = geometry ] || [ "$stage" = all ]; then
    for w in $WINDOWS; do
      for k in $KS; do
        cap=$(cap_for "$k" "$w")
        find=$(( w > 2000 ? w : 2000 ))
        run "geometry $sub w=$w k=$k" --data-dir "$sub" \
            --chunks-source bm25 --chunks-k "$k" \
            --search-window "$w" --search-overlap $(( w / 5 )) \
            --reduce-finding-chars "$find" --max-subcall-chars "$cap"
      done
    done
  fi

  if [ "$stage" = exposure ] || [ "$stage" = all ]; then
    w=$EXPOSURE_WINDOW
    for e in $EXPOSURES; do
      run "exposure $sub map_chars=$e (window $w)" --data-dir "$sub" \
          --chunks-source bm25 --chunks-k 5 \
          --search-window "$w" --search-overlap $(( w / 5 )) \
          --structured-map-chars "$e" \
          --reduce-finding-chars $(( e > 2000 ? e : 2000 )) \
          --max-subcall-chars "$(cap_for 5 "$w")"
    done
  fi

  if [ "$stage" = oracle ] || [ "$stage" = all ]; then
    CH="$CHUNKSETS/${sub}.oracle.w2000o400k5.jsonl"
    if [ -f "$CH" ]; then
      run "oracle $sub" --data-dir "$sub" --chunks-source file --chunks-file "$CH" --chunks-k 5
    else
      echo "################ oracle $sub SKIPPED: $CH missing (build it with evaluation.rlm.oracle_chunks)"
    fi
  fi

  if [ "$stage" = hops ] || [ "$stage" = all ]; then
    for h in 1 2; do
      run "hops=$h $sub" --data-dir "$sub" --chunks-source bm25 --chunks-k 5 --structured-hops "$h"
    done
  fi
done
echo "################ DONE $stage $(date +%T)"
