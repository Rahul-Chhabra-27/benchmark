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
#   qbare     the geometry grid again, with BM25 asked the QUESTION instead of
#             LOFT's page of instructions. Measured offline: the wrapper costs
#             recall@2 nearly everything (nq 0.00 -> 0.95) and recall@5 a third
#             (0.65 -> 0.96). The first grid measures a handicapped retriever, so
#             this is the one that answers "the upper limit"; both are kept, since
#             the published arms all used the wrapper.
#
# --reduce-finding-chars tracks the window: left at 2000 a wide window is truncated
# back at the reduce step, so the top of the sweep would measure that cap instead.
#
# Usage:  structured_grid.sh <stage> <subset> [subset ...]
#   stage in {geometry, exposure, oracle, hops, qbare, all,
#             ladder, exhaustive, holdout}   -- the query-agnostic stages
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
# The query-agnostic campaign: chunk sets written by query_free_chunks.py, and the
# ids of the rows the chunk-geometry campaign never ran (test 55..99).
# The chunk sets were archived off $HOME (lab storage policy) on 2026-09-11; the ids stayed.
QCHUNKS=${QCHUNKS:-/mnt/nas/gautammahale/archive/qagnostic/chunksets}
IDS=${IDS:-/mnt/home/gautammahale/qagnostic/ids}
LADDER_KS=${LADDER_KS:-"2 5 10"}
LADDER_SEEDS=${LADDER_SEEDS:-"0 1 2"}
LADDER_WINDOW=${LADDER_WINDOW:-2000}
# Exhaustive reads every window, so the geometry is the cost. At 2000/400 a LOFT
# document is ~290 windows and one example is ~290 map calls; at 8000/1600 it is
# ~72 and the reduce prompt still fits. 8000 answers the same question for a third
# of the time, so it is the default and 2000 is opt-in.
EXHAUSTIVE_WINDOW=${EXHAUSTIVE_WINDOW:-8000}

cd "$REPO" || exit 1
export HF_HOME=${HF_HOME:-/mnt/nas/gautammahale/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
COMMON="--root-model $MODEL --sub-model $MODEL --base-url $BASE_URL --sub-backend http --out $RESULTS"
ROWSEL="--limit $LIMIT"

QUERY=${QUERY:-task}   # `bare` strips LOFT's instruction wrapper before BM25 sees it
stage=${1:?stage: geometry|exposure|oracle|hops|qbare|all|ladder|exhaustive|holdout}; shift

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
  # ROWSEL is `--limit N` for the development half and `--ids-file F` for the
  # holdout. The two are mutually exclusive at the CLI on purpose: --ids-file is
  # applied BEFORE --limit, so passing both would silently mean "the first N of the
  # named rows" -- a different experiment that reads like a cap.
  $PY -m evaluation.rlm.harness.run_benchmark --dataset loft --split "$SPLIT" $ROWSEL \
      --mode structured "$@" $COMMON 2>&1 | grep -vE "^\s*$"
}

for s in "$@"; do
  sub=${s}_128k

  if [ "$stage" = geometry ] || [ "$stage" = qbare ] || [ "$stage" = all ]; then
    q=$QUERY; [ "$stage" = qbare ] && q=bare
    for w in $WINDOWS; do
      for k in $KS; do
        cap=$(cap_for "$k" "$w")
        find=$(( w > 2000 ? w : 2000 ))
        run "geometry($q) $sub w=$w k=$k" --data-dir "$sub" \
            --chunks-source bm25 --chunks-k "$k" --chunks-query "$q" \
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
      echo "################ oracle $sub SKIPPED: $CH missing (build it with evaluation.rlm.chunk_sources.oracle_chunks)"
    fi
  fi

  # ---------------------------------------------------------------- ladder --
  # What the QUERY is worth, at a matched read budget. Every rung reads exactly k
  # windows of exactly LADDER_WINDOW chars, which is why they all run at
  # --structured-map-granularity window: `_cluster_hits` merges touching windows,
  # so at the default `passage` granularity lead-k (adjacent windows) would become
  # ONE call of k x window chars while stride-k stayed k calls of one window. At
  # matched k the rungs would differ in call structure as much as in selection, and
  # `bm25_bare - random` would be measuring both.
  #
  # That is also why the bm25 and oracle rungs are re-run here rather than read off
  # the existing campaign: those cells ran at `passage` and are not controls for this.
  if [ "$stage" = ladder ]; then
    w=$LADDER_WINDOW
    o=$(( w / 5 ))
    for k in $LADDER_KS; do
      for q in task bare; do
        run "ladder bm25($q) $sub k=$k" --data-dir "$sub" \
            --chunks-source bm25 --chunks-k "$k" --chunks-query "$q" \
            --search-window "$w" --search-overlap "$o" \
            --structured-map-granularity window --max-subcall-chars "$(cap_for "$k" "$w")"
      done
      for sel in lead stride; do
        CH="$QCHUNKS/${sub}.${sel}.w${w}o${o}k${k}.jsonl"
        [ -f "$CH" ] || { echo "################ ladder $sel $sub k=$k SKIPPED: $CH missing"; continue; }
        run "ladder $sel $sub k=$k" --data-dir "$sub" \
            --chunks-source file --chunks-file "$CH" --chunks-k "$k" \
            --search-window "$w" --search-overlap "$o" \
            --structured-map-granularity window --max-subcall-chars "$(cap_for "$k" "$w")"
      done
    done
    # Three seeds, at k=5 only. One draw at n=55 carries about +/-0.05 of binomial
    # noise -- three times the reproducibility floor -- and the headline of this
    # whole stage is `bm25_bare - random`, so the null model needs an error bar.
    for seed in $LADDER_SEEDS; do
      CH="$QCHUNKS/${sub}.random-s${seed}.w${w}o${o}k5.jsonl"
      [ -f "$CH" ] || { echo "################ ladder random-s$seed $sub SKIPPED: $CH missing"; continue; }
      run "ladder random-s$seed $sub k=5" --data-dir "$sub" \
          --chunks-source file --chunks-file "$CH" --chunks-k 5 \
          --search-window "$w" --search-overlap "$o" \
          --structured-map-granularity window --max-subcall-chars "$(cap_for 5 "$w")"
    done
    # The ceiling, on the same granularity as the rungs it is a ceiling for.
    CH="$CHUNKSETS/${sub}.oracle.w2000o400k5.jsonl"
    if [ -f "$CH" ] && [ "$w" = 2000 ]; then
      run "ladder oracle $sub k=5" --data-dir "$sub" \
          --chunks-source file --chunks-file "$CH" --chunks-k 5 \
          --structured-map-granularity window
    fi
  fi

  # ------------------------------------------------------------ exhaustive --
  # Retrieval removed rather than replaced: every window, in document order.
  # The k is READ OUT of the filename and never guessed -- for a `file` source
  # --chunks-k is a payload cap of k x window and overlap is double-counted, so a
  # single global k passes nq (298 windows at 2000/400) and fails quest (302) with
  # a fatal ChunkSetError.
  if [ "$stage" = exhaustive ]; then
    w=$EXHAUSTIVE_WINDOW
    o=$(( w / 5 ))
    CH=$(ls "$QCHUNKS/${sub}.all.w${w}o${o}k"*.jsonl 2>/dev/null | head -1)
    if [ -z "$CH" ]; then
      echo "################ exhaustive $sub SKIPPED: no $QCHUNKS/${sub}.all.w${w}o${o}k*.jsonl"
    else
      nw=$(basename "$CH" | sed -E "s/.*k([0-9]+)\.jsonl/\1/")
      # --reduce-finding-chars is the binding constraint here, not the window. The
      # reduce prompt is n_findings x this, and n_findings can be every window: at
      # the 2000 default a 75-window sweep would send ~150k chars (~37k tokens) and
      # the server would refuse it with a 400, which the client does NOT retry.
      find=$(( 120000 / nw ))
      [ "$find" -gt 2000 ] && find=2000
      # --max-sub-calls 0 is MANDATORY. Past the budget llm_query returns a notice
      # STRING instead of calling the model, and that string does not start with the
      # "nothing relevant" sentinel -- so the findings filter counts every one of
      # them as a finding. A truncated sweep reports MORE findings than a healthy
      # one and passes every gate that looks for a starved cell. run_benchmark now
      # refuses the combination outright; this is the setting that satisfies it.
      run "exhaustive $sub ${nw} windows w=$w find=$find" --data-dir "$sub" \
          --chunks-source file --chunks-file "$CH" --chunks-k "$nw" \
          --search-window "$w" --search-overlap "$o" \
          --structured-map-granularity window --max-sub-calls 0 --run-timeout 7200 \
          --reduce-finding-chars "$find" --max-subcall-chars "$(cap_for 1 "$w")"
    fi
  fi

  # --------------------------------------------------------------- holdout --
  # The rows the chunk-geometry campaign never ran: test 55..99, n=45. Every cell
  # of that campaign was `--limit 55`, which is df.head(55), so the argmax that
  # picked its headline cells never saw these rows. Running the promoted arms here
  # is the only measurement in the study that is not in-sample.
  #
  # The ids come from the loader via query_free_chunks --ids-out, not by hand, and
  # _restrict raises on any id the split does not have -- so a mistake is caught
  # before an example runs rather than after a night of them.
  if [ "$stage" = holdout ]; then
    IDFILE="$IDS/${sub}.holdout45.ids"
    if [ ! -f "$IDFILE" ]; then
      echo "################ holdout $sub SKIPPED: $IDFILE missing"
      continue
    fi
    ROWSEL="--ids-file $IDFILE"
    k=${HOLDOUT_K:-5}
    w=$LADDER_WINDOW
    o=$(( w / 5 ))
    for q in task bare; do
      run "holdout bm25($q) $sub k=$k" --data-dir "$sub" \
          --chunks-source bm25 --chunks-k "$k" --chunks-query "$q" \
          --search-window "$w" --search-overlap "$o" \
          --structured-map-granularity window --max-subcall-chars "$(cap_for "$k" "$w")"
    done
    for sel in lead stride random-s0; do
      CH="$QCHUNKS/${sub}.${sel}.w${w}o${o}k${k}.jsonl"
      [ -f "$CH" ] || { echo "################ holdout $sel $sub SKIPPED: $CH missing"; continue; }
      run "holdout $sel $sub k=$k" --data-dir "$sub" \
          --chunks-source file --chunks-file "$CH" --chunks-k "$k" \
          --search-window "$w" --search-overlap "$o" \
          --structured-map-granularity window --max-subcall-chars "$(cap_for "$k" "$w")"
    done
    CH="$CHUNKSETS/${sub}.oracle.w2000o400k5.jsonl"
    [ -f "$CH" ] && [ "$w" = 2000 ] && run "holdout oracle $sub" --data-dir "$sub" \
        --chunks-source file --chunks-file "$CH" --chunks-k 5 --structured-map-granularity window
    ew=$EXHAUSTIVE_WINDOW
    eo=$(( ew / 5 ))
    CH=$(ls "$QCHUNKS/${sub}.all.w${ew}o${eo}k"*.jsonl 2>/dev/null | head -1)
    if [ -n "$CH" ]; then
      nw=$(basename "$CH" | sed -E "s/.*k([0-9]+)\.jsonl/\1/")
      find=$(( 120000 / nw )); [ "$find" -gt 2000 ] && find=2000
      run "holdout exhaustive $sub ${nw} windows" --data-dir "$sub" \
          --chunks-source file --chunks-file "$CH" --chunks-k "$nw" \
          --search-window "$ew" --search-overlap "$eo" \
          --structured-map-granularity window --max-sub-calls 0 --run-timeout 7200 \
          --reduce-finding-chars "$find" --max-subcall-chars "$(cap_for 1 "$ew")"
    fi
    ROWSEL="--limit $LIMIT"
  fi

  if [ "$stage" = hops ] || [ "$stage" = all ]; then
    for h in 1 2; do
      run "hops=$h $sub" --data-dir "$sub" --chunks-source bm25 --chunks-k 5 --structured-hops "$h"
    done
  fi
done
echo "################ DONE $stage $(date +%T)"
