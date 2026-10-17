#!/bin/bash
# After an ordered-chunk stage's lanes finish: gate it, stop the server if no row needs a
# rerun (the user's condition for restarting it), report it, and archive it to NAS.
#
# Pull the checkout FIRST, then run this from it -- bash reads a script as it runs, so a
# pull must never rewrite this file under a running copy:
#   ssh cat.cse.iitb.ac.in 'cd /mnt/home/gautammahale/rlmorder/checkout && git pull -q --ff-only \
#     && sg infolab -c "bash scripts/rlm/rlm_order_postrun.sh"'
#
# DATASET=loft (default) gates against the structured graph-order campaign and reports the
# loft128k_guards context rows; DATASET=ruler128k has neither. STAGE names the logs
# (<STAGE>_<arm>_<subset>.log) and the archived copies (<STAGE>_results, _report, _logs).
# Raw output never goes to $HOME (lab storage policy): scratch on cat, archive on NAS.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
DATASET=${DATASET:-loft}
EXPECT=${EXPECT:-55}
STAGE=${STAGE:-full}
SCRATCH=${SCRATCH:-/tmp/gautammahale/rlmorder}
if [ "$DATASET" = loft ]; then
  # Rows 0-2 of every LOFT subset informed revisions 2-5, and the campaign ran revision 5:
  # gate it at that revision, not at the checkout's.
  TUNED_ROWS=${TUNED_ROWS:-3}
  REVISION=${REVISION:-5}
  RESULTS=${RESULTS:-$SCRATCH/results}
  REPORT=${REPORT:-$SCRATCH/report}
else
  # Rows 0-2 of every RULER task informed revision 6. The graph arm's organizations are this
  # campaign's own (rlm_order_lane.sh keeps them next to RESULTS), so they are archived too.
  TUNED_ROWS=${TUNED_ROWS:-3}
  RESULTS=${RESULTS:-$SCRATCH/results_$DATASET}
  REPORT=${REPORT:-$SCRATCH/report_$DATASET}
  ORG=${ORG:-$(dirname "$RESULTS")/organization_$DATASET}
fi
ARCHIVE=${ARCHIVE:-/mnt/nas/gautammahale/archive/rlmorder}
GRAPH=${GRAPH:-/mnt/nas/gautammahale/archive/graphorder}
GUARDS=${GUARDS:-/mnt/nas/gautammahale/benchmark/evaluation/results/loft128k_guards}
SERVER=${SERVER:-ro_root}
PY=${PY:-/mnt/nas/gautammahale/venvs/rlm/bin/python}
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=${HF_HOME:-/mnt/nas/gautammahale/hf_cache} HF_HUB_OFFLINE=1

# Lab storage policy: on a host with an infolab group, write as infolab. A RunPod pod or
# another cloud VM has no such group, and the check does not apply there.
if getent group infolab >/dev/null 2>&1 && [ "$(id -gn)" != infolab ]; then
  echo "Run under primary group infolab (sg infolab -c ...)." >&2
  exit 1
fi
running=$(tmux ls 2>/dev/null | grep -c -E '^ro[0-9]*_(graph|query)_')
if [ "$running" -gt 0 ]; then
  echo "$running lane(s) still running; wait for them to finish." >&2
  exit 1
fi
mkdir -p "$REPORT"
echo "checkout: $(git log --oneline -1)"

comparator=()
reference=()
if [ "$DATASET" = loft ]; then
  if [ -d "$GRAPH/results" ]; then
    comparator=(--structured-results "$GRAPH/results" --organization-dir "$GRAPH/organization")
    reference=(--structured-results "$GRAPH/results" --guards "$GUARDS")
  else
    # Off infolab the structured campaign's results are not there: gate without them, and say so.
    echo "WARN: no structured campaign at $GRAPH/results; gating LOFT with --no-comparator" >&2
    comparator=(--no-comparator)
  fi
fi

# 1. Gate: integrity problems fail; behaviour rows are listed under REVIEW.
# shellcheck disable=SC2086  # the --revision pair, when set, is two words on purpose
$PY -m evaluation.rlm.campaigns.rlm_order_gate "$RESULTS" --dataset "$DATASET" --expect "$EXPECT" \
  ${REVISION:+--revision $REVISION} ${comparator[@]+"${comparator[@]}"} > "$REPORT/${STAGE}_gate.txt" 2>&1
gate=$?
echo "gate exit=$gate"
cat "$REPORT/${STAGE}_gate.txt"

# 2. The server: stop it once no row needs a rerun.
if [ "$gate" -eq 0 ]; then
  tmux kill-session -t "$SERVER" && echo "stopped $SERVER at $(date +%T)"
else
  echo "gate failed: $SERVER kept up in case rows need a rerun"
fi

# 3. Report, with every pair repeated on the rows no revision was tuned on.
$PY -m evaluation.rlm.campaigns.rlm_order_report --results "$RESULTS" --out "$REPORT/$STAGE" \
  ${reference[@]+"${reference[@]}"} --repo . --tuned-rows "$TUNED_ROWS" > "$REPORT/${STAGE}_report_stdout.txt" 2>&1
echo "report exit=$?"

# 4. Archive to NAS and verify: an empty rsync -c -n means the copy is identical.
pairs=("$RESULTS:${STAGE}_results" "$REPORT:${STAGE}_report")
if [ -n "${ORG:-}" ]; then pairs+=("$ORG:${STAGE}_organization"); fi
for pair in "${pairs[@]}"; do
  src=${pair%%:*}
  dst=$ARCHIVE/${pair##*:}
  rsync -a "$src/" "$dst/"
  diffs=$(rsync -a -c -n --itemize-changes "$src/" "$dst/" | wc -l)
  echo "$dst: checksum diffs $diffs, files $(find "$src" -type f | wc -l)/$(find "$dst" -type f | wc -l)"
done
mkdir -p "$ARCHIVE/${STAGE}_logs"
cp -p "$SCRATCH"/logs/"${STAGE}"_*.log "$ARCHIVE/${STAGE}_logs/"
echo "logs archived: $(ls "$ARCHIVE/${STAGE}_logs" | wc -l)"
echo "==== report.md"
cat "$REPORT/$STAGE/report.md"
