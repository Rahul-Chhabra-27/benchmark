#!/bin/bash
# A query-agnostic campaign host script (2026-09-09), kept as it ran; see README.md.
# One lane = one subset, one stage. The campaign convention: several lanes
# concurrently against ONE vLLM (measured 3.28x effective concurrency, so the card
# is not the bottleneck and more lanes are close to free).
set -u
export REPO=/mnt/home/gautammahale/qagnostic/checkout
export RESULTS=/mnt/home/gautammahale/qagnostic/results
export QCHUNKS=/mnt/home/gautammahale/qagnostic/chunksets
export IDS=/mnt/home/gautammahale/qagnostic/ids
export CHUNKSETS=/mnt/nas/gautammahale/chunksets
export HF_HOME=/mnt/nas/gautammahale/hf_cache
export PY=/mnt/nas/gautammahale/venvs/rlm/bin/python
export BASE_URL=http://localhost:8001/v1
export LIMIT=55 SPLIT=test
mkdir -p "$RESULTS" /mnt/home/gautammahale/qagnostic/logs
cd "$REPO"
bash scripts/rlm/structured_grid.sh "$1" "$2" 2>&1 | tee -a "/mnt/home/gautammahale/qagnostic/logs/$1_$2.log"
