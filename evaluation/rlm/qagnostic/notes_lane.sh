#!/bin/bash
# A query-agnostic campaign host script (2026-09-09), kept as it ran; see README.md.
# The query-free index: one pass over the corpus with the question nowhere in it.
# Built ONCE per subset -- every LOFT row shares one document -- so the per-query
# cost is this build divided by 100.
set -u
cd /mnt/home/gautammahale/qagnostic/checkout-notes
export HF_HOME=/mnt/nas/gautammahale/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
W=${2:-8000}; O=${3:-1600}
/mnt/nas/gautammahale/venvs/rlm/bin/python -m evaluation.rlm.window_notes \
  --dataset loft --data-dir "${1}_128k" --split test --window "$W" --overlap "$O" \
  --base-url http://localhost:8001/v1 \
  --out-dir /mnt/home/gautammahale/qagnostic/notes 2>&1 \
  | tee -a "/mnt/home/gautammahale/qagnostic/logs/notes_${1}_w${W}.log"
