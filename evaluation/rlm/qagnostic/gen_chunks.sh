#!/bin/bash
# A query-agnostic campaign host script (2026-09-09), kept as it ran; see README.md.
set -eu
cd /mnt/home/gautammahale/qagnostic/checkout
export HF_HOME=/mnt/nas/gautammahale/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
PY=/mnt/nas/gautammahale/venvs/rlm/bin/python
OUT=/mnt/home/gautammahale/qagnostic/chunksets
IDS=/mnt/home/gautammahale/qagnostic/ids
mkdir -p "$OUT" "$IDS"

gen () {  # subset selector k window overlap [seed]
  $PY -m evaluation.rlm.query_free_chunks --dataset loft --data-dir "${1}_128k" --split test \
      --selector "$2" --k "$3" --window "$4" --overlap "$5" --seed "${6:-0}" \
      --out-dir "$OUT" --force >/dev/null
}

for s in nq hotpotqa musique qampari quest; do
  # The ids the campaign never ran: test rows 55..99. Emitted from the loader, once.
  $PY -m evaluation.rlm.query_free_chunks --dataset loft --data-dir "${s}_128k" --split test \
      --selector lead --k 5 --window 2000 --overlap 400 --out-dir "$OUT" --force \
      --ids-out "$IDS/${s}_128k.holdout45.ids" --ids-out-range 55:100 | tail -3
  for k in 2 5 10; do
    gen "$s" lead   "$k" 2000 400
    gen "$s" stride "$k" 2000 400
  done
  for seed in 0 1 2; do gen "$s" random 5 2000 400 "$seed"; done
  gen "$s" random 2  2000 400 0
  gen "$s" random 10 2000 400 0
  gen "$s" all    0  8000 1600
  echo "  == $s done =="
done
