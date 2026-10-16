#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Wait for a campaign PID, then TERM only the Qwen3.5 vLLM servers on 8012/8013/8014.
set -u
CAMPAIGN_PID="${1:?campaign pid}"
echo "[cleanup] waiting for campaign pid ${CAMPAIGN_PID}"
while kill -0 "$CAMPAIGN_PID" 2>/dev/null; do
  sleep 15
done
echo "[cleanup] campaign gone; stopping 8012/8013/8014"
for spec in "Qwen/Qwen3.5-4B|8012" "Qwen/Qwen3.5-9B|8014" "Qwen/Qwen3.5-27B|8013"; do
  IFS='|' read -r model port <<<"$spec"
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    echo "[cleanup] TERM ${model} pid ${pid}"
    kill "$pid" 2>/dev/null || true
  done < <(pgrep -f "vllm serve ${model} .*--port ${port}" || true)
  fuser -k "${port}/tcp" 2>/dev/null || true
done
sleep 2
echo "[cleanup] done $(date -Iseconds)"
