#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Qwen3-32B BF16 requires two 48 GB L40 GPUs with device_map=auto.
# The matrix includes the no-compression baseline plus 256 MB, 512 MB, 1 GB,
# 2 GB, and 4 GB KV-memory budgets.
#SBATCH --job-name=loft128k-qwen3-32b-yarn4-dgx
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/home/rethinkingai-self/25m0820/logs/%x-%j.out
#SBATCH --error=/home/rethinkingai-self/25m0820/logs/%x-%j.err

set -euo pipefail

BASE_DIR=/home/rethinkingai-self/25m0820
EVAL_DIR="${BASE_DIR}/kvpress/evaluation"
PYTHON="${BASE_DIR}/miniconda3/envs/kvpress/bin/python"

mkdir -p "${BASE_DIR}/logs"
cd "${EVAL_DIR}"

export HF_HOME="${BASE_DIR}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

"${PYTHON}" run_matrix.py \
  --config-file benchmarks/loft/slurm/evaluate_loft128k_qwen3_32b_yarn4_fullcontext.yaml \
  --profile loft128k-all
