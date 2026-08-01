#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=synthetickv-32k-awq-budgets
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
  --config-file evaluate_config.yaml \
  --profile synthetic-kv-32k-native-awq-all-budgets
