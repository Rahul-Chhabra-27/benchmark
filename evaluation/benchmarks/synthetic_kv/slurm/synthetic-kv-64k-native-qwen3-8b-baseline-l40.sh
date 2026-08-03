#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# 64K Synthetic-KV no-YaRN diagnostic baseline for comparison with YaRN-2.
#SBATCH --job-name=synthetickv-64k-native-qwen3-8b
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=18:00:00
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
  --config-file benchmarks/synthetic_kv/slurm/evaluate_synthetic_kv_64k_native_qwen3_8b_baseline.yaml \
  --profile synthetic-kv-64k-baseline
