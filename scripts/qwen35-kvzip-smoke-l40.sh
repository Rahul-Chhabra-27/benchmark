#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Short Qwen3.5 + KVzip adapter smoke test on one L40 GPU.
#SBATCH --job-name=qwen35-kvzip-smoke
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/rethinkingai-self/25m0820/logs/%x-%j.out
#SBATCH --error=/home/rethinkingai-self/25m0820/logs/%x-%j.err

set -euo pipefail

BASE_DIR=/home/rethinkingai-self/25m0820
REPO_DIR="${BASE_DIR}/kvpress"
PYTHON="${BASE_DIR}/miniconda3/envs/kvpress/bin/python"
MODEL_DIR="${REPO_DIR}/Qwen3.5-9B"

mkdir -p "${BASE_DIR}/logs"
cd "${REPO_DIR}"

export HF_HOME="${BASE_DIR}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

"${PYTHON}" scripts/qwen35_kvzip_smoke_test.py \
  --model "${MODEL_DIR}" \
  --context-tokens 256 \
  --compression-ratio 0.25 \
  --device cuda:0

