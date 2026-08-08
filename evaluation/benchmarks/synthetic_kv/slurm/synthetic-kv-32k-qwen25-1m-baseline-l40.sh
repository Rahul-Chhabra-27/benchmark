#!/bin/bash
#SBATCH --job-name=synthetickv-32k-qwen25-1m
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
mkdir -p "${BASE_DIR}/logs"
cd "${BASE_DIR}/kvpress/evaluation"
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
"${BASE_DIR}/miniconda3/envs/kvpress/bin/python" run_matrix.py \
  --config-file benchmarks/synthetic_kv/slurm/evaluate_synthetic_kv_32k_qwen25_1m_baseline.yaml \
  --profile synthetic-kv-32k-qwen25-1m-baseline
