#!/bin/bash
# The one vLLM the structured campaign talks to. Pinned to a single named GPU.
#
# Everything in this campaign -- root calls and sub-calls, every lane -- goes to ONE
# HTTP endpoint, so the whole thing is one GPU's worth of work. Measured on fox:
# 80 cells / 4,400 examples in 77 minutes with four lanes against one card, at 3.28x
# effective concurrency. The card was never the bottleneck; adding lanes is free and
# adding GPUs mostly is not.
#
# SIZING. Qwen3-4B bf16 is ~8 GB of weights, and KV is 147,456 B/token, so a batch of
# eight ~20k-token sequences (the widest cell in the grid) is another ~6 GB. The real
# floor is ~14-16 GB. GPU_UTIL below is what vLLM PRE-ALLOCATES, not what it needs:
# 0.85 is what the 77-minute run used and is kept as the default so that run
# reproduces, but 0.35 fits comfortably and leaves most of the card for other people.
#
# CUDA_DEVICE_ORDER IS LOAD-BEARING. fox mixes RTX 6000 Ada (GPU 0,1) with RTX A6000
# (GPU 2,3,4). CUDA orders devices FASTEST_FIRST by default while nvidia-smi orders
# them by PCI bus, so without this the index you measured as idle is not necessarily
# the index CUDA hands you -- you politely pick an empty card and land on someone
# else's. The two orderings happen to agree on fox today; that is luck, not a
# guarantee, and it costs nothing to remove the luck.
set -u
GPU=${GPU:-4}
PORT=${PORT:-8001}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
MAX_LEN=${MAX_LEN:-65536}
GPU_UTIL=${GPU_UTIL:-0.85}
VLLM=${VLLM:-/mnt/nas/gautammahale/benchmark/.venv/bin/vllm}
HF=${HF_HOME:-/mnt/nas/gautammahale/hf_cache}
LOG=${LOG:-/mnt/nas/gautammahale/logs/root_gpu${GPU}.log}
SESSION=${SESSION:-root_gpu${GPU}}

used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
if [ -z "$used" ]; then echo "no GPU $GPU on this host" >&2; exit 1; fi
# Take a shared card only when taking it costs nobody else anything.
if [ "$used" -gt 1000 ]; then
  echo "GPU $GPU already holds ${used} MiB -- someone else is on it. Refusing." >&2
  nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 1
fi
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "port $PORT is already listening. Check it GENERATES before reusing it:" >&2
  echo "  curl -s --max-time 90 http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\" >&2
  echo "    -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":8}'" >&2
  echo "A wedged vLLM still answers /v1/models and still accepts chat requests." >&2
  exit 1
fi

# CPU BINDING. Shared hosts run other people's CPU-heavy jobs beside ours, and a
# server free to spread over every core competes with all of them. CPUS pins the
# server and every process it forks (the engine core included -- affinity is
# inherited) to a named core list, ideally physical cores local to the GPU's NUMA
# node (`nvidia-smi topo -m`, `lscpu -e`). Unset = unbound, as before.
CPUS=${CPUS:-}
PIN=""
[ -n "$CPUS" ] && PIN="taskset -c $CPUS"
THREADS=${THREADS:-4}

mkdir -p "$(dirname "$LOG")"
# GPU may be an index or a UUID (GPU-xxxx): nvidia-smi -i and CUDA_VISIBLE_DEVICES
# both accept either, and a UUID cannot be renumbered under you.
tmux new-session -d -s "$SESSION" \
  "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$HF HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
   OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
   $PIN $VLLM serve $MODEL --served-model-name $MODEL --port $PORT \
   --max-model-len $MAX_LEN --gpu-memory-utilization $GPU_UTIL --enforce-eager -O0 2>&1 | tee $LOG"
echo "serving $MODEL on GPU $GPU, port $PORT, tmux session $SESSION${CPUS:+, cpus $CPUS}"
echo "weights load from NFS and can take several minutes; wait for a real generation:"
echo "  until curl -s --max-time 60 -o /dev/null -w '%{http_code}' http://localhost:$PORT/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":4}' | grep -q 200;"
echo "  do sleep 15; done; echo READY"
