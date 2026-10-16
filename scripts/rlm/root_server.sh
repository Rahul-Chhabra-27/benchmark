#!/bin/bash
# Root-server helpers for the one-command launchers. Source it, then:
#
#   root_server_up      use the vLLM already generating on PORT, or start one with
#                       serve_root.sh and wait (up to 30 min) until it generates
#   root_server_down    stop the server, only if root_server_up started it
#
# Reads PORT, MODEL, GPU, GPU_UTIL, VLLM, HF_HOME, LOGS, SESSION, SERVER_CPUS and
# KEEP_SERVER=1 (leave it running). The same steps run_bxf.sh takes inline.

_rs_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_rs_started=""

root_server_generates() {
  # A real generation, not /v1/models: a wedged vLLM still answers the static routes.
  curl -s --max-time 60 "http://localhost:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":4}" |
    grep -q '"content"'
}

root_server_up() {
  if root_server_generates; then
    echo "using the server already generating on :$PORT"
    return 0
  fi
  _rs_started=${SESSION:-picked_root}
  GPU=$GPU PORT=$PORT GPU_UTIL=$GPU_UTIL VLLM=$VLLM HF_HOME=$HF_HOME CPUS=${SERVER_CPUS:-} \
    LOG=$LOGS/server.log SESSION=$_rs_started bash "$_rs_here/serve_root.sh" >/dev/null || {
    _rs_started=""
    return 1
  }
  echo "starting the root server on GPU $GPU ($_rs_started); waiting for it to generate..."
  for _ in $(seq 120); do
    root_server_generates && { echo "server ready"; return 0; }
    sleep 15
  done
  echo "the server did not come up in 30 minutes; see $LOGS/server.log" >&2
  return 1
}

root_server_down() {
  [ -n "$_rs_started" ] && [ "${KEEP_SERVER:-0}" != 1 ] || return 0
  tmux kill-session -t "$_rs_started" 2>/dev/null && echo "stopped the server (tmux $_rs_started)"
  return 0
}
