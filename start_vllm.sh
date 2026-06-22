#!/bin/bash
# Start InternVL3-8B as a vLLM OpenAI-compatible server on port 8088
#
# Reads VLN_LOCAL_MODEL from environment (set via .env) for portability.
# Run: set -a; source .env; set +a && bash start_vllm.sh

MODEL="${VLN_LOCAL_MODEL:-/root/autodl-tmp/models/OpenGVLab/InternVL3-8B}"
PORT=8088
VLLM_BIN="/root/miniconda3/envs/habitat/bin/vllm"

echo "[vLLM] Model : $MODEL"
echo "[vLLM] Binary: $VLLM_BIN"
echo "[vLLM] Starting InternVL3-8B server on port $PORT ..."

nohup "$VLLM_BIN" serve "$MODEL" \
  --served-model-name InternVL3-8B \
  --dtype bfloat16 \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port $PORT \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.75 \
  --no-enable-prefix-caching \
  > /tmp/vllm_server.log 2>&1 &

VLLM_PID=$!
echo "[vLLM] Server PID: $VLLM_PID (logs: /tmp/vllm_server.log)"

# Wait until server is ready (up to 120s)
for i in $(seq 1 40); do
  sleep 3
  if curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
    echo "[vLLM] Server ready! (${i}x3s elapsed)"
    exit 0
  fi
done
echo "[vLLM] Timeout waiting for server. Check /tmp/vllm_server.log"
exit 1
