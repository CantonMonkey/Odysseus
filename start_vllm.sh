#!/bin/bash
# Start InternVL3-8B as a vLLM OpenAI-compatible server on port 8088
set -a; source /data3/liangjy/vln/Odysseus/.env; set +a

MODEL=/data3/liangjy/vln/models/OpenGVLab/InternVL3-8B
PORT=8088

echo "[vLLM] Starting InternVL3-8B server on port $PORT ..."
nohup /data3/liangjy/vln/envs/habitat/bin/vllm serve "$MODEL" \
  --dtype bfloat16 \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port $PORT \
  --max-model-len 4096 \
  > /tmp/vllm_server.log 2>&1 &

VLLM_PID=$!
echo "[vLLM] Server PID: $VLLM_PID"

# Wait until server is ready (up to 120s)
for i in $(seq 1 40); do
  sleep 3
  if curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
    echo "[vLLM] Server ready! (${i}x3s elapsed)"
    break
  fi
  if [ $i -eq 40 ]; then
    echo "[vLLM] Timeout waiting for server. Check /tmp/vllm_server.log"
  fi
done
