#!/bin/bash
# Run llama-server natively with ROCm

MODEL="/mnt/nas-moar/models/lmstudio-backup/lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
LLAMA_BIN="/var/mnt/build/llama.cpp-rocm/build/bin/llama-server"
PORT=8080

echo "🦙 Starting llama-server (native with ROCm)"

# Kill existing
pkill -9 -f llama-server 2>/dev/null || true
sleep 2

# Run in background
nohup $LLAMA_BIN \
  -m "$MODEL" \
  --host 0.0.0.0 \
  --port $PORT \
  -ngl 99 \
  -c 8192 \
  --threads 8 \
  --batch-size 512 \
  --ubatch-size 128 \
  --flash-attn auto \
  --cont-batching \
  > /tmp/llama-server.log 2>&1 &

LLAMA_PID=$!
echo "🚀 Started llama-server (PID: $LLAMA_PID)"
echo "📋 Logs: tail -f /tmp/llama-server.log"

sleep 10

# Test
echo ""
echo "🧪 Testing endpoint..."
curl -s http://localhost:$PORT/health | head -5 || echo "Still loading model..."

echo ""
echo "✅ llama-server running!"
echo "   http://localhost:$PORT"
