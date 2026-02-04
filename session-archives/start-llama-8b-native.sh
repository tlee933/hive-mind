#!/bin/bash
# Run llama-server natively with ROCm - Qwen3-8B for tool use

MODEL="/mnt/nas-moar/models/lmstudio-backup/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
LLAMA_BIN="/var/mnt/build/llama.cpp-rocm/build/bin/llama-server"
PORT=8088

echo "🦙 Starting llama-server with Qwen3-8B (native with ROCm)"

# Kill existing on this port
lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
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
  > /tmp/llama-server-8b.log 2>&1 &

LLAMA_PID=$!
echo "🚀 Started llama-server-8b (PID: $LLAMA_PID)"
echo "📋 Logs: tail -f /tmp/llama-server-8b.log"

sleep 15

# Test
echo ""
echo "🧪 Testing endpoint..."
curl -s http://localhost:$PORT/health | head -5 || echo "Still loading model..."

echo ""
echo "✅ llama-server running on port $PORT!"
echo "   http://localhost:$PORT"
