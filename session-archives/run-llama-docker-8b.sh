#!/bin/bash
# Run llama-server in Docker with ROCm - Qwen3-8B for tool use on port 8088

MODEL="/models/lmstudio-backup/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
PORT=8088

echo "🦙 Starting llama-server with Qwen3-8B (Docker)"

# Stop existing instance
docker stop llama-server-8b 2>/dev/null || true
docker rm llama-server-8b 2>/dev/null || true

# Run llama-server with ROCm
docker run -d \
  --name llama-server-8b \
  --restart unless-stopped \
  --network host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /mnt/nas-moar/models:/models:ro \
  -v /var/mnt/build/llama.cpp-rocm/build/bin:/llama:ro \
  -e HSA_OVERRIDE_GFX_VERSION=12.0.1 \
  -e ROCR_VISIBLE_DEVICES=0 \
  rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0 \
  /llama/llama-server \
    -m "$MODEL" \
    --host 0.0.0.0 \
    --port $PORT \
    -ngl 99 \
    -c 8192 \
    --threads 8 \
    --batch-size 512 \
    --ubatch-size 128 \
    --flash-attn auto \
    --cont-batching

echo "⏳ Waiting for server to start..."
sleep 15

# Test
echo ""
echo "🧪 Testing llama-server..."
curl -s http://localhost:$PORT/health || echo "Server starting..."

echo ""
echo "✅ llama-server-8b deployed!"
echo "📍 Endpoint: http://localhost:$PORT"
echo "🧪 Test: curl http://localhost:$PORT/v1/models"
echo "📋 Logs: docker logs -f llama-server-8b"
