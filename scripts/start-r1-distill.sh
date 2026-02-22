#!/bin/bash
# DeepSeek-R1-Distill-Qwen-14B LLM Server startup script
# Reasoning model for the intelligent router — teaches HiveCoder via distillation

# ROCm environment
export HSA_OVERRIDE_GFX_VERSION=12.0.1
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/var/mnt/build/llama.cpp-rocm/build/bin:/opt/rocm/lib:$LD_LIBRARY_PATH

# ROCm optimization for RDNA4
export GPU_MAX_HW_QUEUES=8
export HSA_ENABLE_SDMA=0

# Model path
MODEL_PATH="/home/hashcat/Models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Start llama-server for R1-Distill
# Port 8080, reasoning format for native <think> blocks
exec /usr/local/bin/llama-server \
    -m "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port 8080 \
    -ngl 99 \
    -c 32768 \
    --threads 12 \
    --flash-attn auto \
    --cont-batching \
    --cache-prompt \
    --batch-size 512 \
    --ubatch-size 256 \
    -np 2 \
    --reasoning-format deepseek \
    --reasoning-budget -1
