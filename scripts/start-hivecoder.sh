#!/bin/bash
# HiveCoder (Qwen3-14B) LLM Server startup script
# This script ensures proper ROCm environment for llama-server

# ROCm environment
export HSA_OVERRIDE_GFX_VERSION=12.0.1
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# ROCm optimization for RDNA4
export GPU_MAX_HW_QUEUES=8
export HSA_ENABLE_SDMA=0

# Model path - uses symlink for hot-swap deployments (LoRA-trained versions)
MODEL_PATH="/var/mnt/build/MCP/hive-mind/learning-pipeline/models/foundation_14b_export/HiveCoder-current.gguf"

# Fallback to base Qwen3-14B if symlink doesn't exist
if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH="/home/hashcat/Models/Qwen3-14B-Q4_K_M.gguf"
fi

# Start llama-server
# Context: 32768 / 2 slots = 16384 per request
# KV cache: q8_0 cuts memory bandwidth ~50% vs f16, negligible quality loss
exec /usr/local/bin/llama-server \
    -m "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port 8089 \
    -ngl 99 \
    -c 32768 \
    --threads 12 \
    -np 2 \
    --flash-attn on \
    --cont-batching \
    -ctk q8_0 \
    -ctv q8_0
