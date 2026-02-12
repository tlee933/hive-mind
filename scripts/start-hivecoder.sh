#!/bin/bash
# HiveCoder-7B LLM Server startup script
# This script ensures proper ROCm environment for llama-server

# ROCm environment
export HSA_OVERRIDE_GFX_VERSION=12.0.1
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/var/mnt/build/llama.cpp-rocm/build/bin:/opt/rocm/lib:$LD_LIBRARY_PATH

# ROCm optimization for RDNA4
export GPU_MAX_HW_QUEUES=8
export HSA_ENABLE_SDMA=0

# Context size (8K tokens)
export LLAMA_ARG_CTX_SIZE=8192

# Model path - uses symlink for hot-swap deployments
MODEL_PATH="/var/mnt/build/MCP/hive-mind/learning-pipeline/models/foundation_7b_export/HiveCoder-7B-current.gguf"

# Fallback to original if symlink doesn't exist
if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH="/var/mnt/build/MCP/hive-mind/learning-pipeline/models/foundation_7b_export/HiveCoder-7B-Q5_K_M.gguf"
fi

# Start llama-server
# Note: context (-c) is divided among parallel slots (-np)
# 32768 / 4 slots = 8192 per request
exec /usr/local/bin/llama-server \
    -m "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port 8089 \
    -ngl 99 \
    -c 32768 \
    --threads 12 \
    -np 4
