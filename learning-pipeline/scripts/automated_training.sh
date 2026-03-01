#!/bin/bash
##############################################################################
# Automated Training Script for Hive-Mind
# Runs: Data collection → Training → Model versioning
##############################################################################

set -euo pipefail

# ROCm stability environment variables (fixes HIPBLAS crashes)
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=4
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
DATA_DIR="$PROJECT_DIR/data/automated"
MODELS_DIR="$PROJECT_DIR/models/automated"
CONFIG_FILE="$PROJECT_DIR/../config.yaml"
VENV_PYTHON="$PROJECT_DIR/../.venv/bin/python3"

# Training parameters
BASE_MODEL="Qwen/Qwen3-14B"
MIN_EXAMPLES=10  # Minimum examples needed to train
LORA_R=8
LORA_ALPHA=16
EPOCHS=3
BATCH_SIZE=1     # Qwen3-14B BF16 leaves minimal VRAM headroom
GRAD_ACCUM=8     # Compensate for small batch with more accumulation
LEARNING_RATE=2e-4
VRAM_OVERHEAD=0.25  # Reserve 25% VRAM overhead

# Create directories
mkdir -p "$LOG_DIR" "$DATA_DIR" "$MODELS_DIR"

# Setup logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "🤖 Hive-Mind Automated Training Run"
echo "================================================================================"
echo "Started: $(date)"
echo "Log file: $LOG_FILE"
echo ""

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# Function to handle errors
error_exit() {
    log "❌ ERROR: $1"
    log "Training run failed. Check logs at: $LOG_FILE"
    exit 1
}

# Check prerequisites
log "📋 Checking prerequisites..."

if [ ! -f "$CONFIG_FILE" ]; then
    error_exit "Config file not found: $CONFIG_FILE"
fi

if [ ! -x "$VENV_PYTHON" ]; then
    error_exit "Python venv not found: $VENV_PYTHON"
fi

# Check Redis connection (retry up to 5 times — Docker containers may still be starting)
log "🔌 Testing Redis connection..."
REDIS_OK=0
for attempt in 1 2 3 4 5; do
    if "$VENV_PYTHON" -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from redis.cluster import RedisCluster, ClusterNode
import yaml
with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f)
redis_config = config['redis']
startup_nodes = [ClusterNode(n['host'], n['port']) for n in redis_config['nodes']]
client = RedisCluster(startup_nodes=startup_nodes, password=redis_config['password'])
client.ping()
print('✅ Redis connected')
" 2>/dev/null; then
        REDIS_OK=1
        break
    fi
    log "⏳ Redis not ready (attempt $attempt/5), waiting 30s..."
    sleep 30
done
if [ "$REDIS_OK" -ne 1 ]; then
    log "⚠️  Cannot connect to Redis cluster after 5 attempts"
    log "Skipping training this run. Redis may be down for maintenance."
    exit 0
fi

# Step 1: Collect data from Redis
log "📊 Step 1: Collecting training data from Redis..."
DATASET_FILE="$DATA_DIR/training_data_${TIMESTAMP}.jsonl"

"$VENV_PYTHON" "$SCRIPT_DIR/collect_data.py" \
    --config "$CONFIG_FILE" \
    --output "$DATA_DIR" \
    --max-items 1000 || error_exit "Data collection failed"

# Find the most recent dataset
DATASET_FILE=$(ls -t "$DATA_DIR"/training_data_*.jsonl | head -1)

if [ ! -f "$DATASET_FILE" ]; then
    error_exit "Dataset file not created"
fi

# Check if we have enough examples
NUM_EXAMPLES=$(wc -l < "$DATASET_FILE")
log "📈 Collected $NUM_EXAMPLES examples"

if [ "$NUM_EXAMPLES" -lt "$MIN_EXAMPLES" ]; then
    log "⚠️  Not enough examples for training (need at least $MIN_EXAMPLES)"
    log "Skipping training this run. Try again when more data is available."
    exit 0
fi

# Step 2: Train model
log "🧠 Step 2: Training model with LoRA..."
MODEL_OUTPUT="$MODELS_DIR/model_${TIMESTAMP}"

# Stop llama-server to free GPU VRAM for training
LLM_WAS_RUNNING=0
if systemctl is-active --quiet hivecoder-llm.service 2>/dev/null; then
    log "⏸️  Stopping hivecoder-llm.service to free GPU VRAM..."
    sudo systemctl stop hivecoder-llm.service
    LLM_WAS_RUNNING=1
    sleep 5
fi

TRAIN_EXIT=0
"$VENV_PYTHON" "$SCRIPT_DIR/train_lora.py" \
    --model "$BASE_MODEL" \
    --dataset "$DATASET_FILE" \
    --output "$MODEL_OUTPUT" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LEARNING_RATE" \
    --vram-overhead "$VRAM_OVERHEAD" || TRAIN_EXIT=$?

# Restart llama-server regardless of training outcome
if [ "$LLM_WAS_RUNNING" -eq 1 ]; then
    log "▶️  Restarting hivecoder-llm.service..."
    sudo systemctl start hivecoder-llm.service
fi

if [ "$TRAIN_EXIT" -ne 0 ]; then
    error_exit "Training failed"
fi

# Step 3: Verify model was saved
log "✅ Step 3: Verifying model..."

if [ ! -f "$MODEL_OUTPUT/adapter_model.safetensors" ]; then
    error_exit "Model adapters not found"
fi

ADAPTER_SIZE=$(du -h "$MODEL_OUTPUT/adapter_model.safetensors" | cut -f1)
log "📦 Model saved: $ADAPTER_SIZE"

# Step 4: Update latest symlink
log "🔗 Step 4: Updating 'latest' symlink..."
LATEST_LINK="$MODELS_DIR/latest"
rm -f "$LATEST_LINK"
ln -sf "$(basename "$MODEL_OUTPUT")" "$LATEST_LINK"
log "Latest model: $LATEST_LINK -> $(basename "$MODEL_OUTPUT")"

# Step 5: Generate training report
log "📊 Step 5: Generating training report..."
REPORT_FILE="$LOG_DIR/report_${TIMESTAMP}.txt"

cat > "$REPORT_FILE" << EOF
===============================================================================
Hive-Mind Training Report
===============================================================================
Date: $(date)
Run ID: $TIMESTAMP

DATA COLLECTION
---------------
Dataset: $DATASET_FILE
Examples collected: $NUM_EXAMPLES
Source: Redis learning queue

TRAINING CONFIGURATION
----------------------
Base model: $BASE_MODEL
LoRA rank: $LORA_R
LoRA alpha: $LORA_ALPHA
Epochs: $EPOCHS
Batch size: $BATCH_SIZE
Gradient accumulation: $GRAD_ACCUM
Learning rate: $LEARNING_RATE

MODEL OUTPUT
------------
Location: $MODEL_OUTPUT
Adapter size: $ADAPTER_SIZE
Symlink: $LATEST_LINK

METRICS
-------
EOF

# Extract metrics from training
if [ -f "$MODEL_OUTPUT/training_metrics.json" ]; then
    "$VENV_PYTHON" -c "
import json
with open('$MODEL_OUTPUT/training_metrics.json') as f:
    metrics = json.load(f)
print(f\"Training runtime: {metrics.get('train_runtime', 'N/A')} seconds\")
print(f\"Samples/second: {metrics.get('train_samples_per_second', 'N/A')}\")
print(f\"Steps/second: {metrics.get('train_steps_per_second', 'N/A')}\")
print(f\"Final loss: {metrics.get('train_loss', 'N/A')}\")
" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

LOGS
----
Full log: $LOG_FILE
Report: $REPORT_FILE

===============================================================================
EOF

cat "$REPORT_FILE"

# Step 6: Cleanup old files (keep last 30 days)
log "🧹 Step 6: Cleaning up old files..."
find "$DATA_DIR" -name "*.jsonl" -mtime +30 -delete 2>/dev/null || true
find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true
find "$MODELS_DIR" -type d -name "model_*" -mtime +30 -exec rm -rf {} + 2>/dev/null || true

log "✅ Cleanup complete"

# Summary
echo ""
echo "================================================================================"
echo "🎉 Training Run Complete!"
echo "================================================================================"
echo "Duration: $(date)"
echo "Model: $MODEL_OUTPUT"
echo "Examples: $NUM_EXAMPLES"
echo "Log: $LOG_FILE"
echo "Report: $REPORT_FILE"
echo ""
echo "To use this model:"
echo "  cd $MODELS_DIR"
echo "  ls -la latest  # Points to: $(basename "$MODEL_OUTPUT")"
echo "================================================================================"

exit 0
