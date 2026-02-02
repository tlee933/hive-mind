#!/bin/bash
# Hive-Mind Learning Pipeline Orchestrator
# Manages data collection, training, and model deployment

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
CONFIG_PATH="${CONFIG_PATH:-/workspace/config.yaml}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"

# Functions
log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ${GREEN}INFO${NC}: $1"
}

log_warn() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ${YELLOW}WARN${NC}: $1"
}

log_error() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ${RED}ERROR${NC}: $1"
}

# Step 1: Collect data from Redis
collect_data() {
    log_info "Step 1: Collecting training data from Redis..."

    python3 /workspace/scripts/collect_data.py \
        --config "$CONFIG_PATH" \
        --output "$DATA_DIR" \
        --max-items "${MAX_ITEMS:-1000}"

    if [ $? -eq 0 ]; then
        log_info "✅ Data collection complete"
    else
        log_error "❌ Data collection failed"
        return 1
    fi
}

# Step 2: Train LoRA adapter
train_model() {
    log_info "Step 2: Training LoRA adapter..."

    # Find latest dataset
    LATEST_DATASET=$(ls -t "$DATA_DIR"/training_data_*.jsonl 2>/dev/null | head -1)

    if [ -z "$LATEST_DATASET" ]; then
        log_error "No training dataset found in $DATA_DIR"
        return 1
    fi

    log_info "Using dataset: $LATEST_DATASET"

    # Create output directory with timestamp
    TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
    OUTPUT_DIR="$MODELS_DIR/lora_$TIMESTAMP"

    # Train
    python3 /workspace/scripts/train_lora.py \
        --model "$BASE_MODEL" \
        --dataset "$LATEST_DATASET" \
        --output "$OUTPUT_DIR" \
        --lora-r "${LORA_R:-16}" \
        --lora-alpha "${LORA_ALPHA:-32}" \
        --lora-dropout "${LORA_DROPOUT:-0.05}" \
        --lr "${LEARNING_RATE:-2e-4}" \
        --epochs "${EPOCHS:-3}" \
        --batch-size "${BATCH_SIZE:-4}" \
        --grad-accum "${GRAD_ACCUM:-4}"

    if [ $? -eq 0 ]; then
        log_info "✅ Training complete: $OUTPUT_DIR"
        echo "$OUTPUT_DIR" > "$MODELS_DIR/latest_model.txt"
    else
        log_error "❌ Training failed"
        return 1
    fi
}

# Step 3: Export merged model (optional)
export_model() {
    log_info "Step 3: Exporting merged model..."

    LATEST_MODEL=$(cat "$MODELS_DIR/latest_model.txt" 2>/dev/null)

    if [ -z "$LATEST_MODEL" ]; then
        log_warn "No model to export"
        return 0
    fi

    log_info "Exporting model from: $LATEST_MODEL"

    python3 /workspace/scripts/export_model.py \
        --lora-path "$LATEST_MODEL" \
        --base-model "$BASE_MODEL" \
        --output "$LATEST_MODEL/merged"

    if [ $? -eq 0 ]; then
        log_info "✅ Model exported successfully"
    else
        log_warn "⚠️  Model export failed (non-critical)"
    fi
}

# Step 4: Evaluate model
evaluate_model() {
    log_info "Step 4: Evaluating model..."

    LATEST_MODEL=$(cat "$MODELS_DIR/latest_model.txt" 2>/dev/null)

    if [ -z "$LATEST_MODEL" ]; then
        log_warn "No model to evaluate"
        return 0
    fi

    # TODO: Add evaluation script
    log_warn "Evaluation not yet implemented"
}

# Step 5: Deploy model
deploy_model() {
    log_info "Step 5: Deploying model..."

    LATEST_MODEL=$(cat "$MODELS_DIR/latest_model.txt" 2>/dev/null)

    if [ -z "$LATEST_MODEL" ]; then
        log_warn "No model to deploy"
        return 0
    fi

    # TODO: Add deployment logic (e.g., update llama-server)
    log_info "Model ready at: $LATEST_MODEL"
    log_warn "Automatic deployment not yet implemented"
    log_info "To deploy manually:"
    log_info "  1. Convert to GGUF: python3 scripts/convert_to_gguf.py --model $LATEST_MODEL"
    log_info "  2. Restart llama-server with new model"
}

# Main pipeline
run_pipeline() {
    log_info "🐝 Starting Hive-Mind Learning Pipeline"
    log_info "========================================"
    log_info ""
    log_info "Configuration:"
    log_info "  Config: $CONFIG_PATH"
    log_info "  Data Dir: $DATA_DIR"
    log_info "  Models Dir: $MODELS_DIR"
    log_info "  Base Model: $BASE_MODEL"
    log_info ""

    # Run steps
    collect_data || exit 1
    train_model || exit 1
    export_model
    evaluate_model
    deploy_model

    log_info ""
    log_info "✅ Pipeline complete!"
    log_info "========================================"
}

# Parse arguments
case "${1:-run}" in
    run)
        run_pipeline
        ;;
    collect)
        collect_data
        ;;
    train)
        train_model
        ;;
    export)
        export_model
        ;;
    evaluate)
        evaluate_model
        ;;
    deploy)
        deploy_model
        ;;
    *)
        echo "Usage: $0 {run|collect|train|export|evaluate|deploy}"
        exit 1
        ;;
esac
