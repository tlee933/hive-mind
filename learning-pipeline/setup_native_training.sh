#!/bin/bash
# Setup Native Training Environment
# Uses TheRock ROCm 7.12 build with gfx1201 optimizations

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}🐝 HIVE-MIND NATIVE TRAINING SETUP${NC}"
echo -e "${GREEN}======================================================================${NC}\n"

# Paths
THEROCK="/mnt/build/TheRock"
HIVE_MIND="/var/mnt/build/MCP/hive-mind"
VENV="$HIVE_MIND/.venv"

echo -e "${BLUE}System Configuration:${NC}"
echo "  ROCm Build: $THEROCK"
echo "  Python: $(python3 --version)"
echo "  GPU: gfx1201 (AMD Radeon AI PRO R9700)"
echo "  OS: Fedora 43 Atomic"
echo ""

# Check ROCm
if [ ! -d "$THEROCK/build/artifacts" ]; then
    echo -e "${YELLOW}⚠ TheRock build not found at $THEROCK${NC}"
    exit 1
fi

echo -e "${GREEN}✓ TheRock ROCm 7.12 build found${NC}"

# Setup environment
echo -e "\n${BLUE}Setting up environment...${NC}"

export ROCM_HOME="$THEROCK/build/artifacts/base_run_generic/opt/rocm"
export PATH="$ROCM_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_HOME/lib:$LD_LIBRARY_PATH"
export HSA_OVERRIDE_GFX_VERSION=12.0.1
export PYTORCH_ROCM_ARCH=gfx1201
export HIP_VISIBLE_DEVICES=0

echo "  ROCM_HOME=$ROCM_HOME"
echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "  PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"

# Activate Hive-Mind venv
echo -e "\n${BLUE}Activating virtual environment...${NC}"
source "$VENV/bin/activate"

# Install training dependencies
echo -e "\n${BLUE}Installing training dependencies...${NC}"

pip install --upgrade pip setuptools wheel

echo "  Installing PyTorch with ROCm 6.2 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2 || echo "  (PyTorch already installed)"

echo "  Installing transformers and training libraries..."
pip install \
    transformers>=4.40.0 \
    datasets>=2.18.0 \
    peft>=0.10.0 \
    accelerate>=0.29.0 \
    bitsandbytes>=0.43.0 \
    sentencepiece>=0.2.0 \
    protobuf>=4.25.0 \
    tqdm>=4.66.0 \
    tensorboard>=2.16.0 \
    scipy>=1.12.0 \
    scikit-learn>=1.4.0

echo -e "\n${GREEN}✓ Dependencies installed${NC}"

# Test PyTorch + ROCm
echo -e "\n${BLUE}Testing PyTorch + ROCm...${NC}"

python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  Device name: {torch.cuda.get_device_name(0)}')
    print(f'  Device capability: {torch.cuda.get_device_capability(0)}')
"

echo -e "\n${GREEN}======================================================================${NC}"
echo -e "${GREEN}✅ NATIVE TRAINING ENVIRONMENT READY${NC}"
echo -e "${GREEN}======================================================================${NC}\n"

echo -e "${BLUE}Next steps:${NC}"
echo "  1. source $VENV/bin/activate"
echo "  2. export ROCM_HOME=$THEROCK/build/artifacts/base_run_generic/opt/rocm"
echo "  3. export HSA_OVERRIDE_GFX_VERSION=12.0.1"
echo "  4. cd $HIVE_MIND/learning-pipeline"
echo "  5. python scripts/train_lora.py --config configs/training_config.yaml"
echo ""
