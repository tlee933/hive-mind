# Hive-Mind Learning Pipeline
# ROCm + PyTorch + LoRA Fine-tuning

FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV ROCM_HOME=/opt/rocm
ENV PATH=$ROCM_HOME/bin:$PATH
ENV HSA_OVERRIDE_GFX_VERSION=12.0.1
ENV PYTORCH_ROCM_ARCH=gfx1201

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install training dependencies
RUN pip install \
    transformers>=4.40.0 \
    datasets>=2.18.0 \
    peft>=0.10.0 \
    accelerate>=0.29.0 \
    bitsandbytes>=0.43.0 \
    sentencepiece>=0.2.0 \
    protobuf>=4.25.0 \
    redis[hiredis]>=5.0.0 \
    pyyaml>=6.0 \
    tqdm>=4.66.0 \
    tensorboard>=2.16.0 \
    wandb>=0.16.0 \
    scipy>=1.12.0 \
    scikit-learn>=1.4.0

# Create working directory
WORKDIR /workspace

# Copy pipeline scripts
COPY scripts/ /workspace/scripts/
COPY configs/ /workspace/configs/

# Make scripts executable
RUN chmod +x /workspace/scripts/*.py /workspace/scripts/*.sh 2>/dev/null || true

# Expose tensorboard port
EXPOSE 6006

# Default command
CMD ["/bin/bash"]
