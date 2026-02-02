# 🚀 Learning Pipeline Deployment Guide

**Production deployment checklist and best practices**

---

## 📋 Pre-Deployment Checklist

### Hardware Requirements

- [ ] GPU with 12GB+ VRAM (16GB+ recommended for 7B models)
- [ ] ROCm 6.2+ installed and functional
- [ ] Docker and Docker Compose installed
- [ ] 50GB+ free disk space for models and data
- [ ] Stable network connection to Redis cluster

### Software Requirements

- [ ] Redis cluster operational with learning queue enabled
- [ ] Config file (`config.yaml`) created with correct Redis credentials
- [ ] Base model accessible (downloaded or network-accessible)
- [ ] Training configuration customized for your hardware

### Security

- [ ] Redis password set and secured
- [ ] Data directory permissions configured correctly
- [ ] No sensitive data in training examples
- [ ] Model output directory secured (models are valuable IP)

---

## 🔧 Initial Setup

### 1. Configure Redis Connection

Edit your `config.yaml`:

```yaml
redis:
  cluster_mode: true
  nodes:
    - host: "your-redis-host"  # Change this
      port: 7000
  password: "your-secure-password"  # Change this

learning:
  enabled: true
  queue_name: "learning:queue"
  batch_size: 100
  max_queue_length: 100000
```

### 2. Customize Training Parameters

Copy and edit training configuration:

```bash
cp configs/training_config.example.yaml configs/training_config.yaml
# Edit configs/training_config.yaml with your preferred settings
```

Key parameters to adjust:
- `model.name`: Your base model
- `lora.r`: LoRA rank (8/16/32)
- `training.per_device_train_batch_size`: Based on your VRAM
- `training.num_epochs`: Training duration

### 3. Test Environment

```bash
# Build container
make build

# Run tests
make test

# Check GPU
make gpu-check
```

---

## 🎯 Production Deployment

### Option A: Manual Execution

```bash
# 1. Build container
docker-compose build

# 2. Test with small dataset
docker-compose run --rm -e MAX_ITEMS=10 -e EPOCHS=1 learning-pipeline \
    bash scripts/pipeline.sh run

# 3. Run full pipeline
docker-compose run --rm learning-pipeline bash scripts/pipeline.sh run
```

### Option B: Automated with Make

```bash
# Full pipeline
make run

# Or step-by-step
make collect  # Collect data
make train    # Train model
make export   # Export for deployment
```

### Option C: Scheduled Training (Cron)

Add to crontab:

```bash
# Daily data collection at 2 AM
0 2 * * * cd /path/to/hive-mind/learning-pipeline && make collect

# Weekly training on Sunday at 3 AM
0 3 * * 0 cd /path/to/hive-mind/learning-pipeline && make run
```

---

## 📊 Monitoring

### TensorBoard

```bash
# Start TensorBoard
make logs

# View at http://localhost:6006
```

### Check Training Progress

```bash
# View live logs
docker-compose logs -f learning-pipeline

# Check latest model
cat models/latest_model.txt

# View training metrics
cat models/lora_*/training_metrics.json
```

### GPU Utilization

```bash
# In another terminal while training
watch -n 1 'docker exec hive-mind-learning rocm-smi'
```

---

## 🔄 Model Deployment

### Export Trained Model

```bash
# Export merged model
make export

# Find exported model
ls -lh models/lora_*/merged/
```

### Convert to GGUF (for llama.cpp)

```bash
# Manual conversion steps:

# 1. Get llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 2. Convert to GGUF
python3 convert_hf_to_gguf.py /path/to/hive-mind/learning-pipeline/models/lora_latest/merged/hf

# 3. Quantize (optional but recommended)
./llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M

# 4. Deploy to llama-server
cp model-q4_k_m.gguf /path/to/models/
# Update llama-server config and restart
```

### Update llama-server

```bash
# Stop current llama-server
pkill -f llama-server

# Start with new model
/path/to/llama-server \
    -m /path/to/models/model-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99 \
    -c 8192
```

---

## 🛡️ Best Practices

### Data Quality

1. **Filter Bad Data**: Remove failed or low-quality interactions
2. **Deduplicate**: Remove duplicate training examples
3. **Balance Dataset**: Ensure diverse tool usage examples
4. **Validate Format**: Check examples match instruction format

### Training

1. **Start Small**: Train on 100-1000 examples first
2. **Monitor Loss**: Watch for overfitting (loss increases on validation)
3. **Save Checkpoints**: Keep multiple checkpoints for rollback
4. **Version Models**: Tag models with training date and metrics

### Resource Management

1. **GPU Memory**: Monitor with `rocm-smi`, adjust batch size if OOM
2. **Disk Space**: Clean old checkpoints regularly
3. **Training Time**: Schedule during off-peak hours
4. **Bandwidth**: Cache models locally to avoid repeated downloads

### Security

1. **Protect Models**: Trained models contain your workflow knowledge
2. **Sanitize Data**: Remove PII from training examples
3. **Access Control**: Limit who can trigger training
4. **Audit Trail**: Log all training runs and deployments

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

```bash
# Solution 1: Reduce batch size
docker-compose run --rm \
    -e BATCH_SIZE=1 \
    -e GRAD_ACCUM=8 \
    learning-pipeline bash scripts/pipeline.sh train

# Solution 2: Use gradient checkpointing (edit config)
# In training_config.yaml:
# hardware:
#   gradient_checkpointing: true

# Solution 3: Use smaller LoRA rank
# In training_config.yaml:
# lora:
#   r: 8  # Instead of 16
```

### Training Too Slow

```bash
# Increase batch size (if you have VRAM headroom)
docker-compose run --rm \
    -e BATCH_SIZE=8 \
    -e GRAD_ACCUM=2 \
    learning-pipeline bash scripts/pipeline.sh train

# Use fewer training steps
docker-compose run --rm \
    -e EPOCHS=2 \
    learning-pipeline bash scripts/pipeline.sh train
```

### No Training Data

```bash
# Check Redis connection
docker-compose run --rm learning-pipeline \
    python3 -c "import redis; r = redis.Redis(host='your-host', port=7000, password='your-pass'); print(r.ping())"

# Check queue length
docker-compose run --rm learning-pipeline \
    redis-cli -h your-host -p 7000 -a your-pass XLEN learning:queue

# Enable learning in config.yaml
# learning:
#   enabled: true
```

### GPU Not Detected

```bash
# Check ROCm
docker run --rm --device=/dev/kfd --device=/dev/dri \
    rocm/pytorch:rocm6.2 rocm-smi

# Check HSA_OVERRIDE_GFX_VERSION
# For gfx1201 (R9700 XT):
# In docker-compose.yml:
# environment:
#   - HSA_OVERRIDE_GFX_VERSION=12.0.1
```

---

## 📈 Performance Tuning

### For R9700 XT (32GB VRAM)

```yaml
# Optimal settings
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2

lora:
  r: 16  # or 32 for more capacity
```

Expected: ~20 min/epoch for 7B model with 1000 examples

### For RDNA2 (12GB VRAM)

```yaml
# Conservative settings
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8

lora:
  r: 8  # Lower capacity but fits in memory
```

Expected: ~40 min/epoch for 7B model with 1000 examples

---

## 📝 Logging and Auditing

### Training Logs

```bash
# View training output
tail -f models/lora_*/logs/tensorboard_logs

# Check final metrics
cat models/lora_*/training_metrics.json
```

### Data Collection Logs

```bash
# View collection metadata
cat data/metadata_*.json

# Count training examples
wc -l data/training_data_*.jsonl
```

### Deployment Tracking

Create a deployment log:

```bash
# models/DEPLOYMENT_LOG.md
echo "## $(date '+%Y-%m-%d %H:%M:%S')" >> models/DEPLOYMENT_LOG.md
echo "- Model: $(cat models/latest_model.txt)" >> models/DEPLOYMENT_LOG.md
echo "- Metrics: $(cat models/lora_*/training_metrics.json | jq -r '.train_loss')" >> models/DEPLOYMENT_LOG.md
echo "" >> models/DEPLOYMENT_LOG.md
```

---

## 🔐 Security Hardening

### Production Security Checklist

- [ ] Redis password uses strong credentials (32+ characters)
- [ ] Training data doesn't contain PII or sensitive information
- [ ] Model outputs directory has restricted permissions (700)
- [ ] Docker container runs as non-root user (add to Dockerfile)
- [ ] TensorBoard access restricted (use SSH tunnel, not public)
- [ ] Training logs don't expose credentials
- [ ] Deployment process requires approval/review

### Recommended Permissions

```bash
# Secure data and models directories
chmod 700 learning-pipeline/data
chmod 700 learning-pipeline/models

# Restrict config file
chmod 600 config.yaml

# Ensure scripts are executable but not writable
chmod 755 learning-pipeline/scripts/*.sh
chmod 644 learning-pipeline/scripts/*.py
```

---

## 🎓 Advanced Topics

### Multi-GPU Training

```yaml
# In training_config.yaml
advanced:
  distributed: true
  num_processes: 2  # Number of GPUs
```

### Custom Training Loop

For advanced users who need fine-grained control:

```python
# Create custom_trainer.py
from train_lora import LoRATrainer

class CustomTrainer(LoRATrainer):
    def custom_training_step(self, batch):
        # Your custom logic
        pass
```

### A/B Testing Models

Deploy multiple model versions and compare:

```bash
# Deploy model A on port 8080
# Deploy model B on port 8081
# Route traffic and measure performance
```

---

**Status**: ✅ Production Ready
**Last Updated**: 2026-02-01
**Maintained By**: Hive-Mind Team
