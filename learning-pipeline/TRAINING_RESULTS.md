# 🎯 Training Results: PyTorch 2.9.1 + ROCm 7.12

**Updated:** February 7, 2026
**Hardware:** AMD Radeon AI PRO R9700 (gfx1201, 32GB VRAM)
**PyTorch:** 2.9.1 (custom build for ROCm 7.12)
**Status:** ✅ Production Validated

---

## Training Summary

| Model | Dataset | Final Loss | Time | Throughput | LoRA Size |
|-------|---------|------------|------|------------|-----------|
| Qwen2.5-0.5B | 1,500 samples | 0.34 | 9 min | 8.2 samples/s | 17 MB |
| **Qwen2.5-Coder-7B** | **10,156 samples** | **0.30** | **1h 43min** | **4.9 samples/s** | **155 MB** |

---

## The Challenge

Initial training attempts with pre-built PyTorch wheels (ROCm 7.11) failed consistently:

```
Step 3/282: HIP error: an illegal memory access was encountered
```

**Root cause:** Version mismatch between PyTorch 2.9.1+rocm7.11 and system ROCm 7.12

---

## The Solution

Built PyTorch 2.9.1 from source with ROCm 7.12 compatibility. Full details:
- **Build documentation:** [PYTORCH_ROCM712_BUILD.md](PYTORCH_ROCM712_BUILD.md)
- **TheRock integration:** https://github.com/tlee933/TheRock-Forge-EXPERIMENTAL/tree/fedora-atomic-rocm7.12-ai-pro-experimental/external-builds/pytorch

---

## Successful Training Run

### Configuration
```yaml
Model: Qwen/Qwen2.5-0.5B (498M parameters)
Dataset: 1,500 synthetic examples
LoRA: r=8, alpha=16, dropout=0.05
Training: lr=2e-4, epochs=3, batch=2, grad_accum=4
```

### Results

```
Training runtime: 546.7 seconds (~9 minutes)
Samples/second: 8.231
Steps/second: 1.032
Final loss: 0.3367
Trainable params: 4.4M / 498M (0.88%)
```

**Stability:**
- ✅ Zero HIP errors
- ✅ Clean completion (564/564 steps)
- ✅ Stable gradients
- ✅ Smooth loss convergence

---

## Hardware Utilization

- **VRAM usage:** ~12GB / 32GB (excellent headroom)
- **Throughput:** 8.23 samples/sec
- **GPU:** AMD Radeon AI PRO R9700 (gfx1201)
- **Build:** Custom PyTorch with Flash Attention + FBGEMM

---

## Native vs Docker

Both approaches validated:

**Native** (what we used):
- Direct GPU access
- Easier debugging
- Full system integration

**Docker** (also supported):
- Reproducible environment
- Portable across systems
- Pre-configured dependencies

---

## Model Output

**Location:** `models/qwen-lora-hive/`
**Size:** ~17 MB (LoRA adapters only)
**Format:** SafeTensors

---

## Key Learnings

1. **Version alignment critical** - ROCm/PyTorch must match exactly
2. **Flash Attention works** - With proper build on gfx1201
3. **32GB VRAM ideal** - Excellent headroom for larger models
4. **LoRA efficient** - 0.88% params, <10min training, low VRAM

---

## 7B Foundation Model Training (February 7, 2026)

### Configuration
```yaml
Model: Qwen/Qwen2.5-Coder-7B-Instruct (7.66B parameters)
Dataset: 10,156 foundation samples
LoRA: r=16, alpha=32, dropout=0.05
Training: lr=2e-4, epochs=3, batch=4, grad_accum=4
```

### Results

```
Training runtime: 6208.79 seconds (1h 43min)
Samples/second: 4.907
Steps/second: 0.307
Final loss: 0.2998
Trainable params: 40.3M / 7.66B (0.53%)
```

### Loss Progression

| Epoch | Loss | Notes |
|-------|------|-------|
| 0.1 | 3.72 | Initial high loss |
| 0.2 | 1.46 | Rapid drop |
| 0.3 | 0.48 | Converging |
| 1.0 | 0.36 | Stable |
| 2.0 | 0.31 | Fine-tuning |
| 3.0 | **0.30** | Final |

### Model Output

**Location:** `models/foundation_7b_v1/`
**Size:** 155 MB (LoRA adapters)
**Format:** SafeTensors
**Checkpoints:** 1800, 1900, 1905

---

## Smart Optimizer

Auto-config selection based on hardware and model:

```bash
python scripts/auto_optimize.py --model "Qwen/Qwen2.5-Coder-7B" --task training --quality balanced
```

Detects: GPU arch, VRAM, BF16, Flash Attention
Outputs: Optimal LoRA rank, batch size, precision, quantization

---

**Status:** ✅ Production Ready

The Hive-Mind learning pipeline is fully operational on gfx1201 + ROCm 7.12!
