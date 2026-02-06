# 🎯 Training Results: PyTorch 2.9.1 + ROCm 7.12

**Date:** February 5, 2026
**Hardware:** AMD Radeon AI PRO R9700 (gfx1201, 32GB VRAM)
**PyTorch:** 2.9.1 (custom build for ROCm 7.12)
**Status:** ✅ Production Validated

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

**Status:** ✅ Production Ready

The Hive-Mind learning pipeline is fully operational on gfx1201 + ROCm 7.12!
