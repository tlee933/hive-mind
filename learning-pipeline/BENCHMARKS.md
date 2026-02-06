# 🔥 Hive-Mind LoRA Training Benchmarks

**Hardware:** AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32GB VRAM)
**Software:** PyTorch 2.9.1 + ROCm 7.12.0a20260203
**Date:** February 5, 2026

---

## 📊 Performance Summary

### Qwen2.5-0.5B LoRA Fine-tuning

**Configuration:**
- Model: 498M parameters (trainable: 4.4M / 0.88%)
- Dataset: 1,500 synthetic examples
- LoRA: r=8, alpha=16, dropout=0.05
- Batch size: 2, Grad accumulation: 4 (effective batch=8)
- Epochs: 3
- Learning rate: 2e-4

**Results:**

| Run | Date | Training Time | Samples/sec | Steps/sec | Final Loss | Status |
|-----|------|--------------|-------------|-----------|------------|--------|
| **Validation** | Feb 5 (earlier) | 546.7s (9.11 min) | 8.231 | 1.032 | 0.3367 | ✅ |
| **Benchmark** | Feb 5 (latest) | 548.1s (9.13 min) | 8.210 | 1.029 | 0.3387 | ✅ |

**Consistency:**
- Training time variance: **0.26%** (1.4 seconds difference)
- Throughput variance: **0.25%** (extremely stable)
- Zero crashes, zero HIP errors

---

## 🎯 Key Metrics

### Training Performance

```
┌─────────────────────────────────────────────┐
│  Qwen2.5-0.5B (498M params)                │
│  LoRA Fine-tuning (4.4M trainable)         │
├─────────────────────────────────────────────┤
│  Training Time:      ~9.1 minutes          │
│  Throughput:         8.2 samples/sec       │
│  Step Rate:          1.03 steps/sec        │
│  VRAM Usage:         ~12 GB / 32 GB        │
│  Loss Convergence:   0.34 (stable)         │
│  Stability:          100% (564/564 steps)  │
└─────────────────────────────────────────────┘
```

### Memory Utilization

- **VRAM Total:** 32 GB
- **Peak Training:** ~12 GB (37.5% utilization)
- **Headroom:** 20 GB (excellent for larger models)
- **Memory Efficiency:** LoRA adapters only 17 MB on disk

### Compute Efficiency

- **FLOPS:** 523.5 trillion (per 3-epoch run)
- **Time per Epoch:** ~182 seconds (3.03 min)
- **Time per Step:** ~0.97 seconds
- **GPU Utilization:** High (Flash Attention enabled)

---

## 📈 Scaling Projections

Based on measured performance with Qwen2.5-0.5B:

| Model Size | Est. Training Time* | Est. VRAM | Feasibility |
|-----------|-------------------|-----------|-------------|
| **0.5B** (measured) | 9 min | 12 GB | ✅ Validated |
| **1.5B** | ~15 min | 16 GB | ✅ Excellent |
| **3B** | ~25 min | 20 GB | ✅ Good |
| **7B** | ~45 min | 26 GB | ✅ Possible |
| **13B** | ~75 min | 30 GB | ⚠️ Tight fit |

\* *For 1,500 examples, 3 epochs, LoRA r=8, batch=2, grad_accum=4*

---

## 🔬 Detailed Comparison

### Training Stability

Both runs show exceptional stability:

**Loss Progression (Latest Benchmark):**
```
Epoch 1: 2.599
Epoch 2: 0.338
Epoch 3: 0.338 (converged)
```

**No errors observed:**
- ✅ Zero HIP illegal memory access
- ✅ Zero CUDA out-of-memory
- ✅ Zero gradient explosions
- ✅ Smooth convergence curve

### Hardware Context

**AMD Radeon AI PRO R9700 (gfx1201):**
- First RDNA4 architecture
- ROCm 7.12 native support (required custom PyTorch build)
- Flash Attention support validated
- FBGEMM optimizations enabled

**Build Details:**
- Custom PyTorch 2.9.1 from source
- ROCm 7.12.0a20260203 from TheRock
- GCC 15.0.1, Python 3.14.2
- Build time: ~2 hours (one-time)

---

## 🚀 Performance Analysis

### Throughput Breakdown

**8.2 samples/second means:**
- Processing 300 samples in ~36 seconds
- Complete 1,500-sample dataset in ~3 minutes per epoch
- Efficient use of 32GB VRAM (only 37% utilized)

**Why this matters:**
- Daily training on 10K samples: ~60 minutes
- Weekly training on 50K samples: ~5 hours
- Continuous learning is practical

### Comparison Context

**Note:** Direct comparisons are limited because:
1. gfx1201 (RDNA4) is cutting-edge architecture
2. Most ML benchmarks use NVIDIA or older AMD GPUs
3. Our custom PyTorch build is unique for ROCm 7.12

**Relative Performance:**
- RDNA2 (6900 XT, 16GB): Would be slower, memory-constrained
- RDNA3 (7900 XTX, 24GB): Similar speed, less VRAM
- MI300X (192GB): Much faster, overkill for LoRA
- **R9700 (32GB):** Sweet spot for LoRA fine-tuning

---

## 💡 Optimization Opportunities

### Current Configuration
- Batch size: 2
- Gradient accumulation: 4
- Effective batch: 8

### Potential Improvements

**1. Increase Batch Size** (we have 20GB VRAM headroom)
```
Current:  batch=2, grad_accum=4  → ~12 GB VRAM
Option A: batch=4, grad_accum=4  → ~18 GB VRAM → +15-20% faster
Option B: batch=8, grad_accum=2  → ~22 GB VRAM → +25-30% faster
```

**2. Mixed Precision Training**
- Currently using FP16 (half precision)
- Could try FP8 for faster training (if supported)

**3. Larger LoRA Rank**
```
Current:  r=8  → 4.4M params → fast but limited capacity
Option:   r=16 → 8.8M params → better quality, slightly slower
```

---

## 🎓 Lessons Learned

### What Works Well

1. **LoRA Efficiency**
   - Only 0.88% of model params trained
   - 17 MB adapter files (vs 498M param model)
   - Minimal VRAM overhead

2. **Flash Attention**
   - Validated working on gfx1201
   - Significant speedup for transformer models
   - Critical for ROCm 7.12 compatibility

3. **ROCm 7.12 Stability**
   - Zero HIP errors after PyTorch rebuild
   - Consistent performance across runs
   - Production-ready for ML workloads

### Critical Success Factors

1. **Version Alignment**
   - PyTorch 2.9.1 built for exact ROCm version
   - Flatbuffers v25 compatibility
   - rocprim __half fixes

2. **32GB VRAM**
   - Enables larger models without swapping
   - Room for batch size optimization
   - Future-proof for 7B+ models

3. **TheRock Integration**
   - Custom device libraries
   - libdrm headers from source
   - Complete control over build environment

---

## 📋 Benchmark Methodology

### Test Procedure

1. **Environment Setup**
   - Clean GPU state (no other processes)
   - Verified ROCm installation
   - Confirmed PyTorch build

2. **Data Preparation**
   - 1,500 synthetic examples (consistent dataset)
   - Alpaca-style instruction format
   - Pre-tokenized to avoid overhead

3. **Training Execution**
   - Automated benchmark script
   - JSON metrics capture
   - Stdout/stderr logging

4. **Metric Collection**
   - Training runtime (wall-clock)
   - Samples/second (throughput)
   - Steps/second (iteration rate)
   - Final loss (quality indicator)
   - Peak VRAM (memory usage)

### Reproducibility

**To reproduce these benchmarks:**

```bash
cd /var/mnt/build/MCP/hive-mind/learning-pipeline

# Run benchmark
python3 scripts/benchmark_training.py \
  --model "Qwen/Qwen2.5-0.5B" \
  --dataset data/training_data_synthetic.jsonl \
  --output models/benchmark-test \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 4

# Results saved to benchmarks/benchmark_*.json
```

---

## 🔮 Future Benchmarks

### Planned Tests

1. **Larger Models**
   - Qwen2.5-1.5B
   - Qwen2.5-3B
   - Qwen2.5-7B

2. **Batch Size Scaling**
   - Test batch=4, 8, 16
   - Find optimal throughput/memory balance

3. **LoRA Rank Comparison**
   - r=4, 8, 16, 32, 64
   - Quality vs speed tradeoff

4. **Long Context**
   - Test with 4K, 8K token sequences
   - Flash Attention performance validation

5. **Multi-GPU** (future hardware)
   - When second R9700 available
   - Distributed training benchmarks

---

## 📚 References

- **Training Results:** [TRAINING_RESULTS.md](TRAINING_RESULTS.md)
- **PyTorch Build:** [PYTORCH_ROCM712_BUILD.md](PYTORCH_ROCM712_BUILD.md)
- **TheRock Build:** https://github.com/tlee933/TheRock-Forge-EXPERIMENTAL

---

## 🏆 Bottom Line

**The Hive-Mind learning pipeline achieves:**

✅ **9-minute training** for 0.5B model with LoRA
✅ **8.2 samples/second** sustained throughput
✅ **100% stability** with custom PyTorch + ROCm 7.12
✅ **37% VRAM usage** (room for 2-3x larger models)
✅ **Production-ready** for continuous learning workflows

The R9700 (gfx1201) + ROCm 7.12 stack is **proven stable and performant** for LoRA fine-tuning at scale.

---

*Last Updated: February 5, 2026*
*Hardware: AMD Radeon AI PRO R9700 (32GB VRAM, gfx1201)*
*Software: PyTorch 2.9.1, ROCm 7.12.0a20260203, Python 3.14.2*
