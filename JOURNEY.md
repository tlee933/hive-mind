# 🐝 The Hive-Mind Journey

> **hashcat's first foundation model: HiveCoder-7B**

---

## The Beginning (February 1, 2026)

Started with a simple question: *"How do we fix context loss?"*

Built in one incredible session:
- 6-node Redis cluster with auto-sharding
- 3 Sentinels for high availability
- MCP server for Claude Code integration
- Dual-mode access (HTTP API + MCP Protocol)

---

## The Learning Pipeline (February 3-5, 2026)

### The Challenge
Pre-built PyTorch wheels (ROCm 7.11) failed on our ROCm 7.12 system:
```
Step 3/282: HIP error: an illegal memory access was encountered
```

### The Solution
Built PyTorch 2.9.1 from source with ROCm 7.12 compatibility.

**First successful training run:**
- Model: Qwen2.5-0.5B
- Dataset: 1,500 samples
- Loss: 0.34
- Time: 9 minutes
- Zero HIP errors

---

## HiveCoder-7B: hashcat's First Model (February 7-8, 2026)

### Training the Foundation

**The Numbers:**
| Metric | Value |
|--------|-------|
| Base Model | Qwen2.5-Coder-7B-Instruct |
| Training Dataset | 10,156 samples |
| Training Time | 1h 43min |
| Final Loss | **0.2998** |
| Trainable Params | 40.3M / 7.66B (0.53%) |
| LoRA Config | r=16, alpha=32 |
| VRAM Usage | ~20 GB / 32 GB |

**Loss Progression:**
```
Step 10:   3.72  → Starting high
Step 50:   0.37  → Rapid convergence
Step 160:  0.29  → Sweet spot
Step 1905: 0.30  → Final (3 epochs)
```

### Smart Optimizer

Built an auto-configuration system that detects hardware and selects optimal settings:
- GPU architecture detection (gfx1201)
- VRAM-based batch sizing
- Quality modes: fast / balanced / best
- TorchAO integration for int4/int8

### Export to GGUF

```
HiveCoder-7B-f16.gguf    → 15 GB (full precision)
HiveCoder-7B-Q5_K_M.gguf → 5.1 GB (quantized)
```

---

## The Stack

```
┌─────────────────────────────────────────────────────────────┐
│                      🐝 HIVE-MIND                           │
├─────────────────────────────────────────────────────────────┤
│  Hardware                                                   │
│  └── AMD Radeon AI PRO R9700 (gfx1201, 32GB VRAM)          │
│                                                             │
│  Software                                                   │
│  ├── Fedora 43 bootc Atomic                                │
│  ├── ROCm 7.12 (TheRock build)                             │
│  ├── PyTorch 2.9.1 (custom ROCm 7.12 build)                │
│  └── Python 3.14                                           │
│                                                             │
│  Training                                                   │
│  ├── PEFT + LoRA (no Unsloth needed!)                      │
│  ├── BF16 precision                                        │
│  ├── Gradient checkpointing                                │
│  └── TorchAO for quantization                              │
│                                                             │
│  Inference                                                  │
│  ├── llama.cpp (GGUF format)                               │
│  ├── Q5_K_M quantization (65% smaller)                     │
│  └── llama-server for API                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Milestones

- [x] **Phase 1**: Redis Cluster (Feb 1)
- [x] **Phase 2**: MCP Server (Feb 1)
- [x] **Phase 2.5**: Local LLM Inference (Feb 2)
- [x] **Phase 2.7**: Dual-Mode Access (Feb 3)
- [x] **Phase 4**: Learning Pipeline (Feb 5)
- [x] **Phase 4.5**: Smart Optimizer (Feb 7)
- [x] **HiveCoder-7B**: First Foundation Model (Feb 8) 🎉
- [ ] **Phase 5**: DELL Integration
- [ ] **Phase 6**: Continuous Learning

---

## The Model

**HiveCoder-7B** - hashcat's first foundation model

Trained on:
- Fedora bootc / Atomic operations
- ROCm GPU operations
- SELinux / cgroups
- Container operations
- Python / AI workflows
- Redis cluster operations
- Code generation

Ready for:
- llama.cpp inference
- llama-server API
- Claude Code integration

---

## Credits

Built with:
- 🧠 Claude Code (Opus 4.5)
- ☕ A lot of coffee
- 🔥 Pure determination

**Status**: Production Ready
**Date**: February 8, 2026
**Author**: hashcat

---

*The hive never forgets.* 🐝
