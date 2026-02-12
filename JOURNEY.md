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
- [x] **Phase 5**: HiveCoder Integration (Feb 8) 🔗
- [x] **Phase 6**: R720xd Multi-Node (Feb 8-9) 🖥️
- [x] **Phase 7**: Continuous Learning (Feb 8) 🧠

---

## HiveCoder Integration (February 8, 2026)

### Full Stack Integration

Connected HiveCoder-7B directly into the hive-mind architecture:

```
┌──────────────────────────────────────────────────────────────────┐
│                    🐝 HIVE-MIND + HIVECODER                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ Claude Code │───▶│ MCP Server  │───▶│ HiveCoder-7B        │  │
│  │ (Opus 4.5)  │    │ (Python)    │    │ (llama-server:8089) │  │
│  └─────────────┘    └──────┬──────┘    └─────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│                    ┌───────────────┐                             │
│                    │ Redis Cluster │                             │
│                    │ (6 nodes)     │                             │
│                    └───────────────┘                             │
│                                                                  │
│  MCP Tools:          HTTP API:           LLM Modes:             │
│  • llm_generate      • /llm/generate     • code                 │
│  • llm_code_assist   • /llm/code-assist  • explain              │
│  • llm_complete      • /llm/complete     • debug                │
│                      • /llm/status                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### New Capabilities

| Tool | Description |
|------|-------------|
| `llm_generate` | Text generation with mode selection |
| `llm_code_assist` | Code review, fix, optimize, explain, document |
| `llm_complete` | FIM-style code completion |

### Performance

- **Inference**: 84 tokens/sec (Q5_K_M quantization)
- **Prompt Processing**: 519 tokens/sec
- **VRAM Usage**: ~7 GB (model + KV cache)
- **Response Caching**: Redis-backed with 30min TTL

### Production Services (systemd)

Two systemd services manage the full stack:

| Service | Description | Port |
|---------|-------------|------|
| `hivecoder-llm` | llama-server with HiveCoder-7B | 8089 |
| `hive-mind-http` | HTTP API (depends on hivecoder-llm) | 8090 |

**Service Management:**
```bash
# Enable auto-start on boot
sudo systemctl enable hivecoder-llm hive-mind-http

# Start/Stop/Restart
sudo systemctl start hivecoder-llm hive-mind-http
sudo systemctl stop hivecoder-llm hive-mind-http
sudo systemctl restart hivecoder-llm hive-mind-http

# Check status
sudo systemctl status hivecoder-llm hive-mind-http

# View logs
sudo journalctl -u hivecoder-llm -f
sudo journalctl -u hive-mind-http -f
```

**Health Checks:**
```bash
# LLM server
curl http://localhost:8089/health
# → {"status":"ok"}

# HTTP API + LLM status
curl http://localhost:8090/llm/status
# → {"model":"HiveCoder-7B","status":"online",...}

# Full stats
curl http://localhost:8090/stats
# → Redis info, session counts, LLM status
```

**Service Files:**
- `/etc/systemd/system/hivecoder-llm.service`
- `/etc/systemd/system/hive-mind-http.service`

---

## R720xd Multi-Node Integration (February 8-9, 2026)

### The Second Node - Upgraded!

Originally started with a Dell Precision T3620, but upgraded to a **Dell PowerEdge R720xd** rack server (acquired free!).

| Spec | R720xd | BEAST |
|------|--------|-------|
| Hostname | r720xd | aurora |
| CPU | Dual Xeon E5-2660 (16c/32t) | AMD (ROCm) |
| RAM | 64 GB | 32 GB |
| Storage | 24x 2.5" bays | SSD |
| GPU | External 6700XT planned | AMD R9700 (32GB) |
| Role | Embeddings + Storage | LLM Inference + Training |
| OS | uCore (Fedora 43 Atomic) | Fedora 43 bootc |

### Multi-Node Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        🐝 HIVE-MIND CLUSTER                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐     │
│  │         BEAST               │    │         R720XD              │     │
│  │     (aurora)                │    │      (r720xd)               │     │
│  │                             │    │                             │     │
│  │  ┌─────────────────────┐   │    │  ┌─────────────────────┐   │     │
│  │  │ Redis Cluster       │◀──┼────┼──│ Podman Containers   │   │     │
│  │  │ (6 nodes: 7000-7005)│   │    │  │ (hive-embedding)    │   │     │
│  │  └─────────────────────┘   │    │  └─────────────────────┘   │     │
│  │                             │    │                             │     │
│  │  ┌─────────────────────┐   │    │  ┌─────────────────────┐   │     │
│  │  │ HiveCoder-7B        │   │    │  │ Embedding Service   │   │     │
│  │  │ (llama-server:8089) │   │    │  │ (container:8081)    │   │     │
│  │  └─────────────────────┘   │    │  └─────────────────────┘   │     │
│  │                             │    │                             │     │
│  │  ┌─────────────────────┐   │    │  CPU: Dual E5-2660 (32t)   │     │
│  │  │ HTTP API (:8090)    │   │    │  RAM: 64GB                  │     │
│  │  └─────────────────────┘   │    │  Bays: 24x 2.5" available   │     │
│  │                             │    │                             │     │
│  │  AMD R9700 (32GB VRAM)     │    │  GPU: 6700XT (planned)      │     │
│  └─────────────────────────────┘    └─────────────────────────────┘     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### R720xd Setup

**Hardware acquired (FREE!):**
- Dell PowerEdge R720xd (2U rack server)
- Dual Intel Xeon E5-2660 (16 cores / 32 threads total)
- 64GB DDR3 ECC RAM
- PERC H710P Mini RAID controller
- 24x 2.5" drive bays (empty, ready for expansion)
- Dual redundant PSUs
- iDRAC 7 Enterprise (out-of-band management)

**Upgrade path:**
- CPU: E5-2697 v2 (~$50/pair) → 24c/48t
- GPU: External PCIe box with RX 6700 XT (12GB)

### iDRAC & Firmware (February 9, 2026)

Configured iDRAC for remote management and updated firmware:

| Component | Before | After |
|-----------|--------|-------|
| BIOS | 1.4.8 (2012) | **2.9.0** (latest) |
| iDRAC | 2.65.65.65 | 2.65.65.65 (latest) |

**iDRAC Configuration:**
- Static IP on local network
- Custom hostname
- Local DNS resolver
- SSH enabled (for racadm access)
- Web UI accessible

**BIOS 2.9.0 Benefits:**
- Enhanced security (Intel SINIT v2.5.1)
- CPU microcode updates (Spectre/Meltdown fixes)
- Full E5-2600 v2 processor support
- Improved fan curves (quieter operation!)
- 8 years of stability fixes

**Remote Management:**
```bash
# Check temps via IPMI
sudo ipmitool sdr type Temperature

# Access iDRAC web UI or SSH for racadm
```

### Security Hardening

R720xd secured with:
- SSH key-only authentication
- Firewall (firewalld) with minimal ports:
  - 22 (SSH)
  - 8081 (Embedding service)
  - 7000-7005 (Redis cluster)
  - 26379-26381 (Sentinels)
- uCore immutable OS (atomic updates)
- Containerized services (Podman + Quadlet)

### Embedding Service (Containerized)

Running `sentence-transformers` in Podman container:

```bash
# Health check
curl http://<r720xd>:8081/health
# → {"status":"ok","model_loaded":true}

# Generate embeddings
curl -X POST http://<r720xd>:8081/embed \
  -H "Content-Type: application/json" \
  -d '{"texts":["Hello world","Test embedding"]}'
# → {"embeddings":[[0.1,0.2,...],[0.3,0.4,...]],"dimensions":384}
```

**Service Management (Quadlet on R720xd):**
```bash
# Check status
systemctl --user status hive-embedding

# View logs
journalctl --user -u hive-embedding -f

# Restart
systemctl --user restart hive-embedding
```

**Container location:** `~/.config/containers/systemd/hive-embedding.container`

---

## Continuous Learning (February 8, 2026)

### The Self-Improving System

HiveCoder now learns from every interaction:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    🧠 CONTINUOUS LEARNING PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│   │ Interact │───▶│ Collect  │───▶│  Filter  │───▶│  Train   │        │
│   │ (MCP)    │    │ (Redis)  │    │ (Quality)│    │ (LoRA)   │        │
│   └──────────┘    └──────────┘    └──────────┘    └────┬─────┘        │
│                                                         │              │
│                                                         ▼              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│   │  Serve   │◀───│  Deploy  │◀───│  Export  │◀───│ Evaluate │        │
│   │ (llama)  │    │(hot-swap)│    │  (GGUF)  │    │ (bench)  │        │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Collect**: Every tool interaction goes to Redis learning queue
2. **Filter**: Quality filter removes failed/low-value interactions
3. **Batch**: When threshold (100 samples) is reached, trigger training
4. **Train**: Quick LoRA fine-tuning (1 epoch incremental)
5. **Export**: Convert to GGUF for llama.cpp
6. **Deploy**: Hot-swap model without downtime

### Commands

```bash
# Check status
python learning-pipeline/scripts/continuous_learning.py --status

# Collect data only
python learning-pipeline/scripts/continuous_learning.py --collect-only

# Force training now
python learning-pipeline/scripts/continuous_learning.py --train-now

# Run as daemon (checks every 5 min)
python learning-pipeline/scripts/continuous_learning.py --daemon
```

### Service

```bash
# Enable continuous learning daemon
sudo systemctl enable hivecoder-learning
sudo systemctl start hivecoder-learning

# Check logs
sudo journalctl -u hivecoder-learning -f
```

### Storage Strategy

| Location | Type | Purpose |
|----------|------|---------|
| BEAST SSD | Fast | Active training data, current model |
| R720xd (24 bays) | Expandable | Model archive, datasets, Redis backups |
| NAS (9.2TB) | Archive | Cold storage, backups |

---

## RAG + Fast Tokenization (February 12, 2026)

### The Challenge

Open Interpreter bypassed Hive-Mind's RAG layer - going directly to llama-server without context injection. HiveCoder didn't know user facts (OS, GPU, etc.).

### The Solution

**OpenAI-Compatible RAG Proxy:**
```
Open Interpreter → litellm → hive-mind-http:8090 → llama-server:8089
                                    ↓
                           RAG facts injected
```

Added `/v1/chat/completions` endpoint that:
- Injects Redis facts into system prompt
- Handles streaming responses
- Maintains OpenAI API compatibility

**tiktoken for Python 3.14:**
- Built tiktoken 0.12.0 from source (no official py314 wheels)
- Created custom `hivecoder` encoding
- Added `hivemind_client.tokenizer` module (~10x faster than Python tokenizers)

```python
from hivemind_client import tokenizer

# Fast token counting
count = tokenizer.count_tokens("Your text")

# Chunking for embeddings
chunks = tokenizer.chunk_text(long_text, chunk_size=512, overlap=50)
```

### Canonical AI Venv

Consolidated all AI tools into single venv:
```
/var/mnt/build/.venv → TheRock/.venv
├── ROCm PyTorch 2.9.1
├── open-interpreter
├── hivemind_client
└── tiktoken 0.12.0
```

Bashrc: `$AI_VENV`, `activate-ai`, `oih`

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
**Date**: February 9, 2026
**Author**: hashcat

---

*The hive never forgets.* 🐝
