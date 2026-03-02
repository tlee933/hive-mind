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
│  ├── PyTorch 2.10.0 (custom ROCm 7.12 build)               │
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
- [x] **Phase 8**: PyTorch 2.10 + ROCm 7.12 Native (Feb 14) 🔧
- [x] **Phase 9**: Active Learning - RAG Retrieval Mining (Feb 15) 🔍
- [x] **Phase 10**: First Continuous Training on 7B — v0.9.1 (Feb 16) 🏋️

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
│  │ (Opus 4.6)  │    │ (Python)    │    │ (llama-server:8089) │  │
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

**Keyword-Based RAG Filtering:**

Selective fact injection based on query keywords reduces token overhead:

| Query Keywords | Injected Facts |
|----------------|----------------|
| `gpu`, `amd`, `vram` | gpu, rocm_version, pytorch_location |
| `install`, `package`, `rpm` | package_management, system_type |
| `os`, `linux`, `fedora` | operating_system, desktop_environment |
| `python`, `venv`, `pip` | python_venv, pytorch_location |
| `token`, `chunk`, `encode` | hivemind_tokenizer |

Unmatched queries get core facts only (os, gpu, project) → ~84 tokens vs ~250+ for all facts.

**Test Results:**
```
Q: "What GPU do I have?"
A: "AMD Radeon AI PRO R9700 with 32GB VRAM" ✓

Q: "How do I install htop?"
A: "sudo rpm-ostree install htop" ✓ (Fedora Atomic aware!)
```

**tiktoken for Python 3.14:**
- Built tiktoken 0.12.0 from source (no official py314 wheels)
- Created custom `hivecoder` encoding
- Added `hivemind_client.tokenizer` module

```python
from hivemind_client import tokenizer

# Fast token counting
count = tokenizer.count_tokens("Your text")

# Chunking for embeddings
chunks = tokenizer.chunk_text(long_text, chunk_size=512, overlap=50)
```

**Benchmark (tiktoken vs HuggingFace):**
| Text Size | tiktoken | HuggingFace | Speedup |
|-----------|----------|-------------|---------|
| Medium (900 chars) | 31,000/sec | 4,800/sec | **6.4x** |
| Long (8.6K chars) | 3,400/sec | 574/sec | **6.0x** |

### Canonical AI Venv

Consolidated all AI tools into single venv:
```
/var/mnt/build/.venv → TheRock/.venv
├── ROCm PyTorch 2.10.0
├── open-interpreter
├── hivemind_client
└── tiktoken 0.12.0
```

Bashrc: `$AI_VENV`, `activate-ai`, `oih`

### Future: Semantic Search → **DONE (Day 17)**

Keyword matching worked well but missed synonyms ("graphics card" vs "gpu").
Solved with embedding-based semantic search — see [Day 17](#day-17-semantic-embedding-rag-feb-15-2026).

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

## Day 15: The Brew Upgrade Incident (Feb 13, 2026)

Upgraded Homebrew which pulled llama.cpp b8020 and Claude Code Opus 4.6. The new `convert_hf_to_gguf.py` referenced `MODEL_ARCH.QWEN35` which didn't exist in the released `gguf` pip package (0.17.1). Fix: installed gguf directly from llama.cpp git master.

### What Broke
- GGUF export failed mid-pipeline, learning daemon kept re-triggering training (4 runs in one day)
- Root cause: `should_train()` checked time since last *deployed* version, not last *training attempt*
- Each failed export left the deployed version stale, so the 24h threshold kept firing

### What We Fixed
1. **gguf package** - installed from llama.cpp master to match brew's converter
2. **Training frequency** - added `.last_training` marker file to track attempts, not just deploys
3. **Auto-cleanup** - `cleanup_old_versions()` removes stale model versions after deploy (keeps deployed + 1 previous)
4. **NAS backup timer** - systemd timer for Sun/Wed/Fri 3am, 3-copy rotation to `/var/mnt/ai/hive-mind/`
5. **Disk recovery** - cleaned 4 stale versions, went from 73% to 56% usage (reclaimed ~60GB)

### Performance
- HiveCoder-7B v20260213: 92 tok/s generation, 597 tok/s prompt on R9700
- Q5_K_M quantization: 5.44GB (64.3% smaller than f16)

### Lessons Learned
- Always pin or sync `gguf` pip package version with llama.cpp build
- Track training *attempts* not just successful deploys to prevent runaway retraining
- Auto-cleanup is essential when each model version is 15-35GB

---

## Day 16: PyTorch 2.10 Upgrade (Feb 14, 2026)

PyTorch 2.10 dropped in early February, and ROCm/pytorch had an active `release/2.10` branch with RDNA4-specific fixes. Upgraded from 2.9.1 to 2.10.0, rebuilt against ROCm 7.12.

### What Changed

Switched the build from PyTorch 2.9.1 (detached HEAD, pre-hipified) to the `release/2.10` branch. This required discovering and fixing several issues the old pre-hipified checkout had hidden.

### Build Fixes (4 attempts before success)

| Attempt | Failure | Fix |
|---------|---------|-----|
| 1 | Missing `c10/hip/impl/hip_cmake_macros.h.in` | Added hipify step (`tools/amd_build/build_amd.py`) |
| 2 | `flatbuffers version 24.12 != 25.9` | Replaced patch with sed-based version detection |
| 3 | `cannot find ROCm device library` | Created `amdgcn/bitcode` symlink (TheRock layout quirk) |
| 4 | Same device lib error (stale cmake cache) | Full clean build (`rm -rf build`) |

### The TheRock Path Quirk

Fedora Atomic mounts `/opt` at `/var/opt`. TheRock puts device bitcode at `/opt/rocm/lib/llvm/amdgcn/bitcode/` but clang expects `/opt/rocm/amdgcn/bitcode/`. Previous 2.9 builds used a pre-hipified source tree that avoided this. The fix:
```bash
sudo mkdir -p /opt/rocm/amdgcn
sudo ln -sf /opt/rocm/lib/llvm/amdgcn/bitcode /opt/rocm/amdgcn/bitcode
```

### Results

| Metric | 2.9.1 | 2.10.0 |
|--------|-------|--------|
| Build time | ~2h (est.) | **28 min** |
| FP16 GEMM (1024x) | ~87 TFLOPS | **84.1 TFLOPS** |
| FP32 GEMM (1024x) | - | **14.8 TFLOPS** |
| LoRA training | Pass | **Pass** |
| ROCm mismatch warning | Yes (7.11 vs 7.12) | **None** |
| Wheel size | 301 MB | **335 MB** |

### Build Script Improvements

The build script (`build_pytorch_gfx1201.sh`) is now fully self-contained:
- Auto-detects and fixes flatbuffers version from system `flatc`
- Auto-relaxes numpy/optree pins for Python 3.14
- Creates amdgcn device library symlinks
- Runs hipification automatically
- Works on fresh `release/2.10` checkout with zero manual steps

### TheRock Rebuild Assessment

Checked upstream TheRock (67 commits since our Feb 6 build). No gfx1201-specific fixes. The LLVM compiler bump is the only potentially useful change. Verdict: skip for now, revisit when `therock-7.12` is tagged.

### GGUF Export Fix (Round 2)

The Day 15 `QWEN35` fix had regressed - the gguf package was back to PyPI 0.17.1 (missing `MODEL_ARCH.QWEN35`) after the PyTorch 2.10 `pip install --force-reinstall` pulled in fresh dependencies. Force-reinstalled gguf from llama.cpp b8020 git source and pinned it in `requirements.txt` so it won't regress again.

### End-to-End Pipeline Verification

Forced a training run with PyTorch 2.10 + fixed gguf:

| Stage | Result | Time |
|-------|--------|------|
| Data collection | 31 samples merged | instant |
| LoRA training | loss: 0.0000 | 26s |
| GGUF export (f16 + Q5_K_M) | Success | ~3 min |
| Deploy (llama-server hot-swap) | v20260214 live | 11s |
| Auto-cleanup | 3 old versions removed (~21 GB) | instant |

Full cycle: **~4 minutes** from data to deployed model.

---

## Day 17: Semantic Embedding RAG (Feb 15, 2026)

### The Problem

Keyword-based RAG filtering worked but had blind spots. Queries like "graphics card" wouldn't match the `gpu` fact because the keyword map only had literal terms. Every new concept required manually adding keyword->fact mappings. Time to make it smart.

### The Solution: Embedding-Based Semantic Search

Replaced the keyword-only approach with vector similarity search using `bge-small-en-v1.5` (384-dim, ~130MB). The embedding model runs on CPU to keep the GPU free for LLM inference.

**Architecture:**
```
User Query
    │
    ▼
┌──────────────┐     ┌──────────────────┐
│ Embed Query  │     │ Redis: cached     │
│ (bge-small)  │     │ fact_embeddings:* │
└──────┬───────┘     └────────┬─────────┘
       │                      │
       ▼                      ▼
┌──────────────────────────────────┐
│  Cosine Similarity (dot product) │
│  top_k=5, threshold=0.3         │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Filtered facts + core facts  │──▶ System Prompt
│ (os, gpu, project always)    │
└──────────────────────────────┘
               │
               ▼ (fallback if embeddings unavailable)
┌──────────────────────────────┐
│ Keyword map (legacy)          │
└──────────────────────────────┘
```

### Key Design Decisions

- **Lazy loading**: Embedding model loads on first use, not at startup (keeps MCP server fast to connect)
- **Cached embeddings**: Pre-computed on `fact_store`, stored as base64 float32 arrays in Redis with 30-day TTL
- **Bootstrap on connect**: On startup, checks for any facts missing embeddings and batch-computes them
- **Redis pipelines**: Batch-retrieves all cached embeddings in one round-trip instead of N individual GETs
- **Graceful fallback**: If the embedding model fails to load or embeddings are missing, falls back to the keyword map

### Semantic Search Results

| Query | Top Match | Score |
|-------|-----------|-------|
| "How do I build PyTorch for ROCm?" | rocm_version | 0.883 |
| "What GPU do I have?" | gpu | 0.760 |
| "How to benchmark GPU performance?" | gpu_benchmarking | 0.852 |
| "How to tokenize text with hivemind?" | hivemind_tokenizer | 0.870 |
| "Is this system immutable?" | package_management | 0.660 |
| "How to export GGUF model?" | gguf_pinning | 0.744 |

All 20 stored facts get embedded and cached. Query time is dominated by the model encode (~150ms on CPU), Redis retrieval is negligible with pipelines.

### GPU Benchmarks (PyTorch 2.10 + ROCm 7.12)

While testing, ran a proper GEMM benchmark to establish the R9700's actual capabilities:

| Precision | Matrix Size | TFLOPS |
|-----------|-------------|--------|
| FP32 | 8192x8192 | 16.88 |
| FP16 | 4096x4096 | **122.88** |
| BF16 | 4096x4096 | **122.50** |

Previous benchmark showed 3.98 TFLOPS — that was FP32 on undersized matrices. The R9700 actually does **120+ TFLOPS** in half precision with WMMA.

### Nova Fractal (bonus)

Generated a 5th-power Nova fractal on the GPU as a wallpaper:
- Formula: `z = z - (z^5 - 1)/(5z^4) + c` where `c = 0.52 + 0.3i`
- Orbit trap coloring with neon palette
- 3840x2160, 400 iterations, 8.5 seconds on R9700

### Files Changed

- `mcp-server/server.py` — Added `EmbeddingManager` class, semantic search in `_get_facts_context()`, Redis pipeline optimization, embedding bootstrap
- `config.yaml` — `embedding` section (model, device, top_k, similarity_threshold)
- `requirements.txt` — `sentence-transformers>=2.2.0` (already listed for Phase 2, now active)

---

## Day 18: Active Learning — RAG Retrieval Mining (Feb 15, 2026)

### The Problem

The semantic embedding RAG system (shipped earlier today) returns relevant facts for queries, but had zero visibility into when it fails. Queries that don't match any facts well silently fall back to default core facts. No way to know what knowledge gaps exist or what facts to add next.

### The Solution: Retrieval Quality Tracking

Every RAG retrieval now logs its method, quality classification, and similarity scores to Redis. Three data structures capture different views:

| Redis Key | Type | Purpose |
|-----------|------|---------|
| `rag:retrieval_log` | Stream (maxlen 5000) | Full audit trail — every query with method, quality, scores |
| `rag:stats` | Hash (HINCRBY) | Aggregate counters — total, per-method, poor retrievals |
| `rag:missed_queries` | Sorted Set (capped 500) | Failed queries ranked by frequency |

**Quality Classification:**

| Quality | Condition |
|---------|-----------|
| `good` | Semantic hit, top score >= 0.6 |
| `weak` | Semantic hit, top score 0.45–0.6 |
| `fallback` | Fell back to keyword filter |
| `miss` | Only default core facts returned |

### Implementation

**`RetrievalResult` dataclass** — `EmbeddingManager.find_relevant()` now returns structured results with keys, all scored matches, and top score instead of a bare `Set[str]`.

**`_log_retrieval()` method** — fires via `asyncio.create_task()` (non-blocking, fire-and-forget). Writes all three Redis structures in a single pipeline for minimal overhead.

**`_keyword_filter()` refactored** — now returns a `(text, method)` tuple distinguishing `"keyword"` matches from `"default"` fallback, so the logger knows which path was taken.

**`fact_suggestions()` MCP tool** — reads `rag:missed_queries`, extracts topic words (stop-word filtered), groups by frequency, cross-references against existing facts, and returns actionable suggestions for new facts to add.

### New Tools & Endpoints

| Interface | Name | Description |
|-----------|------|-------------|
| MCP Tool | `fact_suggestions` | Missed query analysis + suggested topics |
| HTTP GET | `/rag/suggestions?limit=N` | Same, over HTTP |
| MCP Tool | `get_stats` (extended) | Now includes `rag_retrieval` section |

### Test Results

```
# After 4 test queries:
Stream length: 4
Stats: {total: 4, method_semantic: 4}

# All queries classified as "good" — embedding model performing well:
"What GPU do I have?"                           → top_score: 0.772
"Deploy kubernetes cluster on my machine?"      → top_score: 0.616
"Configure nginx reverse proxy?"                → top_score: 0.626
"What color is the sky?"                        → top_score: 0.571
```

The `bge-small-en-v1.5` model finds relevant facts even for queries with no direct keyword match ("kubernetes" → matched system facts at 0.62). As fact gaps emerge over real usage, `fact_suggestions` will surface them.

### Files Changed

- `mcp-server/server.py` — `RetrievalResult` dataclass, `_log_retrieval()`, refactored `_get_facts_context()` and `_keyword_filter()`, `fact_suggestions()` method + MCP tool, extended `get_stats()`
- `mcp-server/http_server.py` — `GET /rag/suggestions` endpoint

### Stress Test & Threshold Tuning

Ran 28 diverse queries through the RAG endpoint — direct matches, semantic synonyms, training questions, and completely irrelevant queries (pasta, quantum physics, dog training). The initial thresholds were too generous.

**The Problem:** Queries like "pasta carbonara" (0.52) and "quantum entanglement" (0.53) were classified as `good` and injecting irrelevant facts into the LLM prompt. The 0.5 threshold couldn't distinguish real matches from noise.

**Threshold Changes:**

| Parameter | Before | After |
|-----------|--------|-------|
| `good` quality threshold | >= 0.5 | >= 0.6 |
| `weak` quality threshold | >= 0.3 | >= 0.45 |
| `similarity_threshold` (min cutoff) | 0.3 | 0.45 |

**Before vs After (28 queries):**

| Metric | Before | After |
|--------|--------|-------|
| Good (correct injection) | 93.8% | 71.4% |
| Weak (skipped) | 6.2% | 28.6% |
| Avg score | 0.642 | 0.650 |

**Queries correctly reclassified as weak:**
- "How do I make pasta carbonara?" → 0.52 (was `good`, now `weak`)
- "Explain quantum entanglement" → 0.53 (was `good`, now `weak`)
- "Best practices for React hooks" → 0.54 (was `good`, now `weak`)
- "How to train a dog to sit" → 0.53 (was `good`, now `weak`)
- "What's the weather like today?" → 0.46 (was `weak`, stays `weak`)

**New fact: `training_config`**

Added a fact covering LoRA training configuration (r=16, alpha=32, 100-sample threshold, GGUF export). This fixed queries that previously had no good match:

| Query | Before | After |
|-------|--------|-------|
| "What LoRA rank should I use?" | 0.53 → `package_management` | **0.665 → `training_config`** |
| "How do I fine-tune a language model?" | 0.61 → random | **0.644 → `training_config`** |
| "How does the learning pipeline work?" | 0.61 → random | **0.621 → `training_config`** |

**Score distribution (after tuning):**
```
0.8-1.0: #               (1)   good
0.7-0.8: ########        (8)   good
0.6-0.7: ###########     (11)  good
0.5-0.6: #######         (7)   weak - correctly filtered
0.45-0.5: #              (1)   weak - correctly filtered
```

The system now only injects facts when it's genuinely confident about relevance. Irrelevant queries get core facts only instead of polluting context with noise.

### Files Changed

- `mcp-server/server.py` — Raised `good` threshold to 0.6, `weak` to 0.45, default `similarity_threshold` to 0.45
- `config.yaml` — `similarity_threshold: 0.45`

### Scaling to 31 Facts

Added 10 new system facts to broaden RAG coverage:

| Fact | Content |
|------|---------|
| `cpu` | Ryzen 9 5900X, 12C/24T, Zen 3 |
| `memory` | 32GB DDR4, separate from 32GB VRAM |
| `storage` | 2x NVMe — 952GB btrfs (OS) + 403GB ext4 (builds) |
| `kernel` | Linux 6.18.5-200.fc43.x86_64 |
| `containers` | Docker with 9 Redis containers, Podman 5.7.1, Toolbox, Flatpak |
| `tailscale` | Mesh VPN at 100.107.161.22 |
| `network` | 192.168.1.100/24 on enp5s0 |
| `dev_tools` | Python 3.14.2, Git 2.52, Homebrew 5.0.14, llama.cpp |
| `ujust` | Universal Blue command runner, ~40 recipes |
| `universal_blue` | Aurora-DX based on Fedora Kinoite 43 |

Re-ran the stress test with 40 queries (vs 28 before) — more diverse, targeting the new facts plus irrelevant queries.

**v2 vs v3 comparison:**

| Metric | v2 (21 facts) | v3 (31 facts) |
|--------|---------------|---------------|
| Queries tested | 28 | 40 |
| Good | 71.4% | **87.5%** |
| Weak | 28.6% | **12.5%** |
| Avg score | 0.650 | **0.703** |
| Median | 0.639 | **0.734** |

**New fact highlights:**
```
"What CPU do I have?"                 → 0.794  cpu
"How many cores does my processor?"   → 0.820  cpu
"How much RAM is installed?"          → 0.786  memory
"How much disk space is left?"        → 0.720  storage
"What kernel version?"                → 0.750  kernel
"What containers are running?"        → 0.745  containers
"What is my Tailscale IP?"            → 0.772  tailscale
"What ujust commands are available?"  → 0.797  ujust
"What is Universal Blue?"             → 0.747  universal_blue
"How do I benchmark GPU performance?" → 0.832  gpu_benchmarking
```

**All 5 irrelevant queries correctly weak:** pizza (0.40), Minecraft (0.54), France (0.44), blockchain (0.56), tomatoes (0.52). Zero false-positive fact injection.

**Score distribution shifted right:**
```
0.8-1.0: ##                    (2)   good
0.7-0.8: ######################  (22)  good
0.6-0.7: ###########            (11)  good
0.5-0.6: ###                    (3)   weak
0.45-0.5:                       (0)   weak
<0.45:   ##                    (2)   weak
```

The bulk of legitimate queries now land in the 0.7–0.8 band, well above the 0.6 good threshold. The system reliably distinguishes relevant queries from noise.

---

## Day 19: v0.9.0 — Dual LLM, Architecture Diagram, GitHub Release (Feb 15, 2026)

### The Crash That Wasn't

System went down mid-session. Dug through `journalctl -b -1` expecting a kernel panic — found an orderly `sudo reboot` triggered after an rpm-ostree package update (Google Chrome). The real problem: a new user service for **Qwen3-30B-A3B** was crash-looping on startup due to an invalid `--reasoning-budget 4096` flag (llama-server only accepts `-1` or `0`).

### VRAM Math: 30B Won't Fit

Quick math killed the 30B idea:
```
HiveCoder-7B:     ~7.7 GB
Qwen3-30B-A3B:   ~18 GB model + KV cache (65536 ctx × 4 parallel)
Total:            ~30+ GB → maxes out the 32 GB card
```

Swapped to **Qwen3-14B** (Q4_K_M, 8.4 GB) — fits comfortably alongside HiveCoder:
```
HiveCoder-7B:     ~7.7 GB  (:8089, system service)
Qwen3-14B:        ~13.3 GB (:8080, user service)
Total:            ~21 GB / 32 GB — 11 GB free headroom
```

Fixed the service file:
- Model: 30B-A3B → **14B**
- `--reasoning-budget`: `4096` → `-1` (unrestricted)
- `--ctx-size`: 65536 → 32768

Both models verified working:
```
HiveCoder-7B:  596 tok/s prompt, 56 tok/s generation
Qwen3-14B:     241 tok/s prompt, 54 tok/s generation (thinking mode active)
```

### Architecture Diagram

Created a full block diagram with Pillow (3400×1480px) showing all components and data flow:
- Clients → MCP/HTTP → Redis Cluster → LLM Inference
- Embedding Engine → RAG Pipeline → Prompt Injection
- Continuous Learning → LoRA → GGUF → Hot Swap
- Multi-node layout (aurora + r720xd planned)

### Versioning

Established semver starting from 0.5.0:

| Version | Milestone |
|---------|-----------|
| 0.5.0 | Redis Cluster + MCP Server |
| 0.6.0 | Dual-Mode HTTP API + Local LLM |
| 0.7.0 | Learning Pipeline + HiveCoder-7B |
| 0.8.0 | Semantic RAG + Fact Storage |
| **0.9.0** | **Dual LLM + Architecture Overhaul (current)** |
| 1.0.0 | Production release (planned) |

Added `VERSION` file, wired into HTTP API (`GET /` returns `"version": "0.9.0"`), created git tag `v0.9.0`.

### README Rewrite

Gutted the old README (509 lines of emoji soup) and rewrote it (313 lines):
- Accurate models, ports, and stats
- Architecture table + data flow diagram
- Comparison table vs Mem0, LangMem, Qdrant MCP
- Current performance numbers
- Version badge

### GitHub Release

Published `v0.9.0` release with full release notes and architecture diagram.

### Full System Test

Verified all layers post-reboot:
- **HTTP API**: 12 endpoints tested (/, /health, /stats, /llm/status, /v1/models, /v1/chat/completions, /memory/store, /memory/recall, /fact/get, /rag/suggestions, /llm/generate, /learning/queue/add)
- **MCP Tools**: 13/13 pass — full write/read/delete round-trip on every tool
- **LLM Inference**: Both models serving, RAG injection working
- **Learning Queue**: 51 samples accumulated (threshold 100 for auto-training)

### Key Numbers

| Metric | Value |
|--------|-------|
| Version | 0.9.0 |
| VRAM usage | 21 GB / 32 GB |
| Models serving | 2 (HiveCoder-7B + Qwen3-14B) |
| MCP tools | 13/13 passing |
| HTTP endpoints | 12/12 passing |
| RAG hit rate | 84% |
| Learning queue | 51 samples |
| Git commits | 51 |

---

## Day 20: First Continuous Training Run on 7B (Feb 16, 2026)

### The OOM Bug

Triggered `--train-now` with 32 filtered samples and the GPU immediately OOM'd. Root cause: the auto batch size calculator in `train_lora.py` estimated **40 MB per sample** — calibrated for Qwen2.5-**0.5B**, not the 7B model we'd been running since Day 8.

```
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 1.12 GiB.
GPU 0 has a total capacity of 31.86 GiB of which 30.00 MiB is free.
```

**First attempt**: OOM because llama-server was holding the GPU. Stopped `hivecoder-llm` to free VRAM.

**Second attempt**: Still OOM — 34GB free, but batch_size=32 meant the forward+backward pass needed ~18GB on top of the 14GB model weights. The `logits.float()` call during loss computation pushed it over.

### The Fix

Updated `get_optimal_batch_size()`:

| Parameter | Before | After |
|-----------|--------|-------|
| `memory_per_sample` | 40 MB (0.5B estimate) | **2 GB** (7B estimate) |
| Auto batch_size (32GB) | 32 | **~6** |
| Effective batch (with grad_accum=4) | 128 | **24** |

### Successful Training

With the fix, the full pipeline ran clean:

| Stage | Result | Time |
|-------|--------|------|
| LoRA training | 32 samples, loss: 0.0000 | ~20s |
| GGUF export | Q5_K_M (5.1 GB) | ~2 min |
| Deploy + health check | llama-server hot-swap | ~10s |
| Version bump | 0.9.0 → **0.9.1** | instant |
| Auto-cleanup | 4 old versions removed (~50 GB) | instant |

### Semantic Versioning

This was also the first deploy to use the semantic versioning system added in the previous session. The daemon now:
- Reads `learning-pipeline/MODEL_VERSION` for current semver
- Bumps patch version on successful deploy
- Publishes `model_version` in Redis stats (visible in `get_stats`)

### Lessons Learned

- Always calibrate batch size estimates to the actual model size, not the prototype
- The `PYTORCH_ALLOC_CONF=expandable_segments:True` env var helps with HIP memory fragmentation
- Stop llama-server before training — can't share 32GB between inference and training on the same card
- Training loss of 0.0000 on 32 samples suggests the model is memorizing — need more diverse data or multiple epochs to get meaningful gradients

### Files Changed

- `learning-pipeline/scripts/train_lora.py` — Fixed batch size estimate (40MB → 2GB per sample)
- `learning-pipeline/scripts/continuous_learning.py` — Semantic versioning (`_get_model_version`, `_bump_model_version`), `model_version` in published stats
- `mcp-server/server.py` — `model_version` field in `get_stats` response
- `learning-pipeline/MODEL_VERSION` — New file tracking semver (0.9.1)

---

## Day 21 — GPU Orchestration, Validation, and Talos (February 17, 2026)

### The Problems

Two issues from Day 20:
1. **OOM during training** — llama-server holds ~22GB VRAM, leaving nothing for PyTorch. Had to manually stop it before training.
2. **Loss 0.0000** — 1 epoch on 32 samples = pure memorization. No validation split, no way to detect overfitting.

### GPU Orchestration

Added automatic GPU clearing to the continuous learning daemon. New context manager `_gpu_training_context()` handles the full lifecycle:

- **Enter**: Stop `hivecoder-llm` (required — aborts if it fails), optionally stop user `llama-server`, wait 5s for VRAM to free, log usage via `rocm-smi`
- **Exit**: Restart user `llama-server` if stopped, restart `hivecoder-llm` only if deploy didn't already bring it back

Also added KDE desktop notifications (`notify-send`) so you know when training starts and when a new model goes live.

### Multi-Epoch Validation

Updated `train_lora.py` with proper training hygiene:

- `--val-split 0.2` — 80/20 train/val split (skipped if <10 samples)
- `--early-stopping-patience 2` — stops if val loss doesn't improve for 2 epochs
- `--epochs 3` (was 1) — enough to actually learn, early stopping prevents overfit
- Final `trainer.evaluate()` produces `eval_loss` in metrics JSON
- `continuous_learning.py` parses and stores `eval_loss` in the model registry

### Talos — New Project

Realized the hive-mind backend + open-interpreter combo needed a proper desktop frontend. Born: **Talos** — a local-first desktop AI for Fedora Kinoite + KDE Plasma 6.

Named after the bronze automaton from Greek mythology.

**What it does:**
- Agentic shell execution via HiveCoder-7B (through hive-mind HTTP API with RAG)
- Obsidian vault integration — search, read, create, open notes from terminal
- KDE desktop tools — notifications, clipboard (Wayland), Baloo file search
- Bronze/amber themed TUI via Rich

**Repo**: https://github.com/tlee933/talos

```
talos/
├── cli.py       — click CLI (talos, talos ask, talos do, talos vault)
├── agent.py     — Hive-Mind LLM client
├── shell.py     — async command execution
├── kde.py       — notify, clipboard, baloo
├── obsidian.py  — vault search/read/create/open/daily/tags
├── theme.py     — bronze automaton palette
└── tui.py       — interactive REPL
```

Also set up Obsidian (Flatpak) with a vault at `~/Documents/Vault` containing system docs, project notes, and reference material. Wired into Talos config so `talos vault search pytorch` just works.

### Housekeeping

- Fixed git config: proper username, noreply email, SSH protocol for `gh`
- `gh config set git_protocol ssh` — no more HTTPS credential dialogs

### Files Changed (hive-mind)

- `learning-pipeline/scripts/train_lora.py` — val split, early stopping, EarlyStoppingCallback, eval_loss in metrics
- `learning-pipeline/scripts/continuous_learning.py` — GPU orchestration (_gpu_training_context, _run_systemctl, _is_service_active, _check_vram_used), desktop notifications (_notify), epochs=3, val-split args, eval_loss parsing

---

## Days 22-25: Qwen3-14B Migration + QLoRA (February 22-25, 2026)

### The Big Model Switch

Replaced the dual-LLM setup (HiveCoder-7B + R1-Distill-14B) with a single **Qwen3-14B** model. Qwen3-14B handles both code and reasoning tasks well enough to eliminate the routing complexity, and frees VRAM headroom for training.

### transformers 5.0 Compatibility

Upgraded to transformers 5.0, which introduced breaking changes:
- `no_cuda` argument removed from `TrainingArguments` — replaced with `use_cpu`
- Both training paths (automated + continuous) updated

### bitsandbytes QLoRA from Source

Built bitsandbytes 0.50.0.dev0 from source for ROCm/gfx1201 (RDNA4):
```bash
cmake -B build -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=gfx1201
```

Enabled 4-bit QLoRA training (NF4 + double quantization) on Qwen3-14B — uses ~8-10GB VRAM after loading vs ~28GB for BF16 LoRA. Both modes still require stopping the LLM server due to VRAM peak during model loading.

### Training Pipeline Fixes

- **Batch size**: Fixed to `batch_size=1, grad_accum=8` — auto-calculation didn't account for 14B model weight size
- **Learning daemon**: Fixed batch size in continuous_learning.py to match
- **LLM restart gap**: Fixed timing issue where LLM server wasn't fully ready after training restart

### Model Registry

Model version bumped to **0.9.5** through automated continuous training cycles on Qwen3-14B.

### Files Changed

- `learning-pipeline/scripts/train_lora.py` — `use_cpu` instead of `no_cuda`, `--quantize` flag for QLoRA, batch_size=1
- `learning-pipeline/scripts/continuous_learning.py` — batch size fix, LLM restart timing
- `learning-pipeline/scripts/automated_training.sh` — batch_size=1, grad_accum=8, Qwen3-14B target

---

## Days 26-28: Talos AI Suricata (February 28 - March 1, 2026)

### A New Project

Built **Talos AI Suricata** — an autonomous network threat analysis system for OPNsense. Suricata IPS generates thousands of alerts daily, but most are noise. Talos uses HiveCoder (Qwen3-14B) to classify each alert, correlate attack patterns, and automatically block confirmed threats via the OPNsense firewall API.

**Repo**: [talos-ai-suricata](https://github.com/tlee933/talos-ai-suricata)

### Architecture

```
OPNsense (Suricata IPS)
    |  syslog (TCP 5140)
    v
eve_receiver (alderlake)
    |  Redis push
    v
ai_suricata daemon (alderlake)
    |  LLM call
    v
HiveCoder / Qwen3-14B (aurora :8090)
    |  structured JSON
    v
4-tier auto-block --> OPNsense API --> ai_blocklist alias
```

### Multi-Node Deployment

Deployed the 3 Talos services (eve_receiver, ai_suricata, dashboard) to **alderlake** as a podman container, keeping the GPU and Redis on aurora. This was the first real multi-node deployment — alderlake calls aurora's LLM and Redis over the LAN.

| Node | Role |
|------|------|
| **alderlake** | Talos AI Suricata container (eve_receiver, ai_suricata, dashboard) |
| **aurora** | HiveCoder LLM (:8090), Redis cluster (7000-7002), training pipeline |
| **NAS** | Persistent EVE logs, AI analysis results, model backups |
| **OPNsense** | Suricata IPS, firewall, DNSBL, DNS redirect |

### Security Hardening

- 24+ scanner CIDRs blocked at firewall level
- 24 dangerous inbound ports blocked (SSH, SMB, RDP, Redis, Docker API, etc.)
- ICMP blocking, RFC1918 anti-spoofing
- 7 DNSBL blocklists (Hagezi, ThreatFox, Steven Black)
- DNS redirect (force all LAN DNS through OPNsense)
- Suricata detect profile set to HIGH

### alderlake: The New Second Node

The R720xd was replaced by **alderlake** — an i7-12700 (8P+4E, 20 threads) with 64GB RAM running Fedora CoreOS. Already running Redis replicas and the embedding service, now also hosts Talos AI Suricata.

### Backup Script Fix

Fixed the `hivecoder-backup.service` failure (rsync exit code 3 — missing intermediate directories). Added `mkdir -p` before rsync and a cleanup routine for old training logs, data, and model checkpoints.

---

## Milestones (updated)

- [x] **Phase 1**: Redis Cluster (Feb 1)
- [x] **Phase 2**: MCP Server (Feb 1)
- [x] **Phase 2.5**: Local LLM Inference (Feb 2)
- [x] **Phase 2.7**: Dual-Mode Access (Feb 3)
- [x] **Phase 4**: Learning Pipeline (Feb 5)
- [x] **Phase 4.5**: Smart Optimizer (Feb 7)
- [x] **HiveCoder-7B**: First Foundation Model (Feb 8)
- [x] **Phase 5**: HiveCoder Integration (Feb 8)
- [x] **Phase 6**: Multi-Node — R720xd / alderlake (Feb 8-9)
- [x] **Phase 7**: Continuous Learning (Feb 8)
- [x] **Phase 8**: PyTorch 2.10 + ROCm 7.12 Native (Feb 14)
- [x] **Phase 9**: Active Learning — RAG Retrieval Mining (Feb 15)
- [x] **Phase 10**: First Continuous Training on 7B — v0.9.1 (Feb 16)
- [x] **Phase 11**: Qwen3-14B Migration + QLoRA (Feb 22-25)
- [x] **Phase 12**: Talos AI Suricata — Autonomous Threat Defense (Feb 28 - Mar 1)
- [x] **Phase 13**: 4-Node Architecture — Aurora, Alderlake, NAS, OPNsense (Mar 1)

---

## Credits

Built with:
- Claude Code (Opus 4.6)
- A lot of coffee
- Pure determination

**Status**: Production Ready
**Version**: 0.9.5
**Date**: March 1, 2026
**Author**: hashcat

---

*The hive never forgets.*
