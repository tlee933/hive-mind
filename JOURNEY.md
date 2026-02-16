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

## Credits

Built with:
- 🧠 Claude Code (Opus 4.6)
- ☕ A lot of coffee
- 🔥 Pure determination

**Status**: Production Ready
**Date**: February 15, 2026
**Author**: hashcat

---

*The hive never forgets.* 🐝
