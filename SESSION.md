# 🐝 Hive-Mind: Session State

**Last Updated**: 2026-02-01
**Status**: Phase 1 - Documentation Complete, Ready for Implementation
**Next Session**: Start with Phase 1 Redis deployment on NAS

---

## 🎯 Project Goal

Build a **distributed AI memory and learning cluster** using:
- **Redis on NAS** (192.168.1.7) - Central nervous system
- **BEAST** (RDNA4 R9700 32GB) - Heavy compute, training
- **DELL** (Alderlake + RDNA2 12GB) - Inference, embeddings
- **Self-learning** via LoRA fine-tuning from interaction data

## 🏗️ Architecture Overview

```
┌──────────────────────┐    ┌─────────────────────┐    ┌──────────────────┐
│ BEAST (RDNA4)        │    │ DELL (RDNA2)        │    │ NAS              │
│ ─────────────────    │    │ ────────────────    │    │ ───────────      │
│ R9700 32GB VRAM      │    │ 12GB VRAM           │    │ Dual GbE NICs    │
│ PyTorch 125 TFLOPS   │    │ 32GB DDR            │    │ Redis Server     │
│ Heavy compute        │    │ llama-server        │    │ 192.168.1.7      │
│ Training/fine-tune   │    │ Embeddings          │    │ Persistent store │
└──────────────────────┘    │ Tool-use model      │    │ Vector cache     │
                            └─────────────────────┘    └──────────────────┘
         ↑                           ↑                          ↑
         └───────────────────────────┴──────────────────────────┘
                        Redis backbone (1Gbps)
```

## ✅ What's Done

### Documentation (Phase 1 Complete)
- [x] `/mnt/build/MCP/hive-mind/README.md` - Project overview
- [x] `/mnt/build/MCP/hive-mind/docs/ARCHITECTURE.md` - System design
- [x] `/mnt/build/MCP/hive-mind/docs/DEPLOYMENT.md` - Deployment guide
- [x] `/mnt/build/MCP/hive-mind/schemas/redis-schema.md` - Data schemas

### Code (Scaffolded, Needs Testing)
- [x] `/mnt/build/MCP/hive-mind/mcp-server/server.py` - MCP server implementation
- [x] `/mnt/build/MCP/hive-mind/requirements.txt` - Python dependencies
- [x] `/mnt/build/MCP/hive-mind/config.example.yaml` - Config template

### Scripts (Ready to Run)
- [x] `/mnt/build/MCP/hive-mind/scripts/setup-redis-nas.sh` - Deploy Redis on NAS
- [x] `/mnt/build/MCP/hive-mind/scripts/test-connection.sh` - Test Redis connectivity
- [x] `/mnt/build/MCP/hive-mind/scripts/install-mcp-server.sh` - Install MCP server

## 🚧 What's Next

### Phase 1: Redis on NAS (IMMEDIATE NEXT STEP)

**Prerequisites**:
- [ ] SSH access to Netgear ReadyNAS (192.168.1.7)
- [ ] Docker installed on NAS OR ability to install Redis natively
- [ ] Firewall rules allowing Redis port 6379 from local network

**Action Items**:
1. **SSH into NAS**: Get credentials from user
2. **Run setup script**:
   ```bash
   # Transfer script to NAS
   scp /mnt/build/MCP/hive-mind/scripts/setup-redis-nas.sh admin@192.168.1.7:/tmp/

   # SSH and run
   ssh admin@192.168.1.7
   bash /tmp/setup-redis-nas.sh
   ```
3. **Save Redis password** from script output
4. **Test from BEAST**:
   ```bash
   cd /mnt/build/MCP/hive-mind
   REDIS_PASSWORD=<from_nas> ./scripts/test-connection.sh
   ```
5. **Benchmark latency**: Target < 5ms avg

**Success Criteria**:
- Redis running on NAS with persistence (RDB + AOF)
- Network latency < 5ms from BEAST
- Password authentication working
- Test operations (SET/GET) successful

### Phase 2: MCP Server on BEAST

**Action Items**:
1. **Install dependencies**:
   ```bash
   cd /mnt/build/MCP/hive-mind
   ./scripts/install-mcp-server.sh
   ```
2. **Configure** `config.yaml` with Redis password
3. **Test MCP server**:
   ```bash
   source .venv/bin/activate
   python mcp-server/server.py --debug
   ```
4. **Integrate with Claude Code**: Add MCP server to Claude config

**Success Criteria**:
- MCP server connects to Redis on NAS
- Basic operations work (memory_store, memory_recall)
- Session persists across terminal restarts
- Claude Code can use MCP tools

### Phase 3: llama-server on DELL

**Action Items**:
1. SSH into DELL
2. Deploy llama.cpp with ROCm support
3. Download 8B tool-use model
4. Set up embedding service
5. Integrate with Redis

**Success Criteria**:
- llama-server running on port 8080
- Embedding service running on port 8081
- Can query model from BEAST
- Embeddings cached in Redis

### Phase 4: Training Pipeline on BEAST

**Action Items**:
1. Activate PyTorch ROCm environment
2. Build training pipeline script
3. Pull from learning:queue
4. Fine-tune LoRA adapters
5. Deploy updated models

**Success Criteria**:
- Can process learning queue
- LoRA fine-tuning works on R9700
- Model improvements measurable
- Automated nightly runs

## 📊 Project Status

| Phase | Status | Started | Completed |
|-------|--------|---------|-----------|
| Planning & Design | ✅ DONE | 2026-02-01 | 2026-02-01 |
| Redis on NAS | 🔄 READY | - | - |
| MCP Server | 📝 WAITING | - | - |
| Inference (DELL) | 📝 WAITING | - | - |
| Training (BEAST) | 📝 WAITING | - | - |

## 🔑 Key Information

### Hardware Details

**BEAST** (Primary workstation):
- GPU: AMD Radeon AI PRO R9700 (RDNA4, gfx1201)
- VRAM: 32GB GDDR6
- OS: Fedora Atomic 43
- ROCm: 7.12.0 (built from TheRock)
- PyTorch: 2.9.1+rocm.gfx12 (125 TFLOPS FP16)
- Location: `/var/mnt/build/TheRock`

**DELL** (Inference node):
- CPU: Intel Alderlake
- RAM: 32GB DDR
- GPU: RDNA2 12GB VRAM
- SSD storage
- Purpose: llama-server + embeddings

**NAS** (Storage):
- Model: Netgear ReadyNAS
- Network: Dual gigabit NICs
- IP: 192.168.1.7
- Purpose: Redis server

### Network
- Topology: 1Gbps Ethernet
- Latency target: < 5ms between nodes
- All nodes on same subnet (192.168.1.x)

### Project Paths
- **Project root**: `/mnt/build/MCP/hive-mind`
- **PyTorch venv**: `/mnt/build/torch-venv-py312`
- **TheRock build**: `/var/mnt/build/TheRock`
- **Session file**: `/mnt/build/MCP/hive-mind/SESSION.md` (this file)

## 💬 Session Continuity

**To resume in a new session**, have Claude read this file:

```
Read /mnt/build/MCP/hive-mind/SESSION.md and continue where we left off.
```

**Current blocker**: Need SSH credentials for NAS to run Phase 1 setup.

**User context**:
- Has Redis expertise (built Redis stacks in past)
- Comfortable with system administration
- Wants hands-on involvement in setup
- Building for AI development and experimentation

## 📝 Notes & Decisions

### Design Decisions Made
1. **Redis over SQLite**: Chose Redis on NAS for multi-machine access
2. **Dual-phase learning**: RAG first (immediate), LoRA later (periodic)
3. **Separate inference node**: Keep DELL for 24/7 inference, BEAST for heavy compute
4. **Async everything**: Non-blocking writes to avoid latency spikes

### Open Questions
- [ ] Redis AUTH password strength requirements?
- [ ] NAS backup schedule for Redis data?
- [ ] Which 8B model for DELL? (Llama 3.2, Qwen, Mistral?)
- [ ] LoRA update frequency? (nightly, weekly, manual?)

### User Feedback
- User excited about distributed architecture
- Likes "hive-mind" name (chosen over "distributed-memory-mcp")
- Wants to leverage existing Redis knowledge
- Impressed with TheRock ROCm build (125 TFLOPS!)

---

## 🚀 Next Actions for New Session

1. **Confirm NAS access**: User provides SSH key or credentials
2. **Run Redis setup**: Execute `setup-redis-nas.sh` on NAS
3. **Test connectivity**: Verify < 5ms latency from BEAST
4. **Install MCP server**: Run `install-mcp-server.sh` on BEAST
5. **Update this file**: Mark tasks complete, add new blockers

---

**Remember**: This project is about building a **learning system** that gets smarter with use. The goal isn't just memory - it's continuous improvement through distributed compute and persistent knowledge. 🐝🧠🔥
