# 🐝 Hive-Mind Project Inventory

**Generated**: 2026-02-01  
**Status**: Production Ready

---

## 📦 Complete File Structure

```
hive-mind/
├── .git/                          # Git repository
├── .gitignore                     # Git ignore rules (secrets excluded)
├── LICENSE                        # MIT License
├── README.md                      # Main documentation ⭐
├── INVENTORY.md                   # This file
│
├── Core Configuration
├── config.yaml                    # Redis + MCP config (gitignored - contains password)
├── config.example.yaml            # Template config (safe for git)
├── requirements.txt               # Python dependencies
│
├── Documentation
├── COMPLETE.md                    # Achievement summary
├── CLUSTER_STATUS.md              # Redis cluster ops manual
├── MCP_SERVER_READY.md            # MCP usage guide
├── PERFORMANCE.md                 # Benchmark results
├── SESSION.md                     # Resume guide
├── STATUS.md                      # System status
├── REDIS_INFO.md                  # Redis information
│
├── docs/
│   ├── ARCHITECTURE.md            # System design deep dive
│   ├── CLUSTER_ARCHITECTURE.md    # Cluster architecture
│   └── DEPLOYMENT.md              # Deployment guide
│
├── MCP Server
├── mcp-server/
│   └── server.py                  # MCP server implementation (Python 3.14)
│
├── schemas/
│   └── redis-schema.md            # Redis data structures
│
├── Learning Pipeline
├── learning-pipeline/
│   ├── Dockerfile                 # ROCm + PyTorch container
│   ├── docker-compose.yml         # GPU orchestration
│   ├── Makefile                   # Easy commands
│   ├── README.md                  # Pipeline guide
│   ├── DEPLOYMENT.md              # Production deployment
│   │
│   ├── configs/
│   │   ├── .gitkeep
│   │   └── training_config.example.yaml
│   │
│   ├── scripts/
│   │   ├── collect_data.py       # Data collection from Redis
│   │   ├── train_lora.py         # LoRA fine-tuning
│   │   ├── export_model.py       # Model export
│   │   ├── pipeline.sh           # Full automation
│   │   └── test_pipeline.sh      # Validation suite
│   │
│   ├── data/                      # Training datasets (gitignored)
│   │   └── .gitkeep
│   │
│   └── models/                    # Trained models (gitignored)
│       └── .gitkeep
│
├── Deployment Scripts
├── scripts/
│   ├── install-mcp-server.sh
│   ├── setup-redis-local.sh
│   ├── setup-redis-nas.sh
│   └── test-connection.sh
│
├── Testing
├── tests/
│   ├── benchmark-hive-mind.py     # Performance benchmarks
│   └── test-hive-mind-stack.py    # Integration tests
│
├── Session Archives
├── session-archives/
│   ├── SESSION_2026-02-01.md      # Build session summary
│   ├── llama-7b.log               # Inference logs
│   └── llama-8b.log
│
└── scripts-archive/
    ├── add-hive-mind-mcp.sh       # MCP integration script
    ├── benchmark-hive-mind.py     # Performance tests
    ├── build-redis-cluster.sh     # Redis deployment
    ├── start-llama-8b-native.sh   # 8B model startup
    ├── start-llama-native.sh      # 7B model startup
    └── test-hive-mind-stack.py    # Stack validation
```

---

## 📊 Statistics

| Metric | Count |
|--------|-------|
| **Total Files** | 37+ |
| **Lines of Code** | 6,000+ |
| **Documentation** | 15 files |
| **Python Scripts** | 8 |
| **Shell Scripts** | 10 |
| **Config Files** | 5 |
| **Tests** | 8/8 passing |

---

## 🎯 Key Components

### 1. Redis Cluster (Running)
- 6 nodes: 3 masters + 3 replicas
- 3 Sentinels for HA
- Password protected
- Performance: 12K-60K ops/sec

### 2. MCP Server (Ready)
- Cluster-aware Redis client
- Session management
- Tool caching
- Learning queue
- Ready for Claude Code

### 3. LLM Inference (Running)
- Port 8080: Qwen2.5-Coder-7B (89 tok/s)
- Port 8088: Qwen3-8B (74 tok/s)
- ROCm acceleration
- 11.2 GB / 31.9 GB VRAM used

### 4. Learning Pipeline (Deployed)
- Docker-based, portable
- Data collection working
- LoRA training ready
- Production deployment guide

---

## 🔐 Security

### Protected (Not in Git)
- ✅ config.yaml (contains password)
- ✅ learning-pipeline/data/* (training data)
- ✅ learning-pipeline/models/* (trained models)
- ✅ .venv/ (Python virtual environment)

### Safe in Git
- ✅ All documentation
- ✅ All source code
- ✅ Example configs (no secrets)
- ✅ Scripts and tests

---

## 🚀 Quick Reference

### Start Everything
```bash
# Redis Cluster (already running)
docker ps | grep redis

# Llama Servers (already running)
curl http://localhost:8080/health
curl http://localhost:8088/health

# MCP Server (connect via Claude Code)
# Config: ~/.config/claude-code/mcp_config.json
```

### Learning Pipeline
```bash
cd learning-pipeline
make build    # Build container
make test     # Run tests
make collect  # Collect data
make train    # Train model
```

### Development
```bash
# Activate Python environment
source .venv/bin/activate

# Run tests
python tests/test-hive-mind-stack.py

# Benchmarks
python tests/benchmark-hive-mind.py
```

---

## 📝 Important Files

### Must Read
1. **README.md** - Start here
2. **COMPLETE.md** - What we built
3. **PERFORMANCE.md** - Benchmark results

### Operations
4. **CLUSTER_STATUS.md** - Redis ops
5. **MCP_SERVER_READY.md** - MCP usage
6. **learning-pipeline/DEPLOYMENT.md** - Training

### Reference
7. **docs/ARCHITECTURE.md** - System design
8. **SESSION_2026-02-01.md** - Build notes

---

## 🌐 GitHub

**Repository**: https://github.com/tlee933/hive-mind  
**Status**: ✅ Public  
**Stars**: (waiting for them!)

---

## 💪 Next Steps

1. ✅ Everything archived
2. ✅ All scripts saved
3. → Restart Claude Code
4. → MCP connects automatically
5. → Start learning from interactions!

---

**Status**: 🔥 PRODUCTION READY  
**Archive**: ✅ COMPLETE  
**Safety**: 🛡️ NO SECRETS IN GIT

**Ready to restart Claude Code!** 🐝
