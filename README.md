# Hive-Mind

> **Distributed AI Memory System with Semantic RAG, Dual-LLM Routing, and Knowledge Distillation**

[![Version](https://img.shields.io/badge/version-0.9.1-blue.svg)](VERSION)
[![Redis](https://img.shields.io/badge/Redis_Cluster-7.4.7-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ROCm](https://img.shields.io/badge/ROCm-7.12-FF6600?logo=amd&logoColor=white)](https://rocm.docs.amd.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A self-improving AI backend that persists context across sessions, enriches LLM prompts with semantic RAG, routes queries to the optimal model, and continuously fine-tunes a local model from its own interactions. The bigger model teaches the smaller one. Fully local, zero cloud dependencies.

![Architecture](hivemind_architecture.png)

---

## What It Does

- **Intelligent model routing** — queries auto-classify and route to the best model: HiveCoder-7B (88 tok/s) for code/shell, R1-Distill-14B (55 tok/s) for reasoning
- **Knowledge distillation** — R1's high-quality reasoning answers auto-feed HiveCoder's LoRA training pipeline. The bigger model teaches the smaller one through the existing learning loop
- **Persistent memory** across terminal restarts, shared across machines via Redis cluster
- **Semantic RAG** that automatically injects relevant context into every LLM prompt
- **Continuous learning** pipeline that collects interactions, fine-tunes via LoRA, exports to GGUF, and hot-swaps the model into production
- **Dual access modes** via MCP protocol (Claude Code) and HTTP API (any client)
- **Active retrieval tracking** with hit rate monitoring and gap analysis

---

## Architecture

| Component | Port | Role |
|-----------|------|------|
| **Redis Cluster** | 7000-7005 | 3 masters + 3 replicas, sessions, facts, caches, streams |
| **HiveCoder-7B** | 8089 | LoRA fine-tuned Qwen2.5-Coder-7B, code/shell/tools, 88 tok/s |
| **R1-Distill-14B** | 8080 | DeepSeek-R1-Distill-Qwen-14B, reasoning/analysis, 55 tok/s |
| **HTTP API** | 8090 | FastAPI server, OpenAI-compatible proxy with model routing + RAG |
| **Model Router** | - | Pure-function query classifier, sub-millisecond |
| **MCP Server** | stdio | Claude Code integration, 12 tools |
| **Learning Daemon** | - | 5-min interval, drains queue, triggers LoRA training |

### Data Flow

```
User Query
  -> MCP Server or HTTP API
    -> Model Router (intent classify -> code or reasoning)
      -> Semantic RAG (embed query, cosine sim against facts)
        -> Inject matching facts into system prompt
          -> HiveCoder-7B (:8089) or R1-Distill-14B (:8080)
            -> Response returned + routing headers
              -> Interaction logged to learning queue
                -> R1 responses auto-rated positive (distillation)

Learning Queue (Redis Stream)
  -> Continuous Learning Daemon (every 5 min)
    -> Quality filter (r1-distill -> always pass)
      -> LoRA fine-tune -> GGUF export -> Hot swap HiveCoder
```

### MCP Tools

| Category | Tools |
|----------|-------|
| **Memory** | `memory_store`, `memory_recall`, `memory_list_sessions` |
| **Facts / RAG** | `fact_store`, `fact_get`, `fact_delete`, `fact_suggestions` |
| **LLM** | `llm_generate`, `llm_code_assist`, `llm_complete` |
| **System** | `tool_cache_get`, `tool_cache_set`, `learning_queue_add`, `get_stats` |

---

## Performance

| Component | Metric | Value |
|-----------|--------|-------|
| HiveCoder-7B | Generation speed | 88 tok/s |
| R1-Distill-14B | Generation speed | 55 tok/s |
| Redis Cluster | Pipeline throughput | 59,763 ops/s |
| Redis Cluster | Latency | < 1ms |
| RAG | Hit rate | 84% (31 facts, semantic) |
| Training | Full cycle | ~4 min (LoRA + GGUF export) |
| VRAM | Total / Used / Free | 32 GB / 21 GB / 11 GB |

---

## Quick Start

### Prerequisites

- Docker (for Redis cluster)
- Python 3.12+ (tested on 3.14)
- AMD GPU with 16GB+ VRAM and ROCm 6.x+ (tested on R9700 32GB, ROCm 7.12)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server` binary)

### Install

```bash
git clone https://github.com/tlee933/hive-mind.git
cd hive-mind

# Deploy Redis Cluster (6 nodes + Docker)
./scripts/deploy-redis-cluster.sh

# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your Redis password and model paths
```

### Start Services

```bash
# Redis (should already be running via Docker)
docker ps | grep redis

# HiveCoder-7B
./scripts/start-hivecoder.sh

# R1-Distill-14B
./scripts/start-r1-distill.sh

# HTTP API
source .venv/bin/activate
CONFIG_PATH=../config.yaml python mcp-server/http_server.py
```

### Claude Code Integration

Add to `.mcp.json` or `~/.config/claude-code/mcp_config.json`:

```json
{
  "mcpServers": {
    "hive-mind": {
      "command": "python",
      "args": ["mcp-server/server.py"],
      "env": {
        "CONFIG_PATH": "/path/to/hive-mind/config.yaml"
      }
    }
  }
}
```

### HTTP API

```bash
# Health check
curl localhost:8090/health

# Chat with model routing + RAG injection
curl localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "explain btrfs snapshots"}]}'
# -> Routes to R1-Distill-14B (reasoning query)

# Interactive docs
open http://localhost:8090/docs
```

---

## Continuous Learning Pipeline

The system improves itself over time via [self-instruct](https://arxiv.org/abs/2212.10560) — R1's high-quality responses become training data for HiveCoder:

1. **Collect** — Every LLM interaction is logged to a Redis stream
2. **Filter** — Quality filter removes low-quality samples; R1 responses always pass
3. **Train** — LoRA fine-tuning (r=16, alpha=32) triggers after 50+ samples
4. **Export** — Trained model exported to GGUF (Q5_K_M, 5.1 GB)
5. **Deploy** — Symlink hot-swap + llama-server restart, zero downtime

```bash
# Check learning status
python learning-pipeline/scripts/continuous_learning.py --status

# Force immediate training
python learning-pipeline/scripts/continuous_learning.py --train-now
```

---

## Semantic RAG

Every LLM call is enriched with relevant facts from the knowledge base:

1. **Embed** the user query using bge-small-en-v1.5 (768-dim, runs on CPU)
2. **Search** against pre-computed fact embeddings via cosine similarity
3. **Filter** by threshold (>= 0.45 good, >= 0.6 strong match)
4. **Inject** matching facts into the system prompt before sending to llama-server
5. **Track** retrieval quality (hit rate, missed queries, weak matches)

---

## Hardware

| Node | Role | Specs |
|------|------|-------|
| **aurora** | GPU inference + training | AMD Ryzen 9 7900X, R9700 32GB VRAM, ROCm 7.12 |
| **r720xd** (planned) | Embeddings + storage | Dual Xeon E5-2660, 64GB RAM, 24x 2.5" bays |

---

## Project Structure

```
hive-mind/
  mcp-server/
    server.py              # MCP server (stdio, 12 tools)
    http_server.py         # FastAPI HTTP API (:8090) with model routing
    router.py              # Query intent classifier (pure functions)
    test_router.py         # Router tests
  learning-pipeline/
    scripts/
      continuous_learning.py   # Training daemon + R1 distillation
      train_lora.py            # LoRA fine-tuning
      export_model.py          # GGUF export
    models/                    # Model registry + exports
  scripts/
    deploy-redis-cluster.sh    # Redis cluster setup
    start-hivecoder.sh         # HiveCoder-7B launcher
    start-r1-distill.sh        # R1-Distill-14B launcher
  config.yaml                  # Multi-model + Redis config
  hivemind_architecture.png    # Architecture diagram
```

---

## How It Compares

| Feature | Hive-Mind | Mem0 | LangMem | Qdrant MCP |
|---------|:---------:|:----:|:-------:|:----------:|
| Persistent memory | Yes | Yes | Yes | Yes |
| Semantic RAG | Yes | Yes | Yes | Yes |
| Local LLM inference | Yes | No | No | No |
| Multi-model routing | **Yes** | No | No | No |
| Knowledge distillation | **Yes** | No | No | No |
| Continuous fine-tuning | **Yes** | No | No | No |
| Self-improving model | **Yes** | No | No | No |
| Retrieval quality tracking | **Yes** | No | No | No |
| MCP protocol | Yes | Yes | No | Yes |
| Fully local / zero cloud | **Yes** | No | No | Partial |

---

## Documentation

| Document | Description |
|----------|-------------|
| [JOURNEY.md](JOURNEY.md) | Full project history, build log |
| [QUICKSTART.md](QUICKSTART.md) | Get running in 2 minutes |
| [MCP_SERVER_READY.md](MCP_SERVER_READY.md) | Operational status, all tools, config |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Deep dive into system design |
| [learning-pipeline/README.md](learning-pipeline/README.md) | Training pipeline docs |

---

## License

MIT

---

<div align="center">

Built on Fedora Atomic (Kinoite 43) with ROCm 7.12 and PyTorch 2.10.0

[Back to Top](#hive-mind)

</div>
