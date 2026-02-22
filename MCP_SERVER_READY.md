# Hive-Mind MCP Server: Production Status

**Date**: 2026-02-22
**Version**: v0.7.3
**Status**: OPERATIONAL — Dual-LLM Routing + Knowledge Distillation

---

## What's Working

### Core Infrastructure
```
Redis Cluster       6 nodes (3 masters + 3 replicas), 4.26M keys, 99+ sessions
MCP Server          Claude Code integration via .mcp.json
HTTP API            :8090 — multi-model routing, streaming, tools
HiveCoder-7B        :8089 — code/shell/tools, 88 tok/s (LoRA fine-tuned Qwen2.5-Coder Q5_K_M)
R1-Distill-14B      :8080 — reasoning/analysis, 55 tok/s (DeepSeek-R1-Distill-Qwen-14B Q4_K_M)
Semantic RAG        bge-small-en-v1.5 embeddings, 768-dim vectors
Learning Pipeline   LoRA -> GGUF -> hot-swap, R1 knowledge distillation
```

### MCP Tools (12 tools)
```
memory_store / recall / list_sessions    Persistent context across sessions
tool_cache_get / set                     Cache expensive tool outputs (1h TTL)
learning_queue_add                       Log interactions for training pipeline
fact_store / get / delete                Semantic RAG fact management
fact_suggestions                         RAG gap analysis, missed query tracking
llm_generate / complete                  HiveCoder-7B text + code generation
llm_code_assist                          Code review, fix, optimize, explain, document
get_stats                                Redis info, session counts, queue lengths, LLM status
```

### Clients
```
Talos TUI           Python REPL — agentic execution, tool-use, reasoning, auto-rating
Firefox/Zen         Svelte sidebar extension — streaming, markdown, suggestions
Claude Code         MCP integration — memory, facts, learning, LLM access
HTTP Clients        curl / any OpenAI-compatible client
```

---

## Intelligent Model Router (v0.7.3)

Queries auto-route to the best model via `mcp-server/router.py`:

```
User Query
    |
    v
Model Router (pure-function classifier, sub-millisecond)
    |
    +-- CODE_SIGNALS (~30 keywords) ---------> HiveCoder-7B  :8089  88 tok/s
    |   code blocks, imports, file paths        code / shell / tools
    |
    +-- REASONING_SIGNALS (~25 keywords) ----> R1-Distill-14B :8080  55 tok/s
    |   explain, why, compare, analyze          reasoning / analysis
    |
    +-- /reason command -----> always R1
    +-- model_hint ----------> explicit override
    +-- default -------------> HiveCoder (faster)
```

Response headers: `X-Model-Used`, `X-Model-Id`, `X-Routing-Reason`

---

## Knowledge Distillation (The Self-Improving Loop)

```
R1-Distill-14B produces high-quality reasoning answers
    |
    v
Auto-rated positive, tagged model_source="r1-distill"
    |
    v
Quality Filter (r1-distill -> ALWAYS PASS)
    |
    v
Format -> JSONL training examples
    |
    v
LoRA fine-tune HiveCoder-7B (1 epoch, incremental)
    |
    v
GGUF export -> hot-swap into running HiveCoder
    |
    v
HiveCoder gets smarter, handles more queries, still 88 tok/s
```

The bigger model teaches the smaller one. The loop tightens over time.

---

## How to Start

### Full Stack
```bash
# 1. Redis cluster (if not running)
cd /mnt/build/MCP/hive-mind
./scripts/start-redis-cluster.sh

# 2. HiveCoder-7B
./scripts/start-hivecoder.sh

# 3. R1-Distill-14B
./scripts/start-r1-distill.sh

# 4. HTTP API
source .venv/bin/activate
CONFIG_PATH=../config.yaml python mcp-server/http_server.py
```

### MCP Server Only (Claude Code)
Configured via `.mcp.json` in project root — starts automatically with Claude Code.

---

## Claude Code Integration

### Config (.mcp.json)
```json
{
  "mcpServers": {
    "hive-mind": {
      "command": "/mnt/build/MCP/hive-mind/.venv/bin/python",
      "args": ["/mnt/build/MCP/hive-mind/mcp-server/server.py"],
      "env": {
        "CONFIG_PATH": "/mnt/build/MCP/hive-mind/config.yaml"
      }
    }
  }
}
```

---

## Configuration

### Multi-Model Config (config.yaml)
```yaml
inference:
  enabled: true
  default_model: "hivecoder-7b"
  timeout: 120
  models:
    hivecoder-7b:
      endpoint: "http://127.0.0.1:8089"
      display_name: "HiveCoder-7B"
      capabilities: ["code", "shell", "tools"]
      max_tokens: 1024
      temperature: 0.7
    r1-distill-14b:
      endpoint: "http://127.0.0.1:8080"
      display_name: "R1-Distill-14B"
      capabilities: ["reasoning", "analysis", "explanation", "general"]
      max_tokens: 2048
      temperature: 0.7
```

Backward compatible — if no `models` dict, falls back to flat `inference.endpoint`.

### Redis Cluster
```yaml
redis:
  cluster_mode: true
  nodes:
    - host: "127.0.0.1"
      port: 7000
    - host: "127.0.0.1"
      port: 7001
    - host: "127.0.0.1"
      port: 7002
```

---

## Performance

| Metric | Value |
|--------|-------|
| Cluster nodes | 6 (3 masters + 3 replicas) |
| Redis keys | 4.26M |
| Sessions | 99+ (28-day TTL) |
| Redis latency | < 1ms (localhost) |
| Redis throughput | ~300K ops/sec |
| HiveCoder-7B | 88 tok/s |
| R1-Distill-14B | 55 tok/s |
| VRAM used | ~21/32 GB |
| GPU | AMD Radeon RX 9070 XT, ROCm 7.12 |

---

## Project Structure

```
/mnt/build/MCP/hive-mind/
├── mcp-server/
│   ├── server.py              MCP server (Claude Code tools)
│   ├── http_server.py         HTTP API with model routing
│   ├── router.py              Query intent classifier
│   └── test_router.py         14 router tests
├── learning-pipeline/
│   └── scripts/
│       └── continuous_learning.py   LoRA training + R1 distillation
├── scripts/
│   ├── start-hivecoder.sh     HiveCoder-7B launcher
│   ├── start-r1-distill.sh    R1-Distill-14B launcher
│   └── start-redis-cluster.sh Redis cluster launcher
├── config.yaml                Multi-model + Redis config
├── hivemind_architecture.png  Architecture diagram
├── requirements.txt           Dependencies
└── .venv/                     Python virtualenv
```

---

## Roadmap

### Done
- [x] Redis Cluster (6 nodes, cluster mode)
- [x] MCP Server + Claude Code integration
- [x] Session management (28-day TTL)
- [x] Memory store/recall
- [x] Tool caching
- [x] Learning queue + training pipeline
- [x] HTTP API (:8090)
- [x] HiveCoder-7B serving (88 tok/s)
- [x] Semantic RAG (bge-small-en-v1.5)
- [x] Fact store/get/delete + suggestions
- [x] LLM generate/complete/code_assist tools
- [x] Continuous learning pipeline (LoRA -> GGUF -> hot-swap)
- [x] R1-Distill-14B serving (55 tok/s)
- [x] Intelligent model router
- [x] Knowledge distillation (R1 teaches HiveCoder)
- [x] Talos TUI client (agentic, tool-use, reasoning)
- [x] Firefox/Zen sidebar extension (v0.7.3)
- [x] 147 Python tests + 69 JS tests

### Future
- [ ] Multi-node deployment (DELL as second machine)
- [ ] Cascade routing (HiveCoder fallback to R1 on low confidence)
- [ ] Evaluation framework (measure distillation effectiveness)
- [ ] Additional specialized models (embedding, vision)
- [ ] TLS for Redis inter-node communication

---

## Security

- Password authentication on Redis cluster
- No TLS (local network only)
- Ports open on all interfaces (host networking)
- API keys for AMO extension signing stored in .env (gitignored)

---

**Status**: OPERATIONAL on BEAST
**Architecture**: Dual-LLM with intelligent routing + knowledge distillation
**Self-improving**: R1 answers train HiveCoder through continuous learning pipeline
