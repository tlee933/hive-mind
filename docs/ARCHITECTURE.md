# Hive-Mind Architecture

## Overview

Hive-Mind is a self-improving AI backend that provides persistent memory, semantic RAG, dual-LLM inference with intelligent routing, and a continuous learning pipeline that trains the code model from the reasoning model's output. Everything runs locally on a single machine.

## Design Principles

1. **Self-improving** — every interaction feeds the learning pipeline; R1 teaches HiveCoder
2. **Local-first** — zero cloud dependencies, all inference and training on-device
3. **Route intelligently** — send each query to the model best suited for it
4. **Learn continuously** — LoRA fine-tune, GGUF export, hot-swap without downtime
5. **Persist everything** — context survives restarts via Redis cluster

---

## System Components

### 1. Model Router (`mcp-server/router.py`)

Pure-function query classifier. Zero I/O, sub-millisecond.

**Priority order:**
1. Explicit `model_hint` from client -> use that model
2. `reason_mode=True` (`/reason` command) -> always R1
3. Heuristic keyword/pattern scoring:
   - `CODE_SIGNALS` (~30 keywords): python, bash, write, fix, debug, code blocks, imports, file paths
   - `REASONING_SIGNALS` (~25 keywords): explain, why, compare, analyze, step by step
   - Higher score wins; ties -> fast model (HiveCoder)
4. Default -> HiveCoder (faster)

Returns `RoutingDecision(model_id, reason, confidence)`.

### 2. HTTP API (`mcp-server/http_server.py`, port 8090)

FastAPI server, OpenAI-compatible. Handles:
- `/v1/chat/completions` — model routing + RAG injection + streaming
- `/v1/models` — dynamically generated from config
- `/fact/*`, `/memory/*`, `/conversation/*`, `/web/*` endpoints
- Routing response headers: `X-Model-Used`, `X-Model-Id`, `X-Routing-Reason`

### 3. LLM Inference (dual model)

| Model | Port | Role | Speed | Quantization |
|-------|------|------|-------|-------------|
| HiveCoder-7B | 8089 | Code, shell, tools | 88 tok/s | Q5_K_M (5.1 GB) |
| R1-Distill-14B | 8080 | Reasoning, analysis | 55 tok/s | Q4_K_M (8.4 GB) |

Both served via `llama-server` (llama.cpp) with ROCm 7.12 on a single AMD R9700 XT (32 GB VRAM). Combined ~21 GB VRAM with 11 GB headroom.

### 4. Redis Cluster (ports 7000-7005)

6 Docker containers: 3 masters + 3 replicas. 12 GB total memory, AOF + RDB persistence.

**Key data structures:**

| Key Pattern | Type | Purpose |
|-------------|------|---------|
| `session:{id}` | Hash | Session context (28-day TTL) |
| `fact:{key}` | String | RAG facts |
| `fact_embedding:{key}` | String | Pre-computed fact embeddings (base64 float32) |
| `learning:queue` | Stream | Interaction log for training pipeline |
| `rag:retrieval_log` | Stream | RAG quality audit trail |
| `rag:stats` | Hash | Aggregate retrieval counters |
| `rag:missed_queries` | Sorted Set | Failed queries ranked by frequency |
| `tool:{name}:{hash}` | String | Tool output cache (1h TTL) |
| `conversation:{id}` | String | Conversation history |

### 5. Semantic RAG (`mcp-server/server.py`)

- **Model**: bge-small-en-v1.5 (768-dim, ~130 MB, CPU)
- **Lazy loading**: loads on first use, not at startup
- **Pre-computed embeddings**: stored in Redis on `fact_store`, base64 float32
- **Retrieval**: cosine similarity, top-k=5, threshold >= 0.45
- **Quality tracking**: every retrieval logged with method, scores, quality classification
- **Fallback**: keyword filter if embedding model unavailable
- **Hit rate**: 84% across 31 stored facts

### 6. MCP Server (`mcp-server/server.py`, stdio)

Claude Code integration. 12 tools across 4 categories:
- Memory (store, recall, list_sessions)
- Facts/RAG (store, get, delete, suggestions)
- LLM (generate, code_assist, complete)
- System (tool_cache, learning_queue, stats)

### 7. Continuous Learning Pipeline (`learning-pipeline/scripts/`)

Daemon runs every 5 minutes:

```
Redis learning:queue
  -> Drain interactions
    -> Quality filter (r1-distill -> always pass, successful -> pass, failed -> skip)
      -> Format as JSONL training examples
        -> LoRA fine-tune HiveCoder (r=16, alpha=32, 1 epoch)
          -> GGUF export (Q5_K_M)
            -> Hot-swap into running llama-server
              -> Version bump (MODEL_VERSION)
```

GPU orchestration: stops llama-server before training, restarts after deploy. Desktop notifications via `notify-send`.

---

## Knowledge Distillation

The key architectural insight: **R1's answers train HiveCoder.**

```
R1-Distill-14B
  -> Produces high-quality reasoning responses
    -> Auto-rated positive by Talos TUI
      -> Tagged model_source="r1-distill"
        -> Quality filter: r1-distill always passes
          -> LoRA training data for HiveCoder
            -> HiveCoder gets smarter over time
              -> Handles more queries itself
                -> Only hardest reasoning goes to R1
```

Pure-reasoning answers (no shell commands) are captured via `build_reasoning_interaction()` — previously these were silently dropped.

---

## Data Flow

### Query Lifecycle

```
1. User query arrives (Talos TUI, Firefox sidebar, or HTTP client)
2. HTTP API extracts query text
3. Model Router classifies intent -> RoutingDecision
4. RAG: embed query, cosine search facts, inject into system prompt
5. Forward to HiveCoder-7B (:8089) or R1-Distill-14B (:8080)
6. Stream response back with routing headers
7. Client displays response, executes any tool calls
8. Auto-rate interaction (exit codes for commands, always-positive for R1)
9. Log to learning:queue with model_source metadata
10. Learning daemon picks up, filters, trains, deploys
```

### Clients

| Client | Protocol | Features |
|--------|----------|----------|
| Talos TUI | HTTP API | Agentic execution, tool-use, reasoning, auto-rating, distillation |
| Firefox/Zen Extension | HTTP API | Streaming, markdown, suggestions, conversation persistence |
| Claude Code | MCP (stdio) | Memory, facts, LLM generation, learning queue |
| curl / scripts | HTTP API | Any OpenAI-compatible client |

---

## Configuration

Multi-model config in `config.yaml`:

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
    r1-distill-14b:
      endpoint: "http://127.0.0.1:8080"
      display_name: "R1-Distill-14B"
      capabilities: ["reasoning", "analysis", "explanation"]
      max_tokens: 2048
```

Backward compatible: if no `models` dict, falls back to flat `inference.endpoint`.

---

## Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 9 7900X (16 threads) |
| GPU | AMD Radeon RX 9070 XT (32 GB VRAM) |
| ROCm | 7.12 (TheRock build) |
| PyTorch | 2.10.0 (custom ROCm 7.12 build) |
| OS | Fedora 43 Kinoite (rpm-ostree, Wayland) |
| Desktop | KDE Plasma 6 |

VRAM allocation: HiveCoder ~7 GB + R1-Distill ~13 GB = ~21 GB / 32 GB.

---

## Future: Multi-Node

```
aurora (current)                    r720xd (planned)
├── Redis Cluster (7000-7005)       ├── Redis replicas
├── HiveCoder-7B (:8089)            ├── Embedding service
├── R1-Distill-14B (:8080)          └── Storage (24x 2.5" bays)
├── HTTP API (:8090)
├── Learning pipeline
└── Training (LoRA + GGUF)

Connected via Tailscale VPN mesh.
```
