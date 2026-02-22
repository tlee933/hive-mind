# Hive-Mind Roadmap

*Last updated: 2026-02-22*

---

## Completed

- [x] Redis Cluster (6 nodes, 3 masters + 3 replicas)
- [x] MCP Server + Claude Code integration (12 tools)
- [x] HTTP API with OpenAI-compatible proxy (:8090)
- [x] HiveCoder-7B — LoRA fine-tuned Qwen2.5-Coder-7B (88 tok/s)
- [x] R1-Distill-14B — DeepSeek-R1-Distill-Qwen-14B (55 tok/s)
- [x] Intelligent model router (pure-function query classifier)
- [x] Knowledge distillation (R1 answers train HiveCoder)
- [x] Semantic RAG (bge-small-en-v1.5 embeddings, 768-dim)
- [x] Active retrieval tracking (hit rate, missed queries, gap analysis)
- [x] Continuous learning pipeline (LoRA -> GGUF -> hot-swap)
- [x] GPU orchestration (stop inference for training, auto-restart)
- [x] Multi-epoch validation with early stopping
- [x] Talos TUI client (agentic execution, tool-use, reasoning)
- [x] Firefox/Zen sidebar extension (v0.7.3)
- [x] Conversation bridge (TUI <-> Firefox via Redis)

---

## Next

### Cascade Routing
If HiveCoder returns < 50 tokens AND confidence < 0.7, re-query R1. Requires response buffering for streaming. Better after base router is proven in production.

### Evaluation Framework
Measure distillation effectiveness over time. Track HiveCoder's accuracy on reasoning queries across LoRA versions. A/B test routing thresholds.

### Multi-Node Deployment
- Deploy R1-Distill or embedding service on r720xd
- Add r720xd as Redis replica nodes
- Offload embedding computation to CPU-heavy server

### Additional Models
- Vision model for screenshot/diagram analysis
- Dedicated embedding model on separate GPU or CPU node

### Infrastructure
- TLS for Redis inter-node communication
- Prometheus metrics for RAG pipeline + routing
- Horizontal scaling for embedding service

---

## Research

- Context distillation — compress/summarize facts to reduce token overhead
- Hybrid RAG — combine semantic + keyword + recency + source weighting via RRF
- Router fine-tuning — train a small classifier on actual routing outcomes
- Cross-session memory linking — build user preference profiles from session history
