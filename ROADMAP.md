# Hive-Mind Roadmap

*Last updated: 2026-03-01*

---

## Completed

- [x] Redis Cluster (6 nodes, 3 masters + 3 replicas across aurora + alderlake)
- [x] MCP Server + Claude Code integration (13 tools)
- [x] HTTP API with OpenAI-compatible proxy (:8090)
- [x] HiveCoder — Qwen3-14B with LoRA fine-tuning
- [x] Semantic RAG (bge-small-en-v1.5 embeddings, 384-dim, 87.5% hit rate)
- [x] Active retrieval tracking (hit rate, missed queries, gap analysis)
- [x] Continuous learning pipeline (QLoRA -> GGUF -> hot-swap)
- [x] QLoRA 4-bit training (bitsandbytes from source for ROCm/gfx1201)
- [x] GPU orchestration (stop inference for training, auto-restart)
- [x] Multi-epoch validation with early stopping
- [x] Multi-node deployment (aurora + alderlake + NAS + OPNsense)
- [x] Embedding service on alderlake (CPU-only, port 8081)
- [x] Redis replicas on alderlake (7003-7005)
- [x] Talos AI Suricata — autonomous IDS threat analysis + auto-blocking
- [x] Talos deployed on alderlake as podman container
- [x] Conversation bridge (TUI <-> Firefox via Redis)
- [x] Firefox/Zen sidebar extension
- [x] Model backup to NAS (daily 3 AM timer, 3-copy rotation)
- [x] Architecture diagram (4-node, auto-generated with Pillow)

---

## Next

### Cascade Routing
If HiveCoder returns < 50 tokens AND confidence < 0.7, re-query with a larger model or different prompt strategy. Requires response buffering for streaming.

### Evaluation Framework
Measure training effectiveness over time. Track HiveCoder's accuracy across LoRA versions. A/B test QLoRA vs BF16 LoRA quality.

### Talos Enhancements
- Repeat offender detection (persistent IP counter across correlation windows)
- GeoIP/ASN enrichment for alert analysis context
- Webhook notifications for critical blocks (Discord, ntfy)
- Historical trend analysis (7-day/30-day baselines)

### Additional Models
- Vision model for screenshot/diagram analysis
- Dedicated IDS-tuned LoRA adapter for Talos

### Infrastructure
- TLS for Redis inter-node communication
- Prometheus metrics for RAG pipeline + training
- Automated model quality regression testing

---

## Research

- Context distillation — compress/summarize facts to reduce token overhead
- Hybrid RAG — combine semantic + keyword + recency + source weighting via RRF
- Router fine-tuning — train a small classifier on actual routing outcomes
- Cross-session memory linking — build user preference profiles from session history
- Multi-sensor correlation — combine Suricata alerts with DNS, auth, and firewall logs
- Honeypot integration — feed low-interaction honeypot data into Talos analysis pipeline
