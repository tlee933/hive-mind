# Hive-Mind Roadmap

Future enhancements and planned features for the Hive-Mind distributed AI memory system.

## Current State (February 2026)

### RAG Fact Injection
- **Implemented**: Keyword-based filtering for selective fact injection
- **How it works**: Query keywords map to relevant fact keys
  - "gpu" query -> `gpu`, `rocm_version`, `pytorch_location` facts
  - "install" query -> `package_management`, `system_type` facts
  - "os/linux" query -> `operating_system`, `desktop_environment` facts
- **Benefits**: Reduces token overhead by only injecting relevant context

### Local LLM
- HiveCoder-7B (Qwen2.5-Coder-7B + LoRA fine-tuned)
- Continuous learning pipeline with 50-sample threshold
- Q5_K_M GGUF quantization (5.1GB)

---

## Planned Enhancements

### Phase 1: Semantic Search for RAG (High Priority)

**Goal**: Replace keyword-based filtering with embedding-based semantic similarity.

**Why**:
- Keyword matching misses synonyms ("graphics card" vs "gpu")
- Semantic search understands meaning, not just words
- Better retrieval quality = more relevant context = better responses

**Implementation Options**:

#### Option A: Small Embedding Model (Recommended)
```
Query -> Embedding Model -> Vector -> Redis Vector Search -> Top-K Facts
```

**Candidate Models**:
| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| all-MiniLM-L6-v2 | 80MB | ~5ms | Good |
| e5-small-v2 | 130MB | ~8ms | Better |
| bge-small-en-v1.5 | 130MB | ~8ms | Best for RAG |

**Infrastructure**:
- Store fact embeddings in Redis with `RediSearch` vector index
- Pre-compute embeddings when facts are stored
- At query time: embed query, vector search, return top-K facts

**Estimated Overhead**:
- Latency: +5-15ms per query
- Memory: +200MB for model
- Storage: ~1.5KB per fact (384-dim float32)

#### Option B: LLM-Based Relevance Scoring
Use HiveCoder-7B to score fact relevance (slower but more accurate).

```
Query + Facts -> HiveCoder -> Relevance Scores -> Top-K Facts
```

**Trade-offs**:
- Slower: +500ms per query
- More accurate for complex queries
- No additional model needed

### Phase 2: Multi-Node Embedding Service

**Goal**: Offload embedding computation to r720xd storage server.

**Why**:
- Free up GPU VRAM on aurora for LLM inference
- r720xd has plenty of CPU/RAM for embedding models
- Network latency acceptable for async embedding

**Architecture**:
```
aurora (LLM inference) <---> Redis Cluster <---> r720xd (embeddings + storage)
```

### Phase 3: Hybrid RAG

Combine multiple retrieval methods:
1. **Keyword matching** - Fast, exact matches
2. **Semantic search** - Meaning-based similarity
3. **Recency weighting** - Prefer recent facts
4. **Source weighting** - Trust authoritative sources more

**Fusion Strategy**: Reciprocal Rank Fusion (RRF) to combine results.

---

## Research Items

### Token-Efficient Context
- Investigate context distillation techniques
- Explore fact compression/summarization

### Active Learning
- Use query logs to identify missing facts
- Suggest fact additions based on failed retrievals

### Cross-Session Memory
- Link related sessions for context carryover
- Build user preference profiles

---

## Infrastructure Improvements

### Short Term
- [ ] Migrate FastAPI to lifespan handlers (deprecation warning)
- [ ] Add Prometheus metrics for RAG pipeline
- [ ] Health check for embedding service

### Long Term
- [ ] Kubernetes deployment manifests
- [ ] Horizontal scaling for embedding service
- [ ] A/B testing framework for retrieval strategies

---

## Contributing

See [JOURNEY.md](JOURNEY.md) for project history and design decisions.

*Last updated: 2026-02-12*
