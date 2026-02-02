# 🏗️ Hive-Mind Architecture

## Overview

Hive-Mind is a distributed AI memory and learning system that uses Redis as a central nervous system to coordinate between multiple GPU nodes, providing persistent memory, semantic search, and continuous learning capabilities.

## Design Principles

1. **Distributed-first**: No single point of failure, work continues if any node is down
2. **Learn continuously**: Every interaction improves the system
3. **Network-efficient**: Minimize round-trips, cache aggressively
4. **GPU-native**: Leverage available compute for embeddings and training
5. **Persistence**: All important context survives restarts

## System Components

### 1. Redis Memory Store (NAS)

**Role**: Central persistent storage and coordination layer

**Key Data Structures**:
```redis
# Session memory (ephemeral, 7-day TTL)
HASH session:{session_id}
  - timestamp
  - context
  - files
  - current_task

# Long-term embeddings (persistent)
ZSET memory:embeddings          # sorted by timestamp
  - score: unix_timestamp
  - member: "{embedding_hash}:{text_hash}"

HASH embedding:{sha256}
  - vector: binary blob (768-dim)
  - text: original text
  - timestamp: creation time
  - source: where it came from

# Learning queue (STREAM for multi-consumer)
STREAM learning:queue
  - user_query
  - tool_used
  - result
  - success
  - timestamp

# Tool output cache (temporary, 1-hour TTL)
STRING tool:{tool_name}:{input_hash}
  - cached output

# Model registry
HASH models:active
  - dell: "llama-3.2-8B-instruct"
  - beast: "qwen3-30B"

# Coordination
PUBSUB channel:events           # real-time coordination
SET nodes:active                # heartbeat tracking
```

**Performance Configuration**:
```redis
# redis.conf optimizations
maxmemory 8gb                   # Use available NAS RAM
maxmemory-policy allkeys-lru    # Evict old embeddings if needed
save 900 1                      # RDB snapshot every 15 min
save 300 10
save 60 10000
appendonly yes                  # AOF for durability
appendfsync everysec            # Balance performance/safety
```

### 2. MCP Server (Runs on any node)

**Role**: Bridge between Claude Code and Redis memory

**Interface**:
```python
class HiveMindMCP:
    async def get_context(session_id: str) -> Dict
    async def set_context(session_id: str, context: Dict)
    async def search_memory(query: str, limit: int) -> List[Dict]
    async def add_to_learning_queue(interaction: Dict)
    async def get_tool_cache(tool: str, inputs: str) -> Optional[str]
    async def set_tool_cache(tool: str, inputs: str, output: str)
```

**MCP Tools Exposed to Claude**:
- `memory_recall`: Search past interactions semantically
- `memory_store`: Explicitly save important context
- `memory_context`: Get current session state
- `memory_clear`: Reset session (with confirmation)

### 3. Embedding Service (DELL)

**Role**: Generate vector embeddings for semantic search

**Stack**:
- sentence-transformers (all-MiniLM-L6-v2 or similar)
- RDNA2 GPU acceleration via ROCm
- Redis integration for caching

**API**:
```python
POST /embed
{
  "text": "What was I doing with PyTorch benchmarks?"
}

Response:
{
  "embedding": [0.123, -0.456, ...],  # 768-dim
  "hash": "sha256_of_text",
  "cached": false
}
```

### 4. llama-server (DELL)

**Role**: Tool-use model for function calling and reasoning

**Configuration**:
- Model: Llama 3.2 8B Instruct (or similar)
- Context: 8K tokens
- GPU offload: Full (12GB VRAM)
- Integration: Queries Redis for context before responding

**Use Cases**:
- Tool selection and parameter extraction
- Short-form reasoning
- Context summarization
- Query reformulation for embeddings

### 5. Training Pipeline (BEAST)

**Role**: Fine-tune models based on successful interactions

**Process**:
```
1. Pull from learning:queue (batch of 100-1000 interactions)
2. Filter for successful tool uses
3. Generate training examples:
   <user_query> → <tool_call> → <result>
4. Fine-tune LoRA adapter on base model
5. Evaluate on held-out test set
6. If improved, push to NAS and notify DELL
7. DELL reloads model with new adapter
```

**Schedule**:
- Nightly for small updates (< 1000 examples)
- Weekly for major retraining

## Data Flow

### Typical Interaction

```
1. User sends query to Claude Code
   ↓
2. MCP server queries Redis:
   - Get session context
   - Search embeddings for relevant past interactions
   ↓
3. Claude generates response with context
   ↓
4. If tool call needed:
   - Check tool cache in Redis
   - If miss, execute tool
   - Cache result
   ↓
5. MCP server:
   - Updates session context
   - Generates embedding (via DELL service)
   - Stores in Redis
   - Adds to learning queue
   ↓
6. (Async) BEAST processes learning queue
```

### Network Topology

```
            ┌─────────────────────┐
            │   NAS (Redis)       │
            │   192.168.1.7:6379  │
            └──────────┬──────────┘
                       │ 1Gbps
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐   ┌────▼────┐   ┌───▼────┐
    │ BEAST   │   │ DELL    │   │ Other  │
    │ :6380   │   │ :8080   │   │ nodes  │
    └─────────┘   └─────────┘   └────────┘
   Training      Inference    Future expansion
```

## Performance Characteristics

### Latency Budget

| Operation | Target | Typical |
|-----------|--------|---------|
| Redis GET | < 5ms | 1-3ms |
| Redis SET | < 10ms | 2-5ms |
| Embedding generation | < 100ms | 50-80ms |
| Semantic search (top-10) | < 50ms | 20-40ms |
| Tool cache hit | < 5ms | 1-3ms |
| Learning queue add | < 5ms | async |

### Throughput

- **Embeddings**: ~100/sec on DELL RDNA2
- **Redis ops**: ~10K/sec (local network)
- **Fine-tuning**: 1 epoch/30min on BEAST

### Storage

- **Session context**: ~10KB per session
- **Embeddings**: 3KB per embedding (768 floats + metadata)
- **Learning queue**: ~5KB per interaction
- **Estimated**: 1M interactions = ~8GB in Redis

## Failure Modes & Recovery

### NAS Down
- MCP falls back to local SQLite cache
- Sessions continue but without persistence
- Auto-reconnect when NAS returns
- Sync local cache to Redis on reconnect

### DELL Down
- Embeddings disabled temporarily
- Fall back to keyword search
- BEAST can take over embedding generation (slower)

### BEAST Down
- No impact on inference
- Learning pipeline paused
- Queue builds up in Redis
- Resumes when BEAST returns

### Network Partition
- Nodes operate independently
- Redis provides last-known-good context
- Manual merge on partition heal (flag conflicts)

## Security Considerations

1. **Redis AUTH**: Password-protect Redis instance
2. **Network**: Firewall Redis port to local network only
3. **Data privacy**: Embeddings don't leak exact text (use hashes)
4. **Tool cache**: TTL prevents stale credentials
5. **Learning queue**: Filter sensitive data before storing

## Scaling Strategy

### Current (Phase 1)
- 1 NAS, 2 compute nodes
- Single Redis instance
- ~1M embeddings in memory

### Future (Phase 2+)
- Redis Cluster for horizontal scaling
- Multiple DELL nodes for inference load balancing
- Distributed training across multiple BEAST-class nodes
- S3-compatible object storage for model checkpoints

---

**Status**: Phase 1 - Design Complete, Implementation Starting
