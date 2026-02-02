# Redis Data Schema

## Key Naming Conventions

```
{namespace}:{entity}:{id}[:{attribute}]

Examples:
  session:abc123:context
  embedding:sha256_hash
  tool:bash:input_hash_xyz
  model:active
```

## Data Structures

### 1. Session Memory

**Purpose**: Track ongoing conversations and context

```redis
# Current session context
HASH session:{session_id}
  timestamp        "2026-02-01T11:00:00Z"
  context          "Working on PyTorch benchmarks with Qwen3-30B"
  files            "TheRock/benchmark.py,TheRock/phoronix_results.txt"
  current_task     "Optimizing inference latency"
  last_tool        "bash"
  node             "beast"

TTL: 7 days (EXPIRE session:{session_id} 604800)

# Session index (for listing recent sessions)
ZSET sessions:recent
  score: unix_timestamp
  member: session_id

# Per-user sessions
ZSET user:{username}:sessions
  score: unix_timestamp
  member: session_id
```

### 2. Embeddings & Vector Memory

**Purpose**: Semantic search over past interactions

```redis
# Embedding metadata
HASH embedding:{sha256}
  vector           "\x00\x01\x02..."  # binary blob, 768 floats (3072 bytes)
  text             "Original text that was embedded"
  timestamp        "2026-02-01T11:00:00Z"
  source           "session:abc123"
  tool_context     "bash"  # if from tool interaction
  success          "true"  # if from successful interaction

# Embedding index (sorted by time)
ZSET embeddings:recent
  score: unix_timestamp
  member: embedding_sha256

# Embedding by source
ZSET embeddings:session:{session_id}
  score: unix_timestamp
  member: embedding_sha256
```

**Vector Storage Format**:
- 768-dimensional float32 vectors
- Stored as binary: `struct.pack('768f', *vector)`
- Retrieve: `struct.unpack('768f', binary_data)`
- Total size: 3,072 bytes per vector

### 3. Learning Queue

**Purpose**: Collect interactions for training pipeline

```redis
# Main learning queue (STREAM)
XADD learning:queue * \
  timestamp "2026-02-01T11:00:00Z" \
  session_id "abc123" \
  user_query "Benchmark Qwen3-30B on the R9700" \
  tool_used "bash" \
  tool_args "llama-cli -m qwen3-30B.gguf" \
  result "89.3 tok/s generation" \
  success "true" \
  duration_ms "1250" \
  node "beast"

# Consumer groups for distributed processing
XGROUP CREATE learning:queue training_pipeline $ MKSTREAM
XGROUP CREATE learning:queue analytics $ MKSTREAM

# Read with consumer group
XREADGROUP GROUP training_pipeline worker1 COUNT 10 STREAMS learning:queue >
```

**Stream Retention**:
```redis
# Keep last 100K entries or 7 days
XTRIM learning:queue MAXLEN ~ 100000
```

### 4. Tool Output Cache

**Purpose**: Avoid re-running expensive operations

```redis
# Cache key format
SET tool:{tool_name}:{input_hash} "{output}"

# Examples
SET tool:bash:sha256abc "total 48\ndrwxr-xr-x..." EX 3600
SET tool:read:/path/to/file.py:sha256xyz "import torch..." EX 7200

# Tool stats
HINCRBY tool:stats bash:hits 1
HINCRBY tool:stats bash:misses 1
```

**Cache Policy**:
- Default TTL: 1 hour
- Read-only ops (cat, ls, grep): 2 hours
- Write ops (git commit, file edit): 5 minutes
- Expensive ops (model inference): 24 hours

### 5. Model Registry

**Purpose**: Track which models are running where

```redis
# Active models per node
HASH models:active
  beast "qwen3-30B-A3B"
  dell "llama-3.2-8B-instruct"

# Model metadata
HASH model:qwen3-30B-A3B
  node "beast"
  size_gb "17.28"
  quantization "Q4_K_M"
  context_length "32768"
  last_updated "2026-02-01T10:00:00Z"
  performance "89.3 tok/s"

# Model usage stats
HINCRBY model:qwen3-30B-A3B:stats requests 1
HINCRBY model:qwen3-30B-A3B:stats tokens_generated 1234
```

### 6. Node Coordination

**Purpose**: Track cluster health and route requests

```redis
# Node heartbeat (SET with TTL)
SETEX node:beast:heartbeat 30 '{"cpu": 45, "gpu": 78, "mem": 62}'
SETEX node:dell:heartbeat 30 '{"cpu": 23, "gpu": 34, "mem": 41}'

# Node capabilities
HASH node:beast
  gpu "AMD R9700 32GB"
  vram_gb "32"
  compute_tflops "125"
  role "training,inference"
  status "online"

HASH node:dell
  gpu "RDNA2 12GB"
  vram_gb "12"
  ram_gb "32"
  role "inference,embeddings"
  status "online"

# Active nodes set
SADD nodes:active "beast" "dell"
```

### 7. Event Bus

**Purpose**: Real-time coordination between nodes

```redis
# Pub/Sub channels
PUBLISH events:model_updated '{"node": "dell", "model": "llama-3.2-8B-v2"}'
PUBLISH events:training_complete '{"model": "qwen3-LoRA-v5", "accuracy": 0.92}'
PUBLISH events:node_status '{"node": "beast", "status": "offline"}'

# Event log (for missed messages)
XADD events:log * \
  type "model_updated" \
  node "dell" \
  model "llama-3.2-8B-v2" \
  timestamp "2026-02-01T12:00:00Z"
```

## Access Patterns

### High-Frequency Operations (< 5ms target)

```python
# Get session context
context = redis.hgetall(f"session:{session_id}")

# Check tool cache
cached = redis.get(f"tool:bash:{hash}")

# Add to learning queue (async)
redis.xadd("learning:queue", {...}, maxlen=100000, approximate=True)

# Node heartbeat
redis.setex(f"node:{node_name}:heartbeat", 30, json.dumps(stats))
```

### Medium-Frequency Operations (< 50ms target)

```python
# Semantic search (retrieve top-K embeddings)
recent_embeddings = redis.zrevrange("embeddings:recent", 0, 99)
# Then compute similarity in Python/PyTorch

# Get recent sessions
sessions = redis.zrevrange("sessions:recent", 0, 9, withscores=True)
```

### Low-Frequency Operations (< 1s acceptable)

```python
# Store embedding
redis.hset(f"embedding:{hash}", mapping={
    "vector": vector_bytes,
    "text": text,
    "timestamp": now,
    "source": session_id
})
redis.zadd("embeddings:recent", {hash: timestamp})

# Process learning queue batch
entries = redis.xreadgroup("training_pipeline", "worker1",
                          {"learning:queue": ">"}, count=100)
```

## Memory Estimation

### Per Entity

- Session: ~1 KB (context + metadata)
- Embedding: ~4 KB (3KB vector + 1KB metadata)
- Learning queue entry: ~2 KB
- Tool cache: Variable (typically < 10 KB)
- Node status: ~500 bytes

### Total at Scale

| Scenario | Sessions | Embeddings | Queue | Total |
|----------|----------|------------|-------|-------|
| 1 week | 100 | 10K | 50K | ~150 MB |
| 1 month | 500 | 50K | 200K | ~600 MB |
| 1 year | 5K | 500K | 2M | ~6 GB |

**NAS allocation**: 8 GB maxmemory allows ~1M embeddings + queue

## Backup & Persistence

### RDB Snapshots
```redis
# redis.conf
save 900 1      # After 900s if 1 key changed
save 300 10     # After 300s if 10 keys changed
save 60 10000   # After 60s if 10000 keys changed

dbfilename dump.rdb
dir /var/redis/
```

### AOF (Append-Only File)
```redis
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec  # fsync every second
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

### Backup Script
```bash
#!/bin/bash
# Daily backup to NAS storage
redis-cli BGSAVE
sleep 5
cp /var/redis/dump.rdb /nas/backups/redis-$(date +%Y%m%d).rdb
```

## Optimization Tips

1. **Pipeline operations**: Batch GET/SET for lower latency
2. **Use MGET/MSET**: Retrieve multiple keys in one round-trip
3. **Hash field access**: HGET is faster than JSON parsing
4. **SCAN vs KEYS**: Always use SCAN for iteration (non-blocking)
5. **Lazy deletion**: Use UNLINK instead of DEL for large keys

---

**Reference Implementation**: See `/mcp-server/redis_client.py`
