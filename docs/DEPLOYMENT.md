# 🚀 Hive-Mind Deployment Guide

## Phase 1: Redis on NAS

### Prerequisites

- Netgear ReadyNAS with SSH access
- Docker support OR native Redis installation capability
- Dual gigabit NICs configured
- Network access from all nodes to 192.168.1.7

### Option A: Docker Deployment (Recommended)

```bash
# SSH into NAS
ssh admin@192.168.1.7

# Create persistent volume
mkdir -p /nas/redis/{data,conf}

# Create redis.conf
cat > /nas/redis/conf/redis.conf <<'EOF'
# Network
bind 0.0.0.0
port 6379
protected-mode yes
requirepass YOUR_STRONG_PASSWORD_HERE

# Memory
maxmemory 8gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
appendfilename "appendonly.aof"
dir /data

# Performance
tcp-backlog 511
timeout 0
tcp-keepalive 300

# Logging
loglevel notice
logfile ""
EOF

# Run Redis container
docker run -d \
  --name hive-mind-redis \
  --restart unless-stopped \
  -p 6379:6379 \
  -v /nas/redis/data:/data \
  -v /nas/redis/conf/redis.conf:/usr/local/etc/redis/redis.conf \
  redis:7-alpine \
  redis-server /usr/local/etc/redis/redis.conf

# Verify
docker logs hive-mind-redis
```

### Option B: Native Installation

```bash
# Install Redis
apt-get update && apt-get install -y redis-server

# Configure
cat > /etc/redis/redis.conf <<'EOF'
# ... (same config as above)
dir /var/lib/redis
EOF

# Enable and start
systemctl enable redis-server
systemctl start redis-server
systemctl status redis-server
```

### Network Optimization

```bash
# If using NIC bonding (LACP)
# Check current bond status
cat /proc/net/bonding/bond0

# Verify dual NICs are active
ip link show
```

### Firewall Configuration

```bash
# Allow Redis from local network only
iptables -A INPUT -p tcp --dport 6379 -s 192.168.1.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 6379 -j DROP

# Save rules
iptables-save > /etc/iptables/rules.v4
```

### Test Connection from Workstation

```bash
# From BEAST or DELL
redis-cli -h 192.168.1.7 -a YOUR_PASSWORD ping
# Should return: PONG

# Benchmark network latency
redis-cli -h 192.168.1.7 -a YOUR_PASSWORD --latency
# Target: < 3ms avg

# Throughput test
redis-cli -h 192.168.1.7 -a YOUR_PASSWORD --latency-history -i 1
```

---

## Phase 2: MCP Server Setup

### Installation on BEAST

```bash
cd /mnt/build/MCP/hive-mind

# Create virtual environment
python3.14 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install redis aioredis mcp numpy sentence-transformers

# Create config
cat > config.yaml <<'EOF'
redis:
  host: "192.168.1.7"
  port: 6379
  password: "YOUR_PASSWORD"
  db: 0
  socket_timeout: 5
  socket_connect_timeout: 5

mcp:
  server_name: "hive-mind"
  version: "0.1.0"

embedding:
  enabled: false  # Phase 2
  endpoint: "http://dell:8080/embed"

cache:
  tool_ttl: 3600
  session_ttl: 604800  # 7 days
EOF

# Test connection
python -c "
import redis
r = redis.Redis(host='192.168.1.7', port=6379, password='YOUR_PASSWORD')
print(r.ping())
print(f'Latency: {r.execute_command(\"PING\")}')
"
```

### Configure Claude Code MCP

```bash
# Edit Claude Code config
mkdir -p ~/.config/claude-code
cat > ~/.config/claude-code/mcp_config.json <<'EOF'
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
EOF
```

### Run MCP Server

```bash
cd /mnt/build/MCP/hive-mind
source .venv/bin/activate

# Test mode (logs to stdout)
python mcp-server/server.py --debug

# Production mode (systemd service)
sudo cp scripts/hive-mind-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hive-mind-mcp
sudo systemctl start hive-mind-mcp
sudo systemctl status hive-mind-mcp
```

---

## Phase 3: llama-server on DELL

### Prerequisites

- llama.cpp built with ROCm support for RDNA2
- 8B-14B parameter model downloaded
- 8-10GB VRAM available

### Installation

```bash
# SSH into DELL
ssh user@dell

# Clone and build llama.cpp (if not already done)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_HIPBLAS=ON
cmake --build . --config Release

# Download model (example: Llama 3.2 8B Instruct)
cd models
wget https://huggingface.co/.../llama-3.2-8B-instruct-Q4_K_M.gguf
```

### Run llama-server

```bash
# Create startup script
cat > ~/run-llama-server.sh <<'EOF'
#!/bin/bash
cd ~/llama.cpp/build/bin

./llama-server \
  --model ../models/llama-3.2-8B-instruct-Q4_K_M.gguf \
  --ctx-size 8192 \
  --n-gpu-layers 99 \
  --host 0.0.0.0 \
  --port 8080 \
  --threads 8 \
  --batch-size 512 \
  --ubatch-size 128 \
  --flash-attn \
  --cont-batching
EOF

chmod +x ~/run-llama-server.sh

# Run in tmux session
tmux new -s llama-server
~/run-llama-server.sh
# Ctrl-B, D to detach
```

### Embedding Service

```bash
# Create embedding service
cat > ~/embedding-service.py <<'EOF'
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import hashlib
import redis
import numpy as np

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')
r = redis.Redis(host='192.168.1.7', port=6379, password='YOUR_PASSWORD')

@app.route('/embed', methods=['POST'])
def embed():
    text = request.json['text']
    text_hash = hashlib.sha256(text.encode()).hexdigest()

    # Check cache
    cached = r.hget(f'embedding:{text_hash}', 'vector')
    if cached:
        return jsonify({
            'embedding': np.frombuffer(cached, dtype=np.float32).tolist(),
            'hash': text_hash,
            'cached': True
        })

    # Generate embedding
    embedding = model.encode(text)

    # Cache in Redis
    r.hset(f'embedding:{text_hash}', mapping={
        'vector': embedding.tobytes(),
        'text': text[:500],  # truncate for storage
        'timestamp': time.time()
    })

    return jsonify({
        'embedding': embedding.tolist(),
        'hash': text_hash,
        'cached': False
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
EOF

# Run embedding service
pip install flask sentence-transformers redis numpy
tmux new -s embedding-service
python ~/embedding-service.py
# Ctrl-B, D to detach
```

---

## Phase 4: Training Pipeline on BEAST

### Setup

```bash
cd /mnt/build/MCP/hive-mind

# Activate PyTorch environment (ROCm)
source /mnt/build/torch-venv-py312/bin/activate

# Install training dependencies
pip install transformers datasets peft accelerate

# Test GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Training Script

See `scripts/training_pipeline.py` (to be created in next phase)

---

## Verification Checklist

### Phase 1: Redis
- [ ] Redis running on NAS (192.168.1.7:6379)
- [ ] Password authentication working
- [ ] Persistence configured (RDB + AOF)
- [ ] Network latency < 5ms from all nodes
- [ ] Firewall rules applied

### Phase 2: MCP Server
- [ ] Python environment set up
- [ ] Config file created with Redis credentials
- [ ] MCP server starts without errors
- [ ] Claude Code can connect to MCP server
- [ ] Basic operations work (memory_store, memory_recall)

### Phase 3: Inference (DELL)
- [ ] llama-server running and accessible
- [ ] Model loaded successfully (check GPU VRAM)
- [ ] Embedding service running on port 8081
- [ ] Redis integration working (cache hits)

### Phase 4: Training (BEAST)
- [ ] PyTorch ROCm environment activated
- [ ] Can access learning queue in Redis
- [ ] Training script runs without errors
- [ ] Model checkpoints save to NAS

---

## Troubleshooting

### Redis Connection Issues

```bash
# Test from client
redis-cli -h 192.168.1.7 -a PASSWORD ping

# Check Redis logs
docker logs hive-mind-redis
# OR
tail -f /var/log/redis/redis-server.log

# Verify network
telnet 192.168.1.7 6379
```

### MCP Server Not Starting

```bash
# Check logs
journalctl -u hive-mind-mcp -f

# Test manually
cd /mnt/build/MCP/hive-mind
source .venv/bin/activate
python mcp-server/server.py --debug
```

### High Latency

```bash
# Benchmark Redis
redis-cli -h 192.168.1.7 -a PASSWORD --intrinsic-latency 100

# Check network path
traceroute 192.168.1.7
mtr 192.168.1.7

# Monitor NAS CPU/network
ssh admin@192.168.1.7 "htop"
```

---

**Next**: Phase 1 implementation - Let's get Redis running! 🚀
