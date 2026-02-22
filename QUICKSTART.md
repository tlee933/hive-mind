# Hive-Mind Quick Start

Get running in 2 minutes.

---

## Prerequisites

- Redis cluster running (6 Docker containers on ports 7000-7005)
- Python 3.12+ with virtualenv
- llama-server binary (llama.cpp)
- AMD GPU with ROCm (tested: R9700 32GB, ROCm 7.12)

---

## Start the Stack

```bash
cd /mnt/build/MCP/hive-mind

# 1. Redis cluster (if not running)
docker ps | grep redis  # should show 6 containers

# 2. HiveCoder-7B (code/shell/tools, 88 tok/s)
./scripts/start-hivecoder.sh

# 3. R1-Distill-14B (reasoning/analysis, 55 tok/s)
./scripts/start-r1-distill.sh

# 4. HTTP API with model routing
source .venv/bin/activate
CONFIG_PATH=../config.yaml python mcp-server/http_server.py
```

---

## Verify

```bash
# Health check
curl localhost:8090/health

# Chat (auto-routes to best model)
curl localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "list files in /tmp"}]}'

# Check which model was used
curl -sI localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "explain btrfs snapshots"}]}' \
  | grep X-Model
# X-Model-Used: R1-Distill-14B

# System stats
curl localhost:8090/stats | python3 -m json.tool
```

---

## Claude Code Integration

MCP server auto-starts with Claude Code. Config in `.mcp.json`:

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

## Talos (Desktop Client)

```bash
cd /var/mnt/build/talos
pip install -e .
talos                    # interactive REPL with model routing
```

---

## Service Management

```bash
# Systemd services
sudo systemctl status hivecoder-llm hive-mind-http
sudo journalctl -u hive-mind-http -f

# Redis cluster
docker ps | grep redis

# Learning pipeline
python learning-pipeline/scripts/continuous_learning.py --status
```

---

## Key URLs

| Service | URL |
|---------|-----|
| HTTP API | http://localhost:8090 |
| API Docs | http://localhost:8090/docs |
| HiveCoder-7B | http://localhost:8089 |
| R1-Distill-14B | http://localhost:8080 |

---

For full documentation see [README.md](README.md) and [MCP_SERVER_READY.md](MCP_SERVER_READY.md).
