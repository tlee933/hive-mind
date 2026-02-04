# 🐝 Hive-Mind Quick Start

**Get up and running in 2 minutes!**

---

## 🎯 What You Have

✅ **HTTP API** running on port 8090
✅ **MCP Server** configured for Claude Code
✅ **Redis Cluster** with 9 containers
✅ **Python Client** ready to use

---

## 🚀 Use It Right Now

### Option 1: Open Interpreter / Python Scripts

```bash
cd /mnt/build/MCP/hive-mind
source .venv/bin/activate
python
```

```python
from hivemind_client import HiveMindClient

hive = HiveMindClient()

# Store context
hive.store_memory("Working on AI project", task="Model training")

# Recall context
context = hive.recall_memory()
print(context['context'])  # "Working on AI project"

# Get stats
stats = hive.get_stats()
print(f"Sessions: {stats['total_sessions']}")
```

### Option 2: Claude Code

Just ask naturally:
- "Store context: Working on data analysis"
- "What was I working on last session?"
- "Get Hive-Mind stats"

*(MCP tools will be available when Claude Code connects)*

### Option 3: curl / HTTP API

```bash
# Store memory
curl -X POST http://localhost:8090/memory/store \
  -H 'Content-Type: application/json' \
  -d '{"context": "Quick test", "task": "API demo"}'

# Recall memory
curl -X POST http://localhost:8090/memory/recall \
  -H 'Content-Type: application/json' \
  -d '{}'

# Get stats
curl http://localhost:8090/stats | jq
```

---

## 🔧 Service Commands

```bash
# Check HTTP API status
sudo systemctl status hive-mind-http

# View logs
sudo journalctl -u hive-mind-http -f

# Restart if needed
sudo systemctl restart hive-mind-http

# Check Redis cluster
docker ps | grep redis
```

---

## 📊 Health Check

```bash
curl http://localhost:8090/health
# Should return: {"status": "healthy", "redis": "connected"}
```

---

## 🔗 Quick Links

| What | Where |
|------|-------|
| **Full Setup Guide** | [DUAL_MODE_SETUP.md](DUAL_MODE_SETUP.md) |
| **Open Interpreter Guide** | [docs/OPEN_INTERPRETER_INTEGRATION.md](docs/OPEN_INTERPRETER_INTEGRATION.md) |
| **Project Overview** | [README.md](README.md) |
| **HTTP API** | http://localhost:8090 |
| **API Docs** | http://localhost:8090/docs *(auto-generated)* |

---

## 💡 Example: Cross-Tool Workflow

```bash
# 1. In Python/Open Interpreter
from hivemind_client import HiveMindClient
hive = HiveMindClient()
hive.store_memory("Analyzing Q4 sales", task="Generate report")

# 2. Switch to Claude Code
# Ask: "What was I working on?"
# Response: "You were analyzing Q4 sales"

# 3. Context persists across tools!
```

---

## 🐝 That's It!

You're ready to use Hive-Mind's distributed memory system.

**HTTP API**: For Open Interpreter, scripts, any tool
**MCP Protocol**: For Claude Code integration
**Shared Backend**: Context available everywhere

🚀 **Start using it now!**
