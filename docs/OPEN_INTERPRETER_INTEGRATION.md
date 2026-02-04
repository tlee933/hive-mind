# 🤖 Open Interpreter Integration with Hive-Mind

This guide shows how to integrate Open Interpreter with the Hive-Mind distributed memory system.

---

## 🎯 Overview

Hive-Mind provides **dual-mode access**:

1. **HTTP API** (Port 8090) - For Open Interpreter, scripts, external apps
2. **MCP Protocol** (stdio) - For Claude Code integration

Both interfaces share the same Redis backend, enabling **cross-tool context sharing**.

---

## 🚀 Quick Start

### 1. Ensure HTTP API is Running

```bash
# Check service status
sudo systemctl status hive-mind-http

# View logs
sudo journalctl -u hive-mind-http -f

# Restart if needed
sudo systemctl restart hive-mind-http
```

### 2. Test the API

```bash
# Health check
curl http://localhost:8090/health

# Get stats
curl http://localhost:8090/stats
```

---

## 🐍 Python Client for Open Interpreter

Create a helper module for Open Interpreter to use:

```python
# hivemind_client.py
import requests
from typing import Optional, List, Dict, Any

class HiveMindClient:
    """Python client for Hive-Mind HTTP API"""

    def __init__(self, base_url: str = "http://localhost:8090"):
        self.base_url = base_url
        self.session_id = None

    def store_memory(self, context: str, files: Optional[List[str]] = None,
                     task: Optional[str] = None) -> Dict[str, Any]:
        """Store context in distributed memory"""
        payload = {"context": context}
        if files:
            payload["files"] = files
        if task:
            payload["task"] = task

        response = requests.post(f"{self.base_url}/memory/store", json=payload)
        result = response.json()

        if result.get("success"):
            self.session_id = result.get("session_id")

        return result

    def recall_memory(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Recall context from distributed memory"""
        payload = {}
        if session_id:
            payload["session_id"] = session_id

        response = requests.post(f"{self.base_url}/memory/recall", json=payload)
        return response.json()

    def list_sessions(self, limit: int = 10) -> Dict[str, Any]:
        """List recent sessions"""
        response = requests.post(
            f"{self.base_url}/memory/list-sessions",
            json={"limit": limit}
        )
        return response.json()

    def cache_tool_output(self, tool_name: str, input_hash: str,
                         output: str, ttl: Optional[int] = None) -> Dict[str, Any]:
        """Cache tool output"""
        payload = {
            "tool_name": tool_name,
            "input_hash": input_hash,
            "output": output
        }
        if ttl:
            payload["ttl"] = ttl

        response = requests.post(f"{self.base_url}/tool/cache/set", json=payload)
        return response.json()

    def get_cached_output(self, tool_name: str, input_hash: str) -> Optional[str]:
        """Get cached tool output"""
        response = requests.post(
            f"{self.base_url}/tool/cache/get",
            json={"tool_name": tool_name, "input_hash": input_hash}
        )
        result = response.json()
        return result.get("cached_output")

    def add_to_learning_queue(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Add interaction to learning queue"""
        response = requests.post(
            f"{self.base_url}/learning/queue/add",
            json={"interaction": interaction}
        )
        return response.json()

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        response = requests.get(f"{self.base_url}/stats")
        return response.json()
```

---

## 💡 Usage Examples

### Example 1: Store Context from Open Interpreter

```python
from hivemind_client import HiveMindClient

# Initialize client
hive = HiveMindClient()

# Store what you're working on
hive.store_memory(
    context="Analyzing sales data for Q4 report",
    files=["sales_q4.csv", "analysis.py"],
    task="Generate quarterly sales insights"
)

print(f"Session ID: {hive.session_id}")
```

### Example 2: Recall Context in Claude Code

When you switch to Claude Code, it can access the same context:

```python
# In Claude Code (via MCP tools)
# Automatically has access to the same Redis backend
result = memory_recall()
# Returns: "Analyzing sales data for Q4 report"
```

### Example 3: Tool Output Caching

```python
import hashlib

# Generate hash of inputs
input_data = "analyze data.csv with pandas"
input_hash = hashlib.sha256(input_data.encode()).hexdigest()[:16]

# Check cache first
cached = hive.get_cached_output("data_analysis", input_hash)

if cached:
    print("Using cached result!")
    result = cached
else:
    # Run expensive operation
    result = run_expensive_analysis()

    # Cache for next time
    hive.cache_tool_output("data_analysis", input_hash, result, ttl=3600)
```

### Example 4: Cross-Tool Learning

```python
# Log successful interactions for training
hive.add_to_learning_queue({
    "tool_used": "pandas_analysis",
    "user_query": "Show me sales trends",
    "result": analysis_output,
    "success": True,
    "execution_time": 2.5
})
```

---

## 🔌 Open Interpreter Configuration

### Option 1: Add to Open Interpreter's System Message

Edit your Open Interpreter config to include the Hive-Mind client:

```python
# ~/.config/open-interpreter/config.yaml
system_message: |
  You are Open Interpreter with access to Hive-Mind memory system.

  Use the HiveMindClient to:
  - Store context between sessions
  - Share context with Claude Code
  - Cache expensive operations

  Always store important context before long operations.
```

### Option 2: Custom Tool for Open Interpreter

```python
# oi_hivemind_tool.py
def use_hivemind():
    """Tool for Open Interpreter to access Hive-Mind"""
    from hivemind_client import HiveMindClient

    hive = HiveMindClient()

    # Get current stats
    stats = hive.get_stats()
    print(f"Connected to Hive-Mind: {stats['redis_version']}")
    print(f"Total sessions: {stats['total_sessions']}")

    return hive

# In Open Interpreter session
hive = use_hivemind()
```

---

## 🌐 API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/stats` | GET | System statistics |
| `/memory/store` | POST | Store context |
| `/memory/recall` | POST | Recall context |
| `/memory/list-sessions` | POST | List recent sessions |
| `/tool/cache/get` | POST | Get cached output |
| `/tool/cache/set` | POST | Cache tool output |
| `/learning/queue/add` | POST | Add to learning queue |

### Base URL

- **Local**: `http://localhost:8090`
- **Network**: `http://BEAST_IP:8090` (from other machines)

---

## 📊 Example: Full Integration

```python
#!/usr/bin/env python3
"""
Example: Open Interpreter session with Hive-Mind integration
"""

from hivemind_client import HiveMindClient
import hashlib

class OpenInterpreterWithMemory:
    def __init__(self):
        self.hive = HiveMindClient()
        print("🐝 Connected to Hive-Mind")

        # Show stats
        stats = self.hive.get_stats()
        print(f"   Redis: {stats['redis_version']}")
        print(f"   Sessions: {stats['total_sessions']}")
        print(f"   Current Session: {stats['current_session']}")

    def execute_with_context(self, task: str, code: str):
        """Execute code with context tracking"""

        # Store context
        self.hive.store_memory(
            context=f"Executing: {task}",
            task=task
        )

        # Check for cached result
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        cached = self.hive.get_cached_output("code_execution", code_hash)

        if cached:
            print("📦 Using cached result")
            return cached

        # Execute (placeholder for actual execution)
        result = f"Result of: {task}"

        # Cache result
        self.hive.cache_tool_output("code_execution", code_hash, result)

        # Log to learning queue
        self.hive.add_to_learning_queue({
            "tool_used": "code_execution",
            "user_query": task,
            "success": True
        })

        return result

    def recall_last_session(self):
        """Recall what was worked on last"""
        sessions = self.hive.list_sessions(limit=1)

        if sessions.get("sessions"):
            last = sessions["sessions"][0]
            print(f"📚 Last session: {last['context']}")
            print(f"   Time: {last['timestamp']}")

            # Recall full context
            context = self.hive.recall_memory(last['session_id'])
            return context

        return None

# Usage
if __name__ == "__main__":
    oi = OpenInterpreterWithMemory()

    # Check last session
    oi.recall_last_session()

    # Execute with memory
    result = oi.execute_with_context(
        task="Analyze data",
        code="import pandas as pd; df = pd.read_csv('data.csv')"
    )

    print(f"✅ Result: {result}")
```

---

## 🔄 Cross-Tool Workflow

1. **Open Interpreter** stores context via HTTP API
2. **Claude Code** recalls context via MCP protocol
3. Both tools share the same Redis backend
4. Context persists across tool switches
5. Learning queue collects data from both tools

Example workflow:

```bash
# In Open Interpreter
> Store context: "Working on sales analysis"
✅ Stored to session: abc123

# Switch to Claude Code
$ claude-code
> What was I working on?
📚 You were working on: "Sales analysis" (session: abc123)
```

---

## 🛠️ Troubleshooting

### HTTP API Not Responding

```bash
# Check if service is running
sudo systemctl status hive-mind-http

# Check logs
sudo journalctl -u hive-mind-http -n 50

# Restart service
sudo systemctl restart hive-mind-http
```

### Connection Refused

```bash
# Check if port 8090 is listening
sudo netstat -tulpn | grep 8090

# Check firewall
sudo firewall-cmd --list-ports
sudo firewall-cmd --add-port=8090/tcp --permanent
sudo firewall-cmd --reload
```

### Redis Not Connected

```bash
# Verify Redis cluster is running
docker ps | grep redis

# Check Redis connection
redis-cli -p 7000 -a YOUR_PASSWORD ping
```

---

## 📝 Configuration

The HTTP API reads from the same `config.yaml`:

```yaml
redis:
  cluster_mode: true
  nodes:
    - host: "127.0.0.1"
      port: 7000
    # ... more nodes
  password: "YOUR_PASSWORD"

cache:
  tool_ttl: 3600        # 1 hour
  session_ttl: 604800   # 7 days
```

---

## 🚀 Advanced: Remote Access

### Access from DELL or Other Machines

```python
# On remote machine
from hivemind_client import HiveMindClient

# Connect to BEAST's HTTP API
hive = HiveMindClient(base_url="http://192.168.1.100:8090")

# Same API, remote access
stats = hive.get_stats()
print(f"Connected to BEAST's Hive-Mind: {stats}")
```

---

## 📚 Next Steps

1. ✅ HTTP API running on port 8090
2. ✅ Python client ready for Open Interpreter
3. 🔄 MCP protocol for Claude Code
4. 🧠 Learning queue collecting data
5. 🚀 Ready for cross-tool context sharing!

**Status**: Both interfaces operational and sharing Redis backend! 🎉
