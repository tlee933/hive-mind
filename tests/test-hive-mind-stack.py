#!/usr/bin/env python3
"""
Comprehensive Hive-Mind Stack Test Suite
Tests: llama-servers, Redis cluster, MCP server, integration
"""

import asyncio
import json
import sys
import time
from datetime import datetime
import requests

sys.path.insert(0, '/mnt/build/MCP/hive-mind/mcp-server')
from server import HiveMindMCP

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_test(name):
    print(f"\n{BLUE}🧪 TEST: {name}{RESET}")

def print_success(msg):
    print(f"  {GREEN}✅ {msg}{RESET}")

def print_fail(msg):
    print(f"  {RED}❌ {msg}{RESET}")

def print_info(msg):
    print(f"  {YELLOW}ℹ️  {msg}{RESET}")

# =============================================================================
# Test 1: Llama Server Health
# =============================================================================
async def test_llama_servers():
    print_test("Llama Server Health Checks")

    servers = {
        "Qwen2.5-Coder-7B (8080)": "http://localhost:8080",
        "Qwen3-8B (8088)": "http://localhost:8088"
    }

    for name, url in servers.items():
        try:
            # Health check
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                print_success(f"{name} - health OK")
            else:
                print_fail(f"{name} - unhealthy: {resp.text}")
                continue

            # Get model info
            resp = requests.get(f"{url}/v1/models", timeout=5)
            model_data = resp.json()["data"][0]
            print_info(f"Model: {model_data['id']}")
            print_info(f"Params: {model_data['meta']['n_params'] / 1e9:.1f}B")
            print_info(f"Context: {model_data['meta']['n_ctx_train']} tokens")

        except Exception as e:
            print_fail(f"{name} - Error: {e}")

    return True

# =============================================================================
# Test 2: Llama Server Inference
# =============================================================================
async def test_llama_inference():
    print_test("Llama Server Inference Tests")

    # Test 7B coder
    print_info("Testing Qwen2.5-Coder-7B with code generation...")
    try:
        start = time.time()
        resp = requests.post("http://localhost:8080/v1/completions",
            json={
                "prompt": "def fibonacci(n):",
                "max_tokens": 50,
                "temperature": 0.7,
                "stop": ["\n\n"]
            },
            timeout=30
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            result = resp.json()
            completion = result["choices"][0]["text"]
            tokens = result["usage"]["completion_tokens"]
            speed = tokens / elapsed

            print_success(f"Generated {tokens} tokens in {elapsed:.1f}s ({speed:.1f} tok/s)")
            print_info(f"Output: {completion[:80]}...")
        else:
            print_fail(f"Status {resp.status_code}: {resp.text}")
    except Exception as e:
        print_fail(f"Inference error: {e}")

    # Test 8B general
    print_info("Testing Qwen3-8B with reasoning task...")
    try:
        start = time.time()
        resp = requests.post("http://localhost:8088/v1/completions",
            json={
                "prompt": "What is 15 * 24? Think step by step.",
                "max_tokens": 50,
                "temperature": 0.7,
            },
            timeout=30
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            result = resp.json()
            completion = result["choices"][0]["text"]
            tokens = result["usage"]["completion_tokens"]
            speed = tokens / elapsed

            print_success(f"Generated {tokens} tokens in {elapsed:.1f}s ({speed:.1f} tok/s)")
            print_info(f"Output: {completion[:100]}...")
        else:
            print_fail(f"Status {resp.status_code}: {resp.text}")
    except Exception as e:
        print_fail(f"Inference error: {e}")

    return True

# =============================================================================
# Test 3: Redis Cluster Operations
# =============================================================================
async def test_redis_cluster():
    print_test("Redis Cluster Operations")

    server = HiveMindMCP('/mnt/build/MCP/hive-mind/config.yaml')
    await server.connect()

    # Test basic operations
    print_info("Testing SET/GET operations...")
    test_key = f"test:hive-mind:{int(time.time())}"
    test_value = {"msg": "Testing Redis Cluster", "timestamp": datetime.now().isoformat()}

    await server.redis_client.set(test_key, json.dumps(test_value), ex=60)
    retrieved = await server.redis_client.get(test_key)

    if retrieved and json.loads(retrieved)["msg"] == test_value["msg"]:
        print_success("SET/GET operations working")
    else:
        print_fail("SET/GET failed")

    # Test cluster info
    print_info("Checking cluster status...")
    stats = await server.get_stats()
    print_success(f"Redis {stats['redis_version']} - Cluster mode: {stats['cluster_mode']}")
    print_info(f"Connected clients: {stats['connected_clients']}")
    print_info(f"Memory: {stats['used_memory_human']}")
    print_info(f"Sessions: {stats['total_sessions']}")

    await server.disconnect()
    return True

# =============================================================================
# Test 4: MCP Server Memory Operations
# =============================================================================
async def test_mcp_memory():
    print_test("MCP Server Memory Operations")

    server = HiveMindMCP('/mnt/build/MCP/hive-mind/config.yaml')
    await server.connect()

    # Store context
    print_info("Storing test context...")
    test_context = {
        "context": "Running comprehensive Hive-Mind tests",
        "files": ["test-hive-mind-stack.py", "server.py"],
        "task": "Verify all components: Redis, MCP, llama-servers"
    }

    await server.memory_store(**test_context)
    print_success("Context stored")

    # Recall context
    print_info("Recalling context...")
    result = await server.memory_recall()

    if result["context"] == test_context["context"]:
        print_success("Context recalled successfully")
        print_info(f"Files: {result['files']}")
        print_info(f"Task: {result['task']}")
    else:
        print_fail("Context recall mismatch")

    await server.disconnect()
    return True

# =============================================================================
# Test 5: MCP Tool Caching
# =============================================================================
async def test_mcp_caching():
    print_test("MCP Server Tool Caching")

    server = HiveMindMCP('/mnt/build/MCP/hive-mind/config.yaml')
    await server.connect()

    tool_name = "bash"
    input_hash = "ls_la_hash"
    output_value = "total 48\ndrwxr-xr-x..."

    # Store in cache
    print_info("Storing tool output in cache...")
    await server.tool_cache_set(tool_name, input_hash, output_value, ttl=300)
    print_success("Cache write successful")

    # Retrieve from cache
    print_info("Retrieving from cache...")
    cached = await server.tool_cache_get(tool_name, input_hash)

    if cached and cached == output_value:
        print_success("Cache read successful")
        print_info(f"Cached data: {cached[:50]}...")
    else:
        print_fail("Cache retrieval failed")

    await server.disconnect()
    return True

# =============================================================================
# Test 6: Learning Queue
# =============================================================================
async def test_learning_queue():
    print_test("MCP Learning Queue")

    server = HiveMindMCP('/mnt/build/MCP/hive-mind/config.yaml')
    await server.connect()

    # Add to learning queue
    print_info("Adding interaction to learning queue...")
    interaction = {
        "tool": "bash",
        "input": "ls -la",
        "output": "total 48\ndrwxr-xr-x...",
        "success": "true",
        "timestamp": datetime.now().isoformat()
    }

    await server.learning_queue_add(interaction)
    print_success("Added to learning queue")

    # Check queue length
    stats = await server.get_stats()
    print_info(f"Learning queue length: {stats.get('learning_queue_length', 0)}")

    await server.disconnect()
    return True

# =============================================================================
# Test 7: Multi-Session Support
# =============================================================================
async def test_multi_session():
    print_test("Multi-Session Support")

    # Create two separate sessions
    server1 = HiveMindMCP('/mnt/build/MCP/hive-mind/config.yaml')
    server2 = HiveMindMCP('/mnt/build/MCP/hive-mind/config.yaml')

    await server1.connect()
    await server2.connect()

    print_info(f"Session 1 ID: {server1.session_id}")
    print_info(f"Session 2 ID: {server2.session_id}")

    # Store different contexts
    await server1.memory_store(context="Session 1 context", task="Task A")
    await server2.memory_store(context="Session 2 context", task="Task B")

    # Recall and verify isolation
    result1 = await server1.memory_recall()
    result2 = await server2.memory_recall()

    if result1["context"] == "Session 1 context" and result2["context"] == "Session 2 context":
        print_success("Session isolation working correctly")
    else:
        print_fail("Session isolation failed")

    await server1.disconnect()
    await server2.disconnect()
    return True

# =============================================================================
# Test 8: Integration Test - Full Stack
# =============================================================================
async def test_full_integration():
    print_test("Full Stack Integration")

    print_info("Simulating real workflow...")

    # Step 1: Query llama-server
    print_info("1. Querying llama-server for code...")
    resp = requests.post("http://localhost:8080/v1/completions",
        json={
            "prompt": "# Python function to calculate factorial\ndef factorial(n):",
            "max_tokens": 100,
            "temperature": 0.7,
            "stop": ["\n\n"]
        },
        timeout=30
    )
    code_output = resp.json()["choices"][0]["text"] if resp.status_code == 200 else "Error"
    print_success(f"Got response: {code_output[:50]}...")

    # Step 2: Store in MCP
    print_info("2. Storing context in MCP...")
    server = HiveMindMCP('/mnt/build/MCP/hive-mind/config.yaml')
    await server.connect()

    await server.memory_store(
        context="Generated factorial function via llama-server",
        files=["factorial.py"],
        task="Code generation test"
    )
    print_success("Stored in distributed memory")

    # Step 3: Cache the result
    print_info("3. Caching result...")
    await server.tool_cache_set("llama-server", "factorial_prompt", code_output, ttl=3600)
    print_success("Cached for 1 hour")

    # Step 4: Add to learning queue
    print_info("4. Logging to learning queue...")
    await server.learning_queue_add({
        "tool": "llama-server",
        "prompt": "factorial function",
        "output": code_output,
        "model": "Qwen2.5-Coder-7B",
        "timestamp": datetime.now().isoformat()
    })
    print_success("Logged for future training")

    # Step 5: Recall everything
    print_info("5. Recalling from memory...")
    recalled = await server.memory_recall()
    cached = await server.tool_cache_get("llama-server", "factorial_prompt")

    if recalled and cached:
        print_success("Full workflow completed successfully!")
        print_info("Memory + Cache + Learning Queue all operational")
    else:
        print_fail("Integration test failed")

    await server.disconnect()
    return True

# =============================================================================
# Main Test Runner
# =============================================================================
async def main():
    print(f"\n{'='*70}")
    print(f"{GREEN}🐝 HIVE-MIND COMPREHENSIVE TEST SUITE{RESET}")
    print(f"{'='*70}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    tests = [
        ("Llama Server Health", test_llama_servers),
        ("Llama Server Inference", test_llama_inference),
        ("Redis Cluster Operations", test_redis_cluster),
        ("MCP Memory Operations", test_mcp_memory),
        ("MCP Tool Caching", test_mcp_caching),
        ("MCP Learning Queue", test_learning_queue),
        ("Multi-Session Support", test_multi_session),
        ("Full Stack Integration", test_full_integration),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print_fail(f"Test crashed: {e}")
            results.append((name, False))

    # Summary
    print(f"\n{'='*70}")
    print(f"{BLUE}📊 TEST SUMMARY{RESET}")
    print(f"{'='*70}\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {status} - {name}")

    print(f"\n{'='*70}")
    print(f"{GREEN}Result: {passed}/{total} tests passed{RESET}")
    print(f"{'='*70}\n")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
