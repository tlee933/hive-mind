#!/usr/bin/env python3
"""
Hive-Mind Performance Benchmark Suite
Tests Redis ops/sec and llama-server tokens/sec
"""

import asyncio
import json
import sys
import time
from datetime import datetime
import requests

sys.path.insert(0, '/mnt/build/MCP/hive-mind/mcp-server')
from server import HiveMindMCP

GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def print_header(msg):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{msg}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

def print_metric(name, value, unit):
    print(f"  {GREEN}{name:40s} {value:>15,.2f} {unit}{RESET}")

# =============================================================================
# Redis Performance Benchmark
# =============================================================================
async def benchmark_redis():
    print_header("🔥 REDIS CLUSTER PERFORMANCE BENCHMARK")

    server = HiveMindMCP('/mnt/build/MCP/hive-mind/config.yaml')
    await server.connect()

    # Warm up
    for i in range(100):
        await server.redis_client.set(f"warmup:{i}", f"value{i}")

    # Test 1: SET operations
    print(f"{YELLOW}Testing SET operations...{RESET}")
    iterations = 10000
    start = time.time()

    for i in range(iterations):
        await server.redis_client.set(f"bench:set:{i}", f"value_{i}")

    elapsed = time.time() - start
    set_ops_per_sec = iterations / elapsed
    print_metric("SET ops/sec", set_ops_per_sec, "ops/s")

    # Test 2: GET operations
    print(f"\n{YELLOW}Testing GET operations...{RESET}")
    start = time.time()

    for i in range(iterations):
        await server.redis_client.get(f"bench:set:{i}")

    elapsed = time.time() - start
    get_ops_per_sec = iterations / elapsed
    print_metric("GET ops/sec", get_ops_per_sec, "ops/s")

    # Test 3: HSET operations (hash)
    print(f"\n{YELLOW}Testing HSET operations...{RESET}")
    start = time.time()

    for i in range(iterations):
        await server.redis_client.hset(f"bench:hash:{i % 100}", f"field_{i}", f"value_{i}")

    elapsed = time.time() - start
    hset_ops_per_sec = iterations / elapsed
    print_metric("HSET ops/sec", hset_ops_per_sec, "ops/s")

    # Test 4: Mixed workload
    print(f"\n{YELLOW}Testing mixed workload (70% read, 30% write)...{RESET}")
    start = time.time()

    for i in range(iterations):
        if i % 10 < 7:  # 70% reads
            await server.redis_client.get(f"bench:set:{i % 1000}")
        else:  # 30% writes
            await server.redis_client.set(f"bench:mixed:{i}", f"value_{i}")

    elapsed = time.time() - start
    mixed_ops_per_sec = iterations / elapsed
    print_metric("Mixed ops/sec", mixed_ops_per_sec, "ops/s")

    # Test 5: Batch pipeline (Redis pipelining)
    print(f"\n{YELLOW}Testing pipelined operations...{RESET}")
    batch_size = 100
    num_batches = iterations // batch_size

    start = time.time()
    async with server.redis_client.pipeline(transaction=False) as pipe:
        for i in range(iterations):
            pipe.set(f"bench:pipe:{i}", f"value_{i}")
            if i % batch_size == 0:
                await pipe.execute()
        await pipe.execute()

    elapsed = time.time() - start
    pipe_ops_per_sec = iterations / elapsed
    print_metric("Pipelined ops/sec", pipe_ops_per_sec, "ops/s")

    # Cleanup
    print(f"\n{YELLOW}Cleaning up test keys...{RESET}")
    keys_to_delete = []
    async for key in server.redis_client.scan_iter("bench:*"):
        keys_to_delete.append(key)
        if len(keys_to_delete) >= 1000:
            await server.redis_client.delete(*keys_to_delete)
            keys_to_delete = []

    if keys_to_delete:
        await server.redis_client.delete(*keys_to_delete)

    await server.disconnect()

    print(f"\n{GREEN}Summary:{RESET}")
    print_metric("Average ops/sec", (set_ops_per_sec + get_ops_per_sec + mixed_ops_per_sec) / 3, "ops/s")
    print_metric("Peak (pipelined)", pipe_ops_per_sec, "ops/s")

# =============================================================================
# Llama Server Performance Benchmark
# =============================================================================
def benchmark_llama():
    print_header("🦙 LLAMA-SERVER PERFORMANCE BENCHMARK")

    servers = [
        ("Qwen2.5-Coder-7B", "http://localhost:8080"),
        ("Qwen3-8B", "http://localhost:8088")
    ]

    for model_name, url in servers:
        print(f"\n{BLUE}{'─'*70}{RESET}")
        print(f"{GREEN}Model: {model_name}{RESET}")
        print(f"{BLUE}{'─'*70}{RESET}\n")

        # Test various prompt lengths
        test_cases = [
            ("Short prompt (10 tokens)", "Write a Python function", 50),
            ("Medium prompt (50 tokens)", "Write a detailed Python function that implements a binary search tree with insert, delete, and search operations", 100),
            ("Long prompt (200 tokens)", "Write a comprehensive Python class that implements a distributed key-value store with support for replication, consistency guarantees, failure detection, and automatic failover. Include methods for get, set, delete, and list operations. " * 3, 150),
        ]

        results = []

        for test_name, prompt, max_tokens in test_cases:
            print(f"{YELLOW}{test_name}{RESET}")

            try:
                start = time.time()
                resp = requests.post(f"{url}/v1/completions",
                    json={
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                    },
                    timeout=60
                )
                elapsed = time.time() - start

                if resp.status_code == 200:
                    result = resp.json()
                    tokens_generated = result["usage"]["completion_tokens"]
                    tokens_per_sec = tokens_generated / elapsed
                    ttft = elapsed - (tokens_generated / tokens_per_sec) if tokens_generated > 0 else 0

                    print_metric("Tokens generated", tokens_generated, "tokens")
                    print_metric("Total time", elapsed, "seconds")
                    print_metric("Tokens/second", tokens_per_sec, "tok/s")
                    print_metric("Time to first token", ttft, "seconds")
                    print()

                    results.append(tokens_per_sec)
                else:
                    print(f"  Error: {resp.status_code}")
            except Exception as e:
                print(f"  Error: {e}")

        if results:
            avg_tps = sum(results) / len(results)
            print(f"{GREEN}Average tokens/sec: {avg_tps:.1f} tok/s{RESET}\n")

# =============================================================================
# MCP Server Performance Benchmark
# =============================================================================
async def benchmark_mcp():
    print_header("🐝 MCP SERVER PERFORMANCE BENCHMARK")

    server = HiveMindMCP('/mnt/build/MCP/hive-mind/config.yaml')
    await server.connect()

    # Test 1: Memory store operations
    print(f"{YELLOW}Testing memory_store operations...{RESET}")
    iterations = 1000
    start = time.time()

    for i in range(iterations):
        await server.memory_store(
            context=f"Test context {i}",
            files=[f"file{i}.py"],
            task=f"Task {i}"
        )

    elapsed = time.time() - start
    store_ops_per_sec = iterations / elapsed
    print_metric("memory_store ops/sec", store_ops_per_sec, "ops/s")

    # Test 2: Memory recall operations
    print(f"\n{YELLOW}Testing memory_recall operations...{RESET}")
    start = time.time()

    for i in range(iterations):
        await server.memory_recall()

    elapsed = time.time() - start
    recall_ops_per_sec = iterations / elapsed
    print_metric("memory_recall ops/sec", recall_ops_per_sec, "ops/s")

    # Test 3: Tool cache operations
    print(f"\n{YELLOW}Testing tool_cache operations...{RESET}")
    start = time.time()

    for i in range(iterations):
        await server.tool_cache_set("bash", f"hash_{i}", f"output_{i}", ttl=3600)

    elapsed = time.time() - start
    cache_set_ops_per_sec = iterations / elapsed
    print_metric("tool_cache_set ops/sec", cache_set_ops_per_sec, "ops/s")

    start = time.time()
    for i in range(iterations):
        await server.tool_cache_get("bash", f"hash_{i}")

    elapsed = time.time() - start
    cache_get_ops_per_sec = iterations / elapsed
    print_metric("tool_cache_get ops/sec", cache_get_ops_per_sec, "ops/s")

    await server.disconnect()

# =============================================================================
# Main
# =============================================================================
async def main():
    print(f"\n{GREEN}{'='*70}{RESET}")
    print(f"{GREEN}🚀 HIVE-MIND PERFORMANCE BENCHMARK SUITE{RESET}")
    print(f"{GREEN}{'='*70}{RESET}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Run benchmarks
    await benchmark_redis()
    benchmark_llama()
    await benchmark_mcp()

    print(f"\n{GREEN}{'='*70}{RESET}")
    print(f"{GREEN}✅ BENCHMARK COMPLETE{RESET}")
    print(f"{GREEN}{'='*70}{RESET}\n")

if __name__ == "__main__":
    asyncio.run(main())
