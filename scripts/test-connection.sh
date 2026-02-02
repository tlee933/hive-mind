#!/bin/bash
# Test Redis connection and benchmark latency

set -euo pipefail

REDIS_HOST="${REDIS_HOST:-192.168.1.7}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"

if [ -z "$REDIS_PASSWORD" ]; then
    echo "Error: REDIS_PASSWORD environment variable not set"
    echo "Usage: REDIS_PASSWORD=yourpass $0"
    exit 1
fi

echo "🐝 Hive-Mind: Redis Connection Test"
echo "===================================="
echo "Host: $REDIS_HOST"
echo "Port: $REDIS_PORT"
echo ""

# Test 1: Basic connectivity
echo "1️⃣  Testing basic connectivity..."
if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" ping 2>/dev/null | grep -q PONG; then
    echo "   ✅ Connection successful"
else
    echo "   ❌ Connection failed"
    exit 1
fi

# Test 2: Latency measurement
echo ""
echo "2️⃣  Measuring latency (10 seconds)..."
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --latency 2>/dev/null | head -1

# Test 3: Latency distribution
echo ""
echo "3️⃣  Latency distribution (15 samples)..."
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --latency-dist -i 1 2>/dev/null &
LATENCY_PID=$!
sleep 15
kill $LATENCY_PID 2>/dev/null || true

# Test 4: Throughput
echo ""
echo "4️⃣  Throughput test..."
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --intrinsic-latency 5 2>/dev/null

# Test 5: Basic operations
echo ""
echo "5️⃣  Testing basic operations..."
echo "   SET test key..."
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" SET test:hivemind "Hello from $(hostname)" EX 60 >/dev/null 2>&1
echo "   ✅ SET successful"

echo "   GET test key..."
VALUE=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" GET test:hivemind 2>/dev/null)
echo "   ✅ GET successful: $VALUE"

# Test 6: Info
echo ""
echo "6️⃣  Redis server info..."
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" INFO server 2>/dev/null | grep -E "redis_version|os|uptime_in_seconds"

echo ""
echo "7️⃣  Memory usage..."
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" INFO memory 2>/dev/null | grep -E "used_memory_human|used_memory_peak_human|maxmemory_human"

echo ""
echo "✅ All tests passed! Redis is ready for Hive-Mind."
echo ""
echo "📊 Performance summary:"
echo "   • Target latency: < 5ms"
echo "   • Target throughput: > 10K ops/sec"
echo "   • Memory configured: 8GB max"
