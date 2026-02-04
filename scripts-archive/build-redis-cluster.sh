#!/bin/bash
# Build Redis Cluster: 3 masters + 3 replicas

set -e

CLUSTER_DIR="/mnt/build/redis-cluster"
PASSWORD="P1MC0OSpZ9Iuss5b36Bmpl2U9Yf5JwtFJTXf2Eb5WkQ="
BASE_PORT=7000

echo "🐝 Building Redis Cluster (6 nodes)"

# Stop old standalone instance
echo "🛑 Stopping standalone Redis..."
docker stop hive-mind-redis 2>/dev/null || true
docker rm hive-mind-redis 2>/dev/null || true

# Create cluster directories
mkdir -p "$CLUSTER_DIR"
for port in 7000 7001 7002 7003 7004 7005; do
    mkdir -p "$CLUSTER_DIR/node-$port"/{data,conf}
    
    # Generate node config
    cat > "$CLUSTER_DIR/node-$port/conf/redis.conf" <<CONFIG
# Redis Cluster Node Configuration
port $port
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
appendfsync everysec
dir /data

# Persistence
save 900 1
save 300 10
save 60 10000
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Memory
maxmemory 2gb
maxmemory-policy allkeys-lru

# Security
requirepass $PASSWORD
masterauth $PASSWORD

# Network
bind 0.0.0.0
protected-mode no

# Logging
loglevel notice
CONFIG
    
    echo "✅ Created config for node $port"
done

# Launch nodes
echo ""
echo "🚀 Launching cluster nodes..."
for port in 7000 7001 7002 7003 7004 7005; do
    docker run -d \
      --name redis-cluster-$port \
      --restart unless-stopped \
      --network host \
      -v "$CLUSTER_DIR/node-$port/data:/data" \
      -v "$CLUSTER_DIR/node-$port/conf/redis.conf:/usr/local/etc/redis/redis.conf:ro" \
      redis:7-alpine \
      redis-server /usr/local/etc/redis/redis.conf
    
    echo "  ✅ Node $port started"
done

echo ""
echo "⏳ Waiting for nodes to initialize..."
sleep 5

# Create cluster
echo ""
echo "🔗 Creating cluster..."
docker run --rm --network host redis:7-alpine redis-cli \
  --cluster create \
  127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 \
  127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 \
  --cluster-replicas 1 \
  --cluster-yes \
  -a "$PASSWORD"

echo ""
echo "✅ Redis Cluster Created!"
echo ""
echo "📊 Cluster Info:"
docker exec redis-cluster-7000 redis-cli -p 7000 -a "$PASSWORD" CLUSTER INFO 2>&1 | grep -v "Warning:"
echo ""
echo "🎯 Connect to cluster:"
echo "  redis-cli -c -p 7000 -a '$PASSWORD'"
echo ""
echo "💾 Data persisted to: $CLUSTER_DIR"
echo "🔄 6 nodes: 3 masters + 3 replicas"
