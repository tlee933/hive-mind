#!/bin/bash
# Hive-Mind: Local Redis Setup on BEAST
# Runs Redis locally on /mnt/build with 1TB storage

set -euo pipefail

REDIS_VERSION="7-alpine"
REDIS_DATA_DIR="/mnt/build/redis/data"
REDIS_CONF_DIR="/mnt/build/redis/conf"
REDIS_LOG_DIR="/mnt/build/redis/logs"
REDIS_PORT=6379
REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 32)}"
REDIS_MAXMEMORY="${REDIS_MAXMEMORY:-16gb}"  # Adjust based on available RAM

echo "🐝 Hive-Mind: Local Redis Setup on BEAST"
echo "========================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Installing via podman alias or install Docker."
    echo "   For Fedora: sudo dnf install docker"
    exit 1
fi

# Create directories
echo "📁 Creating Redis directories..."
mkdir -p "$REDIS_DATA_DIR"
mkdir -p "$REDIS_CONF_DIR"
mkdir -p "$REDIS_LOG_DIR"

# Check available space
AVAILABLE_GB=$(df -BG /mnt/build | tail -1 | awk '{print $4}' | tr -d 'G')
echo "   Available space on /mnt/build: ${AVAILABLE_GB}GB"

# Generate redis.conf
echo "⚙️  Generating redis.conf..."
cat > "$REDIS_CONF_DIR/redis.conf" <<EOF
# Network
bind 0.0.0.0
port $REDIS_PORT
protected-mode yes
requirepass $REDIS_PASSWORD

# Memory
maxmemory $REDIS_MAXMEMORY
maxmemory-policy allkeys-lru

# Persistence - optimized for local SSD/NVMe
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
appendfilename "appendonly.aof"
dir /data

# Auto-AOF rewrite (important for long-running instances)
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Performance
tcp-backlog 511
timeout 0
tcp-keepalive 300
databases 16

# Logging
loglevel notice
logfile "/logs/redis.log"

# Slow log
slowlog-log-slower-than 10000
slowlog-max-len 128

# Advanced config
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# RDB compression
rdbcompression yes
rdbchecksum yes

# Streams
stream-node-max-bytes 4096
stream-node-max-entries 100
EOF

# Stop existing container if running
echo "🛑 Stopping existing Redis container (if any)..."
docker stop hive-mind-redis 2>/dev/null || true
docker rm hive-mind-redis 2>/dev/null || true

# Start Redis container
echo "🚀 Starting Redis container..."
docker run -d \
  --name hive-mind-redis \
  --restart unless-stopped \
  --network host \
  -v "$REDIS_DATA_DIR:/data" \
  -v "$REDIS_CONF_DIR/redis.conf:/usr/local/etc/redis/redis.conf:ro" \
  -v "$REDIS_LOG_DIR:/logs" \
  redis:$REDIS_VERSION \
  redis-server /usr/local/etc/redis/redis.conf

# Wait for Redis to start
echo "⏳ Waiting for Redis to start..."
sleep 3

# Test connection
echo "🧪 Testing Redis connection..."
if docker exec hive-mind-redis redis-cli -a "$REDIS_PASSWORD" ping 2>/dev/null | grep -q PONG; then
    echo "✅ Redis is running!"
else
    echo "❌ Redis failed to start"
    docker logs hive-mind-redis
    exit 1
fi

# Get basic info
echo ""
echo "📊 Redis Info:"
docker exec hive-mind-redis redis-cli -a "$REDIS_PASSWORD" INFO server 2>/dev/null | grep -E "redis_version|os|uptime"
docker exec hive-mind-redis redis-cli -a "$REDIS_PASSWORD" INFO memory 2>/dev/null | grep -E "used_memory_human|maxmemory_human"

# Display summary
echo ""
echo "🎉 Hive-Mind Redis is ready!"
echo "========================================="
echo "Host: localhost (127.0.0.1)"
echo "Port: $REDIS_PORT"
echo "Password: $REDIS_PASSWORD"
echo "Data dir: $REDIS_DATA_DIR"
echo "Max memory: $REDIS_MAXMEMORY"
echo ""
echo "⚠️  SAVE THIS PASSWORD!"
echo ""
echo "📝 Add to config.yaml:"
cat <<YAML

redis:
  host: "127.0.0.1"  # localhost for BEAST, IP for DELL
  port: $REDIS_PORT
  password: "$REDIS_PASSWORD"
  db: 0

YAML

# Save password to file (for backup)
echo "$REDIS_PASSWORD" > "$REDIS_CONF_DIR/.redis_password"
chmod 600 "$REDIS_CONF_DIR/.redis_password"
echo ""
echo "💾 Password saved to: $REDIS_CONF_DIR/.redis_password"
echo ""
echo "🔥 Container status:"
docker ps | grep hive-mind-redis
echo ""
echo "📊 Monitor logs: docker logs -f hive-mind-redis"
echo "🧪 Test: redis-cli -a '$REDIS_PASSWORD' ping"
echo ""
echo "🌐 For DELL to connect, use BEAST's IP (not localhost)"
echo "   Update DELL config.yaml with: host: \"$(hostname -I | awk '{print $1}')\" "
