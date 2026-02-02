#!/bin/bash
# Hive-Mind: Redis Setup Script for NAS
# Run this on the Netgear ReadyNAS (192.168.1.7)

set -euo pipefail

REDIS_VERSION="7-alpine"
REDIS_DATA_DIR="/nas/redis/data"
REDIS_CONF_DIR="/nas/redis/conf"
REDIS_PORT=6379
REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 32)}"

echo "🐝 Hive-Mind: Redis Setup for NAS"
echo "=================================="
echo ""

# Create directories
echo "📁 Creating Redis directories..."
mkdir -p "$REDIS_DATA_DIR"
mkdir -p "$REDIS_CONF_DIR"

# Generate redis.conf
echo "⚙️  Generating redis.conf..."
cat > "$REDIS_CONF_DIR/redis.conf" <<EOF
# Network
bind 0.0.0.0
port $REDIS_PORT
protected-mode yes
requirepass $REDIS_PASSWORD

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
databases 16

# Logging
loglevel notice
logfile ""

# Slow log
slowlog-log-slower-than 10000
slowlog-max-len 128

# Advanced
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
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
  -p $REDIS_PORT:6379 \
  -v "$REDIS_DATA_DIR:/data" \
  -v "$REDIS_CONF_DIR/redis.conf:/usr/local/etc/redis/redis.conf" \
  redis:$REDIS_VERSION \
  redis-server /usr/local/etc/redis/redis.conf

# Wait for Redis to start
echo "⏳ Waiting for Redis to start..."
sleep 3

# Test connection
echo "🧪 Testing Redis connection..."
if docker exec hive-mind-redis redis-cli -a "$REDIS_PASSWORD" ping | grep -q PONG; then
    echo "✅ Redis is running!"
else
    echo "❌ Redis failed to start"
    docker logs hive-mind-redis
    exit 1
fi

# Display info
echo ""
echo "🎉 Hive-Mind Redis is ready!"
echo "=================================="
echo "Host: $(hostname -I | awk '{print $1}')"
echo "Port: $REDIS_PORT"
echo "Password: $REDIS_PASSWORD"
echo ""
echo "⚠️  SAVE THIS PASSWORD! Add it to your config.yaml"
echo ""
echo "📊 Container status:"
docker ps | grep hive-mind-redis
echo ""
echo "📝 Next steps:"
echo "  1. Add password to /mnt/build/MCP/hive-mind/config.yaml on workstation"
echo "  2. Test connection: redis-cli -h <NAS_IP> -a <PASSWORD> ping"
echo "  3. Run benchmark: redis-cli -h <NAS_IP> -a <PASSWORD> --latency"
echo ""

# Save password to file (for backup)
echo "$REDIS_PASSWORD" > "$REDIS_CONF_DIR/.redis_password"
chmod 600 "$REDIS_CONF_DIR/.redis_password"
echo "💾 Password saved to: $REDIS_CONF_DIR/.redis_password"
