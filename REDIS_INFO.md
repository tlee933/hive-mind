# 🐝 Hive-Mind Redis Configuration

## Connection Details

**Host**: `192.168.1.100` (BEAST)
**Port**: `6379`
**Password**: `YOUR_REDIS_PASSWORD_HERE`
**Max Memory**: 16GB
**Persistence**: RDB + AOF on `/mnt/build/redis/data` (277GB available)

## Architecture Decision

**Running on BEAST (not NAS)** because:
- NAS has old glibc (2017 kernel) - incompatible with modern Redis builds
- Local storage on BEAST has 277GB available
- Sub-millisecond latency (vs 1-3ms over network)
- DELL can still connect via network (192.168.1.100:6379)
- **NAS used for backups** via NFS mount at `/mnt/nas-moar`

## Quick Commands

### Test Connection (from BEAST)
```bash
docker exec hive-mind-redis redis-cli -a "YOUR_REDIS_PASSWORD_HERE" PING
```

### Test Connection (from DELL)
```bash
# Install redis-cli on DELL first
redis-cli -h 192.168.1.100 -p 6379 -a "YOUR_REDIS_PASSWORD_HERE" PING
```

### View Logs
```bash
docker logs -f hive-mind-redis
# OR
tail -f /mnt/build/redis/logs/redis.log
```

### Check Stats
```bash
docker exec hive-mind-redis redis-cli -a "YOUR_REDIS_PASSWORD_HERE" INFO
```

### Restart Redis
```bash
docker restart hive-mind-redis
```

### Backup to NAS
```bash
# Manual backup
docker exec hive-mind-redis redis-cli -a "YOUR_REDIS_PASSWORD_HERE" BGSAVE
sleep 5
sudo cp /mnt/build/redis/data/dump.rdb /mnt/nas-moar/backups/redis-$(date +%Y%m%d-%H%M%S).rdb
```

### Stop Redis
```bash
docker stop hive-mind-redis
```

### Remove Redis (careful!)
```bash
docker stop hive-mind-redis
docker rm hive-mind-redis
# Data persists in /mnt/build/redis/data
```

## Network Access

- **BEAST (localhost)**: Use `127.0.0.1` or `192.168.1.100`
- **DELL**: Use `192.168.1.100:6379`
- **Other machines**: Use `192.168.1.100:6379`

Firewall: Docker host networking mode exposes port 6379 on all interfaces

## File Locations

```
/mnt/build/redis/
├── data/              # Redis persistence (RDB + AOF)
│   ├── dump.rdb
│   └── appendonlydir/
├── logs/              # Redis logs
│   └── redis.log
└── conf/
    └── redis.conf     # Configuration file
```

## Container Info

```bash
# Container name
hive-mind-redis

# Image
redis:7-alpine (7.4.7)

# Restart policy
unless-stopped (auto-restarts on boot)

# Network
host mode (direct port binding, no NAT)
```

## Security Notes

- ⚠️ Password authentication required
- ✅ Bind to 0.0.0.0 (accessible on local network)
- ✅ Protected mode enabled
- ✅ Running as `redis` user (UID 999) inside container
- ⚠️ No TLS (local network only)

## Backup Strategy

### Automated Backups (to implement)
1. Cron job on BEAST: hourly BGSAVE
2. Copy to NAS via NFS mount
3. Retention: 24 hourly, 7 daily, 4 weekly

### Manual Backup
See "Backup to NAS" command above

---

**Status**: ✅ Running
**Last Updated**: 2026-02-01
**Next**: Set up MCP server to use Redis
