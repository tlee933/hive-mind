# 🐝 Hive-Mind: Redis Cluster + Sentinel Architecture

## Overview

**Design**: Distributed Redis Cluster with high availability monitoring

```
┌──────────────────────────────────┐
│ BEAST (192.168.1.100)            │
│ ──────────────────────────────   │
│                                  │
│ Redis Cluster (6 nodes)          │
│ ┌─────────────────────────────┐  │
│ │ Masters (data sharding)     │  │
│ │ ├─ 7000: slots 0-5460       │  │
│ │ ├─ 7001: slots 5461-10922   │  │
│ │ └─ 7002: slots 10923-16383  │  │
│ │                             │  │
│ │ Replicas (redundancy)       │  │
│ │ ├─ 7003 → replicates 7000   │  │
│ │ ├─ 7004 → replicates 7001   │  │
│ │ └─ 7005 → replicates 7002   │  │
│ └─────────────────────────────┘  │
│                                  │
│ Storage: /mnt/build/redis-cluster│
│ Persistence: AOF + RDB           │
│ Memory: 2GB per node (12GB total)│
└──────────────────────────────────┘
          ↕
    1Gbps Network
          ↕
┌──────────────────────────────────┐
│ NAS (192.168.1.7)                │
│ ──────────────────────────────   │
│                                  │
│ Redis Sentinel (3 instances)     │
│ ┌─────────────────────────────┐  │
│ │ Monitors (quorum-based)     │  │
│ │ ├─ 26379 → monitors all     │  │
│ │ ├─ 26380 → monitors all     │  │
│ │ └─ 26381 → monitors all     │  │
│ │                             │  │
│ │ Responsibilities:           │  │
│ │ • Health monitoring         │  │
│ │ • Automatic failover        │  │
│ │ • Config updates            │  │
│ │ • Notification alerts       │  │
│ └─────────────────────────────┘  │
│                                  │
│ Storage (NFS)                    │
│ └─ /moar/ai/redis-backups        │
│    ├─ Hourly snapshots           │
│    ├─ Daily snapshots            │
│    └─ 9.2TB available            │
│                                  │
│ Memory: ~150MB for Sentinels     │
│ CPU: Intel Atom C3338 (2 cores)  │
└──────────────────────────────────┘
```

## Why This Architecture?

### Redis Cluster (BEAST)
**Purpose**: Fast distributed data storage with automatic sharding

**Benefits**:
- ✅ **Horizontal scaling**: Data sharded across 3 masters
- ✅ **High availability**: Each master has a replica
- ✅ **Performance**: In-memory operations, ~277GB storage
- ✅ **Persistence**: AOF + RDB for durability
- ✅ **Auto-sharding**: Keys distributed by hash slot

**Trade-offs**:
- ⚠️ Single point of failure: All on BEAST
- ⚠️ Manual failover required without Sentinel

### Redis Sentinel (NAS)
**Purpose**: Monitoring and automatic failover

**Benefits**:
- ✅ **Auto-failover**: Promotes replica if master fails
- ✅ **Quorum-based**: 3 sentinels, need 2 to agree
- ✅ **Lightweight**: ~50MB RAM per sentinel
- ✅ **Separate failure domain**: NAS independent of BEAST
- ✅ **Config management**: Updates clients on topology changes

**Trade-offs**:
- ⚠️ NAS has old OS (requires static binary)
- ⚠️ Limited RAM (1.8GB, but 150MB is fine)

### Storage Layer (NAS)
**Purpose**: Persistent backups and disaster recovery

**Benefits**:
- ✅ **9.2TB storage**: Massive backup capacity
- ✅ **24/7 uptime**: NAS always on
- ✅ **RAID redundancy**: Data protected
- ✅ **Network accessible**: Any node can backup

## Data Flow

### Normal Operations
```
Client (MCP Server)
    ↓
Redis Cluster (BEAST)
    ├─ Hash key → determine slot
    ├─ Route to correct master
    ├─ Replicate to replica
    └─ Return result

Sentinel (NAS) [background]
    ├─ Ping masters every 1s
    ├─ Check replica replication lag
    └─ Monitor cluster health
```

### Failover Scenario
```
Master 7000 crashes
    ↓
Sentinels detect (1s ping timeout)
    ↓
Quorum vote (2/3 sentinels agree)
    ↓
Promote replica 7003 to master
    ↓
Update cluster topology
    ↓
Notify clients of new master
    ↓
Service continues (< 5s downtime)
```

### Backup Flow
```
Hourly cron job (BEAST)
    ↓
BGSAVE on each master
    ↓
Copy RDB files via NFS
    ↓
Store on NAS: /moar/ai/redis-backups/
    ↓
Retain: 24h/7d/4w
```

## Configuration

### Cluster Nodes (BEAST)
```yaml
Ports: 7000-7005
Network: host mode (Docker)
Persistence:
  - AOF: appendfsync everysec
  - RDB: save 900 1 / 300 10 / 60 10000
  - Dir: /mnt/build/redis-cluster/node-{port}/data
Memory:
  - maxmemory: 2gb per node
  - maxmemory-policy: allkeys-lru
Security:
  - requirepass: (shared password)
  - masterauth: (for replication)
Cluster:
  - cluster-enabled: yes
  - cluster-node-timeout: 5000
```

### Sentinel Nodes (NAS)
```yaml
Ports: 26379-26381
Monitor:
  - master-7000 (192.168.1.100:7000)
  - master-7001 (192.168.1.100:7001)
  - master-7002 (192.168.1.100:7002)
Quorum: 2 (out of 3 sentinels)
Down-after: 5000ms
Failover-timeout: 60000ms
Parallel-syncs: 1
Auth: (same password as cluster)
```

## Memory Budget

### BEAST
- Redis Cluster: 6 nodes × 2GB = 12GB max
- System overhead: ~4GB
- Available RAM: Likely 32-64GB (plenty of headroom)

### NAS
- Sentinel 1: ~50MB
- Sentinel 2: ~50MB
- Sentinel 3: ~50MB
- **Total**: ~150MB
- **OS + buffers**: ~400MB
- **Reserved buffer**: 1GB
- **Available**: 1.8GB total → **650MB free** ✅

## Failure Modes

### Master Failure
- **Detection**: Sentinels ping timeout (5s)
- **Action**: Promote replica to master
- **Downtime**: < 10s
- **Data loss**: None (if AOF enabled)

### Replica Failure
- **Detection**: Master notices replication lag
- **Action**: Master continues alone
- **Impact**: Reduced redundancy until replica returns

### BEAST Complete Failure
- **Impact**: All data nodes offline
- **Recovery**: Restore from NAS backups
- **Downtime**: Manual intervention required
- **Data loss**: Since last backup (1 hour max)

### NAS Complete Failure
- **Impact**: No monitoring or failover
- **Cluster**: Continues operating normally
- **Risk**: No auto-failover if master fails
- **Backups**: Temporarily unavailable

### Network Partition
- **Split brain protection**: Quorum prevents dual masters
- **Sentinel**: Requires 2/3 to promote replica
- **Cluster**: Majority partition remains writable

## Scaling Strategy

### Current (Phase 1)
- 3 masters on BEAST
- 3 replicas on BEAST
- 3 sentinels on NAS

### Phase 2 (Add DELL)
- Keep 3 masters on BEAST
- Move replicas to DELL (3 × 12GB VRAM nodes)
- Add 3 sentinels on DELL
- **Total**: 6 sentinels, better distribution

### Phase 3 (More compute nodes)
- Add more masters (resharding required)
- Add replicas on new nodes
- Sentinel quorum increases with more nodes

## Monitoring

### Health Checks
```bash
# Cluster health
redis-cli -c -p 7000 -a "$PASSWORD" CLUSTER INFO

# Node status
redis-cli -c -p 7000 -a "$PASSWORD" CLUSTER NODES

# Sentinel status
redis-cli -p 26379 SENTINEL masters

# Replication lag
redis-cli -p 7000 -a "$PASSWORD" INFO replication
```

### Metrics to Watch
- Cluster state: `cluster_state:ok`
- Slots coverage: all 16384 slots assigned
- Replication lag: < 1s ideal
- Memory usage: < 80% of maxmemory
- Connected clients: varies by load

## Backup Schedule

### Automated (to implement)
```bash
# Hourly backup (retain 24)
0 * * * * /mnt/build/MCP/hive-mind/scripts/backup-redis.sh hourly

# Daily backup (retain 7)
0 2 * * * /mnt/build/MCP/hive-mind/scripts/backup-redis.sh daily

# Weekly backup (retain 4)
0 3 * * 0 /mnt/build/MCP/hive-mind/scripts/backup-redis.sh weekly
```

### Manual Backup
```bash
# Trigger BGSAVE on all masters
for port in 7000 7001 7002; do
  redis-cli -p $port -a "$PASSWORD" BGSAVE
done

# Copy to NAS
cp -r /mnt/build/redis-cluster/node-*/data/dump.rdb \
  /mnt/nas-moar/redis-backups/$(date +%Y%m%d-%H%M%S)/
```

## Security Considerations

- ✅ Password auth on all nodes
- ✅ Protected mode disabled (trusted network)
- ⚠️ No TLS (local network only)
- ⚠️ Firewall: Ensure ports 7000-7005, 26379-26381 restricted to local network

## Performance Expectations

### Latency
- **Cluster node**: < 1ms (in-memory)
- **Network RTT**: < 3ms (BEAST ↔ NAS)
- **Failover**: < 10s

### Throughput
- **Per node**: ~100K ops/sec
- **Cluster total**: ~300K ops/sec (3 masters)
- **Network**: 1Gbps = ~125MB/s max

---

**Status**: Cluster deployed, Sentinel ready to deploy
**Next**: Build static Redis binary for NAS Sentinel
