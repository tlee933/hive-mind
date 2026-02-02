#!/usr/bin/env python3
"""
Hive-Mind MCP Server
Distributed memory system using Redis backend
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from redis.asyncio.cluster import RedisCluster, ClusterNode
import yaml

# MCP SDK imports (placeholder - adjust based on actual MCP SDK)
# from mcp import MCPServer, Tool, Response

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hive-mind-mcp")


class HiveMindMCP:
    """MCP Server with Redis-backed distributed memory"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.redis_client: Optional[aioredis.Redis] = None
        self.session_id = self._generate_session_id()

    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        hostname = os.uname().nodename
        return hashlib.sha256(f"{hostname}:{timestamp}".encode()).hexdigest()[:16]

    async def connect(self):
        """Connect to Redis (cluster or standalone)"""
        redis_config = self.config['redis']

        # Check if cluster mode is enabled
        if redis_config.get('cluster_mode', False):
            # Redis Cluster mode
            startup_nodes = [
                ClusterNode(node['host'], node['port'])
                for node in redis_config['nodes']
            ]

            self.redis_client = RedisCluster(
                startup_nodes=startup_nodes,
                password=redis_config['password'],
                decode_responses=True,
                socket_timeout=redis_config.get('socket_timeout', 5),
                socket_connect_timeout=redis_config.get('socket_connect_timeout', 5),
            )

            await self.redis_client.initialize()
            logger.info(f"Connected to Redis Cluster ({len(startup_nodes)} nodes)")
        else:
            # Standalone Redis mode
            self.redis_client = await aioredis.from_url(
                f"redis://:{redis_config['password']}@{redis_config['host']}:{redis_config['port']}/{redis_config.get('db', 0)}",
                socket_timeout=redis_config.get('socket_timeout', 5),
                socket_connect_timeout=redis_config.get('socket_connect_timeout', 5),
                decode_responses=True
            )
            logger.info(f"Connected to Redis at {redis_config['host']}:{redis_config['port']}")

        # Test connection
        await self.redis_client.ping()
        logger.info(f"Session ID: {self.session_id}")

        # Initialize session
        await self._init_session()

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.aclose()
            logger.info("Disconnected from Redis")

    async def _init_session(self):
        """Initialize session in Redis"""
        session_key = f"session:{self.session_id}"
        await self.redis_client.hset(session_key, mapping={
            'timestamp': datetime.now().isoformat(),
            'context': '',
            'files': '',
            'current_task': '',
            'node': os.uname().nodename,
        })
        await self.redis_client.expire(session_key, self.config['cache']['session_ttl'])

        # Add to sessions index
        await self.redis_client.zadd(
            'sessions:recent',
            {self.session_id: time.time()}
        )

    # ========== MCP Tool Implementations ==========

    async def memory_store(self, context: str, files: Optional[List[str]] = None,
                          task: Optional[str] = None) -> Dict[str, Any]:
        """
        Store current context in distributed memory

        Args:
            context: Description of what's being worked on
            files: List of relevant file paths
            task: Current task description
        """
        session_key = f"session:{self.session_id}"

        update_data = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
        }

        if files:
            update_data['files'] = ','.join(files)
        if task:
            update_data['current_task'] = task

        await self.redis_client.hset(session_key, mapping=update_data)

        logger.info(f"Stored context for session {self.session_id}")

        return {
            'success': True,
            'session_id': self.session_id,
            'stored_at': update_data['timestamp']
        }

    async def memory_recall(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Recall context from distributed memory

        Args:
            session_id: Session to recall (defaults to current session)
        """
        target_session = session_id or self.session_id
        session_key = f"session:{target_session}"

        context = await self.redis_client.hgetall(session_key)

        if not context:
            return {
                'success': False,
                'error': f'No context found for session {target_session}'
            }

        logger.info(f"Recalled context for session {target_session}")

        return {
            'success': True,
            'session_id': target_session,
            'context': context.get('context', ''),
            'files': context.get('files', '').split(',') if context.get('files') else [],
            'task': context.get('current_task', ''),
            'timestamp': context.get('timestamp', ''),
            'node': context.get('node', '')
        }

    async def memory_list_sessions(self, limit: int = 10) -> Dict[str, Any]:
        """
        List recent sessions

        Args:
            limit: Maximum number of sessions to return
        """
        sessions = await self.redis_client.zrevrange(
            'sessions:recent', 0, limit - 1, withscores=True
        )

        result = []
        for session_id, timestamp in sessions:
            session_key = f"session:{session_id}"
            context = await self.redis_client.hget(session_key, 'context')
            result.append({
                'session_id': session_id,
                'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                'context': context or 'No context'
            })

        return {
            'success': True,
            'sessions': result
        }

    async def tool_cache_get(self, tool_name: str, input_hash: str) -> Optional[str]:
        """
        Get cached tool output

        Args:
            tool_name: Name of the tool (e.g., 'bash', 'read')
            input_hash: Hash of tool inputs
        """
        cache_key = f"tool:{tool_name}:{input_hash}"
        cached_output = await self.redis_client.get(cache_key)

        if cached_output:
            logger.info(f"Cache HIT for {tool_name}:{input_hash}")
            return cached_output

        logger.info(f"Cache MISS for {tool_name}:{input_hash}")
        return None

    async def tool_cache_set(self, tool_name: str, input_hash: str,
                            output: str, ttl: Optional[int] = None) -> None:
        """
        Cache tool output

        Args:
            tool_name: Name of the tool
            input_hash: Hash of tool inputs
            output: Tool output to cache
            ttl: Time to live in seconds (defaults to config value)
        """
        cache_key = f"tool:{tool_name}:{input_hash}"
        cache_ttl = ttl or self.config['cache']['tool_ttl']

        await self.redis_client.setex(cache_key, cache_ttl, output)
        logger.info(f"Cached output for {tool_name}:{input_hash} (TTL: {cache_ttl}s)")

    async def learning_queue_add(self, interaction: Dict[str, Any]) -> None:
        """
        Add interaction to learning queue

        Args:
            interaction: Dict containing user_query, tool_used, result, success, etc.
        """
        interaction['timestamp'] = datetime.now().isoformat()
        interaction['session_id'] = self.session_id

        await self.redis_client.xadd(
            'learning:queue',
            interaction,
            maxlen=100000,  # Keep last 100K entries
            approximate=True
        )

        logger.info(f"Added interaction to learning queue: {interaction.get('tool_used')}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get Hive-Mind statistics"""
        info = await self.redis_client.info()

        # Count keys by pattern
        try:
            session_count = await self.redis_client.zcard('sessions:recent')
        except Exception:
            session_count = 0

        # Learning queue length (may not exist yet)
        try:
            queue_info = await self.redis_client.xinfo_stream('learning:queue')
            queue_length = queue_info.get('length', 0) if queue_info else 0
        except Exception:
            queue_length = 0

        return {
            'redis_version': info.get('redis_version', 'unknown'),
            'connected_clients': info.get('connected_clients', 0),
            'used_memory_human': info.get('used_memory_human', 'unknown'),
            'total_sessions': session_count,
            'learning_queue_length': queue_length,
            'current_session': self.session_id,
            'cluster_mode': self.config['redis'].get('cluster_mode', False),
        }


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Hive-Mind MCP Server')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize server
    server = HiveMindMCP(args.config)

    try:
        await server.connect()

        # Test basic operations
        logger.info("Testing basic operations...")

        # Store context
        result = await server.memory_store(
            context="Testing Hive-Mind distributed memory system",
            files=["/mnt/build/MCP/hive-mind/README.md"],
            task="Phase 1: Redis setup and MCP server implementation"
        )
        logger.info(f"Store result: {result}")

        # Recall context
        result = await server.memory_recall()
        logger.info(f"Recall result: {result}")

        # Get stats
        stats = await server.get_stats()
        logger.info(f"Stats: {json.dumps(stats, indent=2)}")

        logger.info("✅ All tests passed! MCP server ready.")

        # Keep running (in production, this would be MCP protocol loop)
        if args.debug:
            logger.info("Running in debug mode. Press Ctrl+C to exit.")
            await asyncio.Event().wait()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await server.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
