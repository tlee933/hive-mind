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

import aiohttp
import redis.asyncio as aioredis
from redis.asyncio.cluster import RedisCluster, ClusterNode
import yaml

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Important: log to stderr so stdout is clean for MCP protocol
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

        # Convert all values to strings (Redis streams require string/int/float)
        # Booleans need explicit conversion to string
        redis_interaction = {}
        for k, v in interaction.items():
            if isinstance(v, bool):
                redis_interaction[k] = str(v)
            elif isinstance(v, (str, int, float)):
                redis_interaction[k] = v
            else:
                # Convert any other types (lists, dicts, etc.) to JSON string
                redis_interaction[k] = json.dumps(v) if isinstance(v, (list, dict)) else str(v)

        await self.redis_client.xadd(
            'learning:queue',
            redis_interaction,
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

        # Check LLM inference status
        llm_status = "disabled"
        if self.config.get('inference', {}).get('enabled', False):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.config['inference']['endpoint']}/health",
                        timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            llm_status = "online"
                        else:
                            llm_status = "error"
            except Exception:
                llm_status = "offline"

        return {
            'redis_version': info.get('redis_version', 'unknown'),
            'connected_clients': info.get('connected_clients', 0),
            'used_memory_human': info.get('used_memory_human', 'unknown'),
            'total_sessions': session_count,
            'learning_queue_length': queue_length,
            'current_session': self.session_id,
            'cluster_mode': self.config['redis'].get('cluster_mode', False),
            'llm_model': self.config.get('inference', {}).get('model', 'none'),
            'llm_status': llm_status,
        }

    # ========== LLM Inference Methods ==========

    async def llm_generate(self, prompt: str, mode: str = "code",
                          max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None,
                          use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate text using HiveCoder-7B

        Args:
            prompt: The prompt to send to the model
            mode: System prompt mode ('code', 'explain', 'debug')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to cache the response
        """
        inference_config = self.config.get('inference', {})
        if not inference_config.get('enabled', False):
            return {'success': False, 'error': 'LLM inference is not enabled'}

        # Check cache first
        if use_cache:
            cache_key = hashlib.sha256(f"{prompt}:{mode}:{max_tokens}:{temperature}".encode()).hexdigest()[:16]
            cached = await self.redis_client.get(f"llm:cache:{cache_key}")
            if cached:
                logger.info(f"LLM cache HIT for prompt: {prompt[:50]}...")
                return json.loads(cached)

        # Build request
        system_prompt = inference_config.get('system_prompts', {}).get(mode,
            "You are HiveCoder, a helpful AI coding assistant.")

        payload = {
            "model": inference_config.get('model', 'HiveCoder-7B'),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens or inference_config.get('default_max_tokens', 512),
            "temperature": temperature if temperature is not None else inference_config.get('default_temperature', 0.7),
            "top_p": inference_config.get('default_top_p', 0.9),
        }

        try:
            timeout = aiohttp.ClientTimeout(total=inference_config.get('timeout', 60))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{inference_config['endpoint']}/v1/chat/completions",
                    json=payload
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return {'success': False, 'error': f"LLM request failed: {error_text}"}

                    data = await resp.json()

            # Extract response
            response_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            usage = data.get('usage', {})

            result = {
                'success': True,
                'response': response_text,
                'model': inference_config.get('model', 'HiveCoder-7B'),
                'mode': mode,
                'usage': {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                }
            }

            # Cache the result
            if use_cache:
                cache_ttl = self.config.get('cache', {}).get('inference_ttl', 1800)
                await self.redis_client.setex(
                    f"llm:cache:{cache_key}",
                    cache_ttl,
                    json.dumps(result)
                )

            logger.info(f"LLM generated {usage.get('completion_tokens', 0)} tokens for mode={mode}")
            return result

        except asyncio.TimeoutError:
            return {'success': False, 'error': 'LLM request timed out'}
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def llm_code_assist(self, code: str, task: str = "review",
                             language: str = "python") -> Dict[str, Any]:
        """
        Code assistance using HiveCoder-7B

        Args:
            code: The code to analyze/modify
            task: Task type ('review', 'fix', 'optimize', 'explain', 'document')
            language: Programming language
        """
        task_prompts = {
            'review': f"Review this {language} code for issues, bugs, and improvements:\n\n```{language}\n{code}\n```",
            'fix': f"Fix any bugs in this {language} code and explain what was wrong:\n\n```{language}\n{code}\n```",
            'optimize': f"Optimize this {language} code for better performance:\n\n```{language}\n{code}\n```",
            'explain': f"Explain what this {language} code does, step by step:\n\n```{language}\n{code}\n```",
            'document': f"Add comprehensive docstrings and comments to this {language} code:\n\n```{language}\n{code}\n```",
        }

        prompt = task_prompts.get(task, task_prompts['review'])
        mode_map = {'review': 'debug', 'fix': 'debug', 'optimize': 'code',
                   'explain': 'explain', 'document': 'code'}

        result = await self.llm_generate(prompt, mode=mode_map.get(task, 'code'))
        if result.get('success'):
            result['task'] = task
            result['language'] = language
        return result

    async def llm_complete(self, prefix: str, suffix: str = "",
                          max_tokens: int = 256) -> Dict[str, Any]:
        """
        Code completion using HiveCoder-7B (FIM style)

        Args:
            prefix: Code before cursor
            suffix: Code after cursor
            max_tokens: Maximum tokens to generate
        """
        # Build FIM-style prompt
        if suffix:
            prompt = f"Complete the code between the markers:\n\n```\n{prefix}\n<CURSOR>\n{suffix}\n```\n\nProvide only the code that goes at <CURSOR>:"
        else:
            prompt = f"Complete this code:\n\n```\n{prefix}\n```\n\nProvide the completion:"

        result = await self.llm_generate(prompt, mode='code', max_tokens=max_tokens)
        return result


async def main():
    """Main entry point for MCP server"""
    # Get config path from environment or use default
    config_path = os.environ.get('CONFIG_PATH', 'config.yaml')

    logger.info(f"Starting Hive-Mind MCP Server with config: {config_path}")

    # Initialize Hive-Mind backend
    hive_mind = HiveMindMCP(config_path)
    await hive_mind.connect()

    # Create MCP server
    server = Server("hive-mind")

    # Register tools
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available Hive-Mind tools"""
        return [
            Tool(
                name="memory_store",
                description="Store current context in distributed memory. Survives terminal restarts and shared across machines.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Description of what you're working on"
                        },
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of relevant file paths"
                        },
                        "task": {
                            "type": "string",
                            "description": "Current task description"
                        }
                    },
                    "required": ["context"]
                }
            ),
            Tool(
                name="memory_recall",
                description="Recall context from distributed memory. Retrieve context from current or previous sessions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to recall (optional, defaults to current session)"
                        }
                    }
                }
            ),
            Tool(
                name="memory_list_sessions",
                description="List recent sessions in distributed memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of sessions to return (default: 10)",
                            "default": 10
                        }
                    }
                }
            ),
            Tool(
                name="tool_cache_get",
                description="Get cached tool output from previous executions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool (e.g., 'bash', 'read')"
                        },
                        "input_hash": {
                            "type": "string",
                            "description": "Hash of tool inputs"
                        }
                    },
                    "required": ["tool_name", "input_hash"]
                }
            ),
            Tool(
                name="tool_cache_set",
                description="Cache tool output for future use",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool"
                        },
                        "input_hash": {
                            "type": "string",
                            "description": "Hash of tool inputs"
                        },
                        "output": {
                            "type": "string",
                            "description": "Tool output to cache"
                        },
                        "ttl": {
                            "type": "number",
                            "description": "Time to live in seconds (optional)"
                        }
                    },
                    "required": ["tool_name", "input_hash", "output"]
                }
            ),
            Tool(
                name="learning_queue_add",
                description="Add interaction to learning queue for future model training",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "interaction": {
                            "type": "object",
                            "description": "Interaction data containing user_query, tool_used, result, success, etc."
                        }
                    },
                    "required": ["interaction"]
                }
            ),
            Tool(
                name="get_stats",
                description="Get Hive-Mind system statistics (Redis info, session counts, queue lengths, LLM status, etc.)",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="llm_generate",
                description="Generate text using HiveCoder-7B local LLM. Use for code generation, explanations, or debugging assistance.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to send to HiveCoder"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["code", "explain", "debug"],
                            "description": "System prompt mode (default: code)",
                            "default": "code"
                        },
                        "max_tokens": {
                            "type": "number",
                            "description": "Maximum tokens to generate (default: 512)"
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature 0.0-2.0 (default: 0.7)"
                        },
                        "use_cache": {
                            "type": "boolean",
                            "description": "Whether to cache the response (default: true)",
                            "default": True
                        }
                    },
                    "required": ["prompt"]
                }
            ),
            Tool(
                name="llm_code_assist",
                description="Get code assistance from HiveCoder-7B. Review, fix, optimize, explain, or document code.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to analyze or modify"
                        },
                        "task": {
                            "type": "string",
                            "enum": ["review", "fix", "optimize", "explain", "document"],
                            "description": "The type of assistance needed",
                            "default": "review"
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language (default: python)",
                            "default": "python"
                        }
                    },
                    "required": ["code"]
                }
            ),
            Tool(
                name="llm_complete",
                description="Code completion using HiveCoder-7B. Provide code prefix and optional suffix for FIM-style completion.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prefix": {
                            "type": "string",
                            "description": "Code before the cursor position"
                        },
                        "suffix": {
                            "type": "string",
                            "description": "Code after the cursor position (optional)"
                        },
                        "max_tokens": {
                            "type": "number",
                            "description": "Maximum tokens to generate (default: 256)",
                            "default": 256
                        }
                    },
                    "required": ["prefix"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls"""
        try:
            if name == "memory_store":
                result = await hive_mind.memory_store(
                    context=arguments["context"],
                    files=arguments.get("files"),
                    task=arguments.get("task")
                )
            elif name == "memory_recall":
                result = await hive_mind.memory_recall(
                    session_id=arguments.get("session_id")
                )
            elif name == "memory_list_sessions":
                result = await hive_mind.memory_list_sessions(
                    limit=arguments.get("limit", 10)
                )
            elif name == "tool_cache_get":
                cached = await hive_mind.tool_cache_get(
                    tool_name=arguments["tool_name"],
                    input_hash=arguments["input_hash"]
                )
                result = {"cached_output": cached} if cached else {"cached_output": None}
            elif name == "tool_cache_set":
                await hive_mind.tool_cache_set(
                    tool_name=arguments["tool_name"],
                    input_hash=arguments["input_hash"],
                    output=arguments["output"],
                    ttl=arguments.get("ttl")
                )
                result = {"success": True}
            elif name == "learning_queue_add":
                await hive_mind.learning_queue_add(
                    interaction=arguments["interaction"]
                )
                result = {"success": True}
            elif name == "get_stats":
                result = await hive_mind.get_stats()
            elif name == "llm_generate":
                result = await hive_mind.llm_generate(
                    prompt=arguments["prompt"],
                    mode=arguments.get("mode", "code"),
                    max_tokens=arguments.get("max_tokens"),
                    temperature=arguments.get("temperature"),
                    use_cache=arguments.get("use_cache", True)
                )
            elif name == "llm_code_assist":
                result = await hive_mind.llm_code_assist(
                    code=arguments["code"],
                    task=arguments.get("task", "review"),
                    language=arguments.get("language", "python")
                )
            elif name == "llm_complete":
                result = await hive_mind.llm_complete(
                    prefix=arguments["prefix"],
                    suffix=arguments.get("suffix", ""),
                    max_tokens=arguments.get("max_tokens", 256)
                )
            else:
                result = {"error": f"Unknown tool: {name}"}

            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]

    # Run the server
    logger.info("Hive-Mind MCP Server ready!")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
