#!/usr/bin/env python3
"""
Hive-Mind HTTP API Server
Provides REST API access to the same distributed memory system
"""

import asyncio
import logging
import os
import sys
from typing import Optional, List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import the existing HiveMindMCP class
from server import HiveMindMCP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("hive-mind-http")

# Pydantic models for request/response
class MemoryStoreRequest(BaseModel):
    context: str
    files: Optional[List[str]] = None
    task: Optional[str] = None

class MemoryRecallRequest(BaseModel):
    session_id: Optional[str] = None

class MemoryListSessionsRequest(BaseModel):
    limit: int = 10

class ToolCacheGetRequest(BaseModel):
    tool_name: str
    input_hash: str

class ToolCacheSetRequest(BaseModel):
    tool_name: str
    input_hash: str
    output: str
    ttl: Optional[int] = None

class LearningQueueAddRequest(BaseModel):
    interaction: Dict[str, Any]

# Create FastAPI app
app = FastAPI(
    title="Hive-Mind HTTP API",
    description="Distributed AI Memory System - HTTP Interface",
    version="0.1.0",
)

# Global HiveMind instance
hive_mind: Optional[HiveMindMCP] = None


@app.on_event("startup")
async def startup_event():
    """Initialize Hive-Mind on startup"""
    global hive_mind
    config_path = os.environ.get('CONFIG_PATH', 'config.yaml')
    logger.info(f"Starting Hive-Mind HTTP API with config: {config_path}")

    hive_mind = HiveMindMCP(config_path)
    await hive_mind.connect()
    logger.info("Hive-Mind HTTP API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global hive_mind
    if hive_mind:
        await hive_mind.disconnect()
        logger.info("Hive-Mind disconnected")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Hive-Mind HTTP API",
        "status": "operational",
        "version": "0.1.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    if not hive_mind:
        raise HTTPException(status_code=503, detail="Hive-Mind not initialized")

    try:
        await hive_mind.redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis connection failed: {str(e)}")


@app.post("/memory/store")
async def memory_store(request: MemoryStoreRequest):
    """Store context in distributed memory"""
    if not hive_mind:
        raise HTTPException(status_code=503, detail="Hive-Mind not initialized")

    try:
        result = await hive_mind.memory_store(
            context=request.context,
            files=request.files,
            task=request.task
        )
        return result
    except Exception as e:
        logger.error(f"Error in memory_store: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/recall")
async def memory_recall(request: MemoryRecallRequest):
    """Recall context from distributed memory"""
    if not hive_mind:
        raise HTTPException(status_code=503, detail="Hive-Mind not initialized")

    try:
        result = await hive_mind.memory_recall(session_id=request.session_id)
        return result
    except Exception as e:
        logger.error(f"Error in memory_recall: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/list-sessions")
async def memory_list_sessions(request: MemoryListSessionsRequest):
    """List recent sessions"""
    if not hive_mind:
        raise HTTPException(status_code=503, detail="Hive-Mind not initialized")

    try:
        result = await hive_mind.memory_list_sessions(limit=request.limit)
        return result
    except Exception as e:
        logger.error(f"Error in memory_list_sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool/cache/get")
async def tool_cache_get(request: ToolCacheGetRequest):
    """Get cached tool output"""
    if not hive_mind:
        raise HTTPException(status_code=503, detail="Hive-Mind not initialized")

    try:
        cached = await hive_mind.tool_cache_get(
            tool_name=request.tool_name,
            input_hash=request.input_hash
        )
        return {"cached_output": cached}
    except Exception as e:
        logger.error(f"Error in tool_cache_get: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool/cache/set")
async def tool_cache_set(request: ToolCacheSetRequest):
    """Cache tool output"""
    if not hive_mind:
        raise HTTPException(status_code=503, detail="Hive-Mind not initialized")

    try:
        await hive_mind.tool_cache_set(
            tool_name=request.tool_name,
            input_hash=request.input_hash,
            output=request.output,
            ttl=request.ttl
        )
        return {"success": True}
    except Exception as e:
        logger.error(f"Error in tool_cache_set: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/learning/queue/add")
async def learning_queue_add(request: LearningQueueAddRequest):
    """Add interaction to learning queue"""
    if not hive_mind:
        raise HTTPException(status_code=503, detail="Hive-Mind not initialized")

    try:
        await hive_mind.learning_queue_add(interaction=request.interaction)
        return {"success": True}
    except Exception as e:
        logger.error(f"Error in learning_queue_add: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get Hive-Mind system statistics"""
    if not hive_mind:
        raise HTTPException(status_code=503, detail="Hive-Mind not initialized")

    try:
        stats = await hive_mind.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error in get_stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Start the HTTP server"""
    host = os.environ.get('HTTP_HOST', '0.0.0.0')
    port = int(os.environ.get('HTTP_PORT', '8090'))

    logger.info(f"Starting Hive-Mind HTTP API on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == '__main__':
    main()
