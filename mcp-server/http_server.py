#!/usr/bin/env python3
"""
Hive-Mind HTTP API Server
Provides REST API access to the same distributed memory system
"""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import sys
from typing import Optional, List, Dict, Any

from pathlib import Path
import uvicorn

def _read_version():
    vf = Path(__file__).resolve().parent.parent / "VERSION"
    try:
        return vf.read_text().strip()
    except FileNotFoundError:
        return "0.0.0"

__version__ = _read_version()
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import aiohttp

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


# LLM Inference request models
class LLMGenerateRequest(BaseModel):
    prompt: str
    mode: str = "code"  # code, explain, debug
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    use_cache: bool = True


class LLMCodeAssistRequest(BaseModel):
    code: str
    task: str = "review"  # review, fix, optimize, explain, document
    language: str = "python"


class LLMCompleteRequest(BaseModel):
    prefix: str
    suffix: str = ""
    max_tokens: int = 256


# Fact storage request models (RAG)
class FactStoreRequest(BaseModel):
    key: str
    value: str


class FactGetRequest(BaseModel):
    key: Optional[str] = None


class FactDeleteRequest(BaseModel):
    key: str


class ConversationLogRequest(BaseModel):
    role: str
    content: str
    source: str
    timestamp: Optional[float] = None


class ConversationRecentRequest(BaseModel):
    limit: int = 20
    source: Optional[str] = None


class WebFetchRequest(BaseModel):
    url: str
    max_chars: int = 8000


class WebSearchRequest(BaseModel):
    query: str
    num_results: int = 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Hive-Mind on startup, disconnect on shutdown."""
    config_path = os.environ.get('CONFIG_PATH', 'config.yaml')
    logger.info(f"Starting Hive-Mind HTTP API with config: {config_path}")

    hm = HiveMindMCP(config_path)
    await hm.connect()
    app.state.hive_mind = hm
    logger.info("Hive-Mind HTTP API ready!")

    yield

    await hm.disconnect()
    logger.info("Hive-Mind disconnected")


app = FastAPI(
    title="Hive-Mind HTTP API",
    description="Distributed AI Memory System - HTTP Interface",
    version=__version__,
    lifespan=lifespan,
)


def _hm(request: Request) -> HiveMindMCP:
    """Get HiveMindMCP instance from app state."""
    hm = request.app.state.hive_mind
    if not hm:
        raise HTTPException(status_code=503, detail="Hive-Mind not initialized")
    return hm


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Hive-Mind HTTP API",
        "status": "operational",
        "version": __version__
    }


@app.get("/health")
async def health(request: Request):
    """Detailed health check"""
    hive_mind = _hm(request)
    try:
        await hive_mind.redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis connection failed: {str(e)}")


@app.post("/memory/store")
async def memory_store(body: MemoryStoreRequest, request: Request):
    """Store context in distributed memory"""
    hive_mind = _hm(request)
    try:
        return await hive_mind.memory_store(
            context=body.context, files=body.files, task=body.task
        )
    except Exception as e:
        logger.error(f"Error in memory_store: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/recall")
async def memory_recall(body: MemoryRecallRequest, request: Request):
    """Recall context from distributed memory"""
    hive_mind = _hm(request)
    try:
        return await hive_mind.memory_recall(session_id=body.session_id)
    except Exception as e:
        logger.error(f"Error in memory_recall: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/list-sessions")
async def memory_list_sessions(body: MemoryListSessionsRequest, request: Request):
    """List recent sessions"""
    hive_mind = _hm(request)
    try:
        return await hive_mind.memory_list_sessions(limit=body.limit)
    except Exception as e:
        logger.error(f"Error in memory_list_sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fact/store")
async def fact_store(body: FactStoreRequest, request: Request):
    """Store a fact for RAG retrieval"""
    hive_mind = _hm(request)
    try:
        return await hive_mind.fact_store(key=body.key, value=body.value)
    except Exception as e:
        logger.error(f"Error in fact_store: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fact/get")
async def fact_get(body: FactGetRequest, request: Request):
    """Get stored facts"""
    hive_mind = _hm(request)
    try:
        return await hive_mind.fact_get(key=body.key)
    except Exception as e:
        logger.error(f"Error in fact_get: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/suggestions")
async def rag_suggestions(request: Request, limit: int = 20):
    """Analyze missed RAG queries and suggest new facts to add"""
    hive_mind = _hm(request)
    try:
        return await hive_mind.fact_suggestions(limit=limit)
    except Exception as e:
        logger.error(f"Error in rag_suggestions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fact/delete")
async def fact_delete(body: FactDeleteRequest, request: Request):
    """Delete a stored fact"""
    hive_mind = _hm(request)
    try:
        return await hive_mind.fact_delete(key=body.key)
    except Exception as e:
        logger.error(f"Error in fact_delete: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversation/log")
async def conversation_log(body: ConversationLogRequest, request: Request):
    """Log a conversation message to the shared bridge"""
    hive_mind = _hm(request)
    try:
        return await hive_mind.conversation_log(
            role=body.role, content=body.content,
            source=body.source, timestamp=body.timestamp
        )
    except Exception as e:
        logger.error(f"Error in conversation_log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversation/recent")
async def conversation_recent(body: ConversationRecentRequest, request: Request):
    """Retrieve recent shared conversation messages"""
    hive_mind = _hm(request)
    try:
        return await hive_mind.conversation_recent(
            limit=body.limit, source=body.source
        )
    except Exception as e:
        logger.error(f"Error in conversation_recent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/web/fetch")
async def web_fetch(body: WebFetchRequest, request: Request):
    """Fetch a URL and extract readable text"""
    hive_mind = _hm(request)
    try:
        return await hive_mind.web_fetch(url=body.url, max_chars=body.max_chars)
    except Exception as e:
        logger.error(f"Error in web_fetch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/web/search")
async def web_search(body: WebSearchRequest, request: Request):
    """Search DuckDuckGo for results"""
    hive_mind = _hm(request)
    try:
        return await hive_mind.web_search(
            query=body.query, num_results=body.num_results
        )
    except Exception as e:
        logger.error(f"Error in web_search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool/cache/get")
async def tool_cache_get(body: ToolCacheGetRequest, request: Request):
    """Get cached tool output"""
    hive_mind = _hm(request)
    try:
        cached = await hive_mind.tool_cache_get(
            tool_name=body.tool_name, input_hash=body.input_hash
        )
        return {"cached_output": cached}
    except Exception as e:
        logger.error(f"Error in tool_cache_get: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool/cache/set")
async def tool_cache_set(body: ToolCacheSetRequest, request: Request):
    """Cache tool output"""
    hive_mind = _hm(request)
    try:
        await hive_mind.tool_cache_set(
            tool_name=body.tool_name, input_hash=body.input_hash,
            output=body.output, ttl=body.ttl
        )
        return {"success": True}
    except Exception as e:
        logger.error(f"Error in tool_cache_set: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/learning/queue/add")
async def learning_queue_add(body: LearningQueueAddRequest, request: Request):
    """Add interaction to learning queue"""
    hive_mind = _hm(request)
    try:
        await hive_mind.learning_queue_add(interaction=body.interaction)
        return {"success": True}
    except Exception as e:
        logger.error(f"Error in learning_queue_add: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats(request: Request):
    """Get Hive-Mind system statistics"""
    hive_mind = _hm(request)
    try:
        return await hive_mind.get_stats()
    except Exception as e:
        logger.error(f"Error in get_stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== LLM Inference Endpoints ==========

@app.post("/llm/generate")
async def llm_generate(body: LLMGenerateRequest, request: Request):
    """Generate text using HiveCoder-7B"""
    hive_mind = _hm(request)
    try:
        result = await hive_mind.llm_generate(
            prompt=body.prompt, mode=body.mode,
            max_tokens=body.max_tokens, temperature=body.temperature,
            use_cache=body.use_cache
        )
        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in llm_generate: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/code-assist")
async def llm_code_assist(body: LLMCodeAssistRequest, request: Request):
    """Get code assistance from HiveCoder-7B"""
    hive_mind = _hm(request)
    try:
        result = await hive_mind.llm_code_assist(
            code=body.code, task=body.task, language=body.language
        )
        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in llm_code_assist: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/complete")
async def llm_complete(body: LLMCompleteRequest, request: Request):
    """Code completion using HiveCoder-7B"""
    hive_mind = _hm(request)
    try:
        result = await hive_mind.llm_complete(
            prefix=body.prefix, suffix=body.suffix, max_tokens=body.max_tokens
        )
        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in llm_complete: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/status")
async def llm_status(request: Request):
    """Check HiveCoder-7B status"""
    hive_mind = _hm(request)
    stats = await hive_mind.get_stats()
    return {
        "model": stats.get("llm_model", "none"),
        "status": stats.get("llm_status", "disabled"),
        "endpoint": hive_mind.config.get('inference', {}).get('endpoint', 'not configured')
    }


# ========== OpenAI-Compatible Proxy with RAG ==========

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "HiveCoder-7B"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
async def openai_chat_completions(body: ChatCompletionRequest, request: Request):
    """
    OpenAI-compatible chat completions endpoint with RAG injection.

    Use this endpoint instead of llama-server directly to get
    automatic RAG fact injection into the system prompt.
    """
    hive_mind = _hm(request)

    try:
        # Extract user query for fact filtering
        user_query = None
        for msg in reversed(body.messages):
            if msg.role == "user":
                user_query = msg.content
                break

        # Get RAG facts (semantic search with keyword fallback)
        facts_context = await hive_mind._get_facts_context(query=user_query)

        # Process messages - inject facts into system prompt
        messages = []
        system_found = False

        for msg in body.messages:
            if msg.role == "system" and facts_context:
                enhanced_content = f"{msg.content}\n\n{facts_context}"
                messages.append({"role": "system", "content": enhanced_content})
                system_found = True
            else:
                messages.append({"role": msg.role, "content": msg.content})

        if not system_found and facts_context:
            messages.insert(0, {
                "role": "system",
                "content": f"You are HiveCoder, a helpful AI coding assistant.\n\n{facts_context}"
            })

        # Forward to llama-server
        inference_config = hive_mind.config.get('inference', {})
        endpoint = inference_config.get('endpoint', 'http://127.0.0.1:8089')

        payload = {
            "model": body.model,
            "messages": messages,
            "max_tokens": body.max_tokens,
            "temperature": body.temperature,
            "top_p": body.top_p,
            "stream": body.stream
        }

        timeout = aiohttp.ClientTimeout(total=inference_config.get('timeout', 120))

        if body.stream:
            async def stream_generator():
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{endpoint}/v1/chat/completions",
                        json=payload
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            yield f"data: {json.dumps({'error': error_text})}\n\n"
                            return
                        async for chunk in resp.content.iter_any():
                            yield chunk

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{endpoint}/v1/chat/completions",
                    json=payload
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise HTTPException(status_code=resp.status, detail=error_text)
                    return await resp.json()

    except aiohttp.ClientError as e:
        logger.error(f"Error connecting to LLM backend: {e}")
        raise HTTPException(status_code=502, detail=f"LLM backend error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in chat completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def openai_list_models():
    """OpenAI-compatible models endpoint"""
    return {
        "object": "list",
        "data": [
            {
                "id": "HiveCoder-7B",
                "object": "model",
                "created": 1700000000,
                "owned_by": "hive-mind"
            }
        ]
    }


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
