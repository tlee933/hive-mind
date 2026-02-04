#!/usr/bin/env python3
"""
Hive-Mind HTTP Client
Python client for accessing Hive-Mind distributed memory via HTTP API
"""

import requests
from typing import Optional, List, Dict, Any


class HiveMindClient:
    """Python client for Hive-Mind HTTP API

    Usage:
        >>> hive = HiveMindClient()
        >>> hive.store_memory("Working on data analysis", task="Q4 report")
        >>> context = hive.recall_memory()
        >>> print(context['context'])
    """

    def __init__(self, base_url: str = "http://localhost:8090"):
        """Initialize client

        Args:
            base_url: Base URL of Hive-Mind HTTP API (default: http://localhost:8090)
        """
        self.base_url = base_url.rstrip('/')
        self.session_id = None

    def health_check(self) -> Dict[str, Any]:
        """Check if Hive-Mind is operational"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()

    def store_memory(self, context: str, files: Optional[List[str]] = None,
                     task: Optional[str] = None) -> Dict[str, Any]:
        """Store context in distributed memory

        Args:
            context: Description of what you're working on
            files: List of relevant file paths
            task: Current task description

        Returns:
            Dict with success status, session_id, and timestamp
        """
        payload = {"context": context}
        if files:
            payload["files"] = files
        if task:
            payload["task"] = task

        response = requests.post(f"{self.base_url}/memory/store", json=payload)
        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            self.session_id = result.get("session_id")

        return result

    def recall_memory(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Recall context from distributed memory

        Args:
            session_id: Session to recall (defaults to current session)

        Returns:
            Dict with context, files, task, timestamp, and node
        """
        payload = {}
        if session_id:
            payload["session_id"] = session_id

        response = requests.post(f"{self.base_url}/memory/recall", json=payload)
        response.raise_for_status()
        return response.json()

    def list_sessions(self, limit: int = 10) -> Dict[str, Any]:
        """List recent sessions

        Args:
            limit: Maximum number of sessions to return (default: 10)

        Returns:
            Dict with list of sessions
        """
        response = requests.post(
            f"{self.base_url}/memory/list-sessions",
            json={"limit": limit}
        )
        response.raise_for_status()
        return response.json()

    def cache_tool_output(self, tool_name: str, input_hash: str,
                         output: str, ttl: Optional[int] = None) -> Dict[str, Any]:
        """Cache tool output

        Args:
            tool_name: Name of the tool
            input_hash: Hash of tool inputs
            output: Tool output to cache
            ttl: Time to live in seconds (optional)

        Returns:
            Dict with success status
        """
        payload = {
            "tool_name": tool_name,
            "input_hash": input_hash,
            "output": output
        }
        if ttl:
            payload["ttl"] = ttl

        response = requests.post(f"{self.base_url}/tool/cache/set", json=payload)
        response.raise_for_status()
        return response.json()

    def get_cached_output(self, tool_name: str, input_hash: str) -> Optional[str]:
        """Get cached tool output

        Args:
            tool_name: Name of the tool
            input_hash: Hash of tool inputs

        Returns:
            Cached output if found, None otherwise
        """
        response = requests.post(
            f"{self.base_url}/tool/cache/get",
            json={"tool_name": tool_name, "input_hash": input_hash}
        )
        response.raise_for_status()
        result = response.json()
        return result.get("cached_output")

    def add_to_learning_queue(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Add interaction to learning queue

        Args:
            interaction: Dict containing tool_used, user_query, result, success, etc.

        Returns:
            Dict with success status
        """
        response = requests.post(
            f"{self.base_url}/learning/queue/add",
            json={"interaction": interaction}
        )
        response.raise_for_status()
        return response.json()


def main():
    """Example usage"""
    import sys

    print("🐝 Hive-Mind Client Test\n")

    # Initialize client
    hive = HiveMindClient()

    try:
        # Health check
        health = hive.health_check()
        print(f"✅ Health: {health['status']}")

        # Get stats
        stats = hive.get_stats()
        print(f"✅ Redis: {stats['redis_version']}")
        print(f"✅ Sessions: {stats['total_sessions']}")
        print(f"✅ Current Session: {stats['current_session']}\n")

        # Store memory
        result = hive.store_memory(
            context="Testing Hive-Mind client",
            task="Verify HTTP API integration"
        )
        print(f"✅ Stored memory to session: {result['session_id']}\n")

        # Recall memory
        context = hive.recall_memory()
        print(f"✅ Recalled context: {context['context']}")
        print(f"   Task: {context['task']}")
        print(f"   Node: {context['node']}\n")

        # List sessions
        sessions = hive.list_sessions(limit=3)
        print(f"✅ Recent sessions: {len(sessions['sessions'])}")
        for session in sessions['sessions'][:3]:
            print(f"   - {session['session_id']}: {session['context']}")

        print("\n🎉 All tests passed!")

    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to Hive-Mind HTTP API")
        print("   Make sure the service is running: sudo systemctl status hive-mind-http")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
