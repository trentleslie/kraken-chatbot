"""
Graph Runner: Entry point for executing the KRAKEN discovery workflow.

Provides both synchronous and streaming execution modes.
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Any

from .builder import build_discovery_graph
from .state import DiscoveryState

logger = logging.getLogger(__name__)


async def run_discovery(
    query: str,
    conversation_history: list[tuple[str, str]] | None = None,
) -> DiscoveryState:
    """
    Run the discovery workflow and return final state.

    Args:
        query: User's input query
        conversation_history: Optional list of (role, content) tuples

    Returns:
        Final DiscoveryState with synthesis_report populated
    """
    graph = build_discovery_graph()

    initial_state: DiscoveryState = {
        "raw_query": query,
        "conversation_history": conversation_history or [],
    }

    result = await graph.ainvoke(initial_state)
    return result


async def stream_discovery(
    query: str,
    conversation_history: list[tuple[str, str]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Stream discovery workflow events for real-time updates.

    Yields events as the graph executes, useful for:
    - WebSocket updates to frontend
    - Progress indicators
    - Intermediate result display

    Args:
        query: User's input query
        conversation_history: Optional list of (role, content) tuples

    Yields:
        Event dicts with type, node, and data fields
    """
    query_preview = query[:50] + "..." if len(query) > 50 else query
    logger.info("Stream started — query=%r", query_preview)
    start_time = time.time()

    graph = build_discovery_graph()

    initial_state: DiscoveryState = {
        "raw_query": query,
        "conversation_history": conversation_history or [],
    }

    # Use astream_log for detailed event streaming
    async for event in graph.astream_log(initial_state):
        # Transform LangGraph events into our simplified format
        ops = event.ops if hasattr(event, 'ops') else []
        for op in ops:
            if op.get("path", "").startswith("/logs/"):
                # Extract node name from path
                path_parts = op["path"].split("/")
                if len(path_parts) >= 3:
                    node_name = path_parts[2]

                    yield {
                        "type": "node_event",
                        "node": node_name,
                        "op": op.get("op"),
                        "data": op.get("value"),
                    }

            elif op.get("path") == "/final_output":
                duration = time.time() - start_time
                logger.info("Stream complete in %.1fs", duration)
                yield {
                    "type": "complete",
                    "data": op.get("value"),
                }


# Convenience function for synchronous contexts
def run_discovery_sync(
    query: str,
    conversation_history: list[tuple[str, str]] | None = None,
) -> DiscoveryState:
    """
    Synchronous wrapper for run_discovery.

    Useful for testing or non-async contexts.
    """
    return asyncio.run(run_discovery(query, conversation_history))
