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

    Uses stream_mode="updates" so each yield is {node_name: node_output}.
    The caller accumulates state incrementally.

    Args:
        query: User's input query
        conversation_history: Optional list of (role, content) tuples

    Yields:
        Event dicts with type, node, and node_output fields
    """
    query_preview = query[:50] + "..." if len(query) > 50 else query
    logger.info("Stream started — query=%r", query_preview)
    start_time = time.time()

    graph = build_discovery_graph()

    initial_state: DiscoveryState = {
        "raw_query": query,
        "conversation_history": conversation_history or [],
    }

    async for event in graph.astream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            yield {
                "type": "node_update",
                "node": node_name,
                "node_output": node_output,
            }

    duration = time.time() - start_time
    logger.info("Stream complete in %.1fs", duration)


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
