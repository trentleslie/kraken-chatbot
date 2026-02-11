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

    final_state = None
    prev_keys: set[str] = set()

    # Use astream with stream_mode="values" for full state snapshots
    async for state in graph.astream(initial_state, stream_mode="values"):
        final_state = state  # Each yield is the full accumulated state

        # Detect which node just ran by checking for new keys
        current_keys = set(state.keys())
        new_keys = current_keys - prev_keys
        prev_keys = current_keys

        # Infer node name from new keys added to state
        node_name = None
        if "raw_entities" in new_keys:
            node_name = "intake"
        elif "resolved_entities" in new_keys:
            node_name = "entity_resolution"
        elif "novelty_scores" in new_keys:
            node_name = "triage"
        elif "direct_findings" in new_keys and "disease_associations" in new_keys:
            node_name = "direct_kg"
        elif "cold_start_findings" in new_keys:
            node_name = "cold_start"
        elif "shared_neighbors" in new_keys:
            node_name = "pathway_enrichment"
        elif "bridges" in new_keys:
            node_name = "integration"
        elif "temporal_classifications" in new_keys:
            node_name = "temporal"
        elif "synthesis_report" in new_keys:
            node_name = "synthesis"

        if node_name:
            yield {
                "type": "node_event",
                "node": node_name,
                "op": "add",
                "data": state,
            }

    duration = time.time() - start_time
    logger.info("Stream complete in %.1fs", duration)

    yield {
        "type": "complete",
        "data": final_state,  # Full accumulated state with all fields
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
