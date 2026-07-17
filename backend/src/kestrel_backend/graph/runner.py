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
    config: dict[str, Any] | None = None,
    biomapper_env: str | None = None,
    structured_analytes: list[dict] | None = None,
    selected_groups: list[str] | None = None,
    module_directions: list[dict] | None = None,
) -> DiscoveryState:
    """
    Run the discovery workflow and return final state.

    Args:
        query: User's input query
        conversation_history: Optional list of (role, content) tuples
        config: Optional LangGraph RunnableConfig (e.g. {"callbacks": [handler]}) passed
            through to the graph. The caller owns it; this module stays observability-agnostic.
        biomapper_env: Optional prod/dev biomapper2 API toggle ("production"|"dev"); threaded into
            initial state so this non-streaming path matches stream_discovery.
        structured_analytes: Optional full parsed analyte panel ({name, group?, type?, kme?, kim?}).
            Threaded into initial state; the intake node re-validates it (R19 gate #2) so this
            non-WS entry point (Studio/harness) is guarded identically to main.py.
        selected_groups: Optional group selection for the run-set filter (empty/None = all).
        module_directions: Optional per-module eigengene→outcome direction rows (Axis A); the
            intake node validates + builds ModuleSpine.direction from them (R19 gate #2 parity).

    Returns:
        Final DiscoveryState with synthesis_report populated
    """
    graph = build_discovery_graph()

    initial_state: DiscoveryState = {
        "raw_query": query,
        "conversation_history": conversation_history or [],
        "biomapper_env": biomapper_env,
        "structured_analytes": structured_analytes or [],
        "selected_groups": selected_groups or [],
        "module_directions": module_directions or [],
    }

    result = await graph.ainvoke(initial_state, config=config)
    return result


async def stream_discovery(
    query: str,
    conversation_history: list[tuple[str, str]] | None = None,
    config: dict[str, Any] | None = None,
    biomapper_env: str | None = None,
    structured_analytes: list[dict] | None = None,
    selected_groups: list[str] | None = None,
    module_directions: list[dict] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Stream discovery workflow events for real-time updates.

    Uses stream_mode="updates" so each yield is {node_name: node_output}.
    The caller accumulates state incrementally.

    Args:
        query: User's input query
        conversation_history: Optional list of (role, content) tuples
        config: Optional LangGraph RunnableConfig (e.g. {"callbacks": [handler]}) passed
            through to the graph. The caller owns it; this module stays observability-agnostic.

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
        "biomapper_env": biomapper_env,
        "structured_analytes": structured_analytes or [],
        "selected_groups": selected_groups or [],
        "module_directions": module_directions or [],
    }

    async for event in graph.astream(initial_state, stream_mode="updates", config=config):
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
    biomapper_env: str | None = None,
    structured_analytes: list[dict] | None = None,
    selected_groups: list[str] | None = None,
    module_directions: list[dict] | None = None,
) -> DiscoveryState:
    """
    Synchronous wrapper for run_discovery.

    Useful for testing or non-async contexts.
    """
    return asyncio.run(
        run_discovery(
            query,
            conversation_history,
            biomapper_env=biomapper_env,
            structured_analytes=structured_analytes,
            selected_groups=selected_groups,
            module_directions=module_directions,
        )
    )
