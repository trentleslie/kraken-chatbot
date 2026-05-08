"""Shared SDK utilities for discovery pipeline nodes.

Centralizes the Claude Agent SDK import pattern, MCP configuration factory,
agent options factory, and the chunk() utility that were previously duplicated
across 8 pipeline nodes.

Semaphore definitions and fallback orchestration logic remain per-node
(intentionally different values and patterns).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Centralized SDK availability check — single try/except for all nodes
try:
    from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage  # noqa: F401
    from claude_agent_sdk.types import McpStdioServerConfig  # noqa: F401
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    # Define stubs so importing code doesn't need conditional imports
    query = None  # type: ignore[assignment]
    ClaudeAgentOptions = None  # type: ignore[assignment,misc]
    McpStdioServerConfig = None  # type: ignore[assignment,misc]

    # Sentinel class so isinstance(event, ResultMessage) returns False
    # rather than raising TypeError when SDK is unavailable
    class _ResultMessageStub:  # type: ignore[no-redef]
        pass
    ResultMessage = _ResultMessageStub  # type: ignore[assignment,misc]

# Kestrel MCP server configuration constants
KESTREL_COMMAND = "uvx"
KESTREL_ARGS = ["mcp-client-kestrel"]


def get_kestrel_mcp_config() -> Any:
    """Create a McpStdioServerConfig for the Kestrel MCP server.

    Returns the config object, or None if the SDK is not available.
    """
    if not HAS_SDK or McpStdioServerConfig is None:
        return None
    return McpStdioServerConfig(
        type="stdio",
        command=KESTREL_COMMAND,
        args=KESTREL_ARGS,
    )


# NOTE: create_agent_options is not yet used by existing nodes (they call
# ClaudeAgentOptions directly). It will be used by PR5's LLM-as-judge scorer
# and the config-flagged literature classifier. Kept here to avoid a second
# refactor pass when those features land.
def create_agent_options(
    system_prompt: str,
    allowed_tools: list[str] | None = None,
    max_turns: int = 3,
    mcp_servers: list[Any] | None = None,
    permission_mode: str = "bypassPermissions",
    max_buffer_size: int = 10 * 1024 * 1024,
) -> Any:
    """Create ClaudeAgentOptions with standard defaults.

    Supports both MCP-tool-enabled nodes (with mcp_servers and allowed_tools)
    and pure-reasoning nodes (allowed_tools=[], no mcp_servers) like synthesis.

    Args:
        system_prompt: The system prompt for the agent
        allowed_tools: List of allowed tool names, or [] for pure reasoning
        max_turns: Maximum conversation turns
        mcp_servers: List of MCP server configs (None for pure reasoning)
        permission_mode: SDK permission mode
        max_buffer_size: Maximum buffer size for responses

    Returns:
        ClaudeAgentOptions instance, or None if SDK is not available.
    """
    if not HAS_SDK or ClaudeAgentOptions is None:
        return None

    kwargs: dict[str, Any] = {
        "system_prompt": system_prompt,
        "allowed_tools": allowed_tools or [],
        "max_turns": max_turns,
        "permission_mode": permission_mode,
        "max_buffer_size": max_buffer_size,
    }
    if mcp_servers:
        kwargs["mcp_servers"] = mcp_servers

    return ClaudeAgentOptions(**kwargs)


def chunk(items: list, size: int) -> list[list]:
    """Split a list into chunks of specified size.

    Previously duplicated in entity_resolution, triage, direct_kg, and cold_start.
    """
    return [items[i:i + size] for i in range(0, len(items), size)]


# Default model name constant — all pipeline nodes currently use the same model
DEFAULT_MODEL_NAME = "anthropic/claude-sonnet-4-20250514"


async def query_with_usage(
    prompt: str,
    options: Any,
    node_name: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[str, Any]:
    """Stream a query() call and extract both text and usage metrics.

    Replaces the common pattern of ``async for event in query(...)`` followed
    by text block collection.  Accumulates usage from ALL events that carry a
    ``.usage`` attribute (not just ``ResultMessage``), following the dual-path
    pattern in ``agent.py:510-534``.  This ensures accurate token counts for
    multi-turn nodes like pathway_enrichment (``max_turns=25``).

    Does NOT handle timeouts — nodes that need ``asyncio.wait_for()`` should
    wrap this coroutine themselves, since timeout recovery logic is
    domain-specific.

    Args:
        prompt: The prompt to send to the SDK.
        options: ClaudeAgentOptions instance.
        node_name: Name of the calling graph node (for the usage record).
        model_name: Model identifier string.

    Returns:
        A tuple of ``(text, record)`` where *text* is the joined text content
        and *record* is a ``ModelUsageRecord`` or ``None`` if no usage data
        was found on any event.

    Raises:
        RuntimeError: If the SDK is not available (``HAS_SDK is False``).
    """
    if not HAS_SDK or query is None:
        raise RuntimeError("Claude Agent SDK is not available")

    from .state import ModelUsageRecord

    result_text_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0
    cache_creation_tokens = 0
    cache_read_tokens = 0
    has_usage = False

    async for event in query(prompt=prompt, options=options):
        # Collect text content
        if hasattr(event, "content"):
            for block in event.content:
                if hasattr(block, "text"):
                    result_text_parts.append(block.text)

        # Accumulate usage from ResultMessage (final totals — replaces)
        if isinstance(event, ResultMessage):
            if event.usage:
                usage = event.usage
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
                    cache_read_tokens = usage.get("cache_read_input_tokens", 0)
                else:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)
                    cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0)
                    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0)
                has_usage = True
        # Accumulate usage from other event types (e.g., AssistantMessage)
        elif hasattr(event, "usage") and event.usage:
            usage = event.usage
            if isinstance(usage, dict):
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
                cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
                cache_read_tokens += usage.get("cache_read_input_tokens", 0)
            else:
                input_tokens += getattr(usage, "input_tokens", 0)
                output_tokens += getattr(usage, "output_tokens", 0)
                cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0)
                cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0)
            has_usage = True

    text = "".join(result_text_parts)
    record = None
    if has_usage:
        record = ModelUsageRecord(
            model_name=model_name,
            node_name=node_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )

    return text, record
