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
    from claude_agent_sdk import query, ClaudeAgentOptions  # noqa: F401
    from claude_agent_sdk.types import McpStdioServerConfig  # noqa: F401
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    # Define stubs so importing code doesn't need conditional imports
    query = None  # type: ignore[assignment]
    ClaudeAgentOptions = None  # type: ignore[assignment,misc]
    McpStdioServerConfig = None  # type: ignore[assignment,misc]

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
