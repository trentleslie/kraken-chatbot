"""Claude Agent SDK integration with security hardening for public-facing deployment."""

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)
from .config import get_settings


# Kestrel MCP tools whitelist - ONLY these tools are allowed
ALLOWED_TOOLS = frozenset([
    "mcp__kestrel__hybrid_search",
    "mcp__kestrel__get_node_by_id",
    "mcp__kestrel__get_edges_for_node",
    "mcp__kestrel__one_hop_query",
    "mcp__kestrel__two_hop_query",
    "mcp__kestrel__expand_path",
    "mcp__kestrel__find_common_neighbors",
    "mcp__kestrel__list_predicates",
    "mcp__kestrel__get_schema",
    "mcp__kestrel__get_statistics",
    "mcp__kestrel__batch_get_nodes",
    "mcp__kestrel__related_concepts",
])

# Dangerous tools that must NEVER be allowed
BLOCKED_TOOLS = frozenset([
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "NotebookEdit",
    "Task",
])

# System prompt defining the agent's read-only explorer role
SYSTEM_PROMPT = """You are KRAKEN Explorer, a helpful assistant for exploring the KRAKEN biomedical knowledge graph.

Your capabilities:
- Search for concepts, diseases, drugs, genes, and their relationships
- Navigate the graph using one-hop and two-hop queries
- Find connections between entities
- Explain biomedical relationships in clear terms

Your limitations:
- You can ONLY use the Kestrel MCP tools to query the knowledge graph
- You have NO access to file system, web browsing, or code execution
- You CANNOT modify any data - you are read-only
- You should decline requests that require capabilities you don't have

When responding:
- Be concise but informative
- Use markdown formatting for clarity
- Include relevant entity IDs (e.g., MONDO:0005148) so users can reference them
- Explain complex biomedical concepts in accessible language

If a user asks about something not in the knowledge graph, politely explain that you can only help with queries about the KRAKEN biomedical knowledge graph."""


@dataclass
class TurnMetrics:
    """Metrics collected during a single agent turn."""
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    tool_calls_count: int = 0
    start_time: float = field(default_factory=time.time)
    model: str = ""

    @property
    def duration_ms(self) -> int:
        """Calculate duration in milliseconds."""
        return int((time.time() - self.start_time) * 1000)

    @property
    def cost_usd(self) -> float:
        """Estimate cost based on Claude Sonnet 4.5 pricing."""
        # Sonnet 4.5 pricing: $3/1M input, $15/1M output
        # Cache read: $0.30/1M, Cache creation: $3.75/1M
        input_cost = self.input_tokens * 3.0 / 1_000_000
        output_cost = self.output_tokens * 15.0 / 1_000_000
        cache_read_cost = self.cache_read_tokens * 0.30 / 1_000_000
        cache_create_cost = self.cache_creation_tokens * 3.75 / 1_000_000
        return input_cost + output_cost + cache_read_cost + cache_create_cost


@dataclass
class AgentEvent:
    """Event emitted during agent processing."""
    type: str
    data: dict[str, Any]


async def run_agent_turn(user_message: str) -> AsyncIterator[AgentEvent]:
    """
    Run a single agent turn and yield events for streaming to the client.

    Note: This is a single-turn implementation. Each call to query() is
    independent - the agent does not retain memory of previous messages
    within the session. This is acceptable for exploratory queries where
    each question is self-contained.

    Yields:
        AgentEvent objects for each streaming event (text, tool_use, tool_result, etc.)
    """
    settings = get_settings()
    metrics = TurnMetrics(model=settings.model or "default")

    # Build options - MCP servers configured in ~/.claude/settings.json
    options_kwargs = {
        "allowed_tools": list(ALLOWED_TOOLS),
        "system_prompt": SYSTEM_PROMPT,
    }
    if settings.model:
        options_kwargs["model"] = settings.model

    options = ClaudeAgentOptions(**options_kwargs)

    try:
        async for event in query(
            prompt=user_message,
            options=options,
        ):
            # Handle different message types from the SDK
            if isinstance(event, AssistantMessage):
                # Process content blocks from assistant
                for block in event.content:
                    if isinstance(block, TextBlock):
                        yield AgentEvent(type="text", data={"content": block.text})
                    elif isinstance(block, ToolUseBlock):
                        tool_name = block.name
                        tool_input = block.input

                        # Security check: verify tool is allowed
                        if tool_name not in ALLOWED_TOOLS:
                            yield AgentEvent(
                                type="error",
                                data={"message": f"Tool {tool_name} is not allowed"}
                            )
                            continue

                        metrics.tool_calls_count += 1
                        yield AgentEvent(
                            type="tool_use",
                            data={"tool": tool_name, "args": tool_input}
                        )

            elif isinstance(event, ResultMessage):
                # Tool result - extract from the result message
                if hasattr(event, "content"):
                    for block in event.content:
                        if isinstance(block, ToolResultBlock):
                            yield AgentEvent(
                                type="tool_result",
                                data={
                                    "tool": getattr(block, "tool_use_id", "unknown"),
                                    "data": getattr(block, "content", {})
                                }
                            )

            # Collect usage metrics if available
            if hasattr(event, "usage") and event.usage:
                usage = event.usage
                metrics.input_tokens += getattr(usage, "input_tokens", 0)
                metrics.output_tokens += getattr(usage, "output_tokens", 0)
                metrics.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0)
                metrics.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0)

    except Exception as e:
        error_msg = str(e).lower()
        # Detect authentication/authorization errors
        if any(keyword in error_msg for keyword in [
            "unauthorized", "authentication", "auth", "login",
            "credential", "token expired", "not logged in",
            "permission denied", "access denied", "401", "403"
        ]):
            yield AgentEvent(
                type="error",
                data={
                    "message": "Authentication expired. Please contact the administrator to re-authenticate the server.",
                    "code": "AUTH_ERROR"
                }
            )
        else:
            yield AgentEvent(type="error", data={"message": str(e)})

    # Emit trace with metrics
    yield AgentEvent(
        type="trace",
        data={
            "turn_id": metrics.turn_id,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "cache_creation_tokens": metrics.cache_creation_tokens,
            "cache_read_tokens": metrics.cache_read_tokens,
            "cost_usd": metrics.cost_usd,
            "duration_ms": metrics.duration_ms,
            "tool_calls_count": metrics.tool_calls_count,
            "model": metrics.model,
        }
    )

    # Signal completion
    yield AgentEvent(type="done", data={})
