"""Claude Agent SDK integration with security hardening for public-facing deployment."""

import json
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langfuse import get_client

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    UserMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    HookMatcher,
)
from claude_agent_sdk.types import McpStdioServerConfig
from .config import get_settings
from .bash_sandbox import bash_security_hook


# Langfuse client (lazy initialized)
_langfuse = None  # Type annotation removed since get_client() return type differs


def _get_langfuse():
    """Get or create Langfuse client for observability tracing.

    SDK v3 reads credentials from environment variables:
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST
    """
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    settings = get_settings()
    if settings.langfuse_enabled and settings.langfuse_public_key and settings.langfuse_secret_key:
        # get_client() reads from LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST env vars
        _langfuse = get_client()
    return _langfuse


# Tools whitelist - these tools are allowed
# Tool names follow MCP naming convention: mcp__<server>__<tool>
ALLOWED_TOOLS = frozenset([
    # Kestrel knowledge graph tools (via stdio proxy)
    "mcp__kestrel__one_hop_query",
    "mcp__kestrel__text_search",
    "mcp__kestrel__vector_search",
    "mcp__kestrel__similar_nodes",
    "mcp__kestrel__hybrid_search",
    "mcp__kestrel__get_nodes",
    "mcp__kestrel__get_edges",
    "mcp__kestrel__get_valid_categories",
    "mcp__kestrel__get_valid_predicates",
    "mcp__kestrel__get_valid_prefixes",
    "mcp__kestrel__health_check",
    # Multi-hop graph reasoning tools
    "mcp__kestrel__guilt_by_association",
    "mcp__kestrel__missing_edge_prediction",
    "mcp__kestrel__pathway_enrichment",
    "mcp__kestrel__novelty_score",
    "mcp__kestrel__gap_analysis",
    # Development/testing tools (sandboxed)
    "Bash",
    # Task tracking
    "TodoWrite",
])


def _get_kestrel_mcp_config() -> McpStdioServerConfig:
    """Get the Kestrel MCP server configuration using stdio proxy."""
    # Get the path to the mcp_proxy module
    backend_dir = Path(__file__).parent.parent.parent
    proxy_module = "src.kestrel_backend.mcp_proxy"

    # Pass environment variables to the proxy
    env = {
        "KESTREL_API_KEY": os.getenv("KESTREL_API_KEY", ""),
        "KESTREL_MCP_URL": os.getenv("KESTREL_MCP_URL", "https://kestrel.nathanpricelab.com/mcp"),
    }

    return McpStdioServerConfig(
        type="stdio",
        command=sys.executable,  # Use the current Python interpreter
        args=["-m", proxy_module],
        env=env,
    )

# Dangerous tools that should be blocked in production
# NOTE: Bash temporarily removed for testing
BLOCKED_TOOLS = frozenset([
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

# System prompt defining the agent's explorer role
SYSTEM_PROMPT = """You are KRAKEN Explorer, a helpful assistant for exploring the KRAKEN biomedical knowledge graph.

Your capabilities:
- Search for concepts, diseases, drugs, genes, and their relationships using Kestrel MCP tools
- Navigate the graph using one-hop queries to find connections
- Use Bash for LOCAL data processing and analysis when needed
- Explain biomedical relationships in clear terms

Available tools:
- text_search, vector_search, hybrid_search: Find entities by name/description
- one_hop_query: Find connected entities (diseases, drugs, genes, etc.)
- get_nodes, get_edges: Get detailed information about specific entities
- similar_nodes: Find semantically similar entities
- get_valid_categories, get_valid_predicates, get_valid_prefixes: Query metadata
- Bash: Run shell commands for LOCAL data processing only (30s timeout).
  ALLOWED: jq, python, python3, grep, awk, sed, sort, uniq, wc, head, tail, cat, ls, find, echo
  BLOCKED: ALL network tools (curl, wget, nc, ssh), rm, sudo, system commands
  NOTE: For network/external data, use the Kestrel MCP tools instead.

Multi-hop graph reasoning tools:
- guilt_by_association: Find structurally similar entities via shared neighbors
- missing_edge_prediction: Predict novel associations (2-hop reachable, no direct edge)
- pathway_enrichment: Find shared biological themes across entity sets
- novelty_score: Triage entities by characterization level (well-characterized/moderate/sparse)
- gap_analysis: Find "expected but missing" entities sharing neighbors with input set

## Analytical Workflow for Entity Lists

When a user provides a list of analytes, metabolites, proteins, genes, or other biological entities for analysis, follow this systematic workflow:

### Step 1: Triage with novelty_score
Run `novelty_score` on the full list to classify each entity as well-characterized, moderate, or sparse. This determines your strategy for each entity.

### Step 2: Pathway enrichment
Run `pathway_enrichment` on the full list to find shared biological themes — genes, pathways, diseases, and phenotypes that connect multiple entities in the list. This is the foundation of your narrative.

### Step 3: Direct associations for well-characterized entities
Use `hybrid_search` and `one_hop_query` on well-characterized entities to retrieve known disease associations, pathway memberships, and literature-supported relationships.

### Step 4: Inference for sparse entities
For entities classified as sparse or moderate by novelty_score:
- Use `guilt_by_association` to find structurally similar well-characterized entities
- Use `missing_edge_prediction` with target_category="biolink:Disease" to predict novel disease associations via 2-hop paths
- Clearly label these as **graph-structural inferences**, not established findings

### Step 5: Gap analysis
Run `gap_analysis` on the full list with category="biolink:SmallMolecule" to identify metabolites that would be expected alongside this panel but are absent. These are candidates for follow-up measurement.

### Step 6: Synthesis
Combine all findings into a narrative that:
- Groups entities by biological theme (from pathway_enrichment)
- Presents established associations (from direct lookups) separately from novel predictions (from inference tools)
- Proposes 2-3 biological hypotheses explaining the overall pattern
- Identifies which entities support each hypothesis and which don't fit
- Lists expected-but-absent entities that would confirm or refute each hypothesis
- Recommends specific follow-up analyses

Always be transparent about evidence quality: direct KG edges with publication provenance > multi-source KG edges > graph-structural inferences.

IMPORTANT:
- Use Kestrel MCP tools for knowledge graph queries
- Use Bash ONLY for local data processing (parsing JSON, text manipulation)
- Do NOT attempt network operations via Bash - they will be blocked
- For complex queries requiring multiple genes/entities, you can use Bash with jq or python to find overlaps

When responding:
- Be concise but informative
- Use markdown formatting (tables, lists) for clarity
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

    # Start Langfuse trace for observability
    langfuse = _get_langfuse()
    trace = None
    if langfuse:
        trace = langfuse.start_span(
            name="kraken_agent_turn",
            input={"user_message": user_message},
            metadata={"turn_id": metrics.turn_id, "model": metrics.model},
        )

    # Configure Kestrel MCP server via stdio proxy
    # The proxy handles Kestrel's non-standard MCP-over-HTTP protocol
    kestrel_config = _get_kestrel_mcp_config()

    # Build options with Kestrel MCP server and Bash security hook
    options_kwargs = {
        "allowed_tools": list(ALLOWED_TOOLS),
        "system_prompt": SYSTEM_PROMPT,
        "mcp_servers": {
            "kestrel": kestrel_config,
        },
        # Security: Validate all Bash commands before execution
        "hooks": {
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[bash_security_hook])
            ]
        },
    }
    if settings.model:
        options_kwargs["model"] = settings.model

    options = ClaudeAgentOptions(**options_kwargs)

    # Track tool_use_id -> tool_name mapping for matching results
    tool_id_to_name: dict[str, str] = {}
    # Track Langfuse tool spans for correlation
    tool_spans: dict[str, Any] = {}

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
                        tool_id = getattr(block, "id", None)

                        # Security check: verify tool is allowed
                        if tool_name not in ALLOWED_TOOLS:
                            yield AgentEvent(
                                type="error",
                                data={"message": f"Tool {tool_name} is not allowed"}
                            )
                            continue

                        # Track the mapping for later result matching
                        if tool_id:
                            tool_id_to_name[tool_id] = tool_name
                            # Create Langfuse child span for tool call
                            if trace:
                                tool_spans[tool_id] = trace.start_span(
                                    name=f"tool:{tool_name}",
                                    input=tool_input,
                                )

                        metrics.tool_calls_count += 1
                        yield AgentEvent(
                            type="tool_use",
                            data={"tool": tool_name, "args": tool_input}
                        )

            elif isinstance(event, UserMessage):
                # Tool results come in UserMessage with ToolResultBlock
                if hasattr(event, "content"):
                    for block in event.content:
                        if isinstance(block, ToolResultBlock):
                            tool_use_id = getattr(block, "tool_use_id", None)
                            content = getattr(block, "content", "")

                            # Look up the tool name from our mapping
                            tool_name = tool_id_to_name.get(tool_use_id, "unknown")

                            # Parse JSON content if it's a string
                            if isinstance(content, str):
                                try:
                                    content = json.loads(content)
                                except json.JSONDecodeError:
                                    # Keep as string wrapped in dict
                                    content = {"raw": content}

                            # Ensure content is a dict
                            if not isinstance(content, dict):
                                content = {"result": content}

                            # End Langfuse span for this tool
                            if tool_use_id and tool_use_id in tool_spans:
                                tool_span = tool_spans[tool_use_id]
                                tool_span.update(output=content)
                                tool_span.end()

                            yield AgentEvent(
                                type="tool_result",
                                data={
                                    "tool": tool_name,
                                    "data": content
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

    # Finalize Langfuse trace with collected metrics
    if trace and langfuse:
        trace.update(
            output={"turn_id": metrics.turn_id},
            usage={
                "input": metrics.input_tokens,
                "output": metrics.output_tokens,
                "total": metrics.input_tokens + metrics.output_tokens,
            },
            metadata={
                "cost_usd": metrics.cost_usd,
                "duration_ms": metrics.duration_ms,
                "tool_calls_count": metrics.tool_calls_count,
                "cache_creation_tokens": metrics.cache_creation_tokens,
                "cache_read_tokens": metrics.cache_read_tokens,
            },
        )
        trace.end()  # SDK v3 requires explicit end() for manually created spans
        langfuse.flush()  # Ensure trace is sent before response completes

    # Signal completion
    yield AgentEvent(type="done", data={})
