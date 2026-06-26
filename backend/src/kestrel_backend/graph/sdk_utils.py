"""Shared SDK utilities for discovery pipeline nodes.

Centralizes the Claude Agent SDK import pattern, MCP configuration factory,
agent options factory, and the chunk() utility that were previously duplicated
across 8 pipeline nodes.

Semaphore definitions and fallback orchestration logic remain per-node
(intentionally different values and patterns).
"""

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

# Optional pipeline-wide model override. When KRAKEN_PIPELINE_MODEL is set (e.g.
# "claude-opus-4-8"), every SDK-backed node runs on that model AND the usage label /
# cost estimate attribute to it. Unset -> SDK default model + the Sonnet label below.
_PIPELINE_MODEL = os.getenv("KRAKEN_PIPELINE_MODEL")

logger = logging.getLogger(__name__)


# Langfuse client (lazy, guarded) — mirrors agent.py:_get_langfuse(). Resolved once: a
# disabled/keyless environment caches None so we never re-attempt get_client() per call.
_langfuse = None
_langfuse_resolved = False


def _get_langfuse():
    """Return a Langfuse v3 client if observability is enabled and keys are present, else None.

    Observability must never break the pipeline, so any failure resolves to None.
    """
    global _langfuse, _langfuse_resolved
    if _langfuse_resolved:
        return _langfuse
    _langfuse_resolved = True
    try:
        from ..config import get_settings

        settings = get_settings()
        if (
            settings.langfuse_enabled
            and settings.langfuse_public_key
            and settings.langfuse_secret_key
        ):
            from langfuse import get_client

            _langfuse = get_client()
    except Exception:  # pragma: no cover - defensive: never let tracing setup break a run
        _langfuse = None
    return _langfuse


def reset_langfuse_singleton() -> None:
    """Clear the resolved-once Langfuse client so the next call re-reads settings.

    The client is resolved once and cached (a disabled/keyless env caches None). That
    cache is independent of ``get_settings``'s LRU cache, so any caller that flips
    ``LANGFUSE_*`` at runtime (e.g. ``assessment/runner.py`` forcing
    ``LANGFUSE_ENABLED=false``) must reset BOTH — call this alongside
    ``get_settings.cache_clear()`` or a stale client survives the override.
    """
    global _langfuse, _langfuse_resolved
    _langfuse = None
    _langfuse_resolved = False


# Centralized SDK availability check — single try/except for all nodes
try:
    from claude_agent_sdk import (  # noqa: F401
        query,
        ClaudeAgentOptions,
        ResultMessage,
        SystemMessage,
        ToolUseBlock,
    )
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    # Define stubs so importing code doesn't need conditional imports
    query = None  # type: ignore[assignment]
    ClaudeAgentOptions = None  # type: ignore[assignment,misc]

    # Sentinel classes so isinstance(event, X) returns False rather than raising
    # TypeError when the SDK is unavailable (issue #44 instrumentation).
    class _ResultMessageStub:  # type: ignore[no-redef]
        pass
    ResultMessage = _ResultMessageStub  # type: ignore[assignment,misc]

    class _SystemMessageStub:  # type: ignore[no-redef]
        pass
    SystemMessage = _SystemMessageStub  # type: ignore[assignment,misc]

    class _ToolUseBlockStub:  # type: ignore[no-redef]
        pass
    ToolUseBlock = _ToolUseBlockStub  # type: ignore[assignment,misc]

# NOTE (#61): the Kestrel *stdio* MCP config (get_kestrel_mcp_config, KESTREL_COMMAND,
# KESTREL_ARGS, and the McpStdioServerConfig re-export) was removed. It pointed at
# `uvx mcp-client-kestrel`, a package that does not exist on PyPI, so the subprocess
# could never launch. All discovery-pipeline SDK nodes now use HTTP data-in-prompt
# inference (allowed_tools=[]). Classic-mode agent.py keeps its own _get_kestrel_mcp_config.


@dataclass(frozen=True)
class McpDegradationVerdict:
    """Result of classify_mcp_degradation (issue #44)."""

    degraded: bool
    reason: str
    confidence: str  # "definitive" | "high" | "structural" | "none"


# Fallback phrases the SDK model emits when MCP tools failed to register. These are a
# CORROBORATOR only — never the sole trigger (model wording is brittle). Issue #44.
_MCP_FALLBACK_PHRASES = (
    "not available in my current tool set",
    "tools are not available",
    "are not available in my",
)


def classify_mcp_degradation(
    expected_tools: list[str],
    mcp_tool_calls: int,
    result_text: str = "",
    available_tools: list[str] | None = None,
) -> McpDegradationVerdict:
    """Detect the "MCP tools unavailable -> hallucinated output" condition (issue #44).

    Structural signals are authoritative; the fallback phrase only raises confidence.
    A node that expects no MCP tools (``expected_tools == []``, e.g. data-in-prompt
    inference) can never be MCP-degraded.

    NOTE: the bare ``mcp_tool_calls == 0`` signal is only safe for nodes whose prompt
    *mandates* tool use (e.g. pathway_enrichment requires a one_hop_query per entity).
    A node where zero tool calls is a legitimate model choice must require the init-list
    signal or the corroborating phrase instead of the bare count.
    """
    if not expected_tools:
        return McpDegradationVerdict(False, "no_tools_expected", "none")

    phrase_hit = bool(result_text) and any(
        p in result_text.lower() for p in _MCP_FALLBACK_PHRASES
    )

    # Definitive: the SDK init tool list is known and is missing an expected tool.
    if available_tools is not None:
        missing = [t for t in expected_tools if t not in available_tools]
        if missing:
            return McpDegradationVerdict(
                True, f"tools_missing_from_init:{','.join(missing)}", "definitive"
            )
        # All expected tools confirmed present in the init list. Zero calls WITHOUT the
        # fallback phrase is not structural proof of degradation (the model may have
        # legitimately skipped tools) — avoid a false positive for non-mandating nodes.
        # The phrase still trips the structural check below.
        if mcp_tool_calls == 0 and not phrase_hit:
            return McpDegradationVerdict(False, "tools_registered_zero_calls", "none")

    # Structural: tools were expected but the model made zero MCP tool calls.
    if mcp_tool_calls == 0:
        if phrase_hit:
            return McpDegradationVerdict(True, "zero_mcp_tool_calls+phrase", "high")
        return McpDegradationVerdict(True, "zero_mcp_tool_calls", "structural")

    # Tools were used -> healthy, even if the analysis found nothing.
    return McpDegradationVerdict(False, "tools_used", "none")


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
    model: str | None = None,
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
    # Per-call model wins; otherwise the pipeline-wide override (KRAKEN_PIPELINE_MODEL);
    # otherwise omit `model` so the SDK uses its default.
    effective_model = model or _PIPELINE_MODEL
    if effective_model:
        kwargs["model"] = effective_model

    return ClaudeAgentOptions(**kwargs)


def chunk(items: list, size: int) -> list[list]:
    """Split a list into chunks of specified size.

    Previously duplicated in entity_resolution, triage, direct_kg, and cold_start.
    """
    return [items[i:i + size] for i in range(0, len(items), size)]


# Default model name constant — all pipeline nodes use the same model. Follows the
# KRAKEN_PIPELINE_MODEL override so the usage label + cost estimate match the model the
# SDK is actually told to use (create_agent_options sets the same override).
DEFAULT_MODEL_NAME = _PIPELINE_MODEL or "anthropic/claude-sonnet-4-20250514"


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
    mcp_tool_calls = 0  # count of mcp__* tool-use blocks (issue #44 degradation signal)
    available_tools: list[str] | None = None  # tool names from the SDK init event, if exposed

    # Wrap the LLM call in a Langfuse generation. When the CallbackHandler has set a node
    # span as the active OTel context (pipeline runs), this generation auto-nests under that
    # node span — no parent/trace-id threading. When Langfuse is disabled, this is a no-op
    # nullcontext yielding None. Creating it inside this coroutine (and inside any asyncio
    # task spawned from it, which copies the contextvars context) keeps nesting correct.
    langfuse = _get_langfuse()
    generation_cm = (
        langfuse.start_as_current_generation(
            name=f"llm:{node_name}", model=model_name, input=prompt
        )
        if langfuse is not None
        else nullcontext()
    )

    with generation_cm as generation:
        async for event in query(prompt=prompt, options=options):
            # Capture the available-tool list from the SDK init event (best-effort;
            # stays None if this SDK version/stream doesn't expose it). Issue #44.
            if isinstance(event, SystemMessage) and getattr(event, "subtype", None) == "init":
                data = getattr(event, "data", None)
                tools = data.get("tools") if isinstance(data, dict) else None
                if isinstance(tools, list):
                    available_tools = [t for t in tools if isinstance(t, str)]

            # Collect text content; count MCP tool-use blocks (issue #44).
            if hasattr(event, "content") and event.content:
                for block in event.content:
                    if hasattr(block, "text"):
                        result_text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock) and getattr(block, "name", "").startswith("mcp__"):
                        mcp_tool_calls += 1

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

        # Diagnostic line so degraded MCP runs are observable across SDK nodes (issue #44).
        logger.info(
            "%s SDK call: mcp_tool_calls=%d, available_tools=%s",
            node_name,
            mcp_tool_calls,
            "unknown" if available_tools is None else f"{len(available_tools)} registered",
        )

        record = None
        if has_usage or mcp_tool_calls or available_tools is not None:
            record = ModelUsageRecord(
                model_name=model_name,
                node_name=node_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
                mcp_tool_calls=mcp_tool_calls,
                available_tools=available_tools,
            )

        # Record LLM I/O + token usage on the generation so Langfuse can show latency,
        # tokens, and (for a recognized model) cost. usage_details keys follow Langfuse's
        # Anthropic convention; cost auto-computes from model + usage when the model is known.
        if generation is not None:
            generation.update(
                output=text,
                usage_details={
                    "input": input_tokens,
                    "output": output_tokens,
                    "cache_read_input_tokens": cache_read_tokens,
                    "cache_creation_input_tokens": cache_creation_tokens,
                },
                metadata={
                    "mcp_tool_calls": mcp_tool_calls,
                    "available_tools": (
                        None if available_tools is None else len(available_tools)
                    ),
                },
            )

    return text, record
