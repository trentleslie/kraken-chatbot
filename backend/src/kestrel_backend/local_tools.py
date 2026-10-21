"""Local SDK MCP Tools for data analysis.

These tools run in-process and don't require external services.
They provide safe data processing capabilities without code execution.
"""

import os

import anthropic
from claude_agent_sdk import tool, create_sdk_mcp_server
from .byok import current_api_key
from .config import get_settings
from .graph.sdk_utils import create_agent_options, query_with_usage


# Anthropic client for the analyze tool (None = ambient-env key, i.e. no BYOK active)
_anthropic_client = None


def _get_anthropic_client():
    """Get or create the Anthropic client.

    If a per-request BYOK key is set on the contextvar, creates a fresh client
    scoped to that key. Otherwise returns the cached ambient-env client so that
    trusted / server-key callers don't allocate a new client on every call.
    """
    key = current_api_key.get()
    if key:
        return anthropic.Anthropic(api_key=key)
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


@tool(
    "analyze_results",
    "Analyze and summarize query results. Use this to find patterns, compare datasets, identify overlaps, or extract key insights from knowledge graph query results.",
    {
        "data": str,  # JSON string of the data to analyze
        "task": str,  # What analysis to perform (e.g., "find common diseases", "summarize drugs")
    }
)
async def analyze_results(args):
    """Use Claude to analyze query results without code execution."""
    data = args.get("data", "")
    task = args.get("task", "Summarize this data")

    # Truncate data if too large
    if len(data) > 50000:
        data = data[:50000] + "\n... [truncated]"

    prompt = f"""Analyze the following biomedical knowledge graph data and {task}.

Data:
{data}

Provide a clear, structured analysis. Use markdown formatting. Include relevant entity IDs."""

    try:
        if current_api_key.get() is None and not os.getenv("ANTHROPIC_API_KEY"):
            # SDK-login mode (trusted user, no API key anywhere): the Anthropic client
            # can't use a Claude Code login, so run the analysis through the Agent SDK.
            options = create_agent_options(
                system_prompt="You are a biomedical data analyst.", allowed_tools=[], max_turns=1
            )
            result_text, _ = await query_with_usage(prompt, options, node_name="analyze_results")
            result_text = result_text or "No analysis generated."
            return {"content": [{"type": "text", "text": result_text}]}

        client = _get_anthropic_client()
        response = client.messages.create(
            model=get_settings().model,
            # Opus 5.5 always thinks and thinking counts against max_tokens, so leave
            # headroom; low effort keeps this summarization helper cheap.
            max_tokens=16000,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )

        # Thinking blocks precede the answer, so collect only the text blocks.
        result_text = "".join(b.text for b in response.content if b.type == "text")
        result_text = result_text or "No analysis generated."
        return {"content": [{"type": "text", "text": result_text}]}

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Analysis error: {str(e)}"}],
            "isError": True
        }


def create_local_mcp_server():
    """Create an SDK MCP server with local analysis tools."""
    return create_sdk_mcp_server(
        name="local",
        version="1.0.0",
        tools=[
            analyze_results,
        ]
    )
