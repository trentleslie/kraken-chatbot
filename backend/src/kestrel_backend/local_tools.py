"""Local SDK MCP Tools for data analysis.

These tools run in-process and don't require external services.
They provide safe data processing capabilities without code execution.
"""

import anthropic
from claude_agent_sdk import tool, create_sdk_mcp_server


# Anthropic client for the analyze tool
_anthropic_client = None


def _get_anthropic_client():
    """Get or create the Anthropic client."""
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

    try:
        client = _get_anthropic_client()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze the following biomedical knowledge graph data and {task}.

Data:
{data}

Provide a clear, structured analysis. Use markdown formatting. Include relevant entity IDs."""
                }
            ]
        )

        result_text = response.content[0].text if response.content else "No analysis generated."
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
