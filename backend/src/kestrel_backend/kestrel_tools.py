"""SDK MCP Tools that proxy to the Kestrel MCP server.

These tools use the claude-agent-sdk's in-process MCP server feature to expose
Kestrel's knowledge graph tools to Claude. Each tool wraps a call to the
Kestrel MCP server via our HTTP client.

Note: This file is currently unused - we use the stdio proxy approach instead.
It's kept for reference and potential future use with the SDK MCP server.
"""

from claude_agent_sdk import tool, create_sdk_mcp_server
from .kestrel_client import call_kestrel_tool


# Tool definitions matching Kestrel's actual available tools
# Each tool proxies the call to the Kestrel MCP server


@tool(
    "one_hop_query",
    "Perform a one-hop query to find connected nodes in the knowledge graph",
    {
        "start_node_ids": str,  # Single CURIE or list
        "direction": str,       # 'forward', 'reverse', or 'both'
        "predicate_filter": str,
        "end_node_category": str,
        "end_node_ids": str,
        "mode": str,            # 'slim', 'full', or 'preview'
        "limit": int,
    }
)
async def one_hop_query(args):
    """Traverse one hop from start nodes in the knowledge graph."""
    return await call_kestrel_tool("one_hop_query", args)


@tool(
    "text_search",
    "Search for nodes by text in names, synonyms, and descriptions",
    {
        "search_text": str,
        "limit": int,
        "category_filter": str,
        "prefix_filter": str,
        "truncate_long_fields": bool,
    }
)
async def text_search(args):
    """Search for nodes using text matching."""
    return await call_kestrel_tool("text_search", args)


@tool(
    "vector_search",
    "Search for nodes semantically similar to the search text using vector embeddings",
    {
        "search_text": str,
        "limit": int,
        "category_filter": str,
        "prefix_filter": str,
        "truncate_long_fields": bool,
    }
)
async def vector_search(args):
    """Search for nodes using vector similarity."""
    return await call_kestrel_tool("vector_search", args)


@tool(
    "similar_nodes",
    "Find nodes semantically similar to a given node by its ID",
    {
        "node_id": str,
        "limit": int,
        "category_filter": str,
        "prefix_filter": str,
        "truncate_long_fields": bool,
    }
)
async def similar_nodes(args):
    """Find nodes similar to a given node."""
    return await call_kestrel_tool("similar_nodes", args)


@tool(
    "hybrid_search",
    "Search for nodes using a hybrid approach combining text and vector search",
    {
        "search_text": str,
        "limit": int,
        "category_filter": str,
        "prefix_filter": str,
        "truncate_long_fields": bool,
    }
)
async def hybrid_search(args):
    """Search using combined text and vector matching."""
    return await call_kestrel_tool("hybrid_search", args)


@tool(
    "get_nodes",
    "Retrieve detailed node information for given CURIE identifier(s)",
    {
        "curies": str,  # Single CURIE or list
        "truncate_long_fields": bool,
    }
)
async def get_nodes(args):
    """Get detailed information about specific nodes."""
    return await call_kestrel_tool("get_nodes", args)


@tool(
    "get_edges",
    "Retrieve detailed edge information for given edge ID(s)",
    {"edge_ids": str}  # Single edge ID or list
)
async def get_edges(args):
    """Get detailed information about specific edges."""
    return await call_kestrel_tool("get_edges", args)


@tool(
    "get_valid_categories",
    "Get all valid biolink category values that can be used in queries",
    {}
)
async def get_valid_categories(args):
    """List valid node categories."""
    return await call_kestrel_tool("get_valid_categories", args)


@tool(
    "get_valid_predicates",
    "Get all valid biolink predicate values that can be used in one-hop queries",
    {}
)
async def get_valid_predicates(args):
    """List valid relationship predicates."""
    return await call_kestrel_tool("get_valid_predicates", args)


@tool(
    "get_valid_prefixes",
    "Get all valid CURIE prefixes with example identifiers",
    {}
)
async def get_valid_prefixes(args):
    """List valid CURIE prefixes."""
    return await call_kestrel_tool("get_valid_prefixes", args)


@tool(
    "health_check",
    "Check if the Kestrel API service is healthy and accessible",
    {}
)
async def health_check(args):
    """Check Kestrel service health."""
    return await call_kestrel_tool("health_check", args)


# Create the SDK MCP server with all Kestrel tools
# Note: analyze_results is in local_tools.py, not here (it runs locally, not via Kestrel)
def create_kestrel_mcp_server():
    """Create an SDK MCP server with all Kestrel tools."""
    return create_sdk_mcp_server(
        name="kestrel",
        version="1.0.0",
        tools=[
            one_hop_query,
            text_search,
            vector_search,
            similar_nodes,
            hybrid_search,
            get_nodes,
            get_edges,
            get_valid_categories,
            get_valid_predicates,
            get_valid_prefixes,
            health_check,
        ]
    )
