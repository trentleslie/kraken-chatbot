"""Kestrel MCP Client - HTTP client for the KRAKEN knowledge graph.

Kestrel uses a non-standard MCP-over-HTTP implementation where:
- Requests are sent via POST with JSON-RPC messages
- Responses are returned in SSE format (event: message\ndata: ...)

This client handles this protocol directly without using the mcp library's SSE client.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

KESTREL_MCP_URL = os.getenv("KESTREL_MCP_URL", "https://kestrel.nathanpricelab.com/mcp")
KESTREL_API_KEY = os.getenv("KESTREL_API_KEY", "")


def _get_headers() -> dict[str, str]:
    """Build headers for Kestrel requests."""
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    # Read API key at request time (not module import time) to handle
    # cases where env vars are set after module is first loaded
    api_key = os.getenv("KESTREL_API_KEY", "") or KESTREL_API_KEY
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def parse_sse_response(text: str) -> dict:
    """Parse SSE-formatted response from Kestrel.

    Kestrel returns responses in SSE format:
    event: message
    data: {"jsonrpc":"2.0",...}
    """
    # Find the data line
    for line in text.strip().split("\n"):
        if line.startswith("data: "):
            return json.loads(line[6:])

    # If no data line, try parsing directly as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse Kestrel response: {text[:200]}")


class KestrelClient:
    """Client for interacting with the Kestrel MCP server via HTTP."""

    def __init__(self):
        self._http_client: httpx.AsyncClient | None = None
        self._tools: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._request_id = 0
        self._session_id: str | None = None

    def _next_request_id(self) -> int:
        """Generate next request ID."""
        self._request_id += 1
        return self._request_id

    async def _ensure_client(self):
        """Ensure HTTP client is created with connection pooling."""
        if self._http_client is None:
            headers = _get_headers()
            has_api_key = "X-API-Key" in headers
            logger.info(
                "Creating HTTP client with headers: %s, has_api_key=%s",
                list(headers.keys()), has_api_key
            )

            # Configure connection pooling for improved performance
            # - max_keepalive_connections: Keep up to 20 idle connections alive
            # - max_connections: Allow up to 100 total connections
            # - keepalive_expiry: Keep connections alive for 30 seconds
            limits = httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0,
            )

            # Check if h2 package is available for HTTP/2 support
            try:
                import h2  # noqa: F401
                http2_enabled = True
            except ImportError:
                http2_enabled = False
                logger.debug("h2 package not installed, HTTP/2 support disabled")

            self._http_client = httpx.AsyncClient(
                timeout=60.0,
                headers=headers,
                limits=limits,
                http2=http2_enabled,  # Enable HTTP/2 for multiplexing (if h2 package available)
            )

    async def _send_request(self, method: str, params: dict | None = None, _retry: bool = True) -> dict:
        """Send a JSON-RPC request to Kestrel.

        Automatically recovers from expired sessions by re-establishing the connection.
        """
        await self._ensure_client()

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._next_request_id(),
        }
        if params:
            request["params"] = params

        # Include session ID in headers if we have one
        headers = {}
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        try:
            response = await self._http_client.post(KESTREL_MCP_URL, json=request, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Check for stale session error — recover and retry once
            if e.response.status_code == 400 and _retry:
                error_text = e.response.text
                if "session" in error_text.lower() or "No valid session" in error_text:
                    logger.warning("Kestrel session expired, re-establishing connection...")
                    # Clear session state
                    self._session_id = None
                    self._initialized = False
                    # Re-initialize connection
                    await self.connect()
                    # Retry once with new session
                    return await self._send_request(method, params, _retry=False)
            raise

        # Capture session ID from response headers
        if "mcp-session-id" in response.headers:
            self._session_id = response.headers["mcp-session-id"]

        return parse_sse_response(response.text)

    async def _send_notification(self, method: str, params: dict | None = None):
        """Send a JSON-RPC notification (no response expected)."""
        await self._ensure_client()

        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        # Include session ID in headers if we have one
        headers = {}
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        await self._http_client.post(KESTREL_MCP_URL, json=notification, headers=headers)

    async def connect(self):
        """Initialize connection to the Kestrel MCP server."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            logger.info("Connecting to Kestrel MCP server...")

            # Initialize the session
            init_result = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "kestrel-backend", "version": "1.0.0"},
            })

            if "error" in init_result:
                raise Exception(f"Failed to initialize: {init_result['error']}")

            logger.info(f"Connected to Kestrel: {init_result.get('result', {}).get('serverInfo', {})}")

            # Send initialized notification (required by MCP protocol)
            await self._send_notification("notifications/initialized")

            # List available tools
            tools_result = await self._send_request("tools/list")

            if "result" in tools_result and "tools" in tools_result["result"]:
                for tool in tools_result["result"]["tools"]:
                    self._tools[tool["name"]] = {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "inputSchema": tool.get("inputSchema", {}),
                    }

            logger.info(f"Available Kestrel tools: {list(self._tools.keys())}")
            self._initialized = True

    async def disconnect(self):
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._initialized = False

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the Kestrel MCP server."""
        if not self._initialized:
            await self.connect()

        try:
            result = await self._send_request("tools/call", {
                "name": name,
                "arguments": arguments,
            })

            if "error" in result:
                return {
                    "content": [{"type": "text", "text": f"Error: {result['error']}"}],
                    "isError": True,
                }

            # Extract content from result
            if "result" in result:
                content = result["result"].get("content", [])

                # Format content for SDK
                formatted_content = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text = item["text"]
                        # Pretty-print JSON if it looks like JSON
                        try:
                            if isinstance(text, str) and text.strip().startswith(("{", "[")):
                                parsed = json.loads(text)
                                formatted_content.append({
                                    "type": "text",
                                    "text": json.dumps(parsed, indent=2)
                                })
                            else:
                                formatted_content.append({"type": "text", "text": str(text)})
                        except json.JSONDecodeError:
                            formatted_content.append({"type": "text", "text": str(text)})
                    else:
                        formatted_content.append({"type": "text", "text": str(item)})

                # Check for embedded errors in response content
                # Kestrel sometimes returns auth errors as content with isError=false
                if formatted_content:
                    first_text = formatted_content[0].get("text", "")
                    try:
                        embedded = json.loads(first_text)
                        if isinstance(embedded, dict) and embedded.get("error"):
                            logger.warning(
                                "Kestrel tool %s returned embedded error: %s",
                                name, embedded.get("message", "unknown")
                            )
                            return {
                                "content": formatted_content,
                                "isError": True,  # Mark as error for callers
                            }
                    except (json.JSONDecodeError, TypeError):
                        pass

                return {
                    "content": formatted_content,
                    "isError": result["result"].get("isError", False),
                }

            return {"content": [{"type": "text", "text": str(result)}]}

        except Exception as e:
            logger.error(f"Error calling Kestrel tool {name}: {e}")
            return {
                "content": [{"type": "text", "text": f"Error calling {name}: {str(e)}"}],
                "isError": True,
            }

    def get_tools(self) -> dict[str, dict]:
        """Get the list of available tools."""
        return self._tools.copy()


# Global client instance
_client: KestrelClient | None = None


async def get_kestrel_client() -> KestrelClient:
    """Get or create the global Kestrel client."""
    global _client
    if _client is None:
        _client = KestrelClient()
    if not _client._initialized:
        await _client.connect()
    return _client


async def call_kestrel_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a Kestrel tool. Ensures connection is established."""
    client = await get_kestrel_client()
    return await client.call_tool(name, arguments)


async def check_kestrel_health() -> tuple[bool, float | None, str | None]:
    """Check Kestrel MCP server health.

    Returns:
        tuple: (is_healthy, latency_ms, error_message)
    """
    import time
    start = time.time()

    try:
        # Use health_check tool if available, otherwise try to connect
        async with asyncio.timeout(5.0):
            client = await get_kestrel_client()
            # Simple check: verify we have tools loaded
            if not client.get_tools():
                return False, None, "Kestrel tools not available"
            latency_ms = int((time.time() - start) * 1000)
            return True, latency_ms, None
    except asyncio.TimeoutError:
        return False, None, "Kestrel connection timeout (>5s)"
    except Exception as e:
        return False, None, f"Kestrel error: {str(e)}"


async def multi_hop_query(
    start_node_ids: list[str] | None = None,
    end_node_ids: list[str] | None = None,
    max_hops: int = 3,
    predicate_filter: str | None = None,
    limit: int = 100,
    constraints: list[dict[str, Any]] | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """
    Perform a multi-hop pathfinding query in the knowledge graph.

    Supports two search modes:
    - Singly-pinned: start_node_ids provided, end_node_ids=None → explore from start nodes
    - Doubly-pinned: both start_node_ids and end_node_ids → find paths connecting them

    Args:
        start_node_ids: List of starting CURIEs (required)
        end_node_ids: List of target CURIEs (optional, enables doubly-pinned mode)
        max_hops: Maximum path length (1-5, default 3)
        predicate_filter: Comma-separated predicates to filter edges
        limit: Maximum number of paths to return (default 100)

    Returns:
        dict with "content" list containing path results

    Examples:
        # Singly-pinned: explore 2 hops from glucose
        await multi_hop_query(start_node_ids=["CHEBI:17234"], max_hops=2)

        # Doubly-pinned: find paths from glucose to diabetes
        await multi_hop_query(
            start_node_ids=["CHEBI:17234"],
            end_node_ids=["MONDO:0005148"],
            max_hops=3
        )
    """
    if not start_node_ids:
        return {
            "content": [{"type": "text", "text": "Error: start_node_ids is required"}],
            "isError": True,
        }

    # Validate max_hops
    if max_hops < 1 or max_hops > 5:
        return {
            "content": [{"type": "text", "text": "Error: max_hops must be between 1 and 5"}],
            "isError": True,
        }

    # Build arguments for the MCP tool
    # Note: Kestrel API v1.16.0 uses 'max_path_length' (not 'max_hops')
    # and 'predicate' (not 'predicate_filter')
    arguments = {
        "start_node_ids": start_node_ids if isinstance(start_node_ids, list) else [start_node_ids],
        "max_path_length": max_hops,  # API parameter name differs from wrapper
        "limit": limit,
    }

    if end_node_ids:
        arguments["end_node_ids"] = end_node_ids if isinstance(end_node_ids, list) else [end_node_ids]

    if predicate_filter:
        arguments["predicate"] = predicate_filter  # API uses 'predicate'

    # Optional in-query constraints (e.g. degree/degree_percentile to suppress hubs)
    # and result detail mode (slim|full). The Kestrel MCP tool accepts both.
    if constraints:
        arguments["constraints"] = constraints

    if mode:
        arguments["mode"] = mode

    return await call_kestrel_tool("multi_hop_query", arguments)


def parse_kestrel_response(envelope: dict[str, Any]) -> dict[str, Any]:
    """Parse a multi_hop_query / subgraph_query MCP envelope into normalized paths.

    Replaces the drift-prone ``data.get("paths", data)`` parse that shipped to prod in
    three nodes for ~3.5 months. The real response (see .claude/skills/kestrel-api) has NO
    top-level ``"paths"`` key — it is ``{"results": [...], "nodes": {...}, "edges": {...}}``
    where each ``result["paths"]`` is a list of **CURIE-string lists** (not dicts).

    FAILS LOUDLY: on a missing/mis-shaped ``results`` (or a bad envelope) it returns an
    EMPTY path set and logs — it NEVER falls back to the raw dict (the silent-fallback bug).

    Args:
        envelope: the MCP tool result ``{"content": [{"text": "<json>"}], "isError": ...}``.

    Returns:
        ``{"paths": [{"curies": [...], "names": [...],
        "predicates": [{"predicate": str|None, "forward": bool|None}, ...] (one per hop),
        "end_node_id": str, "degree": int, "score": float}], "nodes": {curie: {...}},
        "end_node_ids": [str, ...], "n_paths": int}`` — ``end_node_ids`` lists every result's
        ``end_node_id`` (order-preserving, deduped), for callers that need reachable end-nodes
        rather than full paths (e.g. pathway_enrichment's shared-neighbor counting).
    """
    empty = {"paths": [], "nodes": {}, "end_node_ids": [], "n_paths": 0}
    try:
        content = envelope.get("content", []) if isinstance(envelope, dict) else []
        if not content:
            return empty
        json_text = content[0].get("text", "")
        if not json_text:
            return empty
        data = json.loads(json_text)
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError) as e:
        logger.warning("parse_kestrel_response: bad envelope (%s)", e)
        return empty

    if not isinstance(data, dict):
        logger.warning("parse_kestrel_response: inner data not a dict: %s", type(data).__name__)
        return empty

    results = data.get("results", [])
    if not isinstance(results, list):
        logger.warning("parse_kestrel_response: 'results' not a list (keys=%s)", list(data)[:5])
        return empty

    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict):
        nodes = {}

    def _name(curie: str) -> str:
        info = nodes.get(curie)
        if isinstance(info, dict) and info.get("name"):
            return str(info["name"])
        return curie

    # --- Per-hop predicate derivation (U0) -------------------------------------------------
    # The response carries `edges` (edge-id -> compact tuple) + `edge_schema` (column order).
    # Map each consecutive path pair to its edge predicate, recording orientation vs the path.
    edges = data.get("edges", {})
    if not isinstance(edges, dict):
        edges = {}
    norm_edges = {str(k): v for k, v in edges.items()}  # edge_ids may be ints; keys may be str
    edge_schema = data.get("edge_schema", [])
    sub_i = pred_i = obj_i = None
    if isinstance(edge_schema, list):
        for _idx, _col in enumerate(edge_schema):
            if _col == "subject":
                sub_i = _idx
            elif _col == "predicate":
                pred_i = _idx
            elif _col == "object":
                obj_i = _idx

    def _edge_triple(e: Any) -> tuple[str, str, str] | None:
        """Extract (subject, predicate, object) from a compact edge tuple via edge_schema.

        Reads positions from edge_schema rather than hardcoding index 1 (robust to reorder).
        """
        if not isinstance(e, list) or sub_i is None or pred_i is None or obj_i is None:
            return None
        if max(sub_i, pred_i, obj_i) >= len(e):
            return None
        s, p, o = e[sub_i], e[pred_i], e[obj_i]
        if not (isinstance(s, str) and isinstance(p, str) and isinstance(o, str)):
            return None
        return s, p, o

    def _triple_map(edge_values: list) -> dict[frozenset, list[tuple[str, str, str]]]:
        """frozenset({subject, object}) -> predicate-sorted (s, p, o) triples (deterministic)."""
        m: dict[frozenset, list[tuple[str, str, str]]] = {}
        for e in edge_values:
            t = _edge_triple(e)
            if t is None or t[0] == t[2]:  # skip unparseable / self-loops
                continue
            m.setdefault(frozenset((t[0], t[2])), []).append(t)
        for _k in m:
            m[_k].sort(key=lambda t: t[1])
        return m

    def _hop_predicates(curies: list[str], tmap: dict) -> list[dict[str, Any]]:
        """Per-hop {predicate, forward}; forward=True when the edge runs A->B along the path.

        Missing edge for a hop -> {None, None} at that position (never a positional shift).
        Prefers a forward-oriented edge; falls back to a reverse one (the direction signal).
        """
        out: list[dict[str, Any]] = []
        for i in range(len(curies) - 1):
            a, b = curies[i], curies[i + 1]
            cands = tmap.get(frozenset((a, b)), []) if a != b else []
            fwd = [t for t in cands if t[0] == a and t[2] == b]
            rev = [t for t in cands if t[0] == b and t[2] == a]
            if fwd:
                out.append({"predicate": fwd[0][1], "forward": True})
            elif rev:
                out.append({"predicate": rev[0][1], "forward": False})
            else:
                out.append({"predicate": None, "forward": None})
        return out

    parsed_paths: list[dict[str, Any]] = []
    end_node_ids: list[str] = []
    seen_ends: set[str] = set()
    for res in results:
        if not isinstance(res, dict):
            continue
        end_id = res.get("end_node_id")
        if isinstance(end_id, str) and end_id and end_id not in seen_ends:
            seen_ends.add(end_id)
            end_node_ids.append(end_id)
        # Scope edges to this result's edge_ids when present. If edge_ids are specified but none
        # resolve in norm_edges, do NOT fall back to all edges — in a multi-result response that
        # borrows edges from other results and mis-attributes predicates/directions to this path's
        # hops. Use the (possibly empty) scoped set instead; _hop_predicates then emits {None, None}
        # for unresolved hops (honest "unknown") rather than a wrong predicate. Only when there is no
        # edge_ids scoping at all do we scan the full edge map.
        eids = res.get("edge_ids")
        if isinstance(eids, list) and eids:
            cand = [norm_edges[str(eid)] for eid in eids if str(eid) in norm_edges]
        else:
            cand = list(edges.values())
        tmap = _triple_map(cand)
        for path in res.get("paths", []) or []:
            # A path is a list of CURIE strings. Reject the old dict shape loudly.
            if not isinstance(path, list) or len(path) < 2:
                continue
            curies = [c for c in path if isinstance(c, str)]
            if len(curies) < 2:
                continue
            parsed_paths.append({
                "curies": curies,
                "names": [_name(c) for c in curies],
                "predicates": _hop_predicates(curies, tmap),
                "end_node_id": res.get("end_node_id", curies[-1]),
                "degree": res.get("degree", 0),
                "score": res.get("score", 0.0),
            })

    return {"paths": parsed_paths, "nodes": nodes,
            "end_node_ids": end_node_ids, "n_paths": len(parsed_paths)}
