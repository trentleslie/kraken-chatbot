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
        """Ensure HTTP client is created."""
        if self._http_client is None:
            headers = _get_headers()
            has_api_key = "X-API-Key" in headers
            logger.info(
                "Creating HTTP client with headers: %s, has_api_key=%s",
                list(headers.keys()), has_api_key
            )
            self._http_client = httpx.AsyncClient(
                timeout=60.0,
                headers=headers,
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
