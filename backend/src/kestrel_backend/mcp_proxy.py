#!/usr/bin/env python3
"""MCP Stdio-to-HTTP Proxy for Kestrel.

This proxy reads JSON-RPC messages from stdin, sends them to the Kestrel MCP
server via HTTP POST with proper headers, and writes responses to stdout.

Usage:
    python -m kestrel_backend.mcp_proxy

The proxy handles Kestrel's non-standard MCP-over-HTTP protocol:
- Requests are sent via POST with JSON-RPC messages
- Responses are returned in SSE format (event: message\ndata: ...)
- Session management via mcp-session-id header
"""

import json
import os
import sys

import httpx

KESTREL_MCP_URL = os.getenv("KESTREL_MCP_URL", "https://kestrel.nathanpricelab.com/mcp")
KESTREL_API_KEY = os.getenv("KESTREL_API_KEY", "")


def get_headers(session_id: str | None = None) -> dict[str, str]:
    """Build headers for Kestrel requests."""
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if KESTREL_API_KEY:
        headers["X-API-Key"] = KESTREL_API_KEY
    if session_id:
        headers["mcp-session-id"] = session_id
    return headers


def parse_sse_response(text: str) -> dict:
    """Parse SSE-formatted response from Kestrel."""
    for line in text.strip().split("\n"):
        if line.startswith("data: "):
            return json.loads(line[6:])
    # Try parsing directly as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"jsonrpc": "2.0", "error": {"code": -32000, "message": f"Parse error: {text[:100]}"}}


def main():
    """Main loop: read JSON-RPC from stdin, forward to Kestrel, write to stdout."""
    session_id = None

    with httpx.Client(timeout=60.0) as client:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"}
                }
                print(json.dumps(error_response), flush=True)
                continue

            try:
                # Send request to Kestrel
                response = client.post(
                    KESTREL_MCP_URL,
                    headers=get_headers(session_id),
                    json=message,
                )

                # Capture session ID from response
                if "mcp-session-id" in response.headers:
                    session_id = response.headers["mcp-session-id"]

                # Handle notifications (no response expected)
                if response.status_code == 202:
                    # For notifications, we might not get a response body
                    continue

                response.raise_for_status()

                # Parse and forward response
                result = parse_sse_response(response.text)
                print(json.dumps(result), flush=True)

                # If this was an initialize request, send initialized notification
                if message.get("method") == "initialize":
                    init_notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
                    client.post(
                        KESTREL_MCP_URL,
                        headers=get_headers(session_id),
                        json=init_notif,
                    )

            except httpx.HTTPStatusError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {"code": -32000, "message": f"HTTP error: {e.response.status_code}"}
                }
                print(json.dumps(error_response), flush=True)
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {"code": -32000, "message": str(e)}
                }
                print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    main()
