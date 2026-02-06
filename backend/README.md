# Kestrel Backend

FastAPI backend for the KRAKEN Knowledge Graph chat interface, powered by Claude Agent SDK.

## Architecture

This backend provides a WebSocket API that:
1. Receives user messages from the React frontend
2. Processes them through Claude using the Claude Code SDK
3. Streams responses (text, tool calls, results) back to the client

## Security Hardening

The agent is configured with strict security controls for public deployment:

- **Tool Whitelist**: Only Kestrel MCP tools are allowed (12 specific tools)
- **Tool Blacklist**: Dangerous tools (Bash, Read, Write, etc.) are explicitly blocked
- **Rate Limiting**: 10 messages per minute per connection
- **Read-Only Role**: System prompt establishes the agent as a read-only explorer

## Setup

```bash
# Create virtual environment with uv
uv venv --python python3.12
source .venv/bin/activate  # or `.venv/Scripts/activate` on Windows

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
```

## Running

```bash
# Development
uv run uvicorn src.kestrel_backend.main:app --reload --host 127.0.0.1 --port 8000

# Production (via systemd on Lightsail)
uv run uvicorn src.kestrel_backend.main:app --host 127.0.0.1 --port 8000
```

## WebSocket Protocol

### Client → Server

```json
{"type": "user_message", "content": "What drugs treat type 2 diabetes?"}
```

### Server → Client

```json
{"type": "text", "content": "Based on my search..."}
{"type": "tool_use", "tool": "mcp__kestrel__hybrid_search", "args": {...}}
{"type": "tool_result", "tool": "mcp__kestrel__hybrid_search", "data": {...}}
{"type": "trace", "input_tokens": 1234, "output_tokens": 567, ...}
{"type": "done"}
```

## Known Limitations

**Single-Turn Context**: The `query()` function is single-turn. Each user message starts fresh - the agent won't remember earlier messages in the session. This is acceptable for exploratory queries where each message is self-contained.

Future enhancement: Accumulate messages for multi-turn conversations.
