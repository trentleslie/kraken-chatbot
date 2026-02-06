"""FastAPI application with WebSocket endpoint for KRAKEN chat interface."""

import asyncio
import json
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .agent import run_agent_turn
from .protocol import (
    TextMessage,
    ToolUseMessage,
    ToolResultMessage,
    ErrorMessage,
    DoneMessage,
    TraceMessage,
)


# Rate limiting state: connection_id -> list of message timestamps
rate_limit_state: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(connection_id: str) -> bool:
    """
    Check if connection is within rate limits.

    Returns True if allowed, False if rate limited.
    Cleans up old timestamps and adds current timestamp if allowed.
    """
    settings = get_settings()
    now = time.time()
    window_start = now - 60  # 1-minute sliding window

    # Clean old timestamps
    timestamps = rate_limit_state[connection_id]
    timestamps[:] = [t for t in timestamps if t > window_start]

    if len(timestamps) >= settings.rate_limit_per_minute:
        return False

    timestamps.append(now)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    settings = get_settings()
    print(f"Starting Kestrel Backend on {settings.host}:{settings.port}")
    print(f"Allowed origins: {settings.allowed_origins}")
    print(f"Rate limit: {settings.rate_limit_per_minute} messages/minute")
    yield
    # Shutdown
    print("Shutting down Kestrel Backend")


app = FastAPI(
    title="Kestrel Backend",
    description="KRAKEN Knowledge Graph Chat Interface",
    version="0.1.0",
    lifespan=lifespan,
)


# Configure CORS for REST endpoints
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "kestrel-backend"}


@app.get("/api/health")
async def api_health_check():
    """API health check endpoint (alternative path)."""
    return {"status": "healthy", "service": "kestrel-backend"}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for chat interactions.

    Protocol:
    - Client sends: {"type": "user_message", "content": "..."}
    - Server sends: text, tool_use, tool_result, trace, done messages
    """
    await websocket.accept()
    connection_id = str(id(websocket))

    try:
        while True:
            # Receive message from client
            raw_data = await websocket.receive_text()

            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_text(
                    ErrorMessage(message="Invalid JSON").model_dump_json()
                )
                continue

            # Validate message type
            if data.get("type") != "user_message":
                await websocket.send_text(
                    ErrorMessage(message="Unknown message type").model_dump_json()
                )
                continue

            content = data.get("content", "").strip()
            if not content:
                await websocket.send_text(
                    ErrorMessage(message="Empty message").model_dump_json()
                )
                continue

            # Check rate limit
            if not check_rate_limit(connection_id):
                await websocket.send_text(
                    ErrorMessage(
                        message="Rate limit exceeded. Please wait before sending more messages."
                    ).model_dump_json()
                )
                continue

            # Process the message through the agent
            try:
                async for event in run_agent_turn(content):
                    # Convert agent events to protocol messages
                    match event.type:
                        case "text":
                            msg = TextMessage(content=event.data["content"])
                        case "tool_use":
                            msg = ToolUseMessage(
                                tool=event.data["tool"],
                                args=event.data["args"]
                            )
                        case "tool_result":
                            msg = ToolResultMessage(
                                tool=event.data["tool"],
                                data=event.data["data"]
                            )
                        case "error":
                            msg = ErrorMessage(
                                message=event.data["message"],
                                code=event.data.get("code")
                            )
                        case "trace":
                            msg = TraceMessage(**event.data)
                        case "done":
                            msg = DoneMessage()
                        case _:
                            continue

                    await websocket.send_text(msg.model_dump_json())

            except Exception as e:
                await websocket.send_text(
                    ErrorMessage(message=f"Agent error: {str(e)}").model_dump_json()
                )
                await websocket.send_text(DoneMessage().model_dump_json())

    except WebSocketDisconnect:
        # Clean up rate limit state for this connection
        rate_limit_state.pop(connection_id, None)
    except Exception as e:
        print(f"WebSocket error: {e}")
        rate_limit_state.pop(connection_id, None)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "kestrel_backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
