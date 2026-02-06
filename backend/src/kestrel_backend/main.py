"""FastAPI application with WebSocket endpoint for KRAKEN chat interface."""

import asyncio
import json
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Tuple

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

# Conversation history per WebSocket connection: {connection_id: [(role, content), ...]}
conversation_history: dict[str, List[Tuple[str, str]]] = defaultdict(list)

# Maximum number of user/assistant exchanges to keep in history
MAX_HISTORY_EXCHANGES = 10


def build_conversation_prompt(history: List[Tuple[str, str]], current_message: str) -> str:
    """
    Build prompt with conversation history for multi-turn context.

    Formats previous exchanges and appends the current message so the agent
    can understand the full conversation context.
    """
    if not history:
        return current_message

    lines = ["Previous conversation:"]
    for role, content in history:
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")

    lines.append("")
    lines.append(f"Current message: {current_message}")
    return "\n".join(lines)


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

            # Build prompt with conversation history for multi-turn context
            history = conversation_history[connection_id]
            full_prompt = build_conversation_prompt(history, content)

            # Process the message through the agent
            try:
                # Track assistant response text for history
                assistant_response_parts: List[str] = []

                async for event in run_agent_turn(full_prompt):
                    # Convert agent events to protocol messages
                    match event.type:
                        case "text":
                            # Capture text content for conversation history
                            text_content = event.data["content"]
                            assistant_response_parts.append(text_content)
                            msg = TextMessage(content=text_content)
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

                # Update conversation history after successful exchange
                history.append(("user", content))
                if assistant_response_parts:
                    history.append(("assistant", "".join(assistant_response_parts)))

                # Trim history to last N exchanges (2 messages per exchange)
                max_messages = MAX_HISTORY_EXCHANGES * 2
                if len(history) > max_messages:
                    conversation_history[connection_id] = history[-max_messages:]

            except Exception as e:
                await websocket.send_text(
                    ErrorMessage(message=f"Agent error: {str(e)}").model_dump_json()
                )
                await websocket.send_text(DoneMessage().model_dump_json())

    except WebSocketDisconnect:
        # Clean up state for this connection
        rate_limit_state.pop(connection_id, None)
        conversation_history.pop(connection_id, None)
    except Exception as e:
        print(f"WebSocket error: {e}")
        rate_limit_state.pop(connection_id, None)
        conversation_history.pop(connection_id, None)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "kestrel_backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
