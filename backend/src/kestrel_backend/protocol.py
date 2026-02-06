"""WebSocket message protocol definitions matching the frontend types."""

from typing import Any, Literal
from pydantic import BaseModel


# Outgoing messages (Server → Client)

class TextMessage(BaseModel):
    """Streaming text content from the agent."""
    type: Literal["text"] = "text"
    content: str


class ToolUseMessage(BaseModel):
    """Agent is calling a tool."""
    type: Literal["tool_use"] = "tool_use"
    tool: str
    args: dict[str, Any]


class ToolResultMessage(BaseModel):
    """Result from a tool call."""
    type: Literal["tool_result"] = "tool_result"
    tool: str
    data: dict[str, Any]


class ErrorMessage(BaseModel):
    """Error occurred during processing."""
    type: Literal["error"] = "error"
    message: str


class DoneMessage(BaseModel):
    """Agent finished responding to the turn."""
    type: Literal["done"] = "done"


class StatusMessage(BaseModel):
    """Status update (e.g., connecting, processing)."""
    type: Literal["status"] = "status"
    status: str


class TraceMessage(BaseModel):
    """Usage statistics for a completed turn."""
    type: Literal["trace"] = "trace"
    turn_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_read_tokens: int | None = None
    cost_usd: float | None = None
    duration_ms: int | None = None
    tool_calls_count: int | None = None
    model: str | None = None


# Incoming messages (Client → Server)

class UserMessageRequest(BaseModel):
    """User sends a chat message."""
    type: Literal["user_message"] = "user_message"
    content: str


# Type alias for all outgoing message types
OutgoingMessage = TextMessage | ToolUseMessage | ToolResultMessage | ErrorMessage | DoneMessage | StatusMessage | TraceMessage
