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
    code: str | None = None  # Optional error code (e.g., "AUTH_ERROR")


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
    correlation_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_read_tokens: int | None = None
    cost_usd: float | None = None
    duration_ms: int | None = None
    tool_calls_count: int | None = None
    model: str | None = None


class ConversationStartedMessage(BaseModel):
    """Sent when a new conversation is created in the database."""
    type: Literal["conversation_started"] = "conversation_started"
    conversation_id: str


# Phase 6: Pipeline-specific message types

class PipelineProgressMessage(BaseModel):
    """Progress update during discovery pipeline execution."""
    type: Literal["pipeline_progress"] = "pipeline_progress"
    node: str                    # Current node name (e.g., "entity_resolution")
    message: str                 # User-friendly status message
    nodes_completed: int         # Number of nodes finished
    total_nodes: int = 10        # Total nodes in pipeline


class PipelineNodeDetailMessage(BaseModel):
    """Intermediate output from a completed pipeline node."""
    type: Literal["pipeline_node_detail"] = "pipeline_node_detail"
    node: str
    summary: str
    duration_ms: int
    details: dict[str, Any]


class PipelineCompleteMessage(BaseModel):
    """Final result from discovery pipeline execution."""
    type: Literal["pipeline_complete"] = "pipeline_complete"
    synthesis_report: str        # The final markdown report
    hypotheses_count: int        # Number of hypotheses generated
    entities_resolved: int       # Number of entities resolved
    duration_ms: int             # Total execution time
    # Token tracking fields for UI compatibility with TraceMessage
    model: str = "claude-sonnet-4-20250514"  # Pipeline uses SDK default
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


# Incoming messages (Client → Server)

class UserMessageRequest(BaseModel):
    """User sends a chat message."""
    type: Literal["user_message"] = "user_message"
    content: str
    agent_mode: str = "classic"  # "classic" or "pipeline"


# Type alias for all outgoing message types
OutgoingMessage = (
    TextMessage
    | ToolUseMessage
    | ToolResultMessage
    | ErrorMessage
    | DoneMessage
    | StatusMessage
    | TraceMessage
    | ConversationStartedMessage
    | PipelineProgressMessage
    | PipelineNodeDetailMessage
    | PipelineCompleteMessage
)


# Node name to user-friendly message mapping
NODE_STATUS_MESSAGES = {
    "intake": "Parsing your query...",
    "entity_resolution": "Resolving entities in knowledge graph...",
    "triage": "Scoring entity novelty...",
    "direct_kg": "Analyzing well-characterized entities...",
    "cold_start": "Investigating sparse entities...",
    "pathway_enrichment": "Finding shared biological pathways...",
    "integration": "Detecting cross-type bridges...",
    "temporal": "Applying temporal reasoning...",
    "synthesis": "Generating discovery report...",
    "literature_grounding": "Grounding hypotheses with literature citations...",
}
