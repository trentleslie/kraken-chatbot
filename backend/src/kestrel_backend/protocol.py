"""WebSocket message protocol definitions matching the frontend types."""

from typing import Any, Literal
from pydantic import BaseModel, Field


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
    trace_id: str | None = None  # Langfuse trace ID for feedback linkage
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
    total_nodes: int = 12        # Total nodes in pipeline (keep in sync with NODE_STATUS_MESSAGES)


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
    turn_id: str | None = None   # Database turn ID for feedback linkage
    trace_id: str | None = None  # Langfuse trace ID for feedback linkage
    # Token tracking fields for UI compatibility with TraceMessage
    model: str = "claude-sonnet-4-20250514"  # Pipeline uses SDK default
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


# Incoming messages (Client → Server)

class StructuredAnalyte(BaseModel):
    """One analyte row from a client-parsed upload panel.

    Sent as part of the structured analyte list on a pipeline ``user_message`` alongside the
    free-text query. ``group`` is the mapped module/category (drives the R16 run-set filter);
    ``type`` is the optional analyte type (``metabolite``|``protein``|``gene``, lowercase, to
    match ``biolink_class_for``). Both optional; validation/normalization is centralized in
    ``analyte_ingest.validate_and_normalize`` (R19), not on this doc-only model.
    """
    name: str = Field(..., description="Analyte name (verbatim from the mapped file column)")
    group: str | None = Field(None, description="Mapped group/category value, if any")
    type: str | None = Field(
        None, description="Optional analyte type hint: metabolite|protein|gene"
    )
    # Signed-weight data spine (Axis A): kME (module-eigengene correlation, [-1,1]) and kIM (raw
    # intramodular connectivity kWithin, unbounded/non-negative) ride each row; validated in
    # analyte_ingest.validate_and_normalize (kME reject-if-outside-[-1,1]; kIM reject-if-negative).
    kme: float | None = Field(None, description="Signed module-eigengene correlation ([-1, 1])")
    kim: float | None = Field(None, description="Raw intramodular connectivity kWithin (>= 0)")


class ModuleDirection(BaseModel):
    """One per-module eigengene→outcome direction row (Axis A).

    Documentation-only, like ``StructuredAnalyte`` — the WS handler reads these via
    ``data.get('module_directions')`` and validation is centralized in
    ``analyte_ingest.validate_and_normalize``. Load-bearing for the sign-inversion metric
    (member-vs-outcome = sign(kME) × sign(direction)).
    """
    group: str = Field(..., description="Group/module the direction applies to")
    eigengene_trait_correlation: float = Field(
        ..., description="Signed correlation of the module eigengene with the trait ([-1, 1])"
    )
    trait_label: str = Field(..., description="Human label of the outcome/trait")


class KeySourceMessage(BaseModel):
    """Server → Client: which key the current turn ran on."""
    type: Literal["key_source"] = "key_source"
    source: Literal["byok", "server"]


class SetKeyRequest(BaseModel):
    """Client → Server: set/clear the per-connection BYOK key."""
    type: Literal["set_key"] = "set_key"
    key: str | None = None


class UserMessageRequest(BaseModel):
    """User sends a chat message.

    NOTE: documentation-only. The WS handler reads fields via ``data.get(...)`` and never
    instantiates this model, so the file-upload OR-semantics guard (content OR analytes) lives
    in ``main.py``, not in a (never-run) ``@model_validator`` here.
    """
    type: Literal["user_message"] = "user_message"
    content: str
    agent_mode: str = "classic"  # "classic" or "pipeline"
    # Analyte file-upload fields (mirror the biomapper_env thread). The full parsed panel plus
    # the client's group selection travel with the message so the backend forms the run set and
    # retains the full group map for future server-side per-group fan-out.
    structured_analytes: list[StructuredAnalyte] | None = Field(
        None, description="Full parsed analyte panel from a client-side file upload"
    )
    selected_groups: list[str] | None = Field(
        None, description="Group values the user chose to run (None/empty = all groups)"
    )
    # Optional per-module eigengene→outcome directions (Axis A). Validated + count-bounded in the
    # shared R19 helper; absent for classic / no-direction runs.
    module_directions: list[ModuleDirection] | None = Field(
        None, description="Per-module eigengene→outcome direction rows for the signed-weight spine"
    )


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
    "bridge_grounding": "Labeling bridge evidence provenance...",
    "temporal": "Applying temporal reasoning...",
    "hypothesis_extraction": "Extracting and validating hypotheses...",
    "literature_grounding": "Grounding hypotheses with literature citations...",
    "synthesis": "Generating discovery report...",
}
