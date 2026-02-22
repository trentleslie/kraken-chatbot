"""FastAPI application with WebSocket endpoint for KRAKEN chat interface."""

import asyncio
import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Tuple, Any
from uuid import UUID

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langfuse import get_client

from .config import get_settings
from .agent import run_agent_turn
from .logging_config import configure_logging, generate_correlation_id, correlation_id

# Configure logging early
settings = get_settings()
configure_logging(
    log_level=settings.log_level,
    log_format=settings.log_format,
    module_levels=settings.log_module_levels
)
logger = logging.getLogger(__name__)

# Langfuse client for pipeline mode tracing (lazy initialized)
_pipeline_langfuse = None


def _get_pipeline_langfuse():
    """Get or create Langfuse client for pipeline mode observability."""
    global _pipeline_langfuse
    if _pipeline_langfuse is not None:
        return _pipeline_langfuse
    settings = get_settings()
    if settings.langfuse_enabled and settings.langfuse_public_key and settings.langfuse_secret_key:
        _pipeline_langfuse = get_client()
    return _pipeline_langfuse


from .database import init_db, close_db, get_conversation_with_turns, create_conversation, add_turn
from .protocol import (
    TextMessage,
    ToolUseMessage,
    ToolResultMessage,
    ErrorMessage,
    DoneMessage,
    TraceMessage,
    ConversationStartedMessage,
    PipelineProgressMessage,
    PipelineNodeDetailMessage,
    PipelineCompleteMessage,
    NODE_STATUS_MESSAGES,
)
from .graph.node_detail_extractors import extract_node_details


# Rate limiting state: connection_id -> list of message timestamps
rate_limit_state: dict[str, list[float]] = defaultdict(list)

# Conversation history per WebSocket connection: {connection_id: [(role, content), ...]}
conversation_history: dict[str, List[Tuple[str, str]]] = defaultdict(list)

# Database conversation IDs per WebSocket connection: {connection_id: UUID}
conversation_ids: dict[str, UUID] = {}

# Turn counter per WebSocket connection: {connection_id: int}
turn_counters: dict[str, int] = defaultdict(int)

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
    logger.info(
        "Starting Kestrel Backend",
        extra={
            "host": settings.host,
            "port": settings.port,
            "allowed_origins": settings.allowed_origins,
            "rate_limit_per_minute": settings.rate_limit_per_minute
        }
    )
    await init_db()
    yield
    # Shutdown
    await close_db()
    logger.info("Shutting down Kestrel Backend")


app = FastAPI(
    title="Kestrel Backend",
    description="KRAKEN Knowledge Graph Chat Interface",
    version="0.1.0",
    lifespan=lifespan,
)


# Correlation ID middleware for HTTP requests
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Add correlation ID to each HTTP request for tracing."""
    # Check if client provided correlation ID, otherwise generate one
    corr_id = request.headers.get("X-Correlation-ID") or generate_correlation_id()
    correlation_id.set(corr_id)

    response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id
    return response


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


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Retrieve conversation by UUID for shared viewing."""
    try:
        uuid_id = UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID format")

    conversation = await get_conversation_with_turns(uuid_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation


async def handle_classic_mode(
    websocket: WebSocket,
    content: str,
    connection_id: str,
) -> None:
    """
    Handle classic agent mode - existing Claude Agent SDK behavior.
    
    This is the original single-agent approach with tool use and streaming.
    """
    history = conversation_history[connection_id]
    full_prompt = build_conversation_prompt(history, content)
    
    # Track assistant response text for history
    assistant_response_parts: List[str] = []
    # Track metrics for database persistence
    turn_metrics: dict = {}

    async for event in run_agent_turn(full_prompt):
        # Convert agent events to protocol messages
        match event.type:
            case "text":
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
                # Add correlation_id to trace message
                trace_data = event.data.copy()
                trace_data["correlation_id"] = correlation_id.get()
                msg = TraceMessage(**trace_data)
                turn_metrics = event.data
            case "done":
                msg = DoneMessage()
            case _:
                continue

        await websocket.send_text(msg.model_dump_json())

    # Update conversation history after successful exchange
    history.append(("user", content))
    if assistant_response_parts:
        history.append(("assistant", "".join(assistant_response_parts)))

    # Persist turn to database if conversation exists
    conv_id = conversation_ids.get(connection_id)
    if conv_id and assistant_response_parts:
        turn_counters[connection_id] += 1
        await add_turn(
            conversation_id=conv_id,
            turn_number=turn_counters[connection_id],
            user_query=content,
            assistant_response="".join(assistant_response_parts),
            metrics=turn_metrics
        )

    # Trim history to last N exchanges (2 messages per exchange)
    max_messages = MAX_HISTORY_EXCHANGES * 2
    if len(history) > max_messages:
        conversation_history[connection_id] = history[-max_messages:]


async def handle_pipeline_mode(
    websocket: WebSocket,
    content: str,
    connection_id: str,
) -> None:
    """
    Handle discovery pipeline mode - LangGraph multi-node workflow.

    Uses stream_mode="updates" with accumulated state tracking.
    After each node completes, sends both PipelineProgressMessage and
    PipelineNodeDetailMessage with extracted structured details.
    """
    from .graph.runner import stream_discovery

    query_preview = content[:50] + "..." if len(content) > 50 else content
    logger.info("Pipeline started — query=%r mode=pipeline", query_preview)

    start_time = time.time()
    nodes_completed = 0
    accumulated_state: dict[str, Any] = {}
    node_timings: dict[str, float] = {}
    node_start_times: dict[str, float] = {}

    langfuse = _get_pipeline_langfuse()
    trace = None
    node_spans: dict[str, Any] = {}

    if langfuse:
        trace = langfuse.start_span(
            name="discovery_pipeline",
            input={"query": content, "connection_id": connection_id},
            metadata={"mode": "pipeline", "version": "2.0"},
        )

    try:
        history = conversation_history.get(connection_id, [])

        async for event in stream_discovery(
            query=content,
            conversation_history=list(history),
        ):
            if event["type"] != "node_update":
                continue

            node_name = event["node"]
            node_output = event["node_output"]

            if node_name == "__start__" or node_name not in NODE_STATUS_MESSAGES:
                continue

            for key, value in node_output.items():
                existing = accumulated_state.get(key)
                if isinstance(existing, list) and isinstance(value, list):
                    accumulated_state[key] = existing + value
                else:
                    accumulated_state[key] = value

            now = time.time()
            node_start = node_start_times.get(node_name, now)
            duration_ms = int((now - node_start) * 1000)
            node_timings[node_name] = now - node_start

            if node_name not in node_start_times:
                node_start_times[node_name] = now

            nodes_completed += 1

            if trace:
                span = node_spans.pop(node_name, None)
                if span:
                    span.update(output={"duration_ms": duration_ms})
                    span.end()
                node_spans[node_name] = trace.start_span(
                    name=f"node_{node_name}",
                    input={"node": node_name, "nodes_completed": nodes_completed},
                )

            progress_msg = PipelineProgressMessage(
                node=node_name,
                message=NODE_STATUS_MESSAGES[node_name],
                nodes_completed=nodes_completed,
            )
            await websocket.send_text(progress_msg.model_dump_json())

            summary, details = extract_node_details(node_name, accumulated_state)
            detail_msg = PipelineNodeDetailMessage(
                node=node_name,
                summary=summary,
                duration_ms=duration_ms,
                details=details,
            )
            await websocket.send_text(detail_msg.model_dump_json())

            if trace:
                span = node_spans.get(node_name)
                if span:
                    span.update(output={"duration_ms": duration_ms, "summary": summary})
                    span.end()
                    node_spans.pop(node_name, None)

        duration_ms = int((time.time() - start_time) * 1000)
        synthesis_report = accumulated_state.get("synthesis_report", "")
        hypotheses = accumulated_state.get("hypotheses", [])
        resolved_entities = accumulated_state.get("resolved_entities", [])

        successful_resolutions = sum(
            1 for e in resolved_entities
            if hasattr(e, "curie") and e.curie is not None
        )

        msg = PipelineCompleteMessage(
            synthesis_report=synthesis_report,
            hypotheses_count=len(hypotheses),
            entities_resolved=len(resolved_entities),
            duration_ms=duration_ms,
            model="claude-sonnet-4-20250514",
        )
        await websocket.send_text(msg.model_dump_json())

        logger.info(
            "Pipeline complete in %.1fs — entities=%d, resolved=%d, hypotheses=%d",
            duration_ms / 1000.0, len(resolved_entities), successful_resolutions, len(hypotheses)
        )

        if trace and langfuse:
            trace.update(
                output={
                    "entities_total": len(resolved_entities),
                    "entities_resolved": successful_resolutions,
                    "resolution_rate": (
                        successful_resolutions / len(resolved_entities)
                        if resolved_entities else 0.0
                    ),
                    "hypotheses_count": len(hypotheses),
                    "duration_ms": duration_ms,
                    "node_timings": node_timings,
                },
                metadata={"status": "completed"},
            )
            trace.end()
            langfuse.flush()

        history = conversation_history[connection_id]
        history.append(("user", content))
        if synthesis_report:
            history.append(("assistant", synthesis_report))

        conv_id = conversation_ids.get(connection_id)
        if conv_id and synthesis_report:
            turn_counters[connection_id] += 1
            await add_turn(
                conversation_id=conv_id,
                turn_number=turn_counters[connection_id],
                user_query=content,
                assistant_response=synthesis_report,
                metrics={
                    "mode": "pipeline",
                    "duration_ms": duration_ms,
                    "hypotheses_count": len(hypotheses),
                    "entities_resolved": successful_resolutions,
                    "entities_total": len(resolved_entities),
                    "node_timings": node_timings,
                }
            )

        max_messages = MAX_HISTORY_EXCHANGES * 2
        if len(history) > max_messages:
            conversation_history[connection_id] = history[-max_messages:]

        await websocket.send_text(DoneMessage().model_dump_json())

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Pipeline failed after %.1fs — error=%s, nodes_completed=%d",
            duration, str(e), nodes_completed, exc_info=True
        )

        if trace and langfuse:
            for span in node_spans.values():
                try:
                    span.end()
                except Exception:
                    pass

            trace.update(
                output={"error": str(e)},
                metadata={"status": "failed"},
            )
            trace.end()
            langfuse.flush()

        error_msg = ErrorMessage(
            message=f"Pipeline error: {str(e)}. Try Classic mode for this query.",
            code="PIPELINE_ERROR"
        )
        await websocket.send_text(error_msg.model_dump_json())
        await websocket.send_text(DoneMessage().model_dump_json())


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for chat interactions.

    Protocol:
    - Client sends: {"type": "user_message", "content": "...", "agent_mode": "classic"|"pipeline"}
    - Server sends: text, tool_use, tool_result, trace, done, pipeline_progress, pipeline_complete messages
    """
    await websocket.accept()
    connection_id = str(id(websocket))

    logger.info(
        "WebSocket connection established",
        extra={"connection_id": connection_id}
    )

    try:
        while True:
            # Receive message from client
            raw_data = await websocket.receive_text()

            # Generate and set correlation ID for each message (per-request tracing)
            corr_id = generate_correlation_id()
            correlation_id.set(corr_id)

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

            # Create conversation on first message
            settings = get_settings()
            if connection_id not in conversation_ids:
                conv_id = await create_conversation(connection_id, settings.model or "default")
                if conv_id:
                    conversation_ids[connection_id] = conv_id
                    # Send conversation_id to frontend for copy link functionality
                    await websocket.send_text(
                        ConversationStartedMessage(conversation_id=str(conv_id)).model_dump_json()
                    )

            # Route based on agent_mode
            agent_mode = data.get("agent_mode", "classic")

            try:
                if agent_mode == "pipeline":
                    await handle_pipeline_mode(websocket, content, connection_id)
                else:
                    await handle_classic_mode(websocket, content, connection_id)
            except Exception as e:
                await websocket.send_text(
                    ErrorMessage(message=f"Agent error: {str(e)}").model_dump_json()
                )
                await websocket.send_text(DoneMessage().model_dump_json())

    except WebSocketDisconnect:
        # Clean up state for this connection
        rate_limit_state.pop(connection_id, None)
        conversation_history.pop(connection_id, None)
        conversation_ids.pop(connection_id, None)
        turn_counters.pop(connection_id, None)
    except Exception as e:
        logger.error(
            "WebSocket error",
            extra={"error": str(e), "connection_id": connection_id},
            exc_info=True
        )
        rate_limit_state.pop(connection_id, None)
        conversation_history.pop(connection_id, None)
        conversation_ids.pop(connection_id, None)
        turn_counters.pop(connection_id, None)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "kestrel_backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
