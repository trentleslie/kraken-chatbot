"""Database persistence for KRAKEN conversations."""
import asyncpg
import hashlib
import json
import os
from decimal import Decimal
from typing import Optional
from uuid import UUID

from .agent import SYSTEM_PROMPT, AGENT_VERSION

_pool: Optional[asyncpg.Pool] = None
MAX_TOOL_RESULT_SIZE = 10 * 1024  # 10KB limit for tool results


def get_prompt_hash() -> str:
    """Return first 16 chars of SHA256 hash of current system prompt."""
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16]


def truncate_tool_result(result: dict) -> tuple[dict, bool]:
    """Truncate tool result if >10KB. Returns (result, was_truncated)."""
    json_str = json.dumps(result)
    if len(json_str) <= MAX_TOOL_RESULT_SIZE:
        return result, False
    # Store raw string preview (not valid JSON, but good for human review)
    truncated = {
        "_truncated": True,
        "_original_size": len(json_str),
        "_preview": json_str[:MAX_TOOL_RESULT_SIZE]
    }
    return truncated, True


async def init_db():
    """Initialize connection pool."""
    global _pool
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("WARNING: DATABASE_URL not set, conversation persistence disabled")
        return
    _pool = await asyncpg.create_pool(database_url)
    print("Database connection pool initialized")


async def close_db():
    """Close connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("Database connection pool closed")


async def create_conversation(session_id: str, model: str) -> Optional[UUID]:
    """Create new conversation on first message, return ID."""
    if not _pool:
        return None
    prompt_hash = get_prompt_hash()
    row = await _pool.fetchrow("""
        INSERT INTO kraken_conversations (session_id, model, system_prompt_hash, agent_version)
        VALUES ($1, $2, $3, $4) RETURNING id
    """, session_id, model, prompt_hash, AGENT_VERSION)
    return row["id"]


async def add_turn(
    conversation_id: UUID,
    turn_number: int,
    user_query: str,
    assistant_response: str,
    metrics: dict
) -> Optional[UUID]:
    """Add turn to conversation, return turn ID."""
    if not _pool or not conversation_id:
        return None

    row = await _pool.fetchrow("""
        INSERT INTO kraken_turns
        (conversation_id, turn_number, user_query, assistant_response,
         input_tokens, output_tokens, cost_usd, duration_ms, tool_calls_count,
         cache_creation_tokens, cache_read_tokens, model)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING id
    """,
        conversation_id,
        turn_number,
        user_query,
        assistant_response,
        metrics.get("input_tokens", 0),
        metrics.get("output_tokens", 0),
        Decimal(str(metrics.get("cost_usd", 0))),
        metrics.get("duration_ms", 0),
        metrics.get("tool_calls_count", 0),
        metrics.get("cache_creation_tokens", 0),
        metrics.get("cache_read_tokens", 0),
        metrics.get("model", "unknown"),
    )

    # Update conversation totals
    await _pool.execute("""
        UPDATE kraken_conversations
        SET total_turns = total_turns + 1,
            total_tokens = total_tokens + $2 + $3,
            total_cost_usd = total_cost_usd + $4
        WHERE id = $1
    """,
        conversation_id,
        metrics.get("input_tokens", 0),
        metrics.get("output_tokens", 0),
        Decimal(str(metrics.get("cost_usd", 0))),
    )

    return row["id"]


async def add_tool_call(
    turn_id: UUID,
    tool_name: str,
    tool_args: dict,
    tool_result: dict,
    sequence: int
):
    """Record a tool call within a turn. Truncates large results."""
    if not _pool or not turn_id:
        return
    result, truncated = truncate_tool_result(tool_result)
    await _pool.execute("""
        INSERT INTO kraken_tool_calls
        (turn_id, tool_name, tool_args, tool_result, result_truncated, sequence_order)
        VALUES ($1, $2, $3, $4, $5, $6)
    """, turn_id, tool_name, json.dumps(tool_args), json.dumps(result), truncated, sequence)


async def end_conversation(conversation_id: UUID, status: str = "completed"):
    """Mark conversation as ended."""
    if not _pool or not conversation_id:
        return
    await _pool.execute("""
        UPDATE kraken_conversations
        SET ended_at = NOW(), status = $2
        WHERE id = $1
    """, conversation_id, status)


async def get_conversation_with_turns(conversation_id: UUID) -> dict | None:
    """Retrieve conversation with all turns and tool calls for sharing.

    Returns full conversation data including metadata, turns, and tool calls
    in a format suitable for rendering in the frontend SharedConversation view.
    """
    if not _pool:
        return None

    # Get conversation metadata
    conv = await _pool.fetchrow("""
        SELECT id, started_at, total_turns, total_tokens, total_cost_usd, model, status
        FROM kraken_conversations WHERE id = $1
    """, conversation_id)

    if not conv:
        return None

    # Get turns with aggregated tool calls
    turns = await _pool.fetch("""
        SELECT t.id, t.turn_number, t.user_query, t.assistant_response,
               t.input_tokens, t.output_tokens, t.cost_usd, t.duration_ms,
               t.tool_calls_count, t.created_at, t.model,
               COALESCE(json_agg(
                   json_build_object(
                       'tool_name', tc.tool_name,
                       'tool_args', tc.tool_args,
                       'tool_result', tc.tool_result,
                       'sequence_order', tc.sequence_order
                   ) ORDER BY tc.sequence_order
               ) FILTER (WHERE tc.id IS NOT NULL), '[]') as tool_calls
        FROM kraken_turns t
        LEFT JOIN kraken_tool_calls tc ON t.id = tc.turn_id
        WHERE t.conversation_id = $1
        GROUP BY t.id
        ORDER BY t.turn_number
    """, conversation_id)

    # Transform to JSON-serializable format
    turns_list = []
    for t in turns:
        turn_dict = dict(t)
        # Convert UUID to string
        turn_dict['id'] = str(turn_dict['id'])
        # Handle Decimal for cost
        if turn_dict.get('cost_usd') is not None:
            turn_dict['cost_usd'] = float(turn_dict['cost_usd'])
        # Parse tool_calls from JSON string to list (asyncpg returns json_agg as string)
        if isinstance(turn_dict.get('tool_calls'), str):
            turn_dict['tool_calls'] = json.loads(turn_dict['tool_calls'])
        turns_list.append(turn_dict)

    return {
        "id": str(conv["id"]),
        "started_at": conv["started_at"].isoformat() if conv["started_at"] else None,
        "total_turns": conv["total_turns"],
        "total_tokens": conv["total_tokens"],
        "total_cost_usd": float(conv["total_cost_usd"]) if conv["total_cost_usd"] else 0,
        "model": conv["model"],
        "status": conv["status"],
        "turns": turns_list
    }
