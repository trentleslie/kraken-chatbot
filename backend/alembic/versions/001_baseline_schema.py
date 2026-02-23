"""Baseline KRAKEN database schema

Revision ID: 001
Revises:
Create Date: 2026-02-22 04:00:00.000000

This migration creates the initial database schema for KRAKEN conversations,
turns, and tool calls tracking. Uses IF NOT EXISTS for idempotency to support
production databases where tables may already exist.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create KRAKEN tables with IF NOT EXISTS for idempotency."""

    # Create kraken_conversations table
    op.execute("""
        CREATE TABLE IF NOT EXISTS kraken_conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id VARCHAR NOT NULL,
            model VARCHAR NOT NULL,
            system_prompt_hash VARCHAR(16) NOT NULL,
            agent_version VARCHAR NOT NULL,
            status VARCHAR DEFAULT 'active' NOT NULL,
            started_at TIMESTAMP DEFAULT NOW() NOT NULL,
            ended_at TIMESTAMP,
            total_turns INTEGER DEFAULT 0 NOT NULL,
            total_tokens INTEGER DEFAULT 0 NOT NULL,
            total_cost_usd NUMERIC(10, 6) DEFAULT 0 NOT NULL
        )
    """)

    # Create index on session_id if it doesn't exist
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kraken_conversations_session_id
        ON kraken_conversations (session_id)
    """)

    # Create kraken_turns table
    op.execute("""
        CREATE TABLE IF NOT EXISTS kraken_turns (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id UUID NOT NULL,
            turn_number INTEGER NOT NULL,
            user_query TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            input_tokens INTEGER DEFAULT 0 NOT NULL,
            output_tokens INTEGER DEFAULT 0 NOT NULL,
            cost_usd NUMERIC(10, 6) DEFAULT 0 NOT NULL,
            duration_ms INTEGER DEFAULT 0 NOT NULL,
            tool_calls_count INTEGER DEFAULT 0 NOT NULL,
            cache_creation_tokens INTEGER DEFAULT 0 NOT NULL,
            cache_read_tokens INTEGER DEFAULT 0 NOT NULL,
            model VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT NOW() NOT NULL,
            CONSTRAINT fk_kraken_turns_conversation
                FOREIGN KEY (conversation_id)
                REFERENCES kraken_conversations(id)
        )
    """)

    # Create index on conversation_id if it doesn't exist
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kraken_turns_conversation_id
        ON kraken_turns (conversation_id)
    """)

    # Create kraken_tool_calls table
    op.execute("""
        CREATE TABLE IF NOT EXISTS kraken_tool_calls (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            turn_id UUID NOT NULL,
            tool_name VARCHAR NOT NULL,
            tool_args JSONB NOT NULL,
            tool_result JSONB NOT NULL,
            result_truncated BOOLEAN DEFAULT FALSE NOT NULL,
            sequence_order INTEGER NOT NULL,
            CONSTRAINT fk_kraken_tool_calls_turn
                FOREIGN KEY (turn_id)
                REFERENCES kraken_turns(id)
        )
    """)

    # Create index on turn_id if it doesn't exist
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kraken_tool_calls_turn_id
        ON kraken_tool_calls (turn_id)
    """)


def downgrade() -> None:
    """Drop all KRAKEN tables."""
    op.drop_table('kraken_tool_calls')
    op.drop_table('kraken_turns')
    op.drop_table('kraken_conversations')
