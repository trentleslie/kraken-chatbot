"""Add feedback table for Langfuse integration

Revision ID: 003
Revises: 002
Create Date: 2026-02-22 10:00:00.000000

This migration adds the kraken_feedback table to store user feedback
(thumbs up/down) on agent responses, with Langfuse trace integration.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create kraken_feedback table with IF NOT EXISTS for idempotency."""

    op.execute("""
        CREATE TABLE IF NOT EXISTS kraken_feedback (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            turn_id UUID NOT NULL,
            conversation_id UUID NOT NULL,
            feedback_type VARCHAR(20) NOT NULL,
            trace_id VARCHAR,
            created_at TIMESTAMP DEFAULT NOW() NOT NULL,
            CONSTRAINT fk_kraken_feedback_turn
                FOREIGN KEY (turn_id)
                REFERENCES kraken_turns(id)
                ON DELETE CASCADE,
            CONSTRAINT fk_kraken_feedback_conversation
                FOREIGN KEY (conversation_id)
                REFERENCES kraken_conversations(id)
                ON DELETE CASCADE,
            CONSTRAINT unique_turn_feedback
                UNIQUE (turn_id)
        )
    """)

    # Create index on conversation_id for efficient lookups
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kraken_feedback_conversation_id
        ON kraken_feedback (conversation_id)
    """)

    # Create index on trace_id for Langfuse correlation
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kraken_feedback_trace_id
        ON kraken_feedback (trace_id)
    """)


def downgrade() -> None:
    """Drop kraken_feedback table."""
    op.drop_table('kraken_feedback')
