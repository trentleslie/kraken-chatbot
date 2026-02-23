"""Add users table and user_id to conversations

Revision ID: 002
Revises: 001
Create Date: 2026-02-22 12:00:00.000000

This migration adds authentication support by creating the kraken_users table
and adding a user_id foreign key to kraken_conversations. The user_id column
is nullable for backwards compatibility with existing conversations.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add kraken_users table and user_id to conversations."""

    # Create kraken_users table
    op.execute("""
        CREATE TABLE IF NOT EXISTS kraken_users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            api_key_hash VARCHAR(64) NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT NOW() NOT NULL,
            last_active TIMESTAMP DEFAULT NOW() NOT NULL,
            is_active BOOLEAN DEFAULT TRUE NOT NULL
        )
    """)

    # Create index on api_key_hash for fast lookups
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kraken_users_api_key_hash
        ON kraken_users (api_key_hash)
    """)

    # Add user_id column to kraken_conversations (nullable for backwards compatibility)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'kraken_conversations'
                AND column_name = 'user_id'
            ) THEN
                ALTER TABLE kraken_conversations
                ADD COLUMN user_id UUID;

                -- Add foreign key constraint
                ALTER TABLE kraken_conversations
                ADD CONSTRAINT fk_kraken_conversations_user
                FOREIGN KEY (user_id) REFERENCES kraken_users(id);

                -- Create index on user_id
                CREATE INDEX ix_kraken_conversations_user_id
                ON kraken_conversations (user_id);
            END IF;
        END $$;
    """)


def downgrade() -> None:
    """Remove user_id from conversations and drop users table."""
    # Remove foreign key and column from conversations
    op.execute("""
        ALTER TABLE kraken_conversations
        DROP CONSTRAINT IF EXISTS fk_kraken_conversations_user
    """)

    op.execute("""
        DROP INDEX IF EXISTS ix_kraken_conversations_user_id
    """)

    op.execute("""
        ALTER TABLE kraken_conversations
        DROP COLUMN IF EXISTS user_id
    """)

    # Drop users table
    op.drop_table('kraken_users')
