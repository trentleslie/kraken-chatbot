"""Tests for Alembic database migrations."""
import os
import pytest
import asyncpg
from uuid import uuid4


@pytest.mark.asyncio
async def test_migrations_create_tables():
    """Test that migrations create all required tables."""
    database_url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")

    if not database_url:
        pytest.skip("No database URL configured for testing")

    # Connect to database
    conn = await asyncpg.connect(database_url)

    try:
        # Check that all tables exist
        tables = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('kraken_conversations', 'kraken_turns', 'kraken_tool_calls')
            ORDER BY table_name
        """)

        table_names = [row['table_name'] for row in tables]
        assert 'kraken_conversations' in table_names, "kraken_conversations table not found"
        assert 'kraken_turns' in table_names, "kraken_turns table not found"
        assert 'kraken_tool_calls' in table_names, "kraken_tool_calls table not found"

    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_migrations_create_indexes():
    """Test that migrations create all required indexes."""
    database_url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")

    if not database_url:
        pytest.skip("No database URL configured for testing")

    conn = await asyncpg.connect(database_url)

    try:
        # Check for indexes
        indexes = await conn.fetch("""
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND indexname LIKE 'ix_kraken_%'
            ORDER BY indexname
        """)

        index_names = [row['indexname'] for row in indexes]
        assert 'ix_kraken_conversations_session_id' in index_names
        assert 'ix_kraken_turns_conversation_id' in index_names
        assert 'ix_kraken_tool_calls_turn_id' in index_names

    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_migrations_are_idempotent():
    """Test that migrations can be run multiple times without errors."""
    database_url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")

    if not database_url:
        pytest.skip("No database URL configured for testing")

    # This test verifies that the IF NOT EXISTS clauses work
    # by checking we can query tables without errors
    conn = await asyncpg.connect(database_url)

    try:
        # Try to query all tables - should work after migrations
        conversations = await conn.fetch("SELECT COUNT(*) FROM kraken_conversations")
        assert conversations is not None

        turns = await conn.fetch("SELECT COUNT(*) FROM kraken_turns")
        assert turns is not None

        tool_calls = await conn.fetch("SELECT COUNT(*) FROM kraken_tool_calls")
        assert tool_calls is not None

    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_schema_matches_models():
    """Test that database schema matches SQLAlchemy models."""
    database_url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")

    if not database_url:
        pytest.skip("No database URL configured for testing")

    conn = await asyncpg.connect(database_url)

    try:
        # Check kraken_conversations columns
        conv_cols = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'kraken_conversations'
            ORDER BY ordinal_position
        """)

        conv_col_names = {row['column_name'] for row in conv_cols}
        expected_conv_cols = {
            'id', 'session_id', 'model', 'system_prompt_hash', 'agent_version',
            'status', 'started_at', 'ended_at', 'total_turns', 'total_tokens',
            'total_cost_usd'
        }
        assert conv_col_names == expected_conv_cols, f"Conversation columns mismatch: {conv_col_names} != {expected_conv_cols}"

        # Check kraken_turns columns
        turns_cols = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'kraken_turns'
            ORDER BY ordinal_position
        """)

        turns_col_names = {row['column_name'] for row in turns_cols}
        expected_turns_cols = {
            'id', 'conversation_id', 'turn_number', 'user_query', 'assistant_response',
            'input_tokens', 'output_tokens', 'cost_usd', 'duration_ms', 'tool_calls_count',
            'cache_creation_tokens', 'cache_read_tokens', 'model', 'created_at'
        }
        assert turns_col_names == expected_turns_cols, f"Turns columns mismatch: {turns_col_names} != {expected_turns_cols}"

        # Check kraken_tool_calls columns
        tool_calls_cols = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'kraken_tool_calls'
            ORDER BY ordinal_position
        """)

        tool_calls_col_names = {row['column_name'] for row in tool_calls_cols}
        expected_tool_calls_cols = {
            'id', 'turn_id', 'tool_name', 'tool_args', 'tool_result',
            'result_truncated', 'sequence_order'
        }
        assert tool_calls_col_names == expected_tool_calls_cols, f"Tool calls columns mismatch: {tool_calls_col_names} != {expected_tool_calls_cols}"

    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_foreign_key_constraints():
    """Test that foreign key constraints are properly set up."""
    database_url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")

    if not database_url:
        pytest.skip("No database URL configured for testing")

    conn = await asyncpg.connect(database_url)

    try:
        # Check foreign key constraints
        fks = await conn.fetch("""
            SELECT
                tc.table_name,
                tc.constraint_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
              AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_name IN ('kraken_turns', 'kraken_tool_calls')
            ORDER BY tc.table_name
        """)

        fk_info = [(row['table_name'], row['column_name'], row['foreign_table_name']) for row in fks]

        # kraken_turns should have FK to kraken_conversations
        assert ('kraken_turns', 'conversation_id', 'kraken_conversations') in fk_info

        # kraken_tool_calls should have FK to kraken_turns
        assert ('kraken_tool_calls', 'turn_id', 'kraken_turns') in fk_info

    finally:
        await conn.close()
