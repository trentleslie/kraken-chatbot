"""Database integration tests for KRAKEN backend.

These tests verify database operations using a real PostgreSQL instance
via testcontainers. They exercise asyncpg-specific behavior including:
- UUID type handling
- JSONB columns
- Decimal precision
- Upsert with ON CONFLICT
- Foreign key constraints

All tests in this file require Docker and are marked with @pytest.mark.integration.
Run with: uv run pytest tests/test_database_integration.py -v -m integration
"""

import json
from decimal import Decimal
from uuid import uuid4

import pytest
import pytest_asyncio


# ============================================================================
# Test Conversation CRUD
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestConversationOperations:
    """Test conversation table operations."""

    async def test_conversation_creation(self, clean_db):
        """Test that create_conversation returns a valid UUID."""
        from kestrel_backend import database as db

        # Patch the pool to use our test database
        original_pool = db._pool
        db._pool = clean_db

        try:
            conv_id = await db.create_conversation(
                session_id="test-session-123",
                model="claude-sonnet-4-20250514",
                user_id=None
            )

            assert conv_id is not None
            assert isinstance(conv_id, type(uuid4()))

            # Verify in database
            async with clean_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM kraken_conversations WHERE id = $1",
                    conv_id
                )

            assert row is not None
            assert row["session_id"] == "test-session-123"
            assert row["model"] == "claude-sonnet-4-20250514"
            assert row["status"] == "active"
            assert row["total_turns"] == 0
            assert row["total_tokens"] == 0
        finally:
            db._pool = original_pool

    async def test_conversation_creation_with_user(self, clean_db):
        """Test conversation creation with associated user."""
        from kestrel_backend import database as db
        from kestrel_backend.auth import hash_api_key

        original_pool = db._pool
        db._pool = clean_db

        try:
            # Create a user first
            async with clean_db.acquire() as conn:
                user_row = await conn.fetchrow("""
                    INSERT INTO kraken_users (api_key_hash)
                    VALUES ($1)
                    RETURNING id
                """, hash_api_key("test-key"))
                user_id = user_row["id"]

            # Create conversation with user
            conv_id = await db.create_conversation(
                session_id="test-session",
                model="test-model",
                user_id=str(user_id)
            )

            # Verify user_id is set
            async with clean_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT user_id FROM kraken_conversations WHERE id = $1",
                    conv_id
                )

            assert row["user_id"] == user_id
        finally:
            db._pool = original_pool

    async def test_end_conversation(self, clean_db):
        """Test that end_conversation sets status and ended_at."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            conv_id = await db.create_conversation(
                session_id="test-session",
                model="test-model"
            )

            await db.end_conversation(conv_id, status="completed")

            async with clean_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT status, ended_at FROM kraken_conversations WHERE id = $1",
                    conv_id
                )

            assert row["status"] == "completed"
            assert row["ended_at"] is not None
        finally:
            db._pool = original_pool


# ============================================================================
# Test Turn Operations
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestTurnOperations:
    """Test turn table operations."""

    async def test_turn_persistence(self, clean_db):
        """Test that add_turn stores user/assistant messages."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            conv_id = await db.create_conversation(
                session_id="test-session",
                model="test-model"
            )

            turn_id = await db.add_turn(
                conversation_id=conv_id,
                turn_number=1,
                user_query="What is caffeine?",
                assistant_response="Caffeine is a stimulant.",
                metrics={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.001,
                    "duration_ms": 500,
                    "tool_calls_count": 0,
                    "model": "claude-sonnet-4-20250514"
                }
            )

            assert turn_id is not None

            async with clean_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM kraken_turns WHERE id = $1",
                    turn_id
                )

            assert row["user_query"] == "What is caffeine?"
            assert row["assistant_response"] == "Caffeine is a stimulant."
            assert row["input_tokens"] == 100
            assert row["output_tokens"] == 50
            assert float(row["cost_usd"]) == pytest.approx(0.001)
            assert row["duration_ms"] == 500
        finally:
            db._pool = original_pool

    async def test_conversation_totals_updated(self, clean_db):
        """Test that adding turns updates conversation totals."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            conv_id = await db.create_conversation(
                session_id="test-session",
                model="test-model"
            )

            # Add first turn
            await db.add_turn(
                conversation_id=conv_id,
                turn_number=1,
                user_query="Query 1",
                assistant_response="Response 1",
                metrics={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.001,
                    "model": "test"
                }
            )

            # Add second turn
            await db.add_turn(
                conversation_id=conv_id,
                turn_number=2,
                user_query="Query 2",
                assistant_response="Response 2",
                metrics={
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "cost_usd": 0.002,
                    "model": "test"
                }
            )

            async with clean_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT total_turns, total_tokens, total_cost_usd FROM kraken_conversations WHERE id = $1",
                    conv_id
                )

            assert row["total_turns"] == 2
            assert row["total_tokens"] == 450  # 100+50+200+100
            assert float(row["total_cost_usd"]) == pytest.approx(0.003)
        finally:
            db._pool = original_pool


# ============================================================================
# Test Tool Call Operations
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestToolCallOperations:
    """Test tool call table operations."""

    async def test_tool_call_logging(self, clean_db):
        """Test that add_tool_call stores tool call data."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            conv_id = await db.create_conversation(
                session_id="test-session",
                model="test-model"
            )

            turn_id = await db.add_turn(
                conversation_id=conv_id,
                turn_number=1,
                user_query="Test",
                assistant_response="Test",
                metrics={"model": "test"}
            )

            await db.add_tool_call(
                turn_id=turn_id,
                tool_name="normalize_curie",
                tool_args={"curie": "CHEBI:12345"},
                tool_result={"curie": "CHEBI:12345", "name": "Test Chemical"},
                sequence=1
            )

            async with clean_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM kraken_tool_calls WHERE turn_id = $1",
                    turn_id
                )

            assert row is not None
            assert row["tool_name"] == "normalize_curie"
            # JSONB columns are returned as Python dicts
            assert row["tool_args"]["curie"] == "CHEBI:12345"
            assert row["tool_result"]["name"] == "Test Chemical"
            assert row["result_truncated"] is False
            assert row["sequence_order"] == 1
        finally:
            db._pool = original_pool

    async def test_tool_result_truncation(self, clean_db):
        """Test that results >10KB are truncated."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            conv_id = await db.create_conversation(
                session_id="test-session",
                model="test-model"
            )

            turn_id = await db.add_turn(
                conversation_id=conv_id,
                turn_number=1,
                user_query="Test",
                assistant_response="Test",
                metrics={"model": "test"}
            )

            # Create result larger than 10KB
            large_result = {"data": "x" * 15000}  # ~15KB

            await db.add_tool_call(
                turn_id=turn_id,
                tool_name="large_result_tool",
                tool_args={},
                tool_result=large_result,
                sequence=1
            )

            async with clean_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM kraken_tool_calls WHERE turn_id = $1",
                    turn_id
                )

            assert row["result_truncated"] is True
            assert "_truncated" in row["tool_result"]
            assert "_original_size" in row["tool_result"]
            assert row["tool_result"]["_original_size"] > 10000
        finally:
            db._pool = original_pool


# ============================================================================
# Test Feedback Operations
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestFeedbackOperations:
    """Test feedback table operations."""

    async def test_feedback_persistence(self, clean_db):
        """Test that record_feedback stores feedback."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            conv_id = await db.create_conversation(
                session_id="test-session",
                model="test-model"
            )

            turn_id = await db.add_turn(
                conversation_id=conv_id,
                turn_number=1,
                user_query="Test",
                assistant_response="Test",
                metrics={"model": "test"}
            )

            feedback_id = await db.record_feedback(
                turn_id=turn_id,
                conversation_id=conv_id,
                feedback_type="positive",
                trace_id="trace-123"
            )

            assert feedback_id is not None

            async with clean_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM kraken_feedback WHERE id = $1",
                    feedback_id
                )

            assert row["turn_id"] == turn_id
            assert row["conversation_id"] == conv_id
            assert row["feedback_type"] == "positive"
            assert row["trace_id"] == "trace-123"
        finally:
            db._pool = original_pool

    async def test_feedback_update_existing(self, clean_db):
        """Test that second feedback replaces first (upsert behavior)."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            conv_id = await db.create_conversation(
                session_id="test-session",
                model="test-model"
            )

            turn_id = await db.add_turn(
                conversation_id=conv_id,
                turn_number=1,
                user_query="Test",
                assistant_response="Test",
                metrics={"model": "test"}
            )

            # First feedback - positive
            feedback_id_1 = await db.record_feedback(
                turn_id=turn_id,
                conversation_id=conv_id,
                feedback_type="positive"
            )

            # Second feedback - negative (should replace)
            feedback_id_2 = await db.record_feedback(
                turn_id=turn_id,
                conversation_id=conv_id,
                feedback_type="negative"
            )

            # Should return same ID due to upsert
            assert feedback_id_1 == feedback_id_2

            # Verify only one record exists
            async with clean_db.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM kraken_feedback WHERE turn_id = $1",
                    turn_id
                )
                row = await conn.fetchrow(
                    "SELECT feedback_type FROM kraken_feedback WHERE turn_id = $1",
                    turn_id
                )

            assert count == 1
            assert row["feedback_type"] == "negative"  # Updated value
        finally:
            db._pool = original_pool


# ============================================================================
# Test Conversation Retrieval
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestConversationRetrieval:
    """Test conversation retrieval with all related data."""

    async def test_get_conversation_with_turns(self, clean_db):
        """Test full conversation retrieval with turns and tool calls."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            conv_id = await db.create_conversation(
                session_id="test-session",
                model="claude-sonnet-4-20250514"
            )

            # Add turns
            turn_id_1 = await db.add_turn(
                conversation_id=conv_id,
                turn_number=1,
                user_query="What is caffeine?",
                assistant_response="Caffeine is a stimulant found in coffee.",
                metrics={"input_tokens": 100, "output_tokens": 50, "model": "test"}
            )

            turn_id_2 = await db.add_turn(
                conversation_id=conv_id,
                turn_number=2,
                user_query="How does it work?",
                assistant_response="It blocks adenosine receptors.",
                metrics={"input_tokens": 80, "output_tokens": 40, "model": "test"}
            )

            # Add tool call to first turn
            await db.add_tool_call(
                turn_id=turn_id_1,
                tool_name="normalize_curie",
                tool_args={"curie": "CHEBI:27732"},
                tool_result={"curie": "CHEBI:27732", "name": "caffeine"},
                sequence=1
            )

            # Retrieve full conversation
            conversation = await db.get_conversation_with_turns(conv_id)

            assert conversation is not None
            assert conversation["id"] == str(conv_id)
            assert conversation["model"] == "claude-sonnet-4-20250514"
            assert conversation["total_turns"] == 2

            # Check turns
            assert len(conversation["turns"]) == 2
            assert conversation["turns"][0]["user_query"] == "What is caffeine?"
            assert conversation["turns"][1]["turn_number"] == 2

            # Check tool calls in first turn
            assert len(conversation["turns"][0]["tool_calls"]) == 1
            assert conversation["turns"][0]["tool_calls"][0]["tool_name"] == "normalize_curie"
        finally:
            db._pool = original_pool

    async def test_get_nonexistent_conversation(self, clean_db):
        """Test retrieval of non-existent conversation returns None."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            result = await db.get_conversation_with_turns(uuid4())
            assert result is None
        finally:
            db._pool = original_pool


# ============================================================================
# Test Health Check
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestDatabaseHealthCheck:
    """Test database health check with real database."""

    async def test_db_health_check_success(self, clean_db):
        """Test health check returns healthy with real database."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            is_healthy, latency_ms, error = await db.check_db_health()

            assert is_healthy is True
            assert latency_ms is not None
            assert latency_ms >= 0
            assert latency_ms < 5000  # Should be fast for local container
            assert error is None
        finally:
            db._pool = original_pool


# ============================================================================
# Test User Operations
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestUserOperations:
    """Test user table operations."""

    async def test_create_user(self, clean_db):
        """Test user creation via auth module."""
        from kestrel_backend import database as db
        from kestrel_backend.auth import create_user, hash_api_key

        original_pool = db._pool
        db._pool = clean_db

        try:
            api_key = "test-api-key-12345"
            user_id = await create_user(api_key)

            assert user_id is not None

            # Verify in database
            async with clean_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM kraken_users WHERE api_key_hash = $1",
                    hash_api_key(api_key)
                )

            assert row is not None
            assert row["is_active"] is True
            assert str(row["id"]) == user_id
        finally:
            db._pool = original_pool

    async def test_create_duplicate_user_fails(self, clean_db):
        """Test that duplicate API key creation fails."""
        from kestrel_backend import database as db
        from kestrel_backend.auth import create_user

        original_pool = db._pool
        db._pool = clean_db

        try:
            api_key = "test-api-key-12345"

            # First creation should succeed
            await create_user(api_key)

            # Second creation should fail
            with pytest.raises(ValueError) as exc_info:
                await create_user(api_key)

            assert "already exists" in str(exc_info.value)
        finally:
            db._pool = original_pool


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_empty_metrics(self, clean_db):
        """Test turn creation with minimal metrics."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            conv_id = await db.create_conversation(
                session_id="test-session",
                model="test-model"
            )

            # Minimal metrics (only required fields)
            turn_id = await db.add_turn(
                conversation_id=conv_id,
                turn_number=1,
                user_query="Test",
                assistant_response="Test",
                metrics={}  # Empty metrics
            )

            assert turn_id is not None

            async with clean_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT input_tokens, output_tokens FROM kraken_turns WHERE id = $1",
                    turn_id
                )

            # Should use defaults
            assert row["input_tokens"] == 0
            assert row["output_tokens"] == 0
        finally:
            db._pool = original_pool

    async def test_unicode_content(self, clean_db):
        """Test handling of Unicode characters in content."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            conv_id = await db.create_conversation(
                session_id="test-session",
                model="test-model"
            )

            unicode_query = "What is α-tocopherol (vitamin E)? 日本語テスト 🧬"
            unicode_response = "α-tocopherol is a fat-soluble vitamin. Réponse en français."

            turn_id = await db.add_turn(
                conversation_id=conv_id,
                turn_number=1,
                user_query=unicode_query,
                assistant_response=unicode_response,
                metrics={"model": "test"}
            )

            async with clean_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT user_query, assistant_response FROM kraken_turns WHERE id = $1",
                    turn_id
                )

            assert row["user_query"] == unicode_query
            assert row["assistant_response"] == unicode_response
        finally:
            db._pool = original_pool

    async def test_very_long_content(self, clean_db):
        """Test handling of very long text content."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            conv_id = await db.create_conversation(
                session_id="test-session",
                model="test-model"
            )

            # Create very long content (100KB)
            long_query = "x" * 100000
            long_response = "y" * 100000

            turn_id = await db.add_turn(
                conversation_id=conv_id,
                turn_number=1,
                user_query=long_query,
                assistant_response=long_response,
                metrics={"model": "test"}
            )

            async with clean_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT LENGTH(user_query), LENGTH(assistant_response) FROM kraken_turns WHERE id = $1",
                    turn_id
                )

            assert row[0] == 100000
            assert row[1] == 100000
        finally:
            db._pool = original_pool

    async def test_decimal_precision(self, clean_db):
        """Test that cost values maintain precision."""
        from kestrel_backend import database as db

        original_pool = db._pool
        db._pool = clean_db

        try:
            conv_id = await db.create_conversation(
                session_id="test-session",
                model="test-model"
            )

            # Use a precise decimal value
            precise_cost = 0.000123

            await db.add_turn(
                conversation_id=conv_id,
                turn_number=1,
                user_query="Test",
                assistant_response="Test",
                metrics={"cost_usd": precise_cost, "model": "test"}
            )

            async with clean_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT cost_usd FROM kraken_turns WHERE conversation_id = $1",
                    conv_id
                )

            # NUMERIC(10,6) should maintain this precision
            assert float(row["cost_usd"]) == pytest.approx(precise_cost, rel=1e-6)
        finally:
            db._pool = original_pool
