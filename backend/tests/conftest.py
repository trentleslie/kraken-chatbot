"""Shared fixtures for KRAKEN backend E2E and integration tests.

This module provides pytest fixtures for:
- PostgreSQL database testing via testcontainers
- WebSocket and HTTP endpoint testing via httpx
- State management and mocking utilities
"""

import asyncio
import json
import os
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio


# ============================================================================
# Pytest Configuration
# ============================================================================

# Note: We use function-scoped event loops (asyncio_default_fixture_loop_scope = "function")
# to avoid conflicts between testcontainers and async fixtures. Each test gets its own loop.


# ============================================================================
# Settings Fixtures
# ============================================================================

@pytest.fixture
def test_settings():
    """Override application settings for testing.

    Returns settings with:
    - auth_enabled=False: Skip authentication by default
    - rate_limit_per_minute=100: Higher limit to avoid rate limiting in tests
    - langfuse_enabled=False: Disable observability
    """
    from kestrel_backend.config import Settings

    return Settings(
        host="127.0.0.1",
        port=8000,
        allowed_origins=["http://test"],
        rate_limit_per_minute=100,
        model="test-model",
        auth_enabled=False,
        jwt_secret_key="test-secret-key-for-testing",
        jwt_algorithm="HS256",
        jwt_expire_minutes=60,
        api_keys=[],
        langfuse_enabled=False,
        langfuse_public_key=None,
        langfuse_secret_key=None,
        log_level="WARNING",
        log_format="text",
        log_module_levels={},
    )


@pytest.fixture
def auth_enabled_settings(test_settings):
    """Settings with authentication enabled."""
    test_settings.auth_enabled = True
    return test_settings


# ============================================================================
# PostgreSQL Database Fixtures (Integration Tests)
# ============================================================================
# NOTE: Tests using postgres_pool must be marked with @pytest.mark.integration
# as the marker does NOT propagate from fixtures to tests automatically.

@pytest.fixture(scope="function")
def postgres_container():
    """Start PostgreSQL container (sync fixture to avoid event loop issues).

    This is a sync fixture that manages the container lifecycle. The async
    pool creation happens in postgres_pool.
    """
    testcontainers = pytest.importorskip("testcontainers")
    from testcontainers.postgres import PostgresContainer

    postgres = PostgresContainer("postgres:15")
    postgres.start()

    yield postgres

    postgres.stop()


@pytest_asyncio.fixture(scope="function")
async def postgres_pool(postgres_container) -> AsyncGenerator:
    """Create a real PostgreSQL database via testcontainers.

    This fixture provides a real asyncpg connection pool connected to a
    PostgreSQL container. Use this for database integration tests that need
    to exercise real asyncpg behavior (JSONB, UUID types, etc.).

    IMPORTANT: Tests using this fixture must be marked with:
        @pytest.mark.integration

    Yields:
        asyncpg.Pool: Connection pool connected to the test database

    Example:
        @pytest.mark.integration
        async def test_conversation_creation(postgres_pool):
            async with postgres_pool.acquire() as conn:
                result = await conn.fetchrow("SELECT 1")
    """
    asyncpg = pytest.importorskip("asyncpg")

    # Get connection URL and create pool
    connection_url = postgres_container.get_connection_url()
    # testcontainers returns psycopg2-style URL, convert to asyncpg format
    connection_url = connection_url.replace("postgresql+psycopg2://", "postgresql://")

    pool = await asyncpg.create_pool(connection_url)

    # Run schema creation (simplified migration for tests)
    async with pool.acquire() as conn:
        await conn.execute("""
            -- Create extension for UUID generation
            CREATE EXTENSION IF NOT EXISTS "pgcrypto";

            -- kraken_users table
            CREATE TABLE IF NOT EXISTS kraken_users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                api_key_hash VARCHAR(64) NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT NOW() NOT NULL,
                last_active TIMESTAMP DEFAULT NOW() NOT NULL,
                is_active BOOLEAN DEFAULT TRUE NOT NULL
            );

            -- kraken_conversations table
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
                total_cost_usd NUMERIC(10, 6) DEFAULT 0 NOT NULL,
                user_id UUID REFERENCES kraken_users(id)
            );

            -- kraken_turns table
            CREATE TABLE IF NOT EXISTS kraken_turns (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                conversation_id UUID NOT NULL REFERENCES kraken_conversations(id),
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
                created_at TIMESTAMP DEFAULT NOW() NOT NULL
            );

            -- kraken_tool_calls table
            CREATE TABLE IF NOT EXISTS kraken_tool_calls (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                turn_id UUID NOT NULL REFERENCES kraken_turns(id),
                tool_name VARCHAR NOT NULL,
                tool_args JSONB NOT NULL,
                tool_result JSONB NOT NULL,
                result_truncated BOOLEAN DEFAULT FALSE NOT NULL,
                sequence_order INTEGER NOT NULL
            );

            -- kraken_feedback table
            CREATE TABLE IF NOT EXISTS kraken_feedback (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                turn_id UUID NOT NULL REFERENCES kraken_turns(id) ON DELETE CASCADE,
                conversation_id UUID NOT NULL REFERENCES kraken_conversations(id) ON DELETE CASCADE,
                feedback_type VARCHAR(20) NOT NULL,
                trace_id VARCHAR,
                created_at TIMESTAMP DEFAULT NOW() NOT NULL,
                CONSTRAINT unique_turn_feedback UNIQUE (turn_id)
            );

            -- Create indexes
            CREATE INDEX IF NOT EXISTS ix_kraken_users_api_key_hash ON kraken_users (api_key_hash);
            CREATE INDEX IF NOT EXISTS ix_kraken_conversations_session_id ON kraken_conversations (session_id);
            CREATE INDEX IF NOT EXISTS ix_kraken_conversations_user_id ON kraken_conversations (user_id);
            CREATE INDEX IF NOT EXISTS ix_kraken_turns_conversation_id ON kraken_turns (conversation_id);
            CREATE INDEX IF NOT EXISTS ix_kraken_tool_calls_turn_id ON kraken_tool_calls (turn_id);
            CREATE INDEX IF NOT EXISTS ix_kraken_feedback_conversation_id ON kraken_feedback (conversation_id);
            CREATE INDEX IF NOT EXISTS ix_kraken_feedback_trace_id ON kraken_feedback (trace_id);
        """)

    yield pool

    # Cleanup
    await pool.close()


@pytest_asyncio.fixture(scope="function")
async def clean_db(postgres_pool):
    """Truncate all tables before and after each test.

    Use this fixture when you need a clean database state for each test.
    """
    async def truncate_tables():
        async with postgres_pool.acquire() as conn:
            await conn.execute("""
                TRUNCATE TABLE kraken_feedback CASCADE;
                TRUNCATE TABLE kraken_tool_calls CASCADE;
                TRUNCATE TABLE kraken_turns CASCADE;
                TRUNCATE TABLE kraken_conversations CASCADE;
                TRUNCATE TABLE kraken_users CASCADE;
            """)

    await truncate_tables()
    yield postgres_pool
    await truncate_tables()


# ============================================================================
# HTTP/WebSocket Client Fixtures (E2E Tests)
# ============================================================================

@pytest_asyncio.fixture
async def async_client(test_settings):
    """Create an httpx AsyncClient with ASGITransport for endpoint testing.

    This tests the actual FastAPI application routing without spinning up
    a real server. Uses ASGITransport to connect directly to the ASGI app.

    Yields:
        httpx.AsyncClient: Client configured to test the FastAPI app
    """
    httpx = pytest.importorskip("httpx")
    from httpx import ASGITransport

    # Patch settings to use test configuration
    with patch("kestrel_backend.config.get_settings", return_value=test_settings):
        # Import app after patching settings to get test config
        from kestrel_backend.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            yield client


# ============================================================================
# Connection State Fixtures
# ============================================================================

@pytest.fixture
def clean_connection_state():
    """Reset global connection state dicts before and after each test.

    This fixture cleans up the module-level state variables in main.py
    that track rate limits, conversation history, etc.
    """
    from kestrel_backend import main

    # Store original state
    original_rate_limit = dict(main.rate_limit_state)
    original_history = dict(main.conversation_history)
    original_conv_ids = dict(main.conversation_ids)
    original_turn_counters = dict(main.turn_counters)

    # Clear state
    main.rate_limit_state.clear()
    main.conversation_history.clear()
    main.conversation_ids.clear()
    main.turn_counters.clear()

    yield

    # Restore original state (in case other tests expect it)
    main.rate_limit_state.clear()
    main.rate_limit_state.update(original_rate_limit)
    main.conversation_history.clear()
    main.conversation_history.update(original_history)
    main.conversation_ids.clear()
    main.conversation_ids.update(original_conv_ids)
    main.turn_counters.clear()
    main.turn_counters.update(original_turn_counters)


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_kestrel_client():
    """Mock Kestrel MCP client for deterministic testing.

    Returns a mock client that provides predictable responses
    for entity resolution and KG queries.
    """
    mock_client = MagicMock()

    # Mock tools
    mock_client.get_tools.return_value = {
        "normalize_curie": {},
        "get_node": {},
        "get_edge": {},
    }

    # Mock call_tool for common operations
    async def mock_call_tool(tool_name: str, args: dict):
        if tool_name == "normalize_curie":
            curie = args.get("curie", "")
            return {
                "curie": curie,
                "name": f"Test Entity for {curie}",
                "category": "biolink:ChemicalEntity"
            }
        elif tool_name == "get_node":
            return {
                "id": args.get("curie", ""),
                "attributes": {"degree": 10}
            }
        return {}

    mock_client.call_tool = AsyncMock(side_effect=mock_call_tool)

    return mock_client


@pytest.fixture
def mock_run_agent_turn():
    """Mock the run_agent_turn generator for classic mode testing.

    Yields a sequence of events simulating a complete agent turn.
    """
    from dataclasses import dataclass

    @dataclass
    class MockEvent:
        type: str
        data: dict

    async def mock_generator(prompt: str, session_id: str = None):
        """Simulate a complete agent turn."""
        # Text response
        yield MockEvent(type="text", data={"content": "This is a test response. "})
        yield MockEvent(type="text", data={"content": "It continues here."})

        # Tool use
        yield MockEvent(type="tool_use", data={
            "tool": "normalize_curie",
            "args": {"curie": "CHEBI:12345"}
        })

        # Tool result
        yield MockEvent(type="tool_result", data={
            "tool": "normalize_curie",
            "data": {"curie": "CHEBI:12345", "name": "Test Chemical"}
        })

        # Trace with metrics
        yield MockEvent(type="trace", data={
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.001,
            "duration_ms": 500,
            "tool_calls_count": 1,
            "model": "test-model"
        })

        # Done
        yield MockEvent(type="done", data={})

    return mock_generator


@pytest.fixture
def mock_stream_discovery():
    """Mock the stream_discovery generator for pipeline mode testing.

    Yields a sequence of node updates simulating pipeline execution.
    """
    async def mock_generator(query: str, conversation_history: list = None):
        """Simulate pipeline node updates."""
        nodes = [
            ("intake", {"entities": ["caffeine", "adenosine"]}),
            ("entity_resolution", {"resolved_entities": [
                {"name": "caffeine", "curie": "CHEBI:27732"},
                {"name": "adenosine", "curie": "CHEBI:16335"}
            ]}),
            ("triage", {"triage_scores": [{"curie": "CHEBI:27732", "score": 0.8}]}),
            ("direct_kg", {"kg_results": [{"type": "pathway", "name": "Purine metabolism"}]}),
            ("pathway_enrichment", {"pathways": ["Purine metabolism"]}),
            ("integration", {"bridges": []}),
            ("synthesis", {
                "synthesis_report": "# Analysis\n\nCaffeine blocks adenosine receptors.",
                "hypotheses": ["Caffeine may affect sleep via adenosine pathway"]
            }),
        ]

        for node_name, node_output in nodes:
            yield {
                "type": "node_update",
                "node": node_name,
                "node_output": node_output
            }

    return mock_generator


@pytest.fixture
def mock_database_pool():
    """Mock asyncpg pool for unit tests that don't need real database.

    Returns a mock pool with common operations pre-configured.
    """
    pool = AsyncMock()

    # Track created records
    created_records = {}

    async def mock_fetchrow(query: str, *args):
        if "INSERT INTO kraken_conversations" in query:
            conv_id = uuid4()
            created_records["conversation_id"] = conv_id
            return {"id": conv_id}
        elif "INSERT INTO kraken_turns" in query:
            turn_id = uuid4()
            created_records["turn_id"] = turn_id
            return {"id": turn_id}
        elif "INSERT INTO kraken_feedback" in query:
            feedback_id = uuid4()
            created_records["feedback_id"] = feedback_id
            return {"id": feedback_id}
        elif "SELECT" in query and "kraken_conversations" in query:
            return {
                "id": created_records.get("conversation_id", uuid4()),
                "started_at": None,
                "total_turns": 0,
                "total_tokens": 0,
                "total_cost_usd": 0,
                "model": "test-model",
                "status": "active"
            }
        return None

    async def mock_fetchval(query: str, *args):
        if "SELECT 1" in query:
            return 1
        return None

    async def mock_execute(query: str, *args):
        pass

    async def mock_fetch(query: str, *args):
        return []

    pool.fetchrow = mock_fetchrow
    pool.fetchval = mock_fetchval
    pool.execute = mock_execute
    pool.fetch = mock_fetch
    pool.created_records = created_records

    return pool


# ============================================================================
# JWT Token Fixtures
# ============================================================================

@pytest.fixture
def valid_jwt_token(test_settings):
    """Generate a valid JWT token for testing authentication."""
    from kestrel_backend.auth import create_access_token

    with patch("kestrel_backend.auth.get_settings", return_value=test_settings):
        token = create_access_token({"sub": str(uuid4())})
    return token


@pytest.fixture
def expired_jwt_token(test_settings):
    """Generate an expired JWT token for testing authentication failures."""
    from datetime import timedelta
    from kestrel_backend.auth import create_access_token

    with patch("kestrel_backend.auth.get_settings", return_value=test_settings):
        token = create_access_token({"sub": str(uuid4())}, expires_delta=timedelta(seconds=-1))
    return token


# ============================================================================
# Utility Functions
# ============================================================================

def create_user_message(content: str, agent_mode: str = "classic") -> str:
    """Create a JSON user message for WebSocket testing."""
    return json.dumps({
        "type": "user_message",
        "content": content,
        "agent_mode": agent_mode
    })


def parse_ws_messages(messages: list) -> list:
    """Parse a list of JSON WebSocket messages."""
    return [json.loads(msg) for msg in messages]
