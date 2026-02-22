"""
Tests for comprehensive health check endpoints.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


class TestDatabaseHealthCheck:
    """Test database health check function."""

    @pytest.mark.asyncio
    async def test_db_health_check_success(self):
        """Verify database health check returns healthy status."""
        from kestrel_backend.database import check_db_health

        # Mock the pool
        mock_pool = AsyncMock()
        mock_pool.fetchval = AsyncMock(return_value=1)

        with patch("kestrel_backend.database._pool", mock_pool):
            is_healthy, latency_ms, error = await check_db_health()

        assert is_healthy is True
        assert latency_ms is not None
        assert latency_ms >= 0
        assert error is None

    @pytest.mark.asyncio
    async def test_db_health_check_no_pool(self):
        """Verify database health check handles missing pool."""
        from kestrel_backend.database import check_db_health

        with patch("kestrel_backend.database._pool", None):
            is_healthy, latency_ms, error = await check_db_health()

        assert is_healthy is False
        assert latency_ms is None
        assert error == "Database pool not initialized"

    @pytest.mark.asyncio
    async def test_db_health_check_timeout(self):
        """Verify database health check handles timeout."""
        from kestrel_backend.database import check_db_health

        # Mock pool that times out
        mock_pool = AsyncMock()

        async def slow_query():
            await asyncio.sleep(10)  # Exceeds 5s timeout
            return 1

        mock_pool.fetchval = slow_query

        with patch("kestrel_backend.database._pool", mock_pool):
            is_healthy, latency_ms, error = await check_db_health()

        assert is_healthy is False
        assert latency_ms is None
        assert "timeout" in error.lower()

    @pytest.mark.asyncio
    async def test_db_health_check_query_error(self):
        """Verify database health check handles query errors."""
        from kestrel_backend.database import check_db_health

        # Mock pool that raises exception
        mock_pool = AsyncMock()
        mock_pool.fetchval = AsyncMock(side_effect=Exception("Connection refused"))

        with patch("kestrel_backend.database._pool", mock_pool):
            is_healthy, latency_ms, error = await check_db_health()

        assert is_healthy is False
        assert latency_ms is None
        assert "Connection refused" in error


class TestKestrelHealthCheck:
    """Test Kestrel MCP health check function."""

    @pytest.mark.asyncio
    async def test_kestrel_health_check_success(self):
        """Verify Kestrel health check returns healthy status."""
        from kestrel_backend.kestrel_client import check_kestrel_health

        # Mock the client
        mock_client = MagicMock()
        mock_client.get_tools = MagicMock(return_value={"tool1": {}, "tool2": {}})

        with patch("kestrel_backend.kestrel_client.get_kestrel_client", AsyncMock(return_value=mock_client)):
            is_healthy, latency_ms, error = await check_kestrel_health()

        assert is_healthy is True
        assert latency_ms is not None
        assert latency_ms >= 0
        assert error is None

    @pytest.mark.asyncio
    async def test_kestrel_health_check_no_tools(self):
        """Verify Kestrel health check detects missing tools."""
        from kestrel_backend.kestrel_client import check_kestrel_health

        # Mock client with no tools
        mock_client = MagicMock()
        mock_client.get_tools = MagicMock(return_value={})

        with patch("kestrel_backend.kestrel_client.get_kestrel_client", AsyncMock(return_value=mock_client)):
            is_healthy, latency_ms, error = await check_kestrel_health()

        assert is_healthy is False
        assert latency_ms is None
        assert "tools not available" in error.lower()

    @pytest.mark.asyncio
    async def test_kestrel_health_check_timeout(self):
        """Verify Kestrel health check handles timeout."""
        from kestrel_backend.kestrel_client import check_kestrel_health

        async def slow_connect():
            await asyncio.sleep(10)  # Exceeds 5s timeout
            return MagicMock()

        with patch("kestrel_backend.kestrel_client.get_kestrel_client", slow_connect):
            is_healthy, latency_ms, error = await check_kestrel_health()

        assert is_healthy is False
        assert latency_ms is None
        assert "timeout" in error.lower()

    @pytest.mark.asyncio
    async def test_kestrel_health_check_connection_error(self):
        """Verify Kestrel health check handles connection errors."""
        from kestrel_backend.kestrel_client import check_kestrel_health

        async def failing_connect():
            raise Exception("Connection refused")

        with patch("kestrel_backend.kestrel_client.get_kestrel_client", failing_connect):
            is_healthy, latency_ms, error = await check_kestrel_health()

        assert is_healthy is False
        assert latency_ms is None
        assert "Connection refused" in error


class TestLangfuseHealthCheck:
    """Test Langfuse health check function."""

    def test_langfuse_health_check_disabled(self):
        """Verify Langfuse health check when disabled."""
        from kestrel_backend.main import check_langfuse_health

        # Mock settings with Langfuse disabled
        with patch("kestrel_backend.main.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.langfuse_enabled = False
            mock_get_settings.return_value = mock_settings

            is_healthy, error = check_langfuse_health()

        assert is_healthy is True
        assert error is None

    def test_langfuse_health_check_enabled_and_initialized(self):
        """Verify Langfuse health check when enabled and initialized."""
        from kestrel_backend.main import check_langfuse_health

        # Mock settings with Langfuse enabled
        with patch("kestrel_backend.main.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.langfuse_enabled = True
            mock_get_settings.return_value = mock_settings

            # Mock initialized client
            with patch("kestrel_backend.main._get_pipeline_langfuse", return_value=MagicMock()):
                is_healthy, error = check_langfuse_health()

        assert is_healthy is True
        assert error is None

    def test_langfuse_health_check_not_initialized(self):
        """Verify Langfuse health check when client not initialized."""
        from kestrel_backend.main import check_langfuse_health

        # Mock settings with Langfuse enabled
        with patch("kestrel_backend.main.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.langfuse_enabled = True
            mock_get_settings.return_value = mock_settings

            # Mock uninitialized client
            with patch("kestrel_backend.main._get_pipeline_langfuse", return_value=None):
                is_healthy, error = check_langfuse_health()

        assert is_healthy is False
        assert "not initialized" in error.lower()


class TestReadinessEndpoint:
    """Test /ready endpoint integration."""

    @pytest.mark.asyncio
    async def test_ready_endpoint_all_healthy(self):
        """Verify /ready endpoint returns 200 when all dependencies healthy."""
        from kestrel_backend.main import app

        # Mock all health checks as healthy
        with patch("kestrel_backend.database.check_db_health", AsyncMock(return_value=(True, 5, None))):
            with patch("kestrel_backend.kestrel_client.check_kestrel_health", AsyncMock(return_value=(True, 120, None))):
                with patch("kestrel_backend.main.check_langfuse_health", return_value=(True, None)):
                    with TestClient(app) as client:
                        response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["checks"]["database"]["status"] == "healthy"
        assert data["checks"]["database"]["latency_ms"] == 5
        assert data["checks"]["kestrel"]["status"] == "healthy"
        assert data["checks"]["kestrel"]["latency_ms"] == 120
        assert data["checks"]["langfuse"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_ready_endpoint_db_unhealthy(self):
        """Verify /ready endpoint returns 503 when database unhealthy."""
        from kestrel_backend.main import app

        # Mock database as unhealthy
        with patch("kestrel_backend.database.check_db_health", AsyncMock(return_value=(False, None, "Connection error"))):
            with patch("kestrel_backend.kestrel_client.check_kestrel_health", AsyncMock(return_value=(True, 120, None))):
                with patch("kestrel_backend.main.check_langfuse_health", return_value=(True, None)):
                    with TestClient(app) as client:
                        response = client.get("/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["database"]["status"] == "unhealthy"
        assert data["checks"]["database"]["error"] == "Connection error"

    @pytest.mark.asyncio
    async def test_ready_endpoint_kestrel_unhealthy(self):
        """Verify /ready endpoint returns 503 when Kestrel unhealthy."""
        from kestrel_backend.main import app

        # Mock Kestrel as unhealthy
        with patch("kestrel_backend.database.check_db_health", AsyncMock(return_value=(True, 5, None))):
            with patch("kestrel_backend.kestrel_client.check_kestrel_health", AsyncMock(return_value=(False, None, "Timeout"))):
                with patch("kestrel_backend.main.check_langfuse_health", return_value=(True, None)):
                    with TestClient(app) as client:
                        response = client.get("/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["kestrel"]["status"] == "unhealthy"
        assert data["checks"]["kestrel"]["error"] == "Timeout"

    @pytest.mark.asyncio
    async def test_ready_endpoint_langfuse_degraded(self):
        """Verify /ready endpoint returns 200 with degraded when Langfuse unhealthy."""
        from kestrel_backend.main import app

        # Mock Langfuse as unhealthy (non-critical)
        with patch("kestrel_backend.database.check_db_health", AsyncMock(return_value=(True, 5, None))):
            with patch("kestrel_backend.kestrel_client.check_kestrel_health", AsyncMock(return_value=(True, 120, None))):
                with patch("kestrel_backend.main.check_langfuse_health", return_value=(False, "Not initialized")):
                    with TestClient(app) as client:
                        response = client.get("/ready")

        # Should return 200 since Langfuse is non-critical
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["checks"]["langfuse"]["status"] == "degraded"
        assert data["checks"]["langfuse"]["error"] == "Not initialized"

    @pytest.mark.asyncio
    async def test_ready_endpoint_all_unhealthy(self):
        """Verify /ready endpoint returns 503 when all dependencies unhealthy."""
        from kestrel_backend.main import app

        # Mock all as unhealthy
        with patch("kestrel_backend.database.check_db_health", AsyncMock(return_value=(False, None, "DB error"))):
            with patch("kestrel_backend.kestrel_client.check_kestrel_health", AsyncMock(return_value=(False, None, "Kestrel error"))):
                with patch("kestrel_backend.main.check_langfuse_health", return_value=(False, "Langfuse error")):
                    with TestClient(app) as client:
                        response = client.get("/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["database"]["status"] == "unhealthy"
        assert data["checks"]["kestrel"]["status"] == "unhealthy"
        assert data["checks"]["langfuse"]["status"] == "degraded"


class TestHealthEndpoint:
    """Test that /health endpoint remains simple."""

    def test_health_endpoint_always_returns_healthy(self):
        """Verify /health endpoint is simple liveness check."""
        from kestrel_backend.main import app

        # No mocking needed - /health should always return 200
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "kestrel-backend"
