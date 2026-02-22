"""Unit tests for authentication module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from jose import jwt

from kestrel_backend.auth import (
    hash_api_key,
    verify_api_key,
    create_access_token,
    validate_api_key,
    validate_ws_token,
    create_user,
)
from kestrel_backend.config import get_settings


class TestHashingFunctions:
    """Test API key hashing functions."""

    def test_hash_api_key(self):
        """Test that API keys are hashed consistently."""
        api_key = "test-api-key-123"
        hash1 = hash_api_key(api_key)
        hash2 = hash_api_key(api_key)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64-char hex
        assert hash1 != api_key  # Should not be plaintext

    def test_hash_different_keys(self):
        """Test that different keys produce different hashes."""
        hash1 = hash_api_key("key1")
        hash2 = hash_api_key("key2")

        assert hash1 != hash2

    def test_verify_api_key_success(self):
        """Test successful API key verification."""
        api_key = "test-key"
        hashed = hash_api_key(api_key)

        assert verify_api_key(api_key, hashed) is True

    def test_verify_api_key_failure(self):
        """Test failed API key verification."""
        api_key = "test-key"
        hashed = hash_api_key(api_key)
        wrong_key = "wrong-key"

        assert verify_api_key(wrong_key, hashed) is False


class TestJWTFunctions:
    """Test JWT token functions."""

    def test_create_access_token(self):
        """Test JWT token creation."""
        data = {"sub": "user123"}
        token = create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token can be decoded
        settings = get_settings()
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        assert payload["sub"] == "user123"
        assert "exp" in payload

    def test_create_access_token_with_expiry(self):
        """Test JWT token creation with custom expiry."""
        data = {"sub": "user123"}
        expires_delta = timedelta(minutes=30)
        token = create_access_token(data, expires_delta)

        settings = get_settings()
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])

        # Check expiry is approximately 30 minutes from now
        exp_time = datetime.utcfromtimestamp(payload["exp"])
        expected_exp = datetime.utcnow() + expires_delta
        time_diff = abs((exp_time - expected_exp).total_seconds())

        assert time_diff < 5  # Allow 5 second variance


@pytest.mark.asyncio
class TestValidateApiKey:
    """Test API key validation for REST endpoints."""

    @patch("kestrel_backend.auth._pool")
    async def test_validate_api_key_auth_disabled(self, mock_pool):
        """Test validation when auth is disabled."""
        with patch("kestrel_backend.auth.get_settings") as mock_settings:
            mock_settings.return_value.auth_enabled = False

            credentials = MagicMock()
            credentials.credentials = "test-key"

            result = await validate_api_key(credentials)

            assert result["authenticated"] is False
            assert result["user_id"] is None
            mock_pool.fetchrow.assert_not_called()

    @patch("kestrel_backend.auth._pool")
    async def test_validate_api_key_success(self, mock_pool):
        """Test successful API key validation."""
        with patch("kestrel_backend.auth.get_settings") as mock_settings:
            mock_settings.return_value.auth_enabled = True

            # Mock database response
            mock_pool.fetchrow = AsyncMock(return_value={
                "id": "user-uuid-123",
                "is_active": True
            })
            mock_pool.execute = AsyncMock()

            credentials = MagicMock()
            credentials.credentials = "valid-key"

            result = await validate_api_key(credentials)

            assert result["authenticated"] is True
            assert result["user_id"] == "user-uuid-123"
            assert mock_pool.execute.called  # Should update last_active

    @patch("kestrel_backend.auth._pool")
    async def test_validate_api_key_not_found(self, mock_pool):
        """Test validation with non-existent API key."""
        with patch("kestrel_backend.auth.get_settings") as mock_settings:
            mock_settings.return_value.auth_enabled = True

            mock_pool.fetchrow = AsyncMock(return_value=None)

            credentials = MagicMock()
            credentials.credentials = "invalid-key"

            with pytest.raises(HTTPException) as exc_info:
                await validate_api_key(credentials)

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in str(exc_info.value.detail)

    @patch("kestrel_backend.auth._pool")
    async def test_validate_api_key_inactive_user(self, mock_pool):
        """Test validation with inactive user."""
        with patch("kestrel_backend.auth.get_settings") as mock_settings:
            mock_settings.return_value.auth_enabled = True

            mock_pool.fetchrow = AsyncMock(return_value={
                "id": "user-uuid-123",
                "is_active": False
            })

            credentials = MagicMock()
            credentials.credentials = "inactive-key"

            with pytest.raises(HTTPException) as exc_info:
                await validate_api_key(credentials)

            assert exc_info.value.status_code == 401
            assert "inactive" in str(exc_info.value.detail).lower()

    @patch("kestrel_backend.auth._pool", None)
    async def test_validate_api_key_no_pool(self):
        """Test validation when database pool is not initialized - should fail closed."""
        with patch("kestrel_backend.auth.get_settings") as mock_settings:
            mock_settings.return_value.auth_enabled = True

            credentials = MagicMock()
            credentials.credentials = "test-key"

            # Should fail closed with 503 when DB unavailable (security best practice)
            with pytest.raises(HTTPException) as exc_info:
                await validate_api_key(credentials)

            assert exc_info.value.status_code == 503
            assert "unavailable" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
class TestValidateWsToken:
    """Test WebSocket token validation."""

    @patch("kestrel_backend.auth._pool")
    async def test_validate_ws_token_auth_disabled(self, mock_pool):
        """Test validation when auth is disabled."""
        with patch("kestrel_backend.auth.get_settings") as mock_settings:
            mock_settings.return_value.auth_enabled = False

            result = await validate_ws_token("some-token")

            assert result is None
            mock_pool.fetchrow.assert_not_called()

    @patch("kestrel_backend.auth._pool")
    async def test_validate_ws_token_missing(self, mock_pool):
        """Test validation with missing token."""
        with patch("kestrel_backend.auth.get_settings") as mock_settings:
            mock_settings.return_value.auth_enabled = True

            with pytest.raises(ValueError) as exc_info:
                await validate_ws_token(None)

            assert "Missing authentication token" in str(exc_info.value)

    @patch("kestrel_backend.auth._pool")
    async def test_validate_ws_token_success(self, mock_pool):
        """Test successful WebSocket token validation."""
        with patch("kestrel_backend.auth.get_settings") as mock_settings:
            settings = get_settings()
            mock_settings.return_value.auth_enabled = True
            mock_settings.return_value.jwt_secret_key = settings.jwt_secret_key
            mock_settings.return_value.jwt_algorithm = settings.jwt_algorithm
            mock_settings.return_value.jwt_expire_minutes = settings.jwt_expire_minutes

            # Create valid token
            token = create_access_token({"sub": "user-uuid-123"})

            # Mock database response
            mock_pool.fetchrow = AsyncMock(return_value={
                "id": "user-uuid-123",
                "is_active": True
            })
            mock_pool.execute = AsyncMock()

            result = await validate_ws_token(token)

            assert result["user_id"] == "user-uuid-123"
            assert mock_pool.execute.called  # Should update last_active

    @patch("kestrel_backend.auth._pool")
    async def test_validate_ws_token_expired(self, mock_pool):
        """Test validation with expired token."""
        with patch("kestrel_backend.auth.get_settings") as mock_settings:
            settings = get_settings()
            mock_settings.return_value.auth_enabled = True
            mock_settings.return_value.jwt_secret_key = settings.jwt_secret_key
            mock_settings.return_value.jwt_algorithm = settings.jwt_algorithm

            # Create expired token
            expires_delta = timedelta(seconds=-1)  # Already expired
            token = create_access_token({"sub": "user-uuid-123"}, expires_delta)

            with pytest.raises(ValueError) as exc_info:
                await validate_ws_token(token)

            assert "Invalid or expired token" in str(exc_info.value)

    @patch("kestrel_backend.auth._pool")
    async def test_validate_ws_token_invalid_user(self, mock_pool):
        """Test validation with non-existent user."""
        with patch("kestrel_backend.auth.get_settings") as mock_settings:
            settings = get_settings()
            mock_settings.return_value.auth_enabled = True
            mock_settings.return_value.jwt_secret_key = settings.jwt_secret_key
            mock_settings.return_value.jwt_algorithm = settings.jwt_algorithm
            mock_settings.return_value.jwt_expire_minutes = settings.jwt_expire_minutes

            token = create_access_token({"sub": "non-existent-user"})

            mock_pool.fetchrow = AsyncMock(return_value=None)

            with pytest.raises(ValueError) as exc_info:
                await validate_ws_token(token)

            assert "User not found" in str(exc_info.value)

    @patch("kestrel_backend.auth._pool")
    async def test_validate_ws_token_inactive_user(self, mock_pool):
        """Test validation with inactive user."""
        with patch("kestrel_backend.auth.get_settings") as mock_settings:
            settings = get_settings()
            mock_settings.return_value.auth_enabled = True
            mock_settings.return_value.jwt_secret_key = settings.jwt_secret_key
            mock_settings.return_value.jwt_algorithm = settings.jwt_algorithm
            mock_settings.return_value.jwt_expire_minutes = settings.jwt_expire_minutes

            token = create_access_token({"sub": "user-uuid-123"})

            mock_pool.fetchrow = AsyncMock(return_value={
                "id": "user-uuid-123",
                "is_active": False
            })

            with pytest.raises(ValueError) as exc_info:
                await validate_ws_token(token)

            assert "User is inactive" in str(exc_info.value)


@pytest.mark.asyncio
class TestCreateUser:
    """Test user creation."""

    @patch("kestrel_backend.auth._pool")
    async def test_create_user_success(self, mock_pool):
        """Test successful user creation."""
        mock_pool.fetchrow = AsyncMock(side_effect=[
            None,  # No existing user
            {"id": "new-user-uuid"}  # Created user
        ])

        api_key = "new-api-key"
        user_id = await create_user(api_key)

        assert user_id == "new-user-uuid"

    @patch("kestrel_backend.auth._pool")
    async def test_create_user_duplicate_key(self, mock_pool):
        """Test user creation with duplicate API key."""
        mock_pool.fetchrow = AsyncMock(return_value={
            "id": "existing-user-uuid"
        })

        api_key = "existing-key"

        with pytest.raises(ValueError) as exc_info:
            await create_user(api_key)

        assert "already exists" in str(exc_info.value)

    @patch("kestrel_backend.auth._pool", None)
    async def test_create_user_no_pool(self):
        """Test user creation when database pool is not initialized."""
        with pytest.raises(ValueError) as exc_info:
            await create_user("test-key")

        assert "Database pool not initialized" in str(exc_info.value)
