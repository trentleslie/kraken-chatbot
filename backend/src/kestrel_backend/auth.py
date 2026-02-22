"""Authentication and authorization utilities for KRAKEN backend.

Provides JWT-based authentication and API key validation for protecting
WebSocket and REST endpoints.
"""

import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import get_settings
from .database import _pool

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme for REST endpoints
security = HTTPBearer()


def hash_api_key(api_key: str) -> str:
    """Hash an API key using SHA256 for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash."""
    return hash_api_key(api_key) == hashed_key


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload to encode in the token
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token string
    """
    settings = get_settings()
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


async def validate_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """
    FastAPI dependency for validating API keys on REST endpoints.

    Args:
        credentials: HTTP Bearer credentials from request header

    Returns:
        Dictionary with user information

    Raises:
        HTTPException: If authentication fails
    """
    settings = get_settings()

    # If auth is disabled, allow all requests
    if not settings.auth_enabled:
        return {"authenticated": False, "user_id": None}

    api_key = credentials.credentials
    api_key_hash = hash_api_key(api_key)

    # Check if API key exists in database
    if not _pool:
        logger.warning("Database pool not initialized, authentication bypassed")
        return {"authenticated": False, "user_id": None}

    try:
        user = await _pool.fetchrow(
            "SELECT id, is_active FROM kraken_users WHERE api_key_hash = $1",
            api_key_hash
        )

        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key")

        if not user["is_active"]:
            raise HTTPException(status_code=401, detail="API key is inactive")

        # Update last_active timestamp
        await _pool.execute(
            "UPDATE kraken_users SET last_active = NOW() WHERE id = $1",
            user["id"]
        )

        return {
            "authenticated": True,
            "user_id": str(user["id"])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API key validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Authentication error")


async def validate_ws_token(token: Optional[str]) -> Optional[dict]:
    """
    Validate a JWT token for WebSocket connections.

    Args:
        token: JWT token from query parameter

    Returns:
        Dictionary with user info if valid, None if auth is disabled

    Raises:
        ValueError: If token is invalid or expired
    """
    settings = get_settings()

    # If auth is disabled, allow all connections
    if not settings.auth_enabled:
        return None

    if not token:
        raise ValueError("Missing authentication token")

    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id = payload.get("sub")

        if not user_id:
            raise ValueError("Invalid token payload")

        # Verify user exists and is active
        if not _pool:
            logger.warning("Database pool not initialized, WebSocket authentication bypassed")
            return None

        user = await _pool.fetchrow(
            "SELECT id, is_active FROM kraken_users WHERE id = $1",
            user_id
        )

        if not user:
            raise ValueError("User not found")

        if not user["is_active"]:
            raise ValueError("User is inactive")

        # Update last_active timestamp
        await _pool.execute(
            "UPDATE kraken_users SET last_active = NOW() WHERE id = $1",
            user["id"]
        )

        return {
            "user_id": str(user["id"])
        }

    except JWTError as e:
        logger.warning(f"JWT validation error: {e}")
        raise ValueError("Invalid or expired token")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"WebSocket token validation error: {e}", exc_info=True)
        raise ValueError("Authentication error")


async def create_user(api_key: str) -> str:
    """
    Create a new user with an API key.

    Args:
        api_key: The API key to associate with the user

    Returns:
        User ID (UUID as string)

    Raises:
        ValueError: If user creation fails
    """
    if not _pool:
        raise ValueError("Database pool not initialized")

    api_key_hash = hash_api_key(api_key)

    try:
        # Check if API key already exists
        existing = await _pool.fetchrow(
            "SELECT id FROM kraken_users WHERE api_key_hash = $1",
            api_key_hash
        )

        if existing:
            raise ValueError("API key already exists")

        # Create new user
        row = await _pool.fetchrow(
            """
            INSERT INTO kraken_users (api_key_hash, created_at, last_active, is_active)
            VALUES ($1, NOW(), NOW(), TRUE)
            RETURNING id
            """,
            api_key_hash
        )

        return str(row["id"])

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"User creation error: {e}", exc_info=True)
        raise ValueError("Failed to create user")
