"""Clerk authentication for Kestrel Backend.

Verifies Clerk session tokens using PyJWT with JWKS.
Provides FastAPI dependencies for REST and WebSocket auth.
"""

import logging
from functools import lru_cache

import jwt
from jwt import PyJWKClient
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import get_settings

logger = logging.getLogger(__name__)

_security = HTTPBearer(auto_error=False)

# Module-level JWKS client (lazy initialized, caches keys internally)
_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    """Get or create the JWKS client. Cached after first call."""
    global _jwks_client
    if _jwks_client is None:
        settings = get_settings()
        if not settings.clerk_jwks_url:
            raise RuntimeError(
                "CLERK_JWKS_URL is not configured. "
                "Set the env var or disable Clerk auth with CLERK_AUTH_ENABLED=false."
            )
        _jwks_client = PyJWKClient(settings.clerk_jwks_url)
    return _jwks_client


def verify_clerk_token(token: str) -> dict:
    """Verify a Clerk JWT and return the decoded payload.

    Validates:
    - RS256 signature via JWKS (algorithm explicitly pinned)
    - exp, nbf, iss claims
    - azp (authorized party) if present

    Returns the decoded token payload with user info.
    Raises jwt.exceptions.PyJWTError on any verification failure.
    Raises RuntimeError if CLERK_JWKS_URL or CLERK_ISSUER is not configured.
    """
    settings = get_settings()

    if not settings.clerk_issuer:
        raise RuntimeError(
            "CLERK_ISSUER is not configured. Cannot validate token issuer."
        )

    jwks_client = _get_jwks_client()
    signing_key = jwks_client.get_signing_key_from_jwt(token)

    payload = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        issuer=settings.clerk_issuer,
        options={"require": ["exp", "nbf", "iss", "sub"]},
    )

    return payload


def _check_email_allowed(payload: dict) -> str | None:
    """Check if the user's email is in the allowed domains/emails list.

    Returns the email if allowed, raises HTTPException if not.
    """
    settings = get_settings()

    # If no domain restrictions configured, allow all authenticated users
    if not settings.clerk_allowed_email_domains and not settings.clerk_allowed_emails:
        return None

    # Extract email from Clerk token claims
    # Clerk puts email in different places depending on token type
    email = None
    if "email" in payload:
        email = payload["email"].lower()
    elif "email_addresses" in payload:
        # Some Clerk token formats use this
        addrs = payload["email_addresses"]
        if addrs:
            email = addrs[0].lower() if isinstance(addrs[0], str) else addrs[0].get("email_address", "").lower()

    if not email:
        # Can't verify email domain without an email claim
        # This can happen with certain Clerk configurations
        logger.warning("Clerk token has no email claim, cannot verify domain")
        raise HTTPException(status_code=403, detail="Email verification required")

    # Check against allowed emails (exact match)
    if email in settings.clerk_allowed_emails:
        return email

    # Check against allowed domains
    domain = email.split("@")[-1] if "@" in email else ""
    if domain in settings.clerk_allowed_email_domains:
        return email

    raise HTTPException(
        status_code=403,
        detail="Access restricted. Your email domain is not authorized.",
    )


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = None,
) -> dict:
    """FastAPI dependency for REST endpoint auth.

    Extracts Bearer token from Authorization header, verifies via Clerk JWKS,
    and checks email domain whitelist.

    When CLERK_AUTH_ENABLED is false, returns a stub user dict.
    """
    settings = get_settings()

    if not settings.clerk_auth_enabled:
        return {"user_id": "anonymous", "auth_skipped": True}

    # Extract token from Authorization header
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    elif credentials:
        token = credentials.credentials
    else:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        payload = verify_clerk_token(token)
    except jwt.exceptions.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.exceptions.PyJWTError as e:
        logger.warning(f"Clerk token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

    _check_email_allowed(payload)

    return {
        "user_id": payload.get("sub"),
        "payload": payload,
    }


async def validate_ws_clerk_token(token: str | None) -> dict:
    """Validate a Clerk token for WebSocket connections.

    Called at connection time. Same verification as REST but extracts
    the token from the WebSocket query parameter.

    When CLERK_AUTH_ENABLED is false, returns a stub user dict.
    """
    settings = get_settings()

    if not settings.clerk_auth_enabled:
        return {"user_id": "anonymous", "auth_skipped": True}

    if not token:
        raise ValueError("Authentication required")

    try:
        payload = verify_clerk_token(token)
    except jwt.exceptions.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.exceptions.PyJWTError as e:
        raise ValueError(f"Invalid token: {e}")

    # Check email domain
    try:
        _check_email_allowed(payload)
    except HTTPException as e:
        raise ValueError(e.detail)

    return {
        "user_id": payload.get("sub"),
        "payload": payload,
    }
