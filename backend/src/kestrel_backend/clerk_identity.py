"""Resolve a user's verified primary email via the Clerk Backend API."""
import logging
import time
import httpx
from .config import get_settings

logger = logging.getLogger(__name__)

# sub -> (verified email or None, monotonic expiry). Bounded TTL so that when a
# user's Clerk email/verification/domain changes, their server-key eligibility is
# re-checked within TTL seconds instead of being frozen for the process lifetime.
# Without this, revoking a trusted email in Clerk would never take effect here.
_CACHE_TTL_SECONDS = 600.0
_email_cache: dict[str, tuple[str | None, float]] = {}


async def _clerk_get(url: str, headers: dict) -> httpx.Response:
    async with httpx.AsyncClient(timeout=5.0) as client:
        return await client.get(url, headers=headers)


def _cache_put(sub: str, email: str | None) -> str | None:
    _email_cache[sub] = (email, time.monotonic() + _CACHE_TTL_SECONDS)
    return email


async def get_verified_email(user_info: dict) -> str | None:
    sub = user_info.get("sub")
    if not sub:
        return None
    cached = _email_cache.get(sub)
    if cached is not None and cached[1] > time.monotonic():
        return cached[0]

    secret = get_settings().clerk_secret_key
    if not secret:
        return _cache_put(sub, None)

    try:
        resp = await _clerk_get(
            f"https://api.clerk.com/v1/users/{sub}",
            {"Authorization": f"Bearer {secret}"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:  # never let identity lookup break a request
        logger.warning("Clerk user lookup failed for sub=%s", sub)
        return None  # not cached: allow retry on a transient failure

    primary_id = data.get("primary_email_address_id")
    email = None
    for addr in data.get("email_addresses", []):
        if addr.get("id") == primary_id and \
           addr.get("verification", {}).get("status") == "verified":
            email = addr.get("email_address")
            break
    return _cache_put(sub, email)
