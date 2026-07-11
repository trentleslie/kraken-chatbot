"""Resolve a user's verified primary email via the Clerk Backend API."""
import logging
import httpx
from .config import get_settings

logger = logging.getLogger(__name__)
_email_cache: dict[str, str | None] = {}  # sub -> verified email or None


async def _clerk_get(url: str, headers: dict) -> httpx.Response:
    async with httpx.AsyncClient(timeout=5.0) as client:
        return await client.get(url, headers=headers)


async def get_verified_email(user_info: dict) -> str | None:
    sub = user_info.get("sub")
    if not sub:
        return None
    if sub in _email_cache:
        return _email_cache[sub]

    secret = get_settings().clerk_secret_key
    if not secret:
        _email_cache[sub] = None
        return None

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
    _email_cache[sub] = email
    return email
