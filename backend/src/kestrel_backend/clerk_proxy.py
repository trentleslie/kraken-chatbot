"""Clerk Frontend API proxy for custom domain support.

Routes requests from /api/__clerk/* to Clerk's Frontend API (frontend-api.clerk.dev)
with the required authentication headers. This avoids needing CNAME DNS records
for Clerk custom domains.

Reference: biomapper-ui/artifacts/api-server/src/middlewares/clerkProxyMiddleware.ts
"""

import logging

import httpx
from fastapi import APIRouter, Request, Response

from .config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared async HTTP client for proxying (reused across requests)
_http_client: httpx.AsyncClient | None = None

# Paths the Clerk React SDK needs. Reject anything outside this allowlist.
ALLOWED_PATH_PREFIXES = (
    "/v1/client",
    "/v1/environment",
    "/v1/dev_browser",
    "/.well-known/",
    "/npm/",  # Clerk JS SDK bundle loaded via proxy
)

CLERK_FAPI_BASE = "https://frontend-api.clerk.dev"


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    return _http_client


async def close_http_client():
    """Close the shared httpx client. Call from app shutdown/lifespan."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


@router.api_route(
    "/api/__clerk/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    include_in_schema=False,
)
async def clerk_proxy(request: Request, path: str) -> Response:
    """Proxy Clerk FAPI requests through the app's own domain.

    Adds required headers: Clerk-Proxy-Url, Clerk-Secret-Key, X-Forwarded-For.
    Only forwards requests to allowlisted path prefixes.
    """
    settings = get_settings()

    if not settings.clerk_auth_enabled or not settings.clerk_secret_key:
        return Response(content="Clerk proxy not configured", status_code=503)

    # Path allowlist check
    check_path = f"/{path}"
    if not any(check_path.startswith(prefix) for prefix in ALLOWED_PATH_PREFIXES):
        logger.warning(f"Clerk proxy: rejected non-allowlisted path: /{path}")
        return Response(content="Forbidden", status_code=403)

    # Build the proxy URL for the Clerk-Proxy-Url header
    proxy_url = f"{settings.clerk_proxy_url}"

    # Build target URL
    target_url = f"{CLERK_FAPI_BASE}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    # Forward headers (strip hop-by-hop headers)
    forward_headers = {}
    for key, value in request.headers.items():
        lower_key = key.lower()
        if lower_key not in ("host", "connection", "transfer-encoding", "content-length"):
            forward_headers[key] = value

    # Add Clerk-required headers
    forward_headers["Clerk-Proxy-Url"] = proxy_url
    forward_headers["Clerk-Secret-Key"] = settings.clerk_secret_key
    forward_headers["X-Forwarded-For"] = request.client.host if request.client else "127.0.0.1"

    # Read request body
    body = await request.body()

    client = _get_http_client()

    try:
        resp = await client.request(
            method=request.method,
            url=target_url,
            headers=forward_headers,
            content=body if body else None,
        )
    except httpx.HTTPError as e:
        logger.error(f"Clerk proxy error: {e}")
        return Response(content="Clerk API unavailable", status_code=502)

    # Forward response headers (including Set-Cookie for Clerk sessions)
    # Note: httpx auto-decompresses gzip responses, so resp.content is always
    # decompressed. We strip content-encoding (body is no longer compressed)
    # and content-length (let Starlette/nginx set correct length for the
    # decompressed body). This allows nginx gzip module to re-compress and
    # correctly set content-encoding: gzip for the browser.
    response_headers = {}
    for key, value in resp.headers.multi_items():
        lower_key = key.lower()
        if lower_key in ("transfer-encoding", "connection", "content-encoding", "content-length"):
            continue
        response_headers[key] = value

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=response_headers,
    )
