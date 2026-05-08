---
title: "Clerk Authentication Integration: React + FastAPI Pattern"
date: 2026-05-06
last_updated: 2026-05-07
category: best-practices
module: authentication
problem_type: best_practice
component: authentication
severity: high
applies_when:
  - Adding Clerk auth to a Python/FastAPI backend (no official Clerk Python SDK)
  - Migrating a reference implementation from Node.js (@clerk/express) to Python
  - Deploying behind nginx with certbot on a VPS (Lightsail, DigitalOcean, etc.)
  - Backend must verify Clerk JWTs without the official SDK
  - Activating a Clerk production instance with custom domain DNS
tags:
  - clerk
  - react
  - fastapi
  - websocket
  - jwt
  - pyjwt
  - nginx-proxy
  - authentication
  - vite
  - custom-domain
  - dns
  - google-oauth
  - production-setup
---

# Clerk Authentication Integration: React + FastAPI Pattern

## Context

Adding Clerk authentication to a full-stack application (kraken-chatbot) where the backend is Python/FastAPI, not Node.js/Express. Most Clerk documentation and SDKs assume a Node.js backend (`@clerk/express`), so adapting to Python requires manual JWT verification and a custom FAPI proxy. The reference implementation was biomapper-ui (same Clerk account, same server, Express backend).

The deployment target is AWS Lightsail with nginx reverse proxy and certbot-managed TLS. The frontend is React 18 + Vite with wouter router. WebSocket connections also require authentication.

## Guidance

### Frontend (React + Vite with wouter)

- Use `@clerk/react` with `ClerkProvider` in `App.tsx`, passing `publishableKey` and optionally `proxyUrl` (undefined in dev, set for prod custom domains).
- Create `/login` and `/sign-up` routes using Clerk's `SignIn`/`SignUp` components with `routing="path"`.
- Implement `ProtectedRoute` using `useUser()` hook with client-side email domain whitelist.
- Obtain tokens for WebSocket auth via `useAuth().getToken()`.
- Pass `routerPush` and `routerReplace` props to `ClerkProvider` for wouter compatibility.

### Backend (FastAPI/Python)

- `clerk_auth.py`: Use `PyJWT[crypto]` + `PyJWKClient` for RS256 verification. Pin algorithm explicitly (`algorithms=["RS256"]`) to prevent alg:none attacks.
- `clerk_proxy.py`: Mount a FastAPI route at `/api/__clerk/{path:path}` that proxies to Clerk FAPI. Use `httpx.AsyncClient` with `follow_redirects=True`. Maintain a path allowlist (must include `/npm/`, `/v1/client`, `/v1/environment`, `/v1/dev_browser`, `/.well-known/`).
- Strip `content-encoding` AND `content-length` headers from proxy responses (httpx auto-decompresses gzip but preserves original headers — browser needs decompressed bytes without compressed-size headers).
- Use `CLERK_AUTH_ENABLED` as an explicit boolean flag. Fail closed: if enabled but `CLERK_SECRET_KEY`, `CLERK_JWKS_URL`, or `CLERK_ISSUER` are missing, log critical error at startup.
- WebSocket auth: validate the JWT once at connection time only. Don't re-verify mid-session.
- Clerk session JWTs do NOT include email claims — skip server-side email domain gating. The frontend ProtectedRoute handles domain whitelisting.

### Deployment

- Never overwrite nginx configs managed by certbot in deploy scripts. Certbot adds SSL directives; overwriting removes them.
- Source `VITE_` environment variables from the server's `.env` file during build: `set -a; source <(grep '^VITE_' "$DEPLOY_DIR/backend/.env"); set +a`
- Clerk dev instances talk directly to `clerk.accounts.dev` — no proxy URL needed in development.
- Add `access_log off;` to nginx `/ws/chat` location block to prevent JWT tokens from appearing in access logs.

### Production Activation (Custom Domain)

Setting up Clerk for production requires a distinct production instance with its own keys and DNS configuration. This was completed for `kraken.expertintheloop.io` on 2026-05-07.

**1. Create a production Clerk instance** — Do not reuse dev keys (`pk_test_*`/`sk_test_*`) in production. Use the Clerk CLI to retrieve production keys:

```bash
npx clerk apps list                              # Find your app
npx clerk env pull --instance prod --app <app_id> # Get pk_live/sk_live keys
```

The publishable key encodes the Clerk FAPI domain. Decode it to derive JWKS/issuer URLs:

```bash
echo "<pk_live_base64_portion>" | base64 -d
# → clerk.<your-domain>
```

**2. Configure five DNS CNAME records** — In the Clerk dashboard, enable the custom domain. Add these records in your DNS provider (Google Cloud DNS for expertintheloop.io):

| Hostname | Target |
|---|---|
| `accounts.<your-domain>` | `accounts.clerk.services` |
| `clerk.<your-domain>` | `frontend-api.clerk.services` |
| `clk._domainkey.<your-domain>` | `dkim1.<instance-id>.clerk.services` |
| `clk2._domainkey.<your-domain>` | `dkim2.<instance-id>.clerk.services` |
| `clkmail.<your-domain>` | `mail.<instance-id>.clerk.services` |

Wait for Clerk dashboard to confirm DNS verification before proceeding.

**3. Remove the FAPI proxy for production** — With custom domain DNS active, Clerk loads directly from `clerk.<your-domain>`. The `/api/__clerk` proxy (`clerk_proxy.py`) is unnecessary and conflicts with custom domain routing. Remove from the server `.env`:

- `VITE_CLERK_PROXY_URL` — do not pass to `npm run build`
- `CLERK_PROXY_URL` — remove from backend config

Keeping `VITE_CLERK_PROXY_URL` set while custom domain DNS is active causes "Failed to load Clerk JS" — the SDK tries to bootstrap through the proxy instead of the custom domain CDN. (session history)

**4. Configure backend JWT verification against the custom domain** — The JWKS endpoint and JWT issuer use the custom domain:

```bash
# Server backend/.env (production)
CLERK_AUTH_ENABLED=true
CLERK_SECRET_KEY=sk_live_...
CLERK_JWKS_URL=https://clerk.<your-domain>/.well-known/jwks.json
CLERK_ISSUER=https://clerk.<your-domain>
ALLOWED_EMAIL_DOMAINS=yourdomain.com
ALLOWED_EMAILS=specific@user.com

# Frontend vars sourced at build time
VITE_CLERK_PUBLISHABLE_KEY=pk_live_...
VITE_ALLOWED_EMAIL_DOMAINS=yourdomain.com
VITE_ALLOWED_EMAILS=specific@user.com
```

**5. Add the deploy workflow guard** — The production `deploy.yml` must source VITE_* vars from the server `.env` and fail fast if the file is missing:

```bash
ENV_FILE=~/kraken-chatbot/backend/.env
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found — frontend build will lack VITE_* vars"
  exit 1
fi
set -a; source <(grep '^VITE_' "$ENV_FILE"); set +a
if [ -z "$VITE_CLERK_PUBLISHABLE_KEY" ]; then
  echo "WARNING: VITE_CLERK_PUBLISHABLE_KEY not set — app will run without Clerk auth"
fi
VITE_WS_URL=wss://<your-domain>/ws/chat npm run build
```

**6. Configure Google OAuth redirect URI** — In Google Cloud Console > APIs & Services > Credentials > OAuth 2.0 Client ID, add:

```
https://clerk.<your-domain>/v1/oauth_callback
```

Without this, Google returns "Error 400: invalid_request" when users attempt sign-in.

## Why This Matters

1. **No official Python SDK for backend auth** — manual JWT verification is required, and getting it wrong opens auth bypass vulnerabilities (algorithm confusion, missing issuer validation).
2. **FAPI proxy subtleties** — Clerk's frontend API issues 307 redirects and serves gzip-compressed bundles; naive proxying breaks the Clerk JS runtime silently with SyntaxErrors.
3. **JWT claim limitations** — Clerk session tokens do NOT include email claims, making server-side email-domain gating impossible from the token alone. This must be understood upfront.
4. **Fail-closed design** — Without explicit enablement flags, a misconfigured deployment (env var typo) silently disables all auth on a public URL.
5. **Dev vs. prod architecture difference** — Dev instances bypass the proxy entirely; applying the production proxy pattern to dev causes 403 errors.

## When to Apply

- Adding Clerk to any Python backend (FastAPI, Django, Flask)
- Replicating a Clerk Express implementation in another language
- Any app that uses WebSocket connections requiring auth tokens
- Deploying to nginx + certbot environments where SSL is managed externally

## Examples

**Token verification (clerk_auth.py):**
```python
from jwt import PyJWKClient, decode as jwt_decode

jwks_client = PyJWKClient(CLERK_JWKS_URL)

def verify_clerk_token(token: str) -> dict:
    signing_key = jwks_client.get_signing_key_from_jwt(token)
    payload = jwt_decode(
        token,
        signing_key.key,
        algorithms=["RS256"],  # Pin explicitly — never derive from token header
        issuer=CLERK_ISSUER,
        options={"require": ["exp", "nbf", "iss", "sub"]},
    )
    return payload
```

**FAPI proxy response handling (clerk_proxy.py):**
```python
# httpx auto-decompresses gzip; strip encoding headers so browser gets raw bytes
STRIP_HEADERS = {"content-encoding", "content-length", "transfer-encoding", "connection"}

resp = await client.request(method=request.method, url=target_url, headers=fwd_headers, content=body)
headers = {k: v for k, v in resp.headers.multi_items() if k.lower() not in STRIP_HEADERS}
return Response(content=resp.content, status_code=resp.status_code, headers=headers)
```

**ClerkProvider with wouter (App.tsx):**
```tsx
<ClerkProvider
  publishableKey={import.meta.env.VITE_CLERK_PUBLISHABLE_KEY}
  proxyUrl={import.meta.env.VITE_CLERK_PROXY_URL}
  routerPush={(to) => setLocation(to)}
  routerReplace={(to) => setLocation(to, { replace: true })}
>
```

**Fail-closed startup guard (main.py):**
```python
if settings.clerk_auth_enabled:
    missing = []
    if not settings.clerk_secret_key: missing.append("CLERK_SECRET_KEY")
    if not settings.clerk_jwks_url: missing.append("CLERK_JWKS_URL")
    if not settings.clerk_issuer: missing.append("CLERK_ISSUER")
    if missing:
        logger.critical("CLERK_AUTH_ENABLED=true but missing: %s", ", ".join(missing))
```

**Deploy script sourcing VITE_ vars (biomapper-ui pattern):**
```bash
set -a; source <(grep '^VITE_' "$DEPLOY_DIR/backend/.env" 2>/dev/null); set +a
VITE_WS_URL=wss://dev-kraken.expertintheloop.io/ws/chat npm run build
```

## Key Pitfalls (session history)

| Pitfall | Consequence | Fix |
|---------|-------------|-----|
| Clerk dev instance + proxyUrl set | Clerk JS fails to load (talks to wrong endpoint) | Leave proxyUrl undefined in dev |
| Assuming email in JWT | Server-side domain check rejects all valid tokens | Do domain gating client-side via `useUser()` |
| Proxy without `follow_redirects` | 307 responses break Clerk JS bundle loading | `follow_redirects=True` on httpx client |
| Proxy keeps `content-encoding` header | Browser receives decompressed bytes but thinks it's gzip | Strip `content-encoding` and `content-length` |
| Deploy script overwrites nginx | Certbot SSL config lost, browser shows "Not Secure" | Skip nginx in deploy; manage separately or re-run certbot |
| Hardcoded VITE_ vars in CI workflow | Keys in repo, inflexible per-environment | Source from server `.env` at build time |
| Missing `/npm/` in proxy allowlist | Clerk JS bundle returns 403 | Add `/npm/` to allowed path prefixes |
| `CLERK_AUTH_ENABLED` inferred from key presence | Env var typo silently disables all auth | Use explicit boolean flag with startup validation |
| Custom domain DNS active + `VITE_CLERK_PROXY_URL` set | "Failed to load Clerk JS" — SDK bootstraps through proxy instead of custom domain | Remove `VITE_CLERK_PROXY_URL` and `CLERK_PROXY_URL` from production `.env` |
| Production deploy missing VITE_* vars | `VITE_CLERK_PUBLISHABLE_KEY` is undefined at build time; app renders without ClerkProvider | Source from server `.env` with fail-fast guard (see Production Activation section) |
| Google OAuth redirect URI not configured for custom domain | "Error 400: invalid_request" on Google sign-in | Add `https://clerk.<domain>/v1/oauth_callback` in Google Cloud Console |

## Related

- **Reference implementation**: `biomapper-ui` (same Clerk account, Express backend, production proxy pattern)
- **GitHub Issue #20**: "Add Authentication and Authorization to FastAPI Backend" — directly solved by this integration
- **GitHub Issue #33**: "feat: implement Google OAuth authentication" — superseded by Clerk (handles OAuth providers internally)
- **Plan**: `docs/plans/2026-05-06-003-feat-clerk-auth-integration-plan.md`
- **PR #52**: feat/clerk-auth → dev (Greptile reviewed, feedback addressed)
- **PR #55**: fix(deploy): source VITE_* vars from .env during production build
- **Clerk docs**: https://clerk.com/docs/react/getting-started/quickstart
- **Clerk JWT verification**: https://clerk.com/docs/guides/sessions/manual-jwt-verification
- **Clerk custom domains**: https://clerk.com/docs/deployments/overview
