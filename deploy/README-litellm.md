# LiteLLM Proxy Local Run

## Deploy prerequisite

On **both** the prod and dev VMs, before starting the proxy unit:

1. Create `deploy/litellm.env` containing the proxy-auth master key:
   ```
   LITELLM_MASTER_KEY=<value>
   ```
   The systemd unit loads this via `EnvironmentFile=` — if the file is missing, the unit
   will not start. This file is gitignored (`deploy/*.env`); it must be created by hand
   (or via secrets tooling) on each host and is never committed.

2. Ensure the **same** `LITELLM_MASTER_KEY` value is also present in `backend/.env`. The
   backend reads it to send `Authorization: Bearer <key>` to the proxy (via
   `ANTHROPIC_AUTH_TOKEN`, see `byok.build_agent_env()`). If it's missing there,
   `build_agent_env()` raises rather than silently sending unauthenticated requests that
   the proxy would 401.

The two values **must match** — they are the same key, one on each side of the proxy call.

## Run the LiteLLM proxy locally

```bash
uv tool install 'litellm[proxy]'
export LITELLM_MASTER_KEY=sk-proxy-local
litellm --config deploy/litellm.config.yaml --port 4000
```

Health check:
```bash
curl -s http://127.0.0.1:4000/health/liveliness
```
Expected response: `"I'm alive!"`

## Notes

**DB-less:** no `DATABASE_URL` is set, so virtual keys / spend UI are intentionally unavailable.
