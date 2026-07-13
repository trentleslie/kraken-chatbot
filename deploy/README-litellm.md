# LiteLLM Proxy Local Run

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
