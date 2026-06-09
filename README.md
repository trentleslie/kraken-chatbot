# KRAKEN Chatbot

AI-powered interface for exploring the KRAKEN knowledge graph (via Kestrel). React frontend + FastAPI backend over WebSocket.

## Run locally

Prerequisites: [`uv`](https://docs.astral.sh/uv/), Node.js, a Kestrel API key, and the Claude CLI (`claude`).

### Backend (terminal 1)

```bash
cd backend
cp .env.example .env
# In .env set:
#   KESTREL_API_KEY=<your key>
#   CLERK_AUTH_ENABLED=false     # no login for local dev
#   LANGFUSE_ENABLED=false       # silence tracing locally

claude login        # one-time; the agent's LLM calls use this (or set ANTHROPIC_API_KEY in .env)
uv sync
uv run uvicorn src.kestrel_backend.main:app --reload --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete` before loading the UI.

### Frontend (terminal 2)

```bash
npm install
VITE_WS_URL=ws://127.0.0.1:8000/ws/chat npm run dev
```

Open the URL Vite prints (http://localhost:5173).

## Gotchas

- **Use `127.0.0.1`, not `localhost`, in `VITE_WS_URL`.** `localhost` can resolve to IPv6 (`::1`) while the backend binds IPv4 (`127.0.0.1`), so the WebSocket silently fails to connect.
- **Start the backend first.** If the UI loads before the backend is accepting connections, the app falls into offline **demo mode** (shows canned data, badge in the upper-left reads `demo`). Hard-refresh once the backend is up.
- Postgres (`DATABASE_URL`) and Langfuse are optional — only needed for conversation persistence and tracing.

## More

See `CLAUDE.md` for architecture, deployment, and the discovery-pipeline details.
