# BYOK Per-Session (kraken) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let untrusted users run kraken's LLM synthesis on their own Anthropic key (per-session, never persisted), while verified `phenomehealth.org` users transparently fall back to the server key.

**Architecture:** A single backend chokepoint (`byok.py`) resolves the effective Anthropic key per request and publishes it on a `ContextVar`. Both LLM paths — the classic single-agent (`agent.py`) and the LangGraph pipeline nodes (`sdk_utils.create_agent_options`) — read that contextvar and inject it into `ClaudeAgentOptions(env=...)` (SDK 0.1.31 supports a per-call `env` override, verified), plus the LangChain `ChatAnthropic` client. The key travels over the existing WebSocket as the first message, lives only in a per-connection dict (mirroring `conversation_history`), and dies on disconnect. No DB, no encryption, no `os.environ` mutation.

**Tech Stack:** Python 3 (uv), FastAPI + WebSockets, `claude_agent_sdk` 0.1.31, LangChain/LangGraph, PyJWT + Clerk JWKS/Backend API, pytest; React/TS client (Vite).

## Global Constraints

- **Never mutate `os.environ`** to set a per-request key — concurrent sessions would race. Inject via `ClaudeAgentOptions(env=...)` and `ChatAnthropic(api_key=...)` only.
- **Key is never persisted** — no DB, no file, no browser `localStorage`/`sessionStorage`. In-memory only (per-connection dict server-side; React state client-side).
- **Key never appears in logs or Langfuse traces.** Redaction is a required, tested deliverable (Task 7).
- **Server-key fallback requires a *verified* email** from Clerk, resolved server-side — never a client-asserted or unverified field (Task 2).
- **Trusted domain list is config-driven:** new setting `byok_trusted_email_domains`, deployed as `["phenomehealth.org"]`. This is distinct from the existing `clerk_allowed_email_domains` (app access gate).
- SDK version pinned at **0.1.31**; `ClaudeAgentOptions` fields available: `model`, `fallback_model`, `env`.
- Anthropic is the only BYOK provider; OpenAI/embeddings run server-side through Kestrel and are out of scope.

---

### Task 1: BYOK core — contextvar, resolution, config

**Files:**
- Create: `backend/src/kestrel_backend/byok.py`
- Modify: `backend/src/kestrel_backend/config.py` (add settings)
- Test: `backend/tests/test_byok.py`

**Interfaces:**
- Produces:
  - `current_api_key: ContextVar[str | None]` — the *resolved effective* key for the active request (default `None`).
  - `class NeedsKeyError(Exception)` — raised when an untrusted user has no BYOK key.
  - `resolve_effective_key(byok_key: str | None, verified_email: str | None) -> tuple[str, str]` — returns `(key, source)` where `source` ∈ `{"byok", "server"}`.
  - `is_trusted_email(verified_email: str | None) -> bool`
- Consumes: `config.get_settings()` → `byok_trusted_email_domains: list[str]`, `server_anthropic_api_key: str | None`.

- [ ] **Step 1: Add settings.** In `config.py`, inside `class Settings(BaseModel)`, add:

```python
    # BYOK: domains whose verified users ride the SERVER key; everyone else must BYOK.
    byok_trusted_email_domains: list[str] = []
    # The Anthropic key used only for the trusted server-key fallback.
    server_anthropic_api_key: str | None = None
```

- [ ] **Step 2: Write failing tests** in `backend/tests/test_byok.py`:

```python
import pytest
from kestrel_backend import byok
from kestrel_backend.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    s = Settings(byok_trusted_email_domains=["phenomehealth.org"],
                 server_anthropic_api_key="sk-server")
    monkeypatch.setattr(byok, "get_settings", lambda: s)


def test_byok_key_wins_even_for_trusted_user():
    key, source = byok.resolve_effective_key("sk-user", "trent@phenomehealth.org")
    assert (key, source) == ("sk-user", "byok")


def test_trusted_user_no_byok_uses_server_key():
    key, source = byok.resolve_effective_key(None, "trent@phenomehealth.org")
    assert (key, source) == ("sk-server", "server")


def test_untrusted_no_byok_raises():
    with pytest.raises(byok.NeedsKeyError):
        byok.resolve_effective_key(None, "someone@gmail.com")


def test_no_verified_email_is_untrusted():
    with pytest.raises(byok.NeedsKeyError):
        byok.resolve_effective_key(None, None)


def test_domain_match_is_case_insensitive_exact_suffix():
    assert byok.is_trusted_email("A@Phenomehealth.org") is True
    assert byok.is_trusted_email("x@evil-phenomehealth.org") is False
    assert byok.is_trusted_email("x@phenomehealth.org.evil.com") is False
```

- [ ] **Step 3: Run tests, verify they fail.** Run: `cd backend && .venv/bin/pytest tests/test_byok.py -v` — Expected: FAIL (module/attrs missing).

- [ ] **Step 4: Implement `byok.py`:**

```python
"""Per-session BYOK key resolution. No persistence, no os.environ mutation."""
from contextvars import ContextVar
from .config import get_settings

# The resolved effective key for the active request (byok or server). Option
# builders in agent.py and sdk_utils.py read this; nodes never do precedence logic.
current_api_key: ContextVar[str | None] = ContextVar("current_api_key", default=None)


class NeedsKeyError(Exception):
    """Untrusted user made an LLM request without providing a BYOK key."""


def is_trusted_email(verified_email: str | None) -> bool:
    if not verified_email or "@" not in verified_email:
        return False
    domain = verified_email.rsplit("@", 1)[-1].lower()
    trusted = {d.lower() for d in get_settings().byok_trusted_email_domains}
    return domain in trusted


def resolve_effective_key(byok_key: str | None, verified_email: str | None) -> tuple[str, str]:
    if byok_key:
        return byok_key, "byok"
    if is_trusted_email(verified_email):
        server_key = get_settings().server_anthropic_api_key
        if not server_key:
            raise NeedsKeyError("Trusted user but no server key configured.")
        return server_key, "server"
    raise NeedsKeyError("No API key provided and user is not trusted.")
```

- [ ] **Step 5: Run tests, verify they pass.** Run: `.venv/bin/pytest tests/test_byok.py -v` — Expected: PASS (5 tests).

- [ ] **Step 6: Commit.**

```bash
git add backend/src/kestrel_backend/byok.py backend/src/kestrel_backend/config.py backend/tests/test_byok.py
git commit -m "feat(byok): key resolution chokepoint + trusted-domain config"
```

---

### Task 2: Server-side verified-email resolution via Clerk Backend API

**Files:**
- Create: `backend/src/kestrel_backend/clerk_identity.py`
- Test: `backend/tests/test_clerk_identity.py`

**Interfaces:**
- Produces: `async def get_verified_email(user_info: dict) -> str | None` — returns the user's **verified primary** email, or `None`. Reads `user_info["sub"]`, calls Clerk Backend API `GET https://api.clerk.com/v1/users/{sub}` with `Authorization: Bearer <clerk_secret_key>`, returns the primary email only if its verification status is `"verified"`. Result cached per `sub`.
- Consumes: `config.get_settings()` → `clerk_secret_key`; `httpx` (already a dependency via clerk_proxy).

**Why this task exists:** `clerk_auth.py` documents that session JWTs often omit email and currently *allows* access when it can't check — safe for a UI gate, unsafe for spending the server key. The server-key fallback must confirm a verified email itself.

- [ ] **Step 1: Write failing test** in `backend/tests/test_clerk_identity.py`:

```python
import pytest
from kestrel_backend import clerk_identity as ci
from kestrel_backend.config import Settings


class _Resp:
    def __init__(self, payload): self._p = payload; self.status_code = 200
    def raise_for_status(self): pass
    def json(self): return self._p


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(ci, "get_settings", lambda: Settings(clerk_secret_key="sk_test"))
    ci._email_cache.clear()


@pytest.mark.asyncio
async def test_returns_verified_primary_email(monkeypatch):
    payload = {
        "primary_email_address_id": "idb- 1",
        "email_addresses": [
            {"id": "idb- 1", "email_address": "trent@phenomehealth.org",
             "verification": {"status": "verified"}},
        ],
    }
    async def fake_get(*a, **k): return _Resp(payload)
    monkeypatch.setattr(ci, "_clerk_get", fake_get)
    assert await ci.get_verified_email({"sub": "user_1"}) == "trent@phenomehealth.org"


@pytest.mark.asyncio
async def test_unverified_primary_returns_none(monkeypatch):
    payload = {
        "primary_email_address_id": "e1",
        "email_addresses": [
            {"id": "e1", "email_address": "x@phenomehealth.org",
             "verification": {"status": "unverified"}},
        ],
    }
    async def fake_get(*a, **k): return _Resp(payload)
    monkeypatch.setattr(ci, "_clerk_get", fake_get)
    assert await ci.get_verified_email({"sub": "user_2"}) is None


@pytest.mark.asyncio
async def test_missing_sub_returns_none():
    assert await ci.get_verified_email({}) is None
```

- [ ] **Step 2: Run test, verify it fails.** Run: `.venv/bin/pytest tests/test_clerk_identity.py -v` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `clerk_identity.py`:**

```python
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
```

- [ ] **Step 4: Run tests, verify they pass.** Run: `.venv/bin/pytest tests/test_clerk_identity.py -v` — Expected: PASS (3 tests).

- [ ] **Step 5: Commit.**

```bash
git add backend/src/kestrel_backend/clerk_identity.py backend/tests/test_clerk_identity.py
git commit -m "feat(byok): server-side verified-email resolution via Clerk Backend API"
```

---

### Task 3: Inject the effective key into classic mode (SDK + LangChain)

**Files:**
- Modify: `backend/src/kestrel_backend/agent.py:394-411` (options_kwargs build)
- Modify: `backend/src/kestrel_backend/local_tools.py:~43` (ChatAnthropic construction)
- Test: `backend/tests/test_byok_injection_classic.py`

**Interfaces:**
- Consumes: `byok.current_api_key` (ContextVar from Task 1).
- Produces: classic-mode `ClaudeAgentOptions` carries `env={"ANTHROPIC_API_KEY": <effective key>}`; LangChain `ChatAnthropic` receives `api_key=<effective key>`.

- [ ] **Step 1: Write failing test** in `backend/tests/test_byok_injection_classic.py`:

```python
from kestrel_backend import agent, byok


def test_classic_options_inject_contextvar_key(monkeypatch):
    captured = {}
    class FakeOptions:
        def __init__(self, **kw): captured.update(kw)
    monkeypatch.setattr(agent, "ClaudeAgentOptions", FakeOptions)
    token = byok.current_api_key.set("sk-abc")
    try:
        agent.build_agent_options()   # extracted builder (see Step 3)
    finally:
        byok.current_api_key.reset(token)
    assert captured["env"]["ANTHROPIC_API_KEY"] == "sk-abc"
```

- [ ] **Step 2: Run test, verify it fails.** Run: `.venv/bin/pytest tests/test_byok_injection_classic.py -v` — Expected: FAIL (`build_agent_options` not defined).

- [ ] **Step 3: Extract + inject in `agent.py`.** Extract the `options_kwargs` dict (lines ~394-409) into a module-level `build_agent_options()` that the turn loop calls, and add the env injection before `ClaudeAgentOptions(**options_kwargs)`:

```python
from .byok import current_api_key

def build_agent_options():
    kestrel_config = _get_kestrel_mcp_config()
    options_kwargs = {
        "allowed_tools": list(ALLOWED_TOOLS),
        "system_prompt": SYSTEM_PROMPT,
        "mcp_servers": {"kestrel": kestrel_config},
        "hooks": {"PreToolUse": [HookMatcher(matcher="Bash", hooks=[bash_security_hook])]},
        "max_buffer_size": 10 * 1024 * 1024,
    }
    settings = get_settings()
    if settings.model:
        options_kwargs["model"] = settings.model
    key = current_api_key.get()
    if key:
        # env is MERGED over the inherited process environment by the SDK, so we
        # only override the credential and leave PATH/etc. intact.
        options_kwargs["env"] = {"ANTHROPIC_API_KEY": key}
    return ClaudeAgentOptions(**options_kwargs)
```

Replace the inline `options = ClaudeAgentOptions(**options_kwargs)` at line ~411 with `options = build_agent_options()`.

- [ ] **Step 4: Inject into LangChain** in `local_tools.py`. At the `ChatAnthropic(model="claude-sonnet-4-20250514", ...)` construction, add `api_key=current_api_key.get()` (import `from .byok import current_api_key` at top). When the contextvar is `None` the SDK/LangChain falls back to ambient env — acceptable for non-request-scoped internal callers.

- [ ] **Step 5: Run test, verify it passes.** Run: `.venv/bin/pytest tests/test_byok_injection_classic.py -v` — Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add backend/src/kestrel_backend/agent.py backend/src/kestrel_backend/local_tools.py backend/tests/test_byok_injection_classic.py
git commit -m "feat(byok): inject effective key into classic SDK + LangChain paths"
```

---

### Task 4: Inject the effective key into pipeline nodes

**Files:**
- Modify: `backend/src/kestrel_backend/graph/sdk_utils.py:207-221` (`create_agent_options`)
- Test: `backend/tests/test_byok_injection_pipeline.py`

**Interfaces:**
- Consumes: `byok.current_api_key`.
- Produces: every pipeline node's `ClaudeAgentOptions` carries `env={"ANTHROPIC_API_KEY": <effective key>}`. No node signatures change — the contextvar is read centrally in `create_agent_options`, which all nodes already call.

- [ ] **Step 1: Write failing test** in `backend/tests/test_byok_injection_pipeline.py`:

```python
from kestrel_backend.graph import sdk_utils
from kestrel_backend import byok


def test_pipeline_options_inject_contextvar_key(monkeypatch):
    captured = {}
    class FakeOptions:
        def __init__(self, **kw): captured.update(kw)
    monkeypatch.setattr(sdk_utils, "ClaudeAgentOptions", FakeOptions)
    monkeypatch.setattr(sdk_utils, "HAS_SDK", True)
    token = byok.current_api_key.set("sk-node")
    try:
        sdk_utils.create_agent_options(system_prompt="x")
    finally:
        byok.current_api_key.reset(token)
    assert captured["env"]["ANTHROPIC_API_KEY"] == "sk-node"
```

- [ ] **Step 2: Run test, verify it fails.** Run: `.venv/bin/pytest tests/test_byok_injection_pipeline.py -v` — Expected: FAIL (no `env` in kwargs).

- [ ] **Step 3: Inject in `create_agent_options`.** Add at top of `sdk_utils.py`: `from ..byok import current_api_key`. Inside `create_agent_options`, before `return ClaudeAgentOptions(**kwargs)`:

```python
    key = current_api_key.get()
    if key:
        kwargs["env"] = {"ANTHROPIC_API_KEY": key}
```

- [ ] **Step 4: Run test, verify it passes.** Run: `.venv/bin/pytest tests/test_byok_injection_pipeline.py -v` — Expected: PASS.

- [ ] **Step 5: Concurrency test** — two contextvar values in two tasks must not bleed. Append to the same test file:

```python
import asyncio, pytest

@pytest.mark.asyncio
async def test_no_cross_task_key_bleed(monkeypatch):
    seen = []
    class FakeOptions:
        def __init__(self, **kw): seen.append(kw.get("env", {}).get("ANTHROPIC_API_KEY"))
    monkeypatch.setattr(sdk_utils, "ClaudeAgentOptions", FakeOptions)
    monkeypatch.setattr(sdk_utils, "HAS_SDK", True)

    async def run(k):
        byok.current_api_key.set(k)          # set inside the task's own context
        await asyncio.sleep(0.01)
        sdk_utils.create_agent_options(system_prompt="x")

    await asyncio.gather(run("sk-A"), run("sk-B"))
    assert set(seen) == {"sk-A", "sk-B"}     # each task kept its own key
```

- [ ] **Step 6: Run it, verify PASS**, then commit.

```bash
.venv/bin/pytest tests/test_byok_injection_pipeline.py -v
git add backend/src/kestrel_backend/graph/sdk_utils.py backend/tests/test_byok_injection_pipeline.py
git commit -m "feat(byok): inject effective key into pipeline nodes + concurrency guard"
```

---

### Task 5: Wire the WebSocket handler — receive key, resolve, gate

**Files:**
- Modify: `backend/src/kestrel_backend/protocol.py` (add `SetKeyRequest` incoming + `KeySourceMessage` outgoing; add `NEEDS_KEY` code convention)
- Modify: `backend/src/kestrel_backend/main.py` (`websocket_chat` ~777-870, message loop, `handle_classic_mode`/`handle_pipeline_mode` call sites)
- Test: `backend/tests/test_ws_byok_wiring.py`

**GROUND TRUTH — the real WS protocol (verified in `protocol.py` / `main.py`, use these EXACTLY):**
- Incoming chat frames are `{"type": "user_message", "content": "...", "agent_mode": "classic"|"pipeline"}` (`UserMessageRequest`). The field is `agent_mode`, NOT `mode`, and the type is `user_message`, NOT `message`.
- The loop at `main.py:~848` rejects **any** `data.get("type") != "user_message"` with `ErrorMessage(message="Unknown message type")`. A `set_key` frame therefore needs its own branch placed **before** that guard, ending in `continue`.
- `ErrorMessage` (protocol.py:29) is `type: Literal["error"]="error"`, `message: str`, `code: str | None = None`. There is **no** `needs_key`/`error=`/`type=` override. Signal "needs key" as `ErrorMessage(message="...", code="NEEDS_KEY")` — this mirrors the existing `AUTH_ERROR` code convention. Existing error paths follow the error frame with `DoneMessage()`; do the same.
- `connection_id = str(id(websocket))` (main.py:803). Per-connection dicts (`conversation_history`, `conversation_ids`, `turn_counters`) live near main.py:94-103 and are cleaned in the disconnect handler.

**Interfaces:**
- Consumes: `byok.resolve_effective_key`, `byok.current_api_key`, `byok.NeedsKeyError`, `clerk_identity.get_verified_email`.
- Message contract: client's **first WS frame** after connect is `{"type": "set_key", "key": "sk-..."}` (or `{"type": "set_key", "key": null}` to declare no key). Stored in a new per-connection dict `connection_api_keys: dict[str, str | None]`, cleaned up alongside `conversation_history` on disconnect.
- Produces (outgoing): `KeySourceMessage(type="key_source", source: Literal["byok","server"])`, sent as the first frame of each successful turn so the UI can show "whose key".
- Per turn, before invoking a handler: resolve `(key, source)`, send `KeySourceMessage`, set `current_api_key`; on `NeedsKeyError` send `ErrorMessage(code="NEEDS_KEY")` + `DoneMessage()` and skip the turn.

- [ ] **Step 1: Add protocol models** to `protocol.py`:

```python
class KeySourceMessage(BaseModel):
    """Server → Client: which key the current turn ran on."""
    type: Literal["key_source"] = "key_source"
    source: Literal["byok", "server"]


class SetKeyRequest(BaseModel):
    """Client → Server: set/clear the per-connection BYOK key."""
    type: Literal["set_key"] = "set_key"
    key: str | None = None
```

- [ ] **Step 2: Write failing test** in `backend/tests/test_ws_byok_wiring.py`:

```python
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from kestrel_backend.main import app


def _drain_until(ws, predicate, limit=5):
    frames = []
    for _ in range(limit):
        f = ws.receive_json()
        frames.append(f)
        if predicate(f):
            return frames
    return frames


def test_untrusted_no_key_is_gated():
    with patch("kestrel_backend.main.validate_ws_clerk_token",
               AsyncMock(return_value={"sub": "u1"})), \
         patch("kestrel_backend.main.get_verified_email",
               AsyncMock(return_value=None)):
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?token=x") as ws:
            ws.send_json({"type": "set_key", "key": None})
            ws.send_json({"type": "user_message", "content": "hi", "agent_mode": "classic"})
            frames = _drain_until(ws, lambda f: f.get("code") == "NEEDS_KEY")
            assert any(f.get("type") == "error" and f.get("code") == "NEEDS_KEY"
                       for f in frames)


def test_trusted_no_key_runs_on_server_key():
    with patch("kestrel_backend.main.validate_ws_clerk_token",
               AsyncMock(return_value={"sub": "u2"})), \
         patch("kestrel_backend.main.get_verified_email",
               AsyncMock(return_value="trent@phenomehealth.org")), \
         patch("kestrel_backend.main.handle_classic_mode", AsyncMock()) as h, \
         patch("kestrel_backend.main.get_settings") as gs:
        gs.return_value.byok_trusted_email_domains = ["phenomehealth.org"]
        gs.return_value.server_anthropic_api_key = "sk-server"
        gs.return_value.max_ws_message_bytes = 1_000_000
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?token=x") as ws:
            ws.send_json({"type": "set_key", "key": None})
            ws.send_json({"type": "user_message", "content": "hi", "agent_mode": "classic"})
            frames = _drain_until(ws, lambda f: f.get("type") == "key_source")
            assert any(f.get("type") == "key_source" and f.get("source") == "server"
                       for f in frames)
            assert h.await_count == 1
```

> Note: `get_settings` is `@lru_cache`'d and imported at module scope in several places; patch it where `main.py` and `byok.py` look it up. If patching proves brittle, set the values via environment/`Settings` construction in a fixture instead — the assertion (server-source frame + handler invoked) is what matters.

- [ ] **Step 3: Run tests, verify they fail.** Run: `.venv/bin/pytest tests/test_ws_byok_wiring.py -v` — Expected: FAIL.

- [ ] **Step 4: Implement.** In `main.py`:
  - Add `connection_api_keys: dict[str, str | None] = {}` near the other per-connection dicts (~line 94-103).
  - Import at module scope: `from .byok import resolve_effective_key, current_api_key, NeedsKeyError` and `from .clerk_identity import get_verified_email` (module-scope so tests can `patch("kestrel_backend.main.get_verified_email", ...)`).
  - `websocket_chat` already binds `user_info` — keep it in scope for the loop.
  - In the message loop, **before** the `!= "user_message"` guard, add:

```python
            if data.get("type") == "set_key":
                connection_api_keys[connection_id] = data.get("key")
                continue
```
  - After the `user_message` type check and content validation, **before** dispatching to `handle_classic_mode` / `handle_pipeline_mode`:

```python
            try:
                verified = await get_verified_email(user_info)
                key, source = resolve_effective_key(
                    connection_api_keys.get(connection_id), verified)
            except NeedsKeyError:
                await websocket.send_text(ErrorMessage(
                    message="Provide your Anthropic API key to run synthesis.",
                    code="NEEDS_KEY").model_dump_json())
                await websocket.send_text(DoneMessage().model_dump_json())
                continue
            await websocket.send_text(KeySourceMessage(source=source).model_dump_json())
            tok = current_api_key.set(key)
            try:
                # existing dispatch to handle_classic_mode / handle_pipeline_mode
                ...
            finally:
                current_api_key.reset(tok)
```
  - Import `KeySourceMessage` from `.protocol` alongside the existing `ErrorMessage`/`DoneMessage` imports.
  - In the disconnect/cleanup block (where `conversation_history` etc. are popped), add `connection_api_keys.pop(connection_id, None)`.

- [ ] **Step 5: Run tests, verify they pass.** Run: `.venv/bin/pytest tests/test_ws_byok_wiring.py -v` — Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add backend/src/kestrel_backend/protocol.py backend/src/kestrel_backend/main.py backend/tests/test_ws_byok_wiring.py
git commit -m "feat(byok): WS key intake, per-turn resolution, untrusted gating"
```

---

### Task 6: Pre-flight key validation endpoint

**Files:**
- Modify: `backend/src/kestrel_backend/main.py` (new REST route)
- Test: `backend/tests/test_validate_key.py`

**Interfaces:**
- Produces: `POST /api/validate-key` body `{"key": "sk-..."}` → `200 {"valid": true}` or `200 {"valid": false, "reason": "..."}`. Validation = one minimal Anthropic call (a 1-token `messages.create`) using the submitted key, nothing stored.

- [ ] **Step 1: Write failing test** in `backend/tests/test_validate_key.py`:

```python
from unittest.mock import patch
from fastapi.testclient import TestClient
from kestrel_backend.main import app


def test_validate_key_ok():
    with patch("kestrel_backend.main._probe_anthropic_key", return_value=(True, None)):
        r = TestClient(app).post("/api/validate-key", json={"key": "sk-good"})
        assert r.status_code == 200 and r.json()["valid"] is True


def test_validate_key_bad():
    with patch("kestrel_backend.main._probe_anthropic_key",
               return_value=(False, "invalid_api_key")):
        r = TestClient(app).post("/api/validate-key", json={"key": "sk-bad"})
        assert r.json() == {"valid": False, "reason": "invalid_api_key"}
```

- [ ] **Step 2: Run test, verify it fails.** Run: `.venv/bin/pytest tests/test_validate_key.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement** `_probe_anthropic_key(key: str) -> tuple[bool, str | None]` (uses the `anthropic` SDK: `Anthropic(api_key=key).messages.create(model="claude-3-5-haiku-latest", max_tokens=1, messages=[{"role":"user","content":"hi"}])`; return `(False, "invalid_api_key")` on `anthropic.AuthenticationError`, `(True, None)` on success) and the route:

```python
@app.post("/api/validate-key")
async def validate_key(request: Request):
    body = await request.json()
    ok, reason = _probe_anthropic_key(body.get("key", ""))
    return {"valid": ok} if ok else {"valid": False, "reason": reason}
```

- [ ] **Step 4: Run tests, verify they pass**, then commit.

```bash
.venv/bin/pytest tests/test_validate_key.py -v
git add backend/src/kestrel_backend/main.py backend/tests/test_validate_key.py
git commit -m "feat(byok): pre-flight key validation endpoint"
```

---

### Task 7: Redact the key from logs and Langfuse traces

**Files:**
- Modify: `backend/src/kestrel_backend/logging_config.py` (add a redaction filter)
- Test: `backend/tests/test_key_redaction.py`

**Interfaces:**
- Produces: a `logging.Filter` that masks anything matching `sk-ant-[A-Za-z0-9_-]+` (and a generic `sk-[A-Za-z0-9_-]{16,}`) as `sk-***REDACTED***` on every log record, installed on the root handler. Langfuse: confirm no trace `input`/`metadata` carries the key (the injection sites in Tasks 3-4 put the key only in `env`, never in prompts/metadata) and add a regression test asserting a crafted record is masked.

- [ ] **Step 1: Write failing test** in `backend/tests/test_key_redaction.py`:

```python
import logging
from kestrel_backend.logging_config import ApiKeyRedactionFilter


def test_filter_masks_anthropic_key():
    f = ApiKeyRedactionFilter()
    rec = logging.LogRecord("x", logging.INFO, "f", 1,
                            "using key sk-ant-abc123DEF456ghi789", None, None)
    f.filter(rec)
    assert "sk-ant-abc123DEF456ghi789" not in rec.getMessage()
    assert "REDACTED" in rec.getMessage()
```

- [ ] **Step 2: Run test, verify it fails.** Run: `.venv/bin/pytest tests/test_key_redaction.py -v` — Expected: FAIL (filter missing).

- [ ] **Step 3: Implement `ApiKeyRedactionFilter`** in `logging_config.py`:

```python
import re
_KEY_RE = re.compile(r"sk-(?:ant-)?[A-Za-z0-9_-]{16,}")

class ApiKeyRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _KEY_RE.sub("sk-***REDACTED***", record.msg)
        if record.args:
            record.args = tuple(
                _KEY_RE.sub("sk-***REDACTED***", a) if isinstance(a, str) else a
                for a in record.args
            )
        return True
```

Install it wherever handlers are configured in `logging_config.py` (add `handler.addFilter(ApiKeyRedactionFilter())` to each configured handler).

- [ ] **Step 4: Run test, verify it passes.** Run: `.venv/bin/pytest tests/test_key_redaction.py -v` — Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add backend/src/kestrel_backend/logging_config.py backend/tests/test_key_redaction.py
git commit -m "feat(byok): redact API keys from logs"
```

---

### Task 8: Frontend — key entry, WS wiring, source indicator

**Files:**
- Create: `client/src/hooks/useApiKey.ts`
- Create: `client/src/components/ApiKeyGate.tsx`
- Modify: `client/src/hooks/useWebSocket.ts` (send `set_key` first; surface needs-key + source)
- Modify: `client/src/pages/chat.tsx` (mount `ApiKeyGate`; show source badge)

**GROUND TRUTH — WS frames (match Task 5 / `protocol.py`):**
- Send first: `{"type":"set_key","key": <string|null>}`.
- "Needs key" arrives as an **error frame**: `{"type":"error","code":"NEEDS_KEY","message":"..."}` — react to `code === "NEEDS_KEY"`, NOT a `needs_key` type.
- Key source arrives as `{"type":"key_source","source":"byok"|"server"}` at the start of each turn.
- Check `client/src/types/` for the existing WS message TS union and extend it with `key_source` + the `code` field / `set_key` request so the client stays type-safe.

**Interfaces:**
- `useApiKey()` → `{ key: string | null, setKey, clearKey, validate }`. Holds the key in React state **only** (never `localStorage`/`sessionStorage`). `validate(key)` calls `POST /api/validate-key`.

- [ ] **Step 1:** Implement `useApiKey.ts` — state + `validate` (fetch `/api/validate-key`). Key lives only in memory; a full page reload clears it (acceptable per per-session design).
- [ ] **Step 2:** Implement `ApiKeyGate.tsx` — a form (paste key → Validate → on success call `setKey`). If `key` is null and the server sent a `NEEDS_KEY` error, block the composer and show the form inline. Show a "clear key" control when a key is set.
- [ ] **Step 3:** In `useWebSocket.ts`, on `onopen` send `{"type":"set_key","key": <current key or null>}` before any message; expose a `needsKey` boolean (set on a `code === "NEEDS_KEY"` error frame) and the latest `source` to consumers. Extend the TS message types in `client/src/types/`.
- [ ] **Step 4:** In `chat.tsx`, mount `ApiKeyGate`; render a small badge from `source` ("Using your key" / "Using server key"). On `needsKey`, reveal the gate.
- [ ] **Step 5:** Manual verification — run the client, connect without a key as a non-phenome user → composer blocked, gate shown; paste a valid key → chat works, badge reads "Using your key". Commit.

```bash
git add client/src/hooks/useApiKey.ts client/src/components/ApiKeyGate.tsx client/src/hooks/useWebSocket.ts client/src/pages/chat.tsx
git commit -m "feat(byok): frontend key entry, WS wiring, source badge"
```

---

### Task 9: ddharmon env-var note

**Files:**
- Modify: `ddharmon/README.md` (or `ddharmon/docs/`)

- [ ] **Step 1:** Add a short "API key" section: ddharmon reads `ANTHROPIC_API_KEY` from the caller-supplied environment/config; there is no settings UI or session handling — the caller (a human, a script, or a host app) is responsible for setting it. Commit.

```bash
git add ddharmon/README.md
git commit -m "docs(byok): document ddharmon env-var key convention"
```

---

## Self-Review

**Spec coverage:**
- §3 decisions 1-6 → Tasks 1 (resolution+config), 2 (verified email = decision 6), 5 (gating/fallback = decision 5). ✓
- §4.1 chokepoint → Task 1. ✓
- §4.2 SDK + LangChain injection → Tasks 3 (classic) + 4 (pipeline). ✓
- §4.3 reusable boundary → `byok.py` + `clerk_identity.py` are app-agnostic; injection sites are the app-specific adapters. (EITL port is a separate future plan.) ✓
- §5 security: no browser storage → Task 8; Langfuse/log scrub → Task 7; TLS/TTL → per-connection lifetime (dies on disconnect), TLS is deployment. ✓
- §6 validation + error UX + "whose key" indicator → Tasks 6, 5, 8. ✓
- §7 testing: precedence (T1), concurrency (T4), scrub (T7), validation (T6), e2e (T5+T8 manual). ✓
- §8 ddharmon → Task 9. ✓

**Placeholder scan:** No TBD/TODO. The WS-protocol shape (Task 5) is now pinned to verified ground truth (`user_message`/`set_key`, `ErrorMessage(code="NEEDS_KEY")`, `KeySourceMessage`). One bounded verify-before-implement note remains: SDK `env` merge-vs-replace (Task 3 — mitigated by only overriding the credential). Acceptable.

**Type consistency:** `current_api_key`, `resolve_effective_key(byok_key, verified_email) -> (key, source)`, `is_trusted_email`, `NeedsKeyError`, `get_verified_email(user_info) -> str | None`, `build_agent_options()`, `_probe_anthropic_key(key) -> (bool, str|None)`, `ApiKeyRedactionFilter` — names used consistently across tasks. ✓
