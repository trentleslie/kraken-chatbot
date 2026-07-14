"""Tests for Task 5: WebSocket BYOK wiring — key intake, per-turn resolution, untrusted gating."""
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from kestrel_backend import byok
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
    # Patch get_settings in both main and byok so BYOK is active (server key configured,
    # trusted domains set), but user is untrusted (no email) and provides no key.
    with patch("kestrel_backend.main.validate_ws_clerk_token",
               AsyncMock(return_value={"sub": "u1"})), \
         patch("kestrel_backend.main.get_verified_email",
               AsyncMock(return_value=None)), \
         patch("kestrel_backend.main.get_settings") as gs, \
         patch("kestrel_backend.byok.get_settings") as gs_byok:
        gs.return_value.server_anthropic_api_key = "sk-server"
        gs.return_value.byok_trusted_email_domains = ["phenomehealth.org"]
        gs.return_value.max_ws_message_bytes = 1_000_000
        gs.return_value.clerk_auth_enabled = False
        gs.return_value.rate_limit_per_minute = 100
        gs_byok.return_value.byok_trusted_email_domains = ["phenomehealth.org"]
        gs_byok.return_value.server_anthropic_api_key = "sk-server"
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?token=x") as ws:
            ws.send_json({"type": "set_key", "key": None})
            ws.send_json({"type": "user_message", "content": "hi", "agent_mode": "classic"})
            frames = _drain_until(ws, lambda f: f.get("code") == "NEEDS_KEY")
            assert any(f.get("type") == "error" and f.get("code") == "NEEDS_KEY"
                       for f in frames)


def test_trusted_no_key_runs_on_server_key():
    # get_settings is lru_cache'd and called from both main.py and byok.py —
    # patch both module references so resolve_effective_key sees trusted domains.
    with patch("kestrel_backend.main.validate_ws_clerk_token",
               AsyncMock(return_value={"sub": "u2"})), \
         patch("kestrel_backend.main.get_verified_email",
               AsyncMock(return_value="trent@phenomehealth.org")), \
         patch("kestrel_backend.main.handle_classic_mode", AsyncMock()) as h, \
         patch("kestrel_backend.main.get_settings") as gs, \
         patch("kestrel_backend.byok.get_settings") as gs_byok:
        gs.return_value.byok_trusted_email_domains = ["phenomehealth.org"]
        gs.return_value.server_anthropic_api_key = "sk-server"
        gs.return_value.max_ws_message_bytes = 1_000_000
        gs.return_value.clerk_auth_enabled = False
        gs.return_value.rate_limit_per_minute = 100
        gs_byok.return_value.byok_trusted_email_domains = ["phenomehealth.org"]
        gs_byok.return_value.server_anthropic_api_key = "sk-server"
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?token=x") as ws:
            ws.send_json({"type": "set_key", "key": None})
            ws.send_json({"type": "user_message", "content": "hi", "agent_mode": "classic"})
            frames = _drain_until(ws, lambda f: f.get("type") == "key_source")
            assert any(f.get("type") == "key_source" and f.get("source") == "server"
                       for f in frames)
            assert h.await_count == 1


def test_ws_sets_provider_contextvar():
    # Drive the trusted-verified-email + server-key branch (same resolution path as
    # test_trusted_no_key_runs_on_server_key) and capture byok.current_provider from
    # *inside* the handler call, since a ContextVar set on the WS task isn't visible
    # from the test's own context.
    captured = {}

    async def _capture_provider(*args, **kwargs):
        captured["provider"] = byok.current_provider.get()

    with patch("kestrel_backend.main.validate_ws_clerk_token",
               AsyncMock(return_value={"sub": "u3"})), \
         patch("kestrel_backend.main.get_verified_email",
               AsyncMock(return_value="trent@phenomehealth.org")), \
         patch("kestrel_backend.main.handle_classic_mode",
               AsyncMock(side_effect=_capture_provider)), \
         patch("kestrel_backend.main.get_settings") as gs, \
         patch("kestrel_backend.byok.get_settings") as gs_byok:
        gs.return_value.byok_trusted_email_domains = ["phenomehealth.org"]
        gs.return_value.server_anthropic_api_key = "sk-server"
        gs.return_value.max_ws_message_bytes = 1_000_000
        gs.return_value.clerk_auth_enabled = False
        gs.return_value.rate_limit_per_minute = 100
        gs_byok.return_value.byok_trusted_email_domains = ["phenomehealth.org"]
        gs_byok.return_value.server_anthropic_api_key = "sk-server"
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?token=x") as ws:
            ws.send_json({"type": "set_key", "key": None})
            ws.send_json({"type": "user_message", "content": "hi", "agent_mode": "classic"})
            _drain_until(ws, lambda f: f.get("type") == "key_source")

    assert captured.get("provider") == "anthropic"
    # Outside the WS task's context, the var must read back as unset (default) —
    # this is a proxy for "reset happened on the task's own context", not a direct
    # observation of that context (ContextVars aren't visible across tasks).
    assert byok.current_provider.get() is None
