"""analyze_results routes to the Agent SDK when no API key is available (SDK-login mode)."""

from types import SimpleNamespace

import pytest

from kestrel_backend import byok, local_tools


def _handler():
    tool = local_tools.analyze_results
    return getattr(tool, "handler", tool)


@pytest.mark.asyncio
async def test_no_key_anywhere_uses_agent_sdk(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = {}

    async def fake_query(prompt, options, node_name):
        calls["prompt"] = prompt
        return "sdk analysis", None

    monkeypatch.setattr(local_tools, "create_agent_options", lambda **kw: object())
    monkeypatch.setattr(local_tools, "query_with_usage", fake_query)
    monkeypatch.setattr(
        local_tools, "_get_anthropic_client",
        lambda: pytest.fail("Anthropic client must not be used in SDK-login mode"),
    )

    tok = byok.current_api_key.set(None)  # explicit: no per-request key
    try:
        result = await _handler()({"data": "[1, 2]", "task": "summarize"})
    finally:
        byok.current_api_key.reset(tok)

    assert result["content"][0]["text"] == "sdk analysis"
    assert "[1, 2]" in calls["prompt"]


@pytest.mark.asyncio
async def test_request_key_uses_anthropic_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = SimpleNamespace(content=[
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text="api analysis"),
    ])
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
    monkeypatch.setattr(local_tools, "_get_anthropic_client", lambda: client)
    monkeypatch.setattr(
        local_tools, "query_with_usage",
        lambda *a, **k: pytest.fail("Agent SDK must not be used when a key is set"),
    )

    tok = byok.current_api_key.set("sk-user")
    try:
        result = await _handler()({"data": "[]", "task": "summarize"})
    finally:
        byok.current_api_key.reset(tok)

    assert result["content"][0]["text"] == "api analysis"
