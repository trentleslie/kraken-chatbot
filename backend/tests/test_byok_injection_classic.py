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


def test_classic_options_route_through_proxy(monkeypatch):
    captured = {}
    class FakeOptions:
        def __init__(self, **kw): captured.update(kw)
    monkeypatch.setattr(agent, "ClaudeAgentOptions", FakeOptions)

    s = byok.get_settings()
    monkeypatch.setattr(s, "kraken_llm_base_url", "http://127.0.0.1:4000", raising=False)
    monkeypatch.setattr(s, "litellm_master_key", "sk-proxy", raising=False)

    token = byok.current_api_key.set("sk-user")
    try:
        agent.build_agent_options()
    finally:
        byok.current_api_key.reset(token)

    assert captured["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-proxy"
    assert captured["env"]["ANTHROPIC_API_KEY"] == "sk-user"
    assert captured["cli_path"] == byok.system_cli_path()
