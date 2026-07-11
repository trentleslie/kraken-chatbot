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
