from kestrel_backend import agent


def test_every_kestrel_tool_allowed():
    for name in (
        "mcp__kestrel__multi_hop_query",
        "mcp__kestrel__get_metagraph",
        "mcp__kestrel__get_valid_primary_knowledge_sources",
        "mcp__kestrel__some_future_tool",
    ):
        assert agent.is_tool_allowed(name)


def test_blocked_and_unknown_tools_rejected():
    for name in ("Write", "Edit", "WebFetch", "mcp__other__tool", "mcp__kestrel"):
        assert not agent.is_tool_allowed(name)
    assert agent.is_tool_allowed("Bash")


def test_options_use_server_level_kestrel_rule(monkeypatch):
    captured = {}

    class FakeOptions:
        def __init__(self, **kw): captured.update(kw)

    monkeypatch.setattr(agent, "ClaudeAgentOptions", FakeOptions)
    agent.build_agent_options()
    assert "mcp__kestrel" in captured["allowed_tools"]
    assert not any(t.startswith("mcp__kestrel__") for t in captured["allowed_tools"])
