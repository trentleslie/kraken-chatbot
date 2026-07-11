import asyncio

import pytest

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
        # After yielding, each task must still see its OWN key.
        # Under any shared-state (module-global / os.environ) implementation,
        # both tasks would see the last writer's key and this assertion would fail.
        assert byok.current_api_key.get() == k
        sdk_utils.create_agent_options(system_prompt="x")

    await asyncio.gather(run("sk-A"), run("sk-B"))
    assert set(seen) == {"sk-A", "sk-B"}     # each task kept its own key
