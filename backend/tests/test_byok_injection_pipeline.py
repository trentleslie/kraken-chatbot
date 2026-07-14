"""Tests that the BYOK per-request key is correctly injected at every SDK call boundary.

Three surfaces are covered:
1. create_agent_options() — still valid for future callers (classic + PR5 scorer).
2. query_with_usage() — the REAL funnel used by all 6 pipeline nodes.
3. _apply_byok_env() — unit test of the helper itself.
4. Concurrency isolation — no cross-task key bleed.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_backend.graph import sdk_utils
from kestrel_backend import byok


# ---------------------------------------------------------------------------
# Helper: async generator that captures the options it was given
# ---------------------------------------------------------------------------

def _make_fake_query(captured: dict):
    """Return an async-generator function that records the options argument."""
    async def fake_query(*, prompt, options):
        captured["options"] = options
        # Yield a minimal ResultMessage-like event so query_with_usage doesn't fail
        event = MagicMock()
        event.usage = None
        event.content = []
        # Make isinstance(event, ResultMessage) return False (MagicMock default)
        yield event
    return fake_query


# ---------------------------------------------------------------------------
# 1. create_agent_options still injects (kept for future callers)
# ---------------------------------------------------------------------------

def test_pipeline_options_inject_contextvar_key(monkeypatch):
    captured = {}

    class FakeOptions:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(sdk_utils, "ClaudeAgentOptions", FakeOptions)
    monkeypatch.setattr(sdk_utils, "HAS_SDK", True)
    token = byok.current_api_key.set("sk-node")
    try:
        sdk_utils.create_agent_options(system_prompt="x")
    finally:
        byok.current_api_key.reset(token)
    assert captured["env"]["ANTHROPIC_API_KEY"] == "sk-node"


def test_pipeline_options_route_through_proxy(monkeypatch):
    """create_agent_options must carry the full proxy env + cli_path when configured,
    and _apply_byok_env must produce the same env shape for the query_with_usage funnel."""
    captured = {}

    class FakeOptions:
        def __init__(self, **kw):
            captured.update(kw)
            self.env = kw.get("env")
            self.cli_path = kw.get("cli_path")

    monkeypatch.setattr(sdk_utils, "ClaudeAgentOptions", FakeOptions)
    monkeypatch.setattr(sdk_utils, "HAS_SDK", True)

    s = byok.get_settings()
    monkeypatch.setattr(s, "kraken_llm_base_url", "http://127.0.0.1:4000", raising=False)
    monkeypatch.setattr(s, "litellm_master_key", "sk-proxy", raising=False)

    token = byok.current_api_key.set("sk-user")
    try:
        opts = sdk_utils.create_agent_options(system_prompt="x")
    finally:
        byok.current_api_key.reset(token)

    assert captured["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-proxy"
    assert captured["env"]["ANTHROPIC_API_KEY"] == "sk-user"
    assert captured["cli_path"] == byok.system_cli_path()

    # _apply_byok_env must produce the same env shape (the real funnel for query_with_usage)
    class BareOptions:
        def __init__(self):
            self.env = None

    token = byok.current_api_key.set("sk-user")
    try:
        bare = sdk_utils._apply_byok_env(BareOptions())
    finally:
        byok.current_api_key.reset(token)

    assert bare.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
    assert bare.env["ANTHROPIC_AUTH_TOKEN"] == "sk-proxy"
    assert bare.env["ANTHROPIC_API_KEY"] == "sk-user"
    assert bare.cli_path == byok.system_cli_path()


# ---------------------------------------------------------------------------
# 2. query_with_usage() injects at the real SDK boundary (funnel for 6 nodes)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_with_usage_injects_byok_key(monkeypatch):
    """Verify that query_with_usage applies the BYOK key to options before calling query."""
    captured = {}

    # Patch the SDK's `query` symbol imported into sdk_utils
    monkeypatch.setattr(sdk_utils, "query", _make_fake_query(captured))
    monkeypatch.setattr(sdk_utils, "HAS_SDK", True)

    # Build a minimal options object with a settable .env attribute
    class FakeOptions:
        def __init__(self):
            self.env = None

    options = FakeOptions()

    token = byok.current_api_key.set("sk-pipeline")
    try:
        await sdk_utils.query_with_usage(
            prompt="test prompt",
            options=options,
            node_name="test_node",
        )
    finally:
        byok.current_api_key.reset(token)

    assert captured["options"] is options, "query() should receive the same options object"
    assert captured["options"].env is not None, "env should have been set by _apply_byok_env"
    assert captured["options"].env["ANTHROPIC_API_KEY"] == "sk-pipeline"


@pytest.mark.asyncio
async def test_query_with_usage_no_key_leaves_env_unchanged(monkeypatch):
    """When no BYOK key is set, query_with_usage must not mutate options.env."""
    captured = {}

    monkeypatch.setattr(sdk_utils, "query", _make_fake_query(captured))
    monkeypatch.setattr(sdk_utils, "HAS_SDK", True)

    class FakeOptions:
        def __init__(self):
            self.env = None

    options = FakeOptions()

    # Ensure no key is set (clear any inherited context)
    token = byok.current_api_key.set(None)
    try:
        await sdk_utils.query_with_usage(
            prompt="test prompt",
            options=options,
            node_name="test_node",
        )
    finally:
        byok.current_api_key.reset(token)

    assert captured["options"].env is None, "env must remain None when no BYOK key is active"


# ---------------------------------------------------------------------------
# 3. _apply_byok_env() unit tests
# ---------------------------------------------------------------------------

def test_apply_byok_env_sets_key_when_present():
    class FakeOptions:
        def __init__(self):
            self.env = None

    options = FakeOptions()
    token = byok.current_api_key.set("sk-unit")
    try:
        result = sdk_utils._apply_byok_env(options)
    finally:
        byok.current_api_key.reset(token)

    assert result is options
    assert result.env["ANTHROPIC_API_KEY"] == "sk-unit"


def test_apply_byok_env_merges_with_existing_env():
    """Existing env entries must be preserved; BYOK key is added/overwritten."""
    class FakeOptions:
        def __init__(self):
            self.env = {"OTHER_VAR": "keep_me"}

    options = FakeOptions()
    token = byok.current_api_key.set("sk-merge")
    try:
        sdk_utils._apply_byok_env(options)
    finally:
        byok.current_api_key.reset(token)

    assert options.env["ANTHROPIC_API_KEY"] == "sk-merge"
    assert options.env["OTHER_VAR"] == "keep_me"


def test_apply_byok_env_noop_when_no_key():
    class FakeOptions:
        def __init__(self):
            self.env = None

    options = FakeOptions()
    token = byok.current_api_key.set(None)
    try:
        sdk_utils._apply_byok_env(options)
    finally:
        byok.current_api_key.reset(token)

    assert options.env is None


def test_apply_byok_env_noop_when_options_is_none():
    token = byok.current_api_key.set("sk-present")
    try:
        result = sdk_utils._apply_byok_env(None)
    finally:
        byok.current_api_key.reset(token)
    assert result is None


# ---------------------------------------------------------------------------
# 4. No cross-task key bleed (scheduling / concurrency isolation)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_cross_task_key_bleed(monkeypatch):
    seen = []

    class FakeOptions:
        def __init__(self, **kw):
            seen.append(kw.get("env", {}).get("ANTHROPIC_API_KEY"))

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
