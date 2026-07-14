"""Task 7: opt-in integration smoke test -- backend -> local LiteLLM proxy -> Anthropic.

Skipped by default (no live proxy / real key in CI or the normal dev suite). Enable with:

    RUN_LITELLM_SMOKE=1 LITELLM_MASTER_KEY=sk-proxy-local USER_ANTHROPIC_KEY=sk-ant-... \\
        uv run pytest tests/test_litellm_integration_smoke.py -v -s

Requires a LiteLLM proxy already running locally on http://127.0.0.1:4000 and a real,
funded Anthropic API key -- this test makes one real, billed LLM call end to end.

Path chosen: `/api/validate-key` (main.py) was considered first (per the task brief's
suggestion) but it calls `anthropic.Anthropic(api_key=key)` directly and never reads
`kraken_llm_base_url` / `litellm_master_key` -- it does not exercise the proxy at all, so
it can't prove this pipeline. Instead this drives the lightest path that actually goes
through the BYOK env injection: `sdk_utils.create_agent_options()` (calls
`byok.build_agent_env()`) followed by `sdk_utils.query_with_usage()` (calls
`byok._apply_byok_env()` on the options right before the SDK `query()` call).
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LITELLM_SMOKE") != "1",
    reason="requires local litellm proxy + real key",
)


@pytest.mark.integration
async def test_pipeline_turn_through_proxy(monkeypatch):
    """One real turn through byok env injection -> SDK -> local LiteLLM proxy -> Anthropic."""
    from kestrel_backend import byok
    from kestrel_backend.graph import sdk_utils

    monkeypatch.setattr(
        byok.get_settings(), "kraken_llm_base_url", "http://127.0.0.1:4000", raising=False
    )
    monkeypatch.setattr(
        byok.get_settings(), "litellm_master_key", os.environ["LITELLM_MASTER_KEY"], raising=False
    )
    token = byok.current_api_key.set(os.environ["USER_ANTHROPIC_KEY"])
    try:
        options = sdk_utils.create_agent_options(
            system_prompt="Reply with exactly one word and nothing else.",
            allowed_tools=[],
            max_turns=1,
        )
        text, record = await sdk_utils.query_with_usage(
            prompt="Say the word 'pong'.",
            options=options,
            node_name="litellm_smoke_test",
        )
    finally:
        byok.current_api_key.reset(token)

    assert "pong" in text.lower()
    assert record is not None
