"""Unit 4: pipeline-mode Langfuse tracing wiring.

Verifies the CallbackHandler is threaded into the graph and the trace_id is surfaced to the
client (enabled), and that the pipeline degrades cleanly with no handler (disabled).

Note: a mocked client cannot prove the enclosing-span trace_id actually equals the trace the
handler writes node spans into, nor that generations nest under node spans — that is verified
live on dev (plan Unit 6). These tests pin the wiring contract only.
"""

import json
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock

import langfuse
import langfuse.langchain
import pytest

# Import runner via the SAME package path as main (src.kestrel_backend) so the module object
# matches main's relative `from .graph.runner import stream_discovery` — otherwise the patch
# lands on a different module object and the real graph runs.
from src.kestrel_backend import main
from src.kestrel_backend.graph import runner


def _empty_stream():
    async def _gen():
        return
        yield  # pragma: no cover - unreachable; makes this an async generator
    return _gen()


def _sent(ws):
    return [json.loads(c.args[0]) for c in ws.send_text.call_args_list]


@pytest.mark.asyncio
async def test_pipeline_attaches_handler_and_sends_trace_id(monkeypatch):
    captured = {}

    def fake_stream(query, conversation_history=None, config=None):
        captured["config"] = config
        return _empty_stream()

    monkeypatch.setattr(runner, "stream_discovery", fake_stream)

    # Langfuse enabled: enclosing observation yields a trace exposing a string trace_id.
    trace = MagicMock()
    trace.trace_id = "trace-xyz"
    obs_cm = MagicMock()
    obs_cm.__enter__ = MagicMock(return_value=trace)
    obs_cm.__exit__ = MagicMock(return_value=False)
    lf = MagicMock()
    lf.start_as_current_observation.return_value = obs_cm
    monkeypatch.setattr(main, "_get_pipeline_langfuse", lambda: lf)

    # Patch the v3 helpers imported locally inside the handler (resolved at call time).
    monkeypatch.setattr(langfuse, "propagate_attributes", lambda **kw: nullcontext())
    sentinel_handler = object()
    monkeypatch.setattr(langfuse.langchain, "CallbackHandler", lambda: sentinel_handler)

    ws = AsyncMock()
    await main.handle_pipeline_mode(ws, "what connects metabolite X and disease Y?", "conn-1")

    # Handler threaded into the graph via config callbacks
    assert captured["config"] == {"callbacks": [sentinel_handler]}
    # trace_id (trace identifier, not span id) surfaced for the feedback round-trip
    completes = [m for m in _sent(ws) if m.get("type") == "pipeline_complete"]
    assert completes, "expected a pipeline_complete message"
    assert completes[0]["trace_id"] == "trace-xyz"


@pytest.mark.asyncio
async def test_pipeline_degrades_when_langfuse_disabled(monkeypatch):
    captured = {}

    def fake_stream(query, conversation_history=None, config=None):
        captured["config"] = config
        return _empty_stream()

    monkeypatch.setattr(runner, "stream_discovery", fake_stream)
    monkeypatch.setattr(main, "_get_pipeline_langfuse", lambda: None)

    ws = AsyncMock()
    await main.handle_pipeline_mode(ws, "q", "conn-2")

    # No handler attached; pipeline still completes cleanly
    assert captured["config"] is None
    types = [m.get("type") for m in _sent(ws)]
    assert "pipeline_complete" in types
    assert "done" in types


@pytest.mark.asyncio
async def test_pipeline_degrades_when_trace_setup_raises(monkeypatch):
    """If Langfuse trace setup throws (bad import/handler/observation), the request must NOT
    hang — it degrades to no tracing and still responds (Greptile P1)."""
    captured = {}

    def fake_stream(query, conversation_history=None, config=None):
        captured["config"] = config
        return _empty_stream()

    monkeypatch.setattr(runner, "stream_discovery", fake_stream)
    # Enabled, but constructing the handler blows up
    monkeypatch.setattr(main, "_get_pipeline_langfuse", lambda: MagicMock())

    def _boom():
        raise RuntimeError("handler init failed")

    monkeypatch.setattr(langfuse.langchain, "CallbackHandler", _boom)

    ws = AsyncMock()
    await main.handle_pipeline_mode(ws, "q", "conn-3")

    # Degraded to no callbacks; client still got a complete response (no hang)
    assert captured["config"] is None
    types = [m.get("type") for m in _sent(ws)]
    assert "pipeline_complete" in types
    assert "done" in types
