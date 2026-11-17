"""Langfuse Python SDK v5 call shapes, checked against the real installed package.

Every test runs offline against a KEYLESS real client. A keyless client is disabled (no
exporter, no API client), so these tests prove that each call the backend makes exists in
the installed SDK and accepts our arguments. They do NOT prove what gets exported: in
particular the propagated session id is never observed here. That is verified on dev.
"""

import re
from pathlib import Path

import pytest
from langfuse import Langfuse, propagate_attributes

from src.kestrel_backend import agent, config, main
from src.kestrel_backend.graph import sdk_utils

_LANGFUSE_ENV = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_BASE_URL",
)

# v3 calls removed from the SDK (plan R2). None may remain under backend/src.
_REMOVED_V3_CALLS = (
    "start_span",
    "start_generation",
    "start_as_current_generation",
    "update_trace",
    "set_trace_io",
    "set_current_trace_io",
)

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(autouse=True)
def _no_langfuse_env(monkeypatch):
    """Strip Langfuse credentials and hosts so nothing here can reach a live project.

    A developer shell with real keys must not send test traces. ``get_settings()`` calls
    ``load_dotenv()``, which could re-read keys from a local ``.env``, so that is stubbed
    out too, and every cached settings object or client singleton is reset before and after.
    """
    for name in _LANGFUSE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **kw: False)

    def _reset():
        config.get_settings.cache_clear()
        sdk_utils.reset_langfuse_singleton()

    monkeypatch.setattr(agent, "_langfuse", None)
    monkeypatch.setattr(main, "_pipeline_langfuse", None)
    _reset()
    yield
    _reset()


@pytest.fixture
def client():
    """A real, keyless (therefore disabled) Langfuse client."""
    lf = Langfuse()
    # No resource manager means no exporter and no API client: nothing can be sent.
    assert lf._resources is None
    return lf


def test_root_observation_create_update_end(client):
    """The agent-turn root: created with input and metadata, updated with output, ended."""
    root = client.start_observation(
        name="kraken_agent_turn",
        as_type="span",
        input={"user_message": "hello"},
        metadata={"turn_id": "t-1", "model": "claude-sonnet"},
    )
    root.update(
        output={"turn_id": "t-1"},
        metadata={
            "cost_usd": 0.01,
            "duration_ms": 1200,
            "tool_calls_count": 2,
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 10,
        },
    )
    # The id sent to the client for feedback is the trace id, not the span id.
    assert isinstance(root.trace_id, str)
    assert isinstance(root.id, str)
    root.end()


def test_pipeline_root_as_current_observation(client):
    """The pipeline root: a current-observation context manager exposing trace_id/update."""
    with client.start_as_current_observation(
        as_type="span",
        name="discovery_pipeline",
        input={"query": "q", "connection_id": "conn-1"},
        metadata={"mode": "pipeline", "version": "2.0"},
    ) as trace:
        assert isinstance(trace.trace_id, str)
        trace.update(output={"error": "boom"}, metadata={"status": "failed"})


def test_child_observation_under_root(client):
    """The tool and node shape: a child created from the root, updated, ended."""
    root = client.start_observation(name="kraken_agent_turn", as_type="span")
    tool = root.start_observation(
        name="tool:one_hop_query", as_type="span", input={"curie": "CHEBI:1"}
    )
    tool.update(output={"result": "ok"})
    tool.end()
    node = root.start_observation(
        name="node_intake",
        as_type="span",
        input={"node": "intake", "nodes_completed": 1},
    )
    node.update(output={"duration_ms": 5})
    node.end()
    root.end()


def test_generation_with_usage_details(client):
    """The per-node LLM generation: name, model, input, then output and four usage keys."""
    with client.start_as_current_observation(
        as_type="generation", name="llm:synthesis", model="claude-sonnet-4", input="prompt"
    ) as generation:
        generation.update(
            output="text",
            usage_details={
                "input": 10,
                "output": 20,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 4,
            },
            metadata={"mcp_tool_calls": 1, "available_tools": None},
        )


def test_session_propagation_wraps_creation(client):
    """Creating a root and a child inside propagate_attributes(session_id=...) does not raise.

    Not observed offline: whether the session id actually lands on the spans.
    """
    with propagate_attributes(session_id="session-123"):
        root = client.start_observation(name="kraken_agent_turn", as_type="span")
    with propagate_attributes(session_id="session-123"):
        child = root.start_observation(name="tool:x", as_type="span", input={})
    child.end()
    root.end()


def test_agent_session_scope_is_noop_without_session_id(client):
    """agent._session_scope: usable as a context manager with and without a session id."""
    with agent._session_scope(None):
        client.start_observation(name="kraken_agent_turn", as_type="span").end()
    with agent._session_scope("session-123"):
        client.start_observation(name="kraken_agent_turn", as_type="span").end()


def test_propagate_attributes_accepts_pipeline_metadata(client):
    """propagate_attributes accepts exactly what main.handle_pipeline_mode passes."""
    metadata = {"mode": "pipeline", "version": "2.0"}
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())
    assert all(len(v) <= 200 for v in metadata.values())
    with client.start_as_current_observation(as_type="span", name="discovery_pipeline"):
        with propagate_attributes(
            trace_name="discovery_pipeline",
            session_id="session-123",
            tags=["pipeline"],
            metadata=metadata,
        ):
            pass


def test_create_score_accepts_feedback_arguments(client):
    """create_score with the argument names /api/feedback uses, then flush."""
    client.create_score(
        trace_id="abcdef1234567890abcdef1234567890",
        name="user_feedback",
        value=1,
        comment="User feedback: positive",
    )
    client.flush()


def test_no_removed_v3_calls_in_src():
    """No file under backend/src names a call that SDK v5 removed."""
    pattern = re.compile(r"\b(" + "|".join(_REMOVED_V3_CALLS) + r")\b")
    offenders = []
    for path in sorted(_SRC_DIR.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(_SRC_DIR)}:{lineno}: {line.strip()}")
    assert not offenders, "removed Langfuse v3 calls remain:\n" + "\n".join(offenders)


def test_client_getters_return_none_without_keys():
    """Disabled path: with keys absent every client getter is a no-op (returns None)."""
    assert agent._get_langfuse() is None
    assert main._get_pipeline_langfuse() is None
    sdk_utils.reset_langfuse_singleton()
    assert sdk_utils._get_langfuse() is None
