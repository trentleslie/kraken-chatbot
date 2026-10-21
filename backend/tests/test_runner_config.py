"""Unit 3: runner threads a LangGraph RunnableConfig (e.g. callbacks) through to the graph."""

from unittest.mock import AsyncMock, MagicMock

from kestrel_backend.graph import runner


async def _astream_gen(events):
    for e in events:
        yield e


class TestRunnerConfigThreading:
    async def test_run_discovery_passes_config(self, monkeypatch):
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"raw_query": "q"})
        monkeypatch.setattr(runner, "build_discovery_graph", lambda: graph)

        sentinel = {"callbacks": ["handler"]}
        await runner.run_discovery("q", config=sentinel)

        graph.ainvoke.assert_awaited_once()
        assert graph.ainvoke.call_args.kwargs["config"] is sentinel

    async def test_run_discovery_default_no_config(self, monkeypatch):
        """Backward compat: no config arg → config=None passed (unchanged behavior)."""
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={})
        monkeypatch.setattr(runner, "build_discovery_graph", lambda: graph)

        await runner.run_discovery("q")

        assert graph.ainvoke.call_args.kwargs["config"] is None

    async def test_stream_discovery_passes_config(self, monkeypatch):
        captured = {}

        def astream(initial_state, stream_mode="updates", config=None):
            captured["config"] = config
            captured["stream_mode"] = stream_mode
            return _astream_gen([{"intake": {"x": 1}}])

        graph = MagicMock()
        graph.astream = astream
        monkeypatch.setattr(runner, "build_discovery_graph", lambda: graph)

        sentinel = {"callbacks": ["h"]}
        events = [e async for e in runner.stream_discovery("q", config=sentinel)]

        assert captured["config"] is sentinel
        assert captured["stream_mode"] == "updates"
        assert events == [{"type": "node_update", "node": "intake", "node_output": {"x": 1}}]

    async def test_stream_discovery_default_no_config(self, monkeypatch):
        captured = {}

        def astream(initial_state, stream_mode="updates", config=None):
            captured["config"] = config
            return _astream_gen([])

        graph = MagicMock()
        graph.astream = astream
        monkeypatch.setattr(runner, "build_discovery_graph", lambda: graph)

        _ = [e async for e in runner.stream_discovery("q")]

        assert captured["config"] is None


class TestRunnerAnalyteThreading:
    """The structured analyte panel + selection reach the graph's initial_state."""

    async def test_run_discovery_injects_analytes(self, monkeypatch):
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={})
        monkeypatch.setattr(runner, "build_discovery_graph", lambda: graph)

        panel = [{"name": "glucose", "group": "Brown"}]
        await runner.run_discovery(
            "q", structured_analytes=panel, selected_groups=["Brown"]
        )

        initial_state = graph.ainvoke.call_args.args[0]
        assert initial_state["structured_analytes"] == panel
        assert initial_state["selected_groups"] == ["Brown"]

    async def test_run_discovery_defaults_empty_lists(self, monkeypatch):
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={})
        monkeypatch.setattr(runner, "build_discovery_graph", lambda: graph)

        await runner.run_discovery("q")

        initial_state = graph.ainvoke.call_args.args[0]
        assert initial_state["structured_analytes"] == []
        assert initial_state["selected_groups"] == []

    async def test_stream_discovery_injects_analytes(self, monkeypatch):
        captured = {}

        def astream(initial_state, stream_mode="updates", config=None):
            captured["initial_state"] = initial_state
            return _astream_gen([])

        graph = MagicMock()
        graph.astream = astream
        monkeypatch.setattr(runner, "build_discovery_graph", lambda: graph)

        panel = [{"name": "IL6", "group": "Blue"}]
        _ = [
            e
            async for e in runner.stream_discovery(
                "q", structured_analytes=panel, selected_groups=["Blue"]
            )
        ]

        assert captured["initial_state"]["structured_analytes"] == panel
        assert captured["initial_state"]["selected_groups"] == ["Blue"]
