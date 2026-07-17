"""Unit 4: runner threads module_directions into initial state (Axis A)."""

from unittest.mock import AsyncMock, MagicMock

from kestrel_backend.graph import runner


async def _astream_gen(events):
    for e in events:
        yield e


class TestRunnerModuleDirections:
    async def test_run_discovery_threads_module_directions(self, monkeypatch):
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"raw_query": "q"})
        monkeypatch.setattr(runner, "build_discovery_graph", lambda: graph)

        directions = [{"group": "Brown", "eigengene_trait_correlation": 0.6, "trait_label": "t"}]
        await runner.run_discovery("q", module_directions=directions)

        initial_state = graph.ainvoke.call_args.args[0]
        assert initial_state["module_directions"] == directions

    async def test_run_discovery_default_module_directions_empty(self, monkeypatch):
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={})
        monkeypatch.setattr(runner, "build_discovery_graph", lambda: graph)

        await runner.run_discovery("q")

        initial_state = graph.ainvoke.call_args.args[0]
        assert initial_state["module_directions"] == []

    async def test_stream_discovery_threads_module_directions(self, monkeypatch):
        captured = {}

        def astream(initial_state, stream_mode="updates", config=None):
            captured["initial_state"] = initial_state
            return _astream_gen([])

        graph = MagicMock()
        graph.astream = astream
        monkeypatch.setattr(runner, "build_discovery_graph", lambda: graph)

        directions = [{"group": "Blue", "eigengene_trait_correlation": -0.3, "trait_label": "t"}]
        _ = [e async for e in runner.stream_discovery("q", module_directions=directions)]

        assert captured["initial_state"]["module_directions"] == directions
