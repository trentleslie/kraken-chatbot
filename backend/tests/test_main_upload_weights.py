"""Axis A Unit 4: WS payload plumbing + runner signatures for module_directions.

kME/kIM ride each row of ``structured_analytes`` (already a ``list[dict]``) — no new WS field.
The per-module eigengene→outcome ``module_directions`` list is a new optional field threaded from
the WS handler through ``handle_pipeline_mode`` / the runner into initial state, and passed into the
shared ``validate_and_normalize`` gate so both entry points (WS gate #1 + intake gate #2) are
guarded identically.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_backend.graph import runner


async def _astream_gen(events):
    for e in events:
        yield e


class TestRunnerDirectionThreading:
    async def test_run_discovery_injects_module_directions(self, monkeypatch):
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={})
        monkeypatch.setattr(runner, "build_discovery_graph", lambda: graph)

        directions = [
            {"group": "Brown", "eigengene_trait_correlation": -0.5, "trait_label": "frailty"}
        ]
        await runner.run_discovery(
            "q",
            structured_analytes=[{"name": "glucose", "group": "Brown", "kme": 0.8}],
            selected_groups=["Brown"],
            module_directions=directions,
        )

        initial_state = graph.ainvoke.call_args.args[0]
        assert initial_state["module_directions"] == directions
        assert initial_state["structured_analytes"][0]["kme"] == 0.8

    async def test_run_discovery_defaults_empty_directions(self, monkeypatch):
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={})
        monkeypatch.setattr(runner, "build_discovery_graph", lambda: graph)

        await runner.run_discovery("q")

        initial_state = graph.ainvoke.call_args.args[0]
        assert initial_state["module_directions"] == []

    async def test_stream_discovery_injects_module_directions(self, monkeypatch):
        captured = {}

        def astream(initial_state, stream_mode="updates", config=None):
            captured["initial_state"] = initial_state
            return _astream_gen([])

        graph = MagicMock()
        graph.astream = astream
        monkeypatch.setattr(runner, "build_discovery_graph", lambda: graph)

        directions = [{"group": "Blue", "eigengene_trait_correlation": 0.3, "trait_label": "age"}]
        _ = [
            e
            async for e in runner.stream_discovery("q", module_directions=directions)
        ]

        assert captured["initial_state"]["module_directions"] == directions


class TestHandlePipelineModeDirectionThreading:
    async def test_module_directions_passed_to_stream_discovery(self):
        from kestrel_backend.main import handle_pipeline_mode

        mock_websocket = AsyncMock()
        captured = {}

        def fake_stream(**kwargs):
            captured.update(kwargs)

            async def _gen():
                yield {
                    "type": "complete",
                    "data": {"synthesis_report": "R", "hypotheses": [], "resolved_entities": []},
                }

            return _gen()

        panel = [{"name": "glucose", "group": "Brown", "kme": 0.8}]
        directions = [
            {"group": "Brown", "eigengene_trait_correlation": -0.5, "trait_label": "frailty"}
        ]
        with patch("kestrel_backend.graph.runner.stream_discovery", side_effect=fake_stream):
            with patch("kestrel_backend.main.conversation_history", {"test": []}):
                with patch("kestrel_backend.main.conversation_ids", {}):
                    with patch("kestrel_backend.main.turn_counters", {"test": 0}):
                        await handle_pipeline_mode(
                            mock_websocket,
                            "study context",
                            "test",
                            structured_analytes=panel,
                            selected_groups=["Brown"],
                            module_directions=directions,
                        )

        assert captured.get("structured_analytes") == panel
        assert captured.get("module_directions") == directions

    async def test_module_directions_optional(self):
        """A pipeline run with no directions still proceeds (direction is optional)."""
        from kestrel_backend.main import handle_pipeline_mode

        mock_websocket = AsyncMock()
        captured = {}

        def fake_stream(**kwargs):
            captured.update(kwargs)

            async def _gen():
                yield {
                    "type": "complete",
                    "data": {"synthesis_report": "R", "hypotheses": [], "resolved_entities": []},
                }

            return _gen()

        with patch("kestrel_backend.graph.runner.stream_discovery", side_effect=fake_stream):
            with patch("kestrel_backend.main.conversation_history", {"test": []}):
                with patch("kestrel_backend.main.conversation_ids", {}):
                    with patch("kestrel_backend.main.turn_counters", {"test": 0}):
                        await handle_pipeline_mode(
                            mock_websocket,
                            "q",
                            "test",
                            structured_analytes=[{"name": "glucose", "group": "Brown", "kme": 0.8}],
                        )

        assert captured.get("module_directions") is None
        assert mock_websocket.send_text.called
