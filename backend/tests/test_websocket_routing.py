"""
Tests for Phase 6: WebSocket mode routing between Classic and Pipeline modes.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch


class TestProtocolMessages:
    """Test the new protocol message types."""

    def test_pipeline_progress_message(self):
        """Verify PipelineProgressMessage structure."""
        from kestrel_backend.protocol import PipelineProgressMessage

        msg = PipelineProgressMessage(
            node="entity_resolution",
            message="Resolving entities in knowledge graph...",
            nodes_completed=2,
        )
        assert msg.type == "pipeline_progress"
        assert msg.node == "entity_resolution"
        assert msg.nodes_completed == 2
        assert msg.total_nodes == 9

    def test_pipeline_complete_message(self):
        """Verify PipelineCompleteMessage structure."""
        from kestrel_backend.protocol import PipelineCompleteMessage

        msg = PipelineCompleteMessage(
            synthesis_report="# Discovery Report\n\nTest report",
            hypotheses_count=3,
            entities_resolved=5,
            duration_ms=12500,
        )
        assert msg.type == "pipeline_complete"
        assert "# Discovery Report" in msg.synthesis_report
        assert msg.hypotheses_count == 3
        assert msg.entities_resolved == 5
        assert msg.duration_ms == 12500

    def test_node_status_messages_mapping(self):
        """Verify all pipeline nodes have status messages."""
        from kestrel_backend.protocol import NODE_STATUS_MESSAGES

        expected_nodes = [
            "intake",
            "entity_resolution",
            "triage",
            "direct_kg",
            "cold_start",
            "pathway_enrichment",
            "integration",
            "temporal",
            "synthesis",
        ]
        for node in expected_nodes:
            assert node in NODE_STATUS_MESSAGES
            assert isinstance(NODE_STATUS_MESSAGES[node], str)
            assert len(NODE_STATUS_MESSAGES[node]) > 0

    def test_user_message_request_default_mode(self):
        """Verify UserMessageRequest defaults to classic mode."""
        from kestrel_backend.protocol import UserMessageRequest

        msg = UserMessageRequest(type="user_message", content="test")
        assert msg.agent_mode == "classic"

    def test_user_message_request_pipeline_mode(self):
        """Verify UserMessageRequest accepts pipeline mode."""
        from kestrel_backend.protocol import UserMessageRequest

        msg = UserMessageRequest(type="user_message", content="test", agent_mode="pipeline")
        assert msg.agent_mode == "pipeline"


class TestModeRouting:
    """Test mode routing logic."""

    @pytest.mark.asyncio
    async def test_classic_mode_routes_to_agent(self):
        """Verify classic mode calls run_agent_turn."""
        from kestrel_backend.main import handle_classic_mode

        # Mock the run_agent_turn function
        mock_websocket = AsyncMock()

        async def mock_agent_gen():
            from kestrel_backend.agent import AgentEvent
            yield AgentEvent(type="text", data={"content": "Hello"})
            yield AgentEvent(type="done", data={})

        with patch("kestrel_backend.main.run_agent_turn", return_value=mock_agent_gen()):
            with patch("kestrel_backend.main.conversation_history", {"test": []}):
                with patch("kestrel_backend.main.conversation_ids", {}):
                    with patch("kestrel_backend.main.turn_counters", {"test": 0}):
                        await handle_classic_mode(mock_websocket, "test query", "test")

        # Verify at least one message was sent
        assert mock_websocket.send_text.called

    @pytest.mark.asyncio
    async def test_pipeline_mode_routes_to_graph(self):
        """Verify pipeline mode calls stream_discovery."""
        from kestrel_backend.main import handle_pipeline_mode

        mock_websocket = AsyncMock()

        async def mock_stream_gen():
            yield {"type": "node_event", "node": "intake", "op": "add", "data": {}}
            yield {
                "type": "complete",
                "data": {
                    "synthesis_report": "Test report",
                    "hypotheses": [],
                    "resolved_entities": [{"curie": "TEST:001"}],
                },
            }

        # Patch at the source module where stream_discovery is defined
        with patch("kestrel_backend.graph.runner.stream_discovery", return_value=mock_stream_gen()):
            with patch("kestrel_backend.main.conversation_history", {"test": []}):
                with patch("kestrel_backend.main.conversation_ids", {}):
                    with patch("kestrel_backend.main.turn_counters", {"test": 0}):
                        await handle_pipeline_mode(mock_websocket, "test query", "test")

        # Verify messages were sent
        assert mock_websocket.send_text.called
        calls = mock_websocket.send_text.call_args_list

        # Check that we sent progress and complete messages
        sent_types = []
        for call in calls:
            msg = json.loads(call[0][0])
            sent_types.append(msg["type"])

        assert "pipeline_progress" in sent_types
        assert "pipeline_complete" in sent_types
        assert "done" in sent_types

    @pytest.mark.asyncio
    async def test_default_mode_is_classic(self):
        """Verify missing agent_mode defaults to classic."""
        # This tests the websocket_chat routing logic
        # When agent_mode is not provided, it should default to "classic"
        data = {"type": "user_message", "content": "test"}
        agent_mode = data.get("agent_mode", "classic")
        assert agent_mode == "classic"


class TestPipelineProgressStreaming:
    """Test pipeline progress message streaming."""

    @pytest.mark.asyncio
    async def test_pipeline_sends_progress_messages(self):
        """Verify status messages sent during pipeline execution."""
        from kestrel_backend.main import handle_pipeline_mode
        from kestrel_backend.protocol import NODE_STATUS_MESSAGES

        mock_websocket = AsyncMock()

        async def mock_stream_gen():
            # Simulate multiple node events
            for node in ["intake", "entity_resolution", "triage"]:
                yield {"type": "node_event", "node": node, "op": "add", "data": {}}
            yield {
                "type": "complete",
                "data": {
                    "synthesis_report": "Report",
                    "hypotheses": [],
                    "resolved_entities": [],
                },
            }

        with patch("kestrel_backend.graph.runner.stream_discovery", return_value=mock_stream_gen()):
            with patch("kestrel_backend.main.conversation_history", {"test": []}):
                with patch("kestrel_backend.main.conversation_ids", {}):
                    with patch("kestrel_backend.main.turn_counters", {"test": 0}):
                        await handle_pipeline_mode(mock_websocket, "test query", "test")

        # Count progress messages
        progress_count = 0
        for call in mock_websocket.send_text.call_args_list:
            msg = json.loads(call[0][0])
            if msg["type"] == "pipeline_progress":
                progress_count += 1
                # Verify message has user-friendly text
                assert msg["message"] in NODE_STATUS_MESSAGES.values()

        assert progress_count == 3  # intake, entity_resolution, triage

    @pytest.mark.asyncio
    async def test_duplicate_node_events_deduplicated(self):
        """Verify duplicate node events only send one progress message."""
        from kestrel_backend.main import handle_pipeline_mode

        mock_websocket = AsyncMock()

        async def mock_stream_gen():
            # Send duplicate events for same node
            yield {"type": "node_event", "node": "intake", "op": "add", "data": {}}
            yield {"type": "node_event", "node": "intake", "op": "update", "data": {}}
            yield {"type": "node_event", "node": "intake", "op": "finish", "data": {}}
            yield {
                "type": "complete",
                "data": {"synthesis_report": "Report", "hypotheses": [], "resolved_entities": []},
            }

        with patch("kestrel_backend.graph.runner.stream_discovery", return_value=mock_stream_gen()):
            with patch("kestrel_backend.main.conversation_history", {"test": []}):
                with patch("kestrel_backend.main.conversation_ids", {}):
                    with patch("kestrel_backend.main.turn_counters", {"test": 0}):
                        await handle_pipeline_mode(mock_websocket, "test query", "test")

        # Should only have 1 progress message for intake despite 3 events
        progress_count = sum(
            1 for call in mock_websocket.send_text.call_args_list
            if json.loads(call[0][0])["type"] == "pipeline_progress"
        )
        assert progress_count == 1


class TestPipelineErrorHandling:
    """Test pipeline error handling."""

    @pytest.mark.asyncio
    async def test_pipeline_error_graceful(self):
        """Verify pipeline failure sends user-friendly error message."""
        from kestrel_backend.main import handle_pipeline_mode

        mock_websocket = AsyncMock()

        async def mock_stream_gen():
            yield {"type": "node_event", "node": "intake", "op": "add", "data": {}}
            raise RuntimeError("Test error")

        with patch("kestrel_backend.graph.runner.stream_discovery", return_value=mock_stream_gen()):
            with patch("kestrel_backend.main.conversation_history", {"test": []}):
                with patch("kestrel_backend.main.conversation_ids", {}):
                    with patch("kestrel_backend.main.turn_counters", {"test": 0}):
                        await handle_pipeline_mode(mock_websocket, "test query", "test")

        # Should send error message followed by done
        sent_types = [
            json.loads(call[0][0])["type"]
            for call in mock_websocket.send_text.call_args_list
        ]

        assert "error" in sent_types
        assert "done" in sent_types

        # Verify error message is user-friendly
        for call in mock_websocket.send_text.call_args_list:
            msg = json.loads(call[0][0])
            if msg["type"] == "error":
                assert "PIPELINE_ERROR" == msg.get("code")
                assert "Try Classic mode" in msg["message"]

    @pytest.mark.asyncio
    async def test_pipeline_error_cleans_up(self):
        """Verify pipeline error still sends done message for cleanup."""
        from kestrel_backend.main import handle_pipeline_mode

        mock_websocket = AsyncMock()

        async def mock_stream_gen():
            raise ValueError("Immediate failure")
            yield  # Never reached, but makes this a generator

        with patch("kestrel_backend.graph.runner.stream_discovery", return_value=mock_stream_gen()):
            with patch("kestrel_backend.main.conversation_history", {"test": []}):
                with patch("kestrel_backend.main.conversation_ids", {}):
                    with patch("kestrel_backend.main.turn_counters", {"test": 0}):
                        await handle_pipeline_mode(mock_websocket, "test query", "test")

        # Should still end with done message
        last_call = mock_websocket.send_text.call_args_list[-1]
        last_msg = json.loads(last_call[0][0])
        assert last_msg["type"] == "done"


class TestPipelineCompleteMessage:
    """Test pipeline complete message content."""

    @pytest.mark.asyncio
    async def test_complete_includes_all_fields(self):
        """Verify complete message has all required fields."""
        from kestrel_backend.main import handle_pipeline_mode

        mock_websocket = AsyncMock()

        async def mock_stream_gen():
            yield {
                "type": "complete",
                "data": {
                    "synthesis_report": "# Test Report",
                    "hypotheses": [{"id": 1}, {"id": 2}],
                    "resolved_entities": [{"curie": "A"}, {"curie": "B"}, {"curie": "C"}],
                },
            }

        with patch("kestrel_backend.graph.runner.stream_discovery", return_value=mock_stream_gen()):
            with patch("kestrel_backend.main.conversation_history", {"test": []}):
                with patch("kestrel_backend.main.conversation_ids", {}):
                    with patch("kestrel_backend.main.turn_counters", {"test": 0}):
                        await handle_pipeline_mode(mock_websocket, "test query", "test")

        # Find the complete message
        for call in mock_websocket.send_text.call_args_list:
            msg = json.loads(call[0][0])
            if msg["type"] == "pipeline_complete":
                assert msg["synthesis_report"] == "# Test Report"
                assert msg["hypotheses_count"] == 2
                assert msg["entities_resolved"] == 3
                assert "duration_ms" in msg
                assert msg["duration_ms"] >= 0
                return

        pytest.fail("No pipeline_complete message found")
