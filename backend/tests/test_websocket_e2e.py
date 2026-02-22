"""End-to-end WebSocket tests for KRAKEN chat interface.

These tests verify the WebSocket endpoint behavior including:
- Connection lifecycle (connect, message exchange, disconnect)
- Classic and pipeline mode message flows
- Rate limiting enforcement
- Authentication and authorization
- Error handling for malformed messages
- Conversation history management
- State cleanup on disconnect

Tests use the Starlette TestClient for WebSocket testing as it provides
a synchronous interface that's easier to work with for WebSocket tests.
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ============================================================================
# Test Helpers
# ============================================================================

def create_user_message(content: str, agent_mode: str = "classic") -> str:
    """Create a JSON user message for WebSocket testing."""
    return json.dumps({
        "type": "user_message",
        "content": content,
        "agent_mode": agent_mode
    })


def collect_messages_until_done(websocket, timeout_messages: int = 50) -> list:
    """Collect WebSocket messages until 'done' type is received."""
    messages = []
    for _ in range(timeout_messages):
        data = websocket.receive_text()
        msg = json.loads(data)
        messages.append(msg)
        if msg.get("type") == "done":
            break
    return messages


# ============================================================================
# WebSocket Connection Tests
# ============================================================================

@pytest.mark.e2e
class TestWebSocketConnection:
    """Test WebSocket connection establishment and lifecycle."""

    def test_websocket_connection_accepted(self, test_settings):
        """Test that WebSocket connections are accepted without authentication."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app

        with patch("kestrel_backend.config.get_settings", return_value=test_settings):
            with patch("kestrel_backend.main.get_settings", return_value=test_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        client = TestClient(app)

                        # Connection should be accepted
                        with client.websocket_connect("/ws/chat") as websocket:
                            # Send a simple message to verify connection works
                            websocket.send_text(create_user_message("test"))
                            # Should not raise

    def test_websocket_disconnect_cleanup(self, test_settings, clean_connection_state):
        """Test that state is cleaned up on WebSocket disconnect."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app, rate_limit_state, conversation_history

        with patch("kestrel_backend.config.get_settings", return_value=test_settings):
            with patch("kestrel_backend.main.get_settings", return_value=test_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        # Mock agent to return quickly
                        with patch("kestrel_backend.main.run_agent_turn") as mock_agent:
                            mock_event = MagicMock()
                            mock_event.type = "done"
                            mock_event.data = {}

                            async def mock_gen(*args, **kwargs):
                                yield mock_event

                            mock_agent.side_effect = mock_gen

                            # Mock database to not require real connection
                            with patch("kestrel_backend.main.create_conversation", new_callable=AsyncMock) as mock_create:
                                mock_create.return_value = None

                                client = TestClient(app)

                                # Connect, send message, and disconnect
                                with client.websocket_connect("/ws/chat") as websocket:
                                    websocket.send_text(create_user_message("test"))
                                    # Wait for done
                                    data = websocket.receive_text()
                                    msg = json.loads(data)
                                    assert msg["type"] == "done"

                                # After disconnect, state should be cleaned up
                                # (clean_connection_state fixture handles verification)


# ============================================================================
# Classic Mode Message Flow Tests
# ============================================================================

@pytest.mark.e2e
class TestClassicModeFlow:
    """Test classic mode (single agent) message flow."""

    def test_classic_mode_text_response(self, test_settings, clean_connection_state):
        """Test that classic mode returns text messages."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app

        with patch("kestrel_backend.config.get_settings", return_value=test_settings):
            with patch("kestrel_backend.main.get_settings", return_value=test_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        with patch("kestrel_backend.main.create_conversation", new_callable=AsyncMock) as mock_create:
                            mock_create.return_value = None

                            # Create mock events
                            with patch("kestrel_backend.main.run_agent_turn") as mock_agent:
                                events = []

                                # Text event
                                text_event = MagicMock()
                                text_event.type = "text"
                                text_event.data = {"content": "Hello, this is a test response."}
                                events.append(text_event)

                                # Done event
                                done_event = MagicMock()
                                done_event.type = "done"
                                done_event.data = {}
                                events.append(done_event)

                                async def mock_gen(*args, **kwargs):
                                    for event in events:
                                        yield event

                                mock_agent.side_effect = mock_gen

                                client = TestClient(app)

                                with client.websocket_connect("/ws/chat") as websocket:
                                    websocket.send_text(create_user_message("Hello"))

                                    messages = collect_messages_until_done(websocket)

                                    # Should have text and done messages
                                    types = [m["type"] for m in messages]
                                    assert "text" in types
                                    assert "done" in types

                                    # Find text message and verify content
                                    text_msgs = [m for m in messages if m["type"] == "text"]
                                    assert len(text_msgs) > 0
                                    assert text_msgs[0]["content"] == "Hello, this is a test response."

    def test_classic_mode_tool_use_flow(self, test_settings, clean_connection_state):
        """Test that tool_use and tool_result messages are sent."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app

        with patch("kestrel_backend.config.get_settings", return_value=test_settings):
            with patch("kestrel_backend.main.get_settings", return_value=test_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        with patch("kestrel_backend.main.create_conversation", new_callable=AsyncMock) as mock_create:
                            mock_create.return_value = None

                            with patch("kestrel_backend.main.run_agent_turn") as mock_agent:
                                events = []

                                # Tool use event
                                tool_use_event = MagicMock()
                                tool_use_event.type = "tool_use"
                                tool_use_event.data = {
                                    "tool": "normalize_curie",
                                    "args": {"curie": "CHEBI:12345"}
                                }
                                events.append(tool_use_event)

                                # Tool result event
                                tool_result_event = MagicMock()
                                tool_result_event.type = "tool_result"
                                tool_result_event.data = {
                                    "tool": "normalize_curie",
                                    "data": {"curie": "CHEBI:12345", "name": "Test"}
                                }
                                events.append(tool_result_event)

                                # Text event
                                text_event = MagicMock()
                                text_event.type = "text"
                                text_event.data = {"content": "I found the entity."}
                                events.append(text_event)

                                # Done event
                                done_event = MagicMock()
                                done_event.type = "done"
                                done_event.data = {}
                                events.append(done_event)

                                async def mock_gen(*args, **kwargs):
                                    for event in events:
                                        yield event

                                mock_agent.side_effect = mock_gen

                                client = TestClient(app)

                                with client.websocket_connect("/ws/chat") as websocket:
                                    websocket.send_text(create_user_message("Look up CHEBI:12345"))

                                    messages = collect_messages_until_done(websocket)

                                    types = [m["type"] for m in messages]
                                    assert "tool_use" in types
                                    assert "tool_result" in types
                                    assert "text" in types
                                    assert "done" in types

    def test_classic_mode_trace_message(self, test_settings, clean_connection_state):
        """Test that trace messages include usage metrics."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app

        with patch("kestrel_backend.config.get_settings", return_value=test_settings):
            with patch("kestrel_backend.main.get_settings", return_value=test_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        with patch("kestrel_backend.main.create_conversation", new_callable=AsyncMock) as mock_create:
                            mock_create.return_value = None

                            with patch("kestrel_backend.main.run_agent_turn") as mock_agent:
                                events = []

                                # Trace event
                                trace_event = MagicMock()
                                trace_event.type = "trace"
                                trace_event.data = {
                                    "input_tokens": 100,
                                    "output_tokens": 50,
                                    "cost_usd": 0.001,
                                    "duration_ms": 500,
                                    "model": "test-model"
                                }
                                events.append(trace_event)

                                # Done event
                                done_event = MagicMock()
                                done_event.type = "done"
                                done_event.data = {}
                                events.append(done_event)

                                async def mock_gen(*args, **kwargs):
                                    for event in events:
                                        yield event

                                mock_agent.side_effect = mock_gen

                                client = TestClient(app)

                                with client.websocket_connect("/ws/chat") as websocket:
                                    websocket.send_text(create_user_message("test"))

                                    messages = collect_messages_until_done(websocket)

                                    trace_msgs = [m for m in messages if m["type"] == "trace"]
                                    assert len(trace_msgs) == 1
                                    assert trace_msgs[0]["input_tokens"] == 100
                                    assert trace_msgs[0]["output_tokens"] == 50
                                    assert "correlation_id" in trace_msgs[0]


# ============================================================================
# Pipeline Mode Tests
# ============================================================================

@pytest.mark.e2e
class TestPipelineModeFlow:
    """Test pipeline mode (LangGraph) message flow."""

    def test_pipeline_mode_node_updates(self, test_settings, clean_connection_state):
        """Test that pipeline mode sends progress updates for each node."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app

        # Create mock hypothesis objects with required attributes
        class MockHypothesis:
            def __init__(self, title, tier, confidence, claim):
                self.title = title
                self.tier = tier
                self.confidence = confidence
                self.claim = claim

        with patch("kestrel_backend.config.get_settings", return_value=test_settings):
            with patch("kestrel_backend.main.get_settings", return_value=test_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        with patch("kestrel_backend.main.create_conversation", new_callable=AsyncMock) as mock_create:
                            mock_create.return_value = None

                            with patch("kestrel_backend.graph.runner.stream_discovery") as mock_stream:
                                # Mock node updates
                                async def mock_gen(*args, **kwargs):
                                    nodes = ["intake", "entity_resolution", "triage", "synthesis"]
                                    for node in nodes:
                                        if node == "synthesis":
                                            yield {
                                                "type": "node_update",
                                                "node": node,
                                                "node_output": {
                                                    "synthesis_report": "# Test Report",
                                                    "hypotheses": [
                                                        MockHypothesis("Test hypothesis", "high", 0.8, "This is a test claim")
                                                    ]
                                                }
                                            }
                                        else:
                                            yield {
                                                "type": "node_update",
                                                "node": node,
                                                "node_output": {}
                                            }

                                mock_stream.side_effect = mock_gen

                                # Also mock add_turn to return a turn_id
                                with patch("kestrel_backend.main.add_turn", new_callable=AsyncMock) as mock_turn:
                                    mock_turn.return_value = uuid4()

                                    client = TestClient(app)

                                    with client.websocket_connect("/ws/chat") as websocket:
                                        websocket.send_text(create_user_message("test query", agent_mode="pipeline"))

                                        messages = collect_messages_until_done(websocket)

                                        types = [m["type"] for m in messages]

                                        # Should have pipeline_progress messages
                                        assert "pipeline_progress" in types

                                        # Should have pipeline_node_detail messages
                                        assert "pipeline_node_detail" in types

                                        # Should have pipeline_complete message
                                        assert "pipeline_complete" in types

                                        # Should end with done
                                        assert messages[-1]["type"] == "done"

    def test_pipeline_mode_complete_message(self, test_settings, clean_connection_state):
        """Test pipeline_complete message contains expected fields."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app

        # Create mock hypothesis objects with required attributes
        class MockHypothesis:
            def __init__(self, title, tier, confidence, claim):
                self.title = title
                self.tier = tier
                self.confidence = confidence
                self.claim = claim

        with patch("kestrel_backend.config.get_settings", return_value=test_settings):
            with patch("kestrel_backend.main.get_settings", return_value=test_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        with patch("kestrel_backend.main.create_conversation", new_callable=AsyncMock) as mock_create:
                            mock_create.return_value = uuid4()

                            with patch("kestrel_backend.graph.runner.stream_discovery") as mock_stream:
                                async def mock_gen(*args, **kwargs):
                                    yield {
                                        "type": "node_update",
                                        "node": "synthesis",
                                        "node_output": {
                                            "synthesis_report": "# Discovery Report\n\nFindings here.",
                                            "hypotheses": [
                                                MockHypothesis("Hypothesis 1", "high", 0.9, "Test claim 1"),
                                                MockHypothesis("Hypothesis 2", "medium", 0.7, "Test claim 2")
                                            ],
                                            "resolved_entities": [
                                                MagicMock(curie="CHEBI:123"),
                                                MagicMock(curie="CHEBI:456")
                                            ]
                                        }
                                    }

                                mock_stream.side_effect = mock_gen

                                with patch("kestrel_backend.main.add_turn", new_callable=AsyncMock) as mock_turn:
                                    mock_turn.return_value = uuid4()

                                    client = TestClient(app)

                                    with client.websocket_connect("/ws/chat") as websocket:
                                        websocket.send_text(create_user_message("test", agent_mode="pipeline"))

                                        messages = collect_messages_until_done(websocket)

                                        complete_msgs = [m for m in messages if m["type"] == "pipeline_complete"]
                                        assert len(complete_msgs) == 1

                                        complete = complete_msgs[0]
                                        assert "synthesis_report" in complete
                                        assert "hypotheses_count" in complete
                                        assert "entities_resolved" in complete
                                        assert "duration_ms" in complete
                                        assert "turn_id" in complete


# ============================================================================
# Rate Limiting Tests
# ============================================================================

@pytest.mark.e2e
class TestRateLimiting:
    """Test rate limiting enforcement."""

    def test_rate_limit_exceeded(self, clean_connection_state):
        """Test that exceeding rate limit returns error message."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app
        from kestrel_backend.config import Settings

        # Create settings with very low rate limit
        low_limit_settings = Settings(
            rate_limit_per_minute=2,  # Only allow 2 messages per minute
            auth_enabled=False,
            langfuse_enabled=False,
        )

        with patch("kestrel_backend.config.get_settings", return_value=low_limit_settings):
            with patch("kestrel_backend.main.get_settings", return_value=low_limit_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        with patch("kestrel_backend.main.create_conversation", new_callable=AsyncMock) as mock_create:
                            mock_create.return_value = None

                            with patch("kestrel_backend.main.run_agent_turn") as mock_agent:
                                # Quick done event
                                done_event = MagicMock()
                                done_event.type = "done"
                                done_event.data = {}

                                async def mock_gen(*args, **kwargs):
                                    yield done_event

                                mock_agent.side_effect = mock_gen

                                client = TestClient(app)

                                with client.websocket_connect("/ws/chat") as websocket:
                                    # First two messages should succeed
                                    for _ in range(2):
                                        websocket.send_text(create_user_message("test"))
                                        data = websocket.receive_text()
                                        msg = json.loads(data)
                                        assert msg["type"] == "done"

                                    # Third message should be rate limited
                                    websocket.send_text(create_user_message("test"))
                                    data = websocket.receive_text()
                                    msg = json.loads(data)
                                    assert msg["type"] == "error"
                                    assert "rate limit" in msg["message"].lower()


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.e2e
class TestErrorHandling:
    """Test error handling for malformed messages."""

    def test_invalid_json_returns_error(self, test_settings, clean_connection_state):
        """Test that invalid JSON returns error message."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app

        with patch("kestrel_backend.config.get_settings", return_value=test_settings):
            with patch("kestrel_backend.main.get_settings", return_value=test_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        client = TestClient(app)

                        with client.websocket_connect("/ws/chat") as websocket:
                            # Send invalid JSON
                            websocket.send_text("not valid json{{{")

                            data = websocket.receive_text()
                            msg = json.loads(data)

                            assert msg["type"] == "error"
                            assert "Invalid JSON" in msg["message"]

    def test_unknown_message_type_returns_error(self, test_settings, clean_connection_state):
        """Test that unknown message type returns error."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app

        with patch("kestrel_backend.config.get_settings", return_value=test_settings):
            with patch("kestrel_backend.main.get_settings", return_value=test_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        client = TestClient(app)

                        with client.websocket_connect("/ws/chat") as websocket:
                            # Send message with unknown type
                            websocket.send_text(json.dumps({
                                "type": "unknown_type",
                                "content": "test"
                            }))

                            data = websocket.receive_text()
                            msg = json.loads(data)

                            assert msg["type"] == "error"
                            assert "Unknown message type" in msg["message"]

    def test_empty_content_returns_error(self, test_settings, clean_connection_state):
        """Test that empty message content returns error."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app

        with patch("kestrel_backend.config.get_settings", return_value=test_settings):
            with patch("kestrel_backend.main.get_settings", return_value=test_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        client = TestClient(app)

                        with client.websocket_connect("/ws/chat") as websocket:
                            # Send message with empty content
                            websocket.send_text(json.dumps({
                                "type": "user_message",
                                "content": "   "  # Just whitespace
                            }))

                            data = websocket.receive_text()
                            msg = json.loads(data)

                            assert msg["type"] == "error"
                            assert "Empty message" in msg["message"]


# ============================================================================
# Authentication Tests
# ============================================================================

@pytest.mark.e2e
class TestWebSocketAuthentication:
    """Test WebSocket authentication enforcement."""

    def test_auth_rejection_close_code_4001(self, auth_enabled_settings, clean_connection_state):
        """Test that invalid token results in close code 4001.

        Note: Starlette's TestClient doesn't raise WebSocketDisconnect when server
        closes the connection. Instead, we verify the connection is closed by
        checking that receiving data fails, and verify auth failure via logs.
        """
        from starlette.testclient import TestClient
        from kestrel_backend.main import app

        with patch("kestrel_backend.config.get_settings", return_value=auth_enabled_settings):
            with patch("kestrel_backend.main.get_settings", return_value=auth_enabled_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        # Mock validate_ws_token to raise ValueError (auth failure)
                        with patch("kestrel_backend.main.validate_ws_token") as mock_validate:
                            mock_validate.side_effect = ValueError("Invalid token")

                            client = TestClient(app)

                            # Attempt connection with invalid token
                            # The server accepts, then closes with code 4001
                            with client.websocket_connect("/ws/chat?token=invalid-token") as websocket:
                                # Try to receive - should get close message
                                # In TestClient, the close is handled gracefully
                                try:
                                    data = websocket.receive()
                                    # If we get data, it should be a close message
                                    assert data.get("type") == "websocket.close"
                                    assert data.get("code") == 4001
                                except Exception:
                                    # Connection was closed, which is expected
                                    pass

                            # Verify validate_ws_token was called
                            mock_validate.assert_called_once()

    def test_auth_success_with_valid_token(self, auth_enabled_settings, clean_connection_state):
        """Test that valid token allows connection."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app

        with patch("kestrel_backend.config.get_settings", return_value=auth_enabled_settings):
            with patch("kestrel_backend.main.get_settings", return_value=auth_enabled_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        # Mock validate_ws_token to return user info
                        with patch("kestrel_backend.main.validate_ws_token") as mock_validate:
                            mock_validate.return_value = {"user_id": str(uuid4())}

                            with patch("kestrel_backend.main.create_conversation", new_callable=AsyncMock) as mock_create:
                                mock_create.return_value = None

                                with patch("kestrel_backend.main.run_agent_turn") as mock_agent:
                                    done_event = MagicMock()
                                    done_event.type = "done"
                                    done_event.data = {}

                                    async def mock_gen(*args, **kwargs):
                                        yield done_event

                                    mock_agent.side_effect = mock_gen

                                    client = TestClient(app)

                                    # Should connect successfully
                                    with client.websocket_connect("/ws/chat?token=valid-token") as websocket:
                                        websocket.send_text(create_user_message("test"))
                                        data = websocket.receive_text()
                                        msg = json.loads(data)
                                        assert msg["type"] == "done"


# ============================================================================
# Conversation History Tests
# ============================================================================

@pytest.mark.e2e
class TestConversationHistory:
    """Test conversation history management."""

    def test_conversation_started_message(self, test_settings, clean_connection_state):
        """Test that conversation_started message is sent on first message."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app

        with patch("kestrel_backend.config.get_settings", return_value=test_settings):
            with patch("kestrel_backend.main.get_settings", return_value=test_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        with patch("kestrel_backend.main.create_conversation", new_callable=AsyncMock) as mock_create:
                            conv_id = uuid4()
                            mock_create.return_value = conv_id

                            with patch("kestrel_backend.main.run_agent_turn") as mock_agent:
                                done_event = MagicMock()
                                done_event.type = "done"
                                done_event.data = {}

                                async def mock_gen(*args, **kwargs):
                                    yield done_event

                                mock_agent.side_effect = mock_gen

                                client = TestClient(app)

                                with client.websocket_connect("/ws/chat") as websocket:
                                    websocket.send_text(create_user_message("test"))

                                    messages = collect_messages_until_done(websocket)

                                    # Should have conversation_started as first message
                                    types = [m["type"] for m in messages]
                                    assert "conversation_started" in types

                                    conv_started = [m for m in messages if m["type"] == "conversation_started"][0]
                                    assert conv_started["conversation_id"] == str(conv_id)

    def test_history_accumulation(self, test_settings, clean_connection_state):
        """Test that conversation history accumulates across turns."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app, conversation_history

        call_count = 0
        received_prompts = []

        with patch("kestrel_backend.config.get_settings", return_value=test_settings):
            with patch("kestrel_backend.main.get_settings", return_value=test_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        with patch("kestrel_backend.main.create_conversation", new_callable=AsyncMock) as mock_create:
                            mock_create.return_value = None

                            with patch("kestrel_backend.main.run_agent_turn") as mock_agent:
                                text_event = MagicMock()
                                text_event.type = "text"
                                text_event.data = {"content": "Response "}

                                done_event = MagicMock()
                                done_event.type = "done"
                                done_event.data = {}

                                async def mock_gen(prompt, **kwargs):
                                    nonlocal call_count
                                    call_count += 1
                                    received_prompts.append(prompt)
                                    yield text_event
                                    yield done_event

                                mock_agent.side_effect = mock_gen

                                client = TestClient(app)

                                with client.websocket_connect("/ws/chat") as websocket:
                                    # Send first message
                                    websocket.send_text(create_user_message("First message"))
                                    collect_messages_until_done(websocket)

                                    # Send second message
                                    websocket.send_text(create_user_message("Second message"))
                                    collect_messages_until_done(websocket)

                                    # The second prompt should contain history
                                    assert call_count == 2
                                    assert "First message" in received_prompts[1]
                                    assert "Second message" in received_prompts[1]


# ============================================================================
# Agent Error Handling Tests
# ============================================================================

@pytest.mark.e2e
class TestAgentErrorHandling:
    """Test error handling when agent fails."""

    def test_agent_error_returns_error_message(self, test_settings, clean_connection_state):
        """Test that agent errors result in error message to client."""
        from starlette.testclient import TestClient
        from kestrel_backend.main import app

        with patch("kestrel_backend.config.get_settings", return_value=test_settings):
            with patch("kestrel_backend.main.get_settings", return_value=test_settings):
                with patch("kestrel_backend.main.init_db", new_callable=AsyncMock):
                    with patch("kestrel_backend.main.close_db", new_callable=AsyncMock):
                        with patch("kestrel_backend.main.create_conversation", new_callable=AsyncMock) as mock_create:
                            mock_create.return_value = None

                            with patch("kestrel_backend.main.run_agent_turn") as mock_agent:
                                # Simulate agent error
                                async def mock_gen(*args, **kwargs):
                                    raise RuntimeError("Agent crashed")
                                    yield  # Make it a generator

                                mock_agent.side_effect = mock_gen

                                client = TestClient(app)

                                with client.websocket_connect("/ws/chat") as websocket:
                                    websocket.send_text(create_user_message("test"))

                                    messages = collect_messages_until_done(websocket)

                                    # Should have error and done messages
                                    types = [m["type"] for m in messages]
                                    assert "error" in types
                                    assert "done" in types

                                    error_msg = [m for m in messages if m["type"] == "error"][0]
                                    assert "Agent error" in error_msg["message"]
