"""Tests for user feedback functionality."""
import pytest
from uuid import uuid4
from src.kestrel_backend.protocol import TraceMessage, PipelineCompleteMessage
from src.kestrel_backend.database import record_feedback


def test_trace_message_with_trace_id():
    """Test that TraceMessage accepts and stores trace_id."""
    trace_id = "langfuse-trace-123"
    turn_id = "turn-456"

    msg = TraceMessage(
        turn_id=turn_id,
        trace_id=trace_id,
        input_tokens=100,
        output_tokens=200
    )

    assert msg.trace_id == trace_id
    assert msg.turn_id == turn_id
    assert msg.type == "trace"


def test_pipeline_complete_message_with_ids():
    """Test that PipelineCompleteMessage accepts turn_id and trace_id."""
    turn_id = str(uuid4())
    trace_id = "langfuse-trace-789"

    msg = PipelineCompleteMessage(
        synthesis_report="Test report",
        hypotheses_count=3,
        entities_resolved=5,
        duration_ms=1000,
        turn_id=turn_id,
        trace_id=trace_id
    )

    assert msg.turn_id == turn_id
    assert msg.trace_id == trace_id
    assert msg.type == "pipeline_complete"


@pytest.mark.asyncio
async def test_record_feedback_without_db():
    """Test that record_feedback handles missing database gracefully."""
    # When no database pool is available, should return None
    result = await record_feedback(
        turn_id=uuid4(),
        conversation_id=uuid4(),
        feedback_type="positive",
        trace_id="test-trace"
    )

    # Should return None when database is not available
    assert result is None
