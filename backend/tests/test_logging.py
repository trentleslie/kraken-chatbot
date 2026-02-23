"""Tests for structured JSON logging with correlation IDs."""

import json
import logging
from io import StringIO

import pytest

from kestrel_backend.logging_config import (
    configure_logging,
    generate_correlation_id,
    correlation_id,
    JSONFormatter,
)


def test_generate_correlation_id():
    """Test correlation ID generation and context setting."""
    corr_id = generate_correlation_id()
    assert corr_id is not None
    assert len(corr_id) == 36  # UUID format
    assert correlation_id.get() == corr_id


def test_correlation_id_isolation():
    """Test that correlation IDs are isolated per context."""
    corr_id_1 = generate_correlation_id()
    assert correlation_id.get() == corr_id_1

    # Generate a new one
    corr_id_2 = generate_correlation_id()
    assert correlation_id.get() == corr_id_2
    assert corr_id_1 != corr_id_2


def test_json_formatter():
    """Test JSONFormatter produces valid JSON with correlation ID."""
    # Create a string buffer to capture log output
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    formatter = JSONFormatter(fmt='%(message)s')
    handler.setFormatter(formatter)

    # Create a test logger
    test_logger = logging.getLogger("test_json_formatter")
    test_logger.handlers.clear()
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)

    # Set correlation ID
    corr_id = generate_correlation_id()

    # Log a message
    test_logger.info("Test message", extra={"custom_field": "custom_value"})

    # Parse the output
    log_output = log_stream.getvalue().strip()
    log_data = json.loads(log_output)

    # Verify JSON structure
    assert log_data["message"] == "Test message"
    assert log_data["correlation_id"] == corr_id
    assert log_data["level"] == "INFO"
    assert log_data["logger"] == "test_json_formatter"
    assert log_data["custom_field"] == "custom_value"
    assert "timestamp" in log_data
    assert "module" in log_data
    assert "function" in log_data
    assert "line" in log_data


def test_configure_logging_text_format():
    """Test logging configuration with text format."""
    configure_logging(log_level="DEBUG", log_format="text")

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG
    assert len(root_logger.handlers) > 0

    # Check that formatter is not JSONFormatter
    handler = root_logger.handlers[0]
    assert not isinstance(handler.formatter, JSONFormatter)


def test_configure_logging_json_format():
    """Test logging configuration with JSON format."""
    configure_logging(log_level="INFO", log_format="json")

    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO
    assert len(root_logger.handlers) > 0

    # Check that formatter is JSONFormatter
    handler = root_logger.handlers[0]
    assert isinstance(handler.formatter, JSONFormatter)


def test_configure_logging_module_levels():
    """Test module-specific log level configuration."""
    module_levels = {
        "test_module_a": "DEBUG",
        "test_module_b": "WARNING",
    }

    configure_logging(
        log_level="INFO",
        log_format="text",
        module_levels=module_levels
    )

    # Check module-specific levels
    assert logging.getLogger("test_module_a").level == logging.DEBUG
    assert logging.getLogger("test_module_b").level == logging.WARNING


def test_json_formatter_with_exception():
    """Test JSONFormatter includes exception info when present."""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    formatter = JSONFormatter(fmt='%(message)s')
    handler.setFormatter(formatter)

    test_logger = logging.getLogger("test_exception_logger")
    test_logger.handlers.clear()
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.ERROR)

    # Log with exception
    try:
        raise ValueError("Test exception")
    except ValueError:
        test_logger.error("Error occurred", exc_info=True)

    # Parse the output
    log_output = log_stream.getvalue().strip()
    log_data = json.loads(log_output)

    # Verify exception info is included
    assert "exception" in log_data
    assert "ValueError: Test exception" in log_data["exception"]


def test_correlation_id_without_setting():
    """Test that correlation_id field is omitted when not set."""
    # Reset correlation ID
    correlation_id.set(None)

    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    formatter = JSONFormatter(fmt='%(message)s')
    handler.setFormatter(formatter)

    test_logger = logging.getLogger("test_no_correlation")
    test_logger.handlers.clear()
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)

    # Log without correlation ID
    test_logger.info("Test without correlation ID")

    # Parse the output
    log_output = log_stream.getvalue().strip()
    log_data = json.loads(log_output)

    # Correlation ID should not be in the output
    assert "correlation_id" not in log_data or log_data["correlation_id"] is None


def test_trace_message_with_correlation_id():
    """Test that TraceMessage includes correlation_id field."""
    from kestrel_backend.protocol import TraceMessage

    corr_id = generate_correlation_id()

    # Create a trace message with correlation ID
    trace = TraceMessage(
        turn_id="test-turn-123",
        correlation_id=corr_id,
        input_tokens=100,
        output_tokens=200,
        cost_usd=0.05,
        duration_ms=1500,
        tool_calls_count=3,
        model="claude-sonnet-4-20250514"
    )

    # Verify correlation_id is in the message
    trace_dict = trace.model_dump()
    assert trace_dict["correlation_id"] == corr_id
    assert trace_dict["type"] == "trace"

    # Verify it serializes to JSON correctly
    trace_json = trace.model_dump_json()
    parsed = json.loads(trace_json)
    assert parsed["correlation_id"] == corr_id
