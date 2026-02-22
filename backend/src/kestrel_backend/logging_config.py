"""Structured JSON logging configuration with correlation ID support."""

import logging
import sys
from contextvars import ContextVar
from typing import Optional
from uuid import uuid4

from pythonjsonlogger.json import JsonFormatter


# ContextVar for correlation ID - propagates across async tasks
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def generate_correlation_id() -> str:
    """Generate a new correlation ID and set it in the context."""
    corr_id = str(uuid4())
    correlation_id.set(corr_id)
    return corr_id


class JSONFormatter(JsonFormatter):
    """Custom JSON formatter that includes correlation ID and structured fields."""

    def add_fields(self, log_record, record, message_dict):
        """Add custom fields to the log record."""
        super().add_fields(log_record, record, message_dict)

        # Add correlation ID if present
        corr_id = correlation_id.get()
        if corr_id:
            log_record['correlation_id'] = corr_id

        # Add standard fields
        log_record['timestamp'] = self.formatTime(record, self.datefmt)
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno

        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "text",
    module_levels: Optional[dict[str, str]] = None
) -> None:
    """
    Configure application logging with JSON or text format.

    Args:
        log_level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: "json" for structured JSON logs, "text" for human-readable
        module_levels: Optional dict of module-specific log levels {"module.name": "DEBUG"}
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set root log level
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)

    # Choose formatter based on log_format
    if log_format.lower() == "json":
        formatter = JSONFormatter(
            fmt='%(timestamp)s %(level)s %(name)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
    else:
        # Text format for development
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Set module-specific log levels
    if module_levels:
        for module_name, level in module_levels.items():
            logging.getLogger(module_name).setLevel(getattr(logging, level.upper()))

    # Log configuration complete
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "log_level": log_level,
            "log_format": log_format,
            "module_levels": module_levels or {}
        }
    )
