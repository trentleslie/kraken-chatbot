"""Structured JSON logging configuration with correlation ID support."""

import logging
import re
import sys
from contextvars import ContextVar
from typing import Optional
from uuid import uuid4

from pythonjsonlogger.json import JsonFormatter


_KEY_RE = re.compile(r"sk-(?:ant-)?[A-Za-z0-9_-]{16,}")
# Secret-bearing query parameters: token= (the WebSocket carries the Clerk session JWT as
# ?token=...) and api_key=/apikey=/key= (OpenAlex and NCBI keys travel as query parameters).
_TOKEN_QS_RE = re.compile(r"([?&](?:token|api_key|apikey|key)=)[^&\s\"']+", re.IGNORECASE)


class ApiKeyRedactionFilter(logging.Filter):
    """Mask Anthropic and generic sk-... API keys in log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _KEY_RE.sub("sk-***REDACTED***", record.msg)
        if isinstance(record.args, dict):
            record.args = {
                k: (_KEY_RE.sub("sk-***REDACTED***", v) if isinstance(v, str) else v)
                for k, v in record.args.items()
            }
        elif record.args:
            record.args = tuple(
                _KEY_RE.sub("sk-***REDACTED***", a) if isinstance(a, str) else a
                for a in record.args
            )
        return True


class QueryTokenRedactionFilter(logging.Filter):
    """Mask secret query values (?token=, api_key=, key=) in any log line.

    Uvicorn's loggers have their own handlers, so the root handler's filters never see them;
    this is attached to those loggers directly as well as to the root handler. The message is
    rendered before redacting because callers such as httpx pass the URL as an ``httpx.URL``
    object, not a string, so redacting string arguments alone would miss it.
    """

    @staticmethod
    def _redact(value):
        return _TOKEN_QS_RE.sub(r"\1***REDACTED***", value) if isinstance(value, str) else value

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:  # noqa: BLE001 - malformed args: fall back to redacting the parts
            record.msg = self._redact(record.msg)
            if isinstance(record.args, dict):
                record.args = {k: self._redact(v) for k, v in record.args.items()}
            elif record.args:
                record.args = tuple(self._redact(a) for a in record.args)
            return True
        redacted = self._redact(rendered)
        if redacted != rendered:
            record.msg, record.args = redacted, None
        return True


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
    # httpx logs every request URL at INFO, and NCBI/OpenAlex keys travel as query parameters.
    # Pin these to WARNING so no root log level can write API keys into the journal.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

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
    console_handler.addFilter(ApiKeyRedactionFilter())
    console_handler.addFilter(QueryTokenRedactionFilter())
    root_logger.addHandler(console_handler)

    # Uvicorn logs WebSocket connects ("WebSocket /ws/chat?token=...") and HTTP access lines
    # through its own loggers/handlers; filter at the logger so the JWT never reaches them.
    for name in ("uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        if not any(isinstance(f, QueryTokenRedactionFilter) for f in uv_logger.filters):
            uv_logger.addFilter(QueryTokenRedactionFilter())

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
