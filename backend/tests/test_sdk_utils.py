"""Tests for shared SDK utilities module."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from kestrel_backend.graph.sdk_utils import (
    DEFAULT_MODEL_NAME,
    HAS_SDK,
    KESTREL_ARGS,
    KESTREL_COMMAND,
    ResultMessage,
    chunk,
    create_agent_options,
    get_kestrel_mcp_config,
    query_with_usage,
)
from kestrel_backend.graph.state import ModelUsageRecord


class TestHasSDK:
    """Test SDK availability detection."""

    def test_has_sdk_is_boolean(self):
        assert isinstance(HAS_SDK, bool)


class TestKestrelConfig:
    """Test Kestrel MCP configuration factory."""

    def test_kestrel_constants(self):
        assert KESTREL_COMMAND == "uvx"
        assert KESTREL_ARGS == ["mcp-client-kestrel"]

    def test_get_kestrel_mcp_config_returns_config_or_none(self):
        config = get_kestrel_mcp_config()
        if HAS_SDK:
            assert config is not None
            # McpStdioServerConfig may be a dict or object depending on SDK version
            if isinstance(config, dict):
                assert config["command"] == "uvx"
                assert config["args"] == ["mcp-client-kestrel"]
            else:
                assert config.command == "uvx"
                assert config.args == ["mcp-client-kestrel"]
        else:
            assert config is None


class TestCreateAgentOptions:
    """Test agent options factory."""

    def test_create_with_tools(self):
        options = create_agent_options(
            system_prompt="Test prompt",
            allowed_tools=["tool1", "tool2"],
            max_turns=5,
        )
        if HAS_SDK:
            assert options is not None
            assert options.system_prompt == "Test prompt"
            assert options.allowed_tools == ["tool1", "tool2"]
            assert options.max_turns == 5
        else:
            assert options is None

    def test_create_pure_reasoning(self):
        """Pure reasoning mode: allowed_tools=[], no mcp_servers (synthesis pattern)."""
        options = create_agent_options(
            system_prompt="Reasoning prompt",
            allowed_tools=[],
            max_turns=1,
        )
        if HAS_SDK:
            assert options is not None
            assert options.allowed_tools == []
            assert options.max_turns == 1
        else:
            assert options is None

    def test_create_with_mcp_servers(self):
        config = get_kestrel_mcp_config()
        if config is not None:
            options = create_agent_options(
                system_prompt="With MCP",
                allowed_tools=["hybrid_search"],
                mcp_servers=[config],
            )
            assert options is not None
            assert options.mcp_servers == [config]


class TestChunk:
    """Test list chunking utility."""

    def test_even_split(self):
        assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

    def test_single_chunk(self):
        assert chunk([1, 2, 3], 5) == [[1, 2, 3]]

    def test_empty_list(self):
        assert chunk([], 3) == []

    def test_chunk_size_one(self):
        assert chunk([1, 2, 3], 1) == [[1], [2], [3]]

    def test_exact_size(self):
        assert chunk([1, 2, 3], 3) == [[1, 2, 3]]


# --- Helper factories for mock SDK events ---

def _make_text_event(text: str):
    """Create a mock event with a text content block."""
    block = MagicMock()
    block.text = text
    event = MagicMock(spec=[])  # no .usage attribute
    event.content = [block]
    return event


def _make_result_message(usage_dict: dict | None = None):
    """Create a mock ResultMessage with optional usage dict."""
    event = MagicMock(spec=ResultMessage)
    event.__class__ = ResultMessage  # so isinstance() works
    event.content = []
    event.usage = usage_dict
    return event


def _make_assistant_event(usage_dict: dict | None = None):
    """Create a mock AssistantMessage-like event with usage."""
    event = MagicMock()
    event.content = []
    event.usage = usage_dict
    return event


async def _mock_query_gen(events):
    """Async generator yielding a list of mock events."""
    for event in events:
        yield event


class TestQueryWithUsage:
    """Test query_with_usage() wrapper for text + usage extraction."""

    async def test_text_and_usage_extraction(self, monkeypatch):
        """Happy path: text events + ResultMessage with usage dict."""
        events = [
            _make_text_event("Hello "),
            _make_text_event("world"),
            _make_result_message({
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 10,
            }),
        ]
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.query", lambda **kw: _mock_query_gen(events))
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.HAS_SDK", True)

        text, record = await query_with_usage("test prompt", MagicMock(), "synthesis")

        assert text == "Hello world"
        assert isinstance(record, ModelUsageRecord)
        assert record.node_name == "synthesis"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.cache_creation_tokens == 5
        assert record.cache_read_tokens == 10
        assert record.model_name == DEFAULT_MODEL_NAME

    async def test_cache_tokens_mapped_correctly(self, monkeypatch):
        """Cache token field names are mapped from SDK naming to model naming."""
        events = [
            _make_result_message({
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 42,
                "cache_read_input_tokens": 99,
            }),
        ]
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.query", lambda **kw: _mock_query_gen(events))
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.HAS_SDK", True)

        _, record = await query_with_usage("p", MagicMock(), "triage")

        assert record.cache_creation_tokens == 42
        assert record.cache_read_tokens == 99

    async def test_multi_turn_usage_accumulation(self, monkeypatch):
        """Usage from multiple events is accumulated (dual-path extraction)."""
        events = [
            _make_assistant_event({
                "input_tokens": 50,
                "output_tokens": 20,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }),
            _make_text_event("response"),
            _make_assistant_event({
                "input_tokens": 30,
                "output_tokens": 10,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }),
        ]
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.query", lambda **kw: _mock_query_gen(events))
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.HAS_SDK", True)

        text, record = await query_with_usage("p", MagicMock(), "pathway_enrichment")

        assert text == "response"
        assert record.input_tokens == 80  # 50 + 30
        assert record.output_tokens == 30  # 20 + 10

    async def test_no_usage_returns_none(self, monkeypatch):
        """Events with no .usage attribute → None record."""
        events = [_make_text_event("just text")]
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.query", lambda **kw: _mock_query_gen(events))
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.HAS_SDK", True)

        text, record = await query_with_usage("p", MagicMock(), "test")

        assert text == "just text"
        assert record is None

    async def test_result_message_no_usage_attr(self, monkeypatch):
        """ResultMessage with usage=None → None record."""
        events = [_make_result_message(None)]
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.query", lambda **kw: _mock_query_gen(events))
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.HAS_SDK", True)

        _, record = await query_with_usage("p", MagicMock(), "test")

        assert record is None

    async def test_has_sdk_false_raises(self, monkeypatch):
        """When HAS_SDK is False, raise RuntimeError."""
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.HAS_SDK", False)

        with pytest.raises(RuntimeError, match="SDK is not available"):
            await query_with_usage("p", MagicMock(), "test")
