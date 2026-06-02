"""Tests for shared SDK utilities module."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from kestrel_backend.graph.sdk_utils import (
    DEFAULT_MODEL_NAME,
    HAS_SDK,
    KESTREL_ARGS,
    KESTREL_COMMAND,
    ResultMessage,
    SystemMessage,
    ToolUseBlock,
    chunk,
    classify_mcp_degradation,
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


# --- Helper factories for MCP-diagnostics events (Unit 1, issue #44) ---

def _make_tool_use_event(name: str):
    """Create a mock event carrying a single ToolUseBlock content block."""
    block = MagicMock(spec=ToolUseBlock)  # spec => isinstance True, no .text attr
    block.name = name
    event = MagicMock(spec=[])  # no .usage
    event.content = [block]
    return event


def _make_init_event(tools):
    """Create a mock SystemMessage(subtype='init', data={'tools': [...]})."""
    event = MagicMock(spec=SystemMessage)
    event.subtype = "init"
    event.data = {"tools": tools}
    return event


class TestQueryWithUsageDiagnostics:
    """Unit 1 (issue #44): MCP tool-availability diagnostics on ModelUsageRecord."""

    async def test_counts_mcp_tool_calls(self, monkeypatch):
        """Only mcp__* tool-use blocks are counted; non-mcp tools are ignored."""
        events = [
            _make_init_event(["Bash", "mcp__kestrel__one_hop_query", "mcp__kestrel__get_nodes"]),
            _make_tool_use_event("mcp__kestrel__one_hop_query"),
            _make_tool_use_event("Bash"),  # not counted
            _make_tool_use_event("mcp__kestrel__get_nodes"),
            _make_text_event("done"),
            _make_result_message({"input_tokens": 1, "output_tokens": 1,
                                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}),
        ]
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.query", lambda **kw: _mock_query_gen(events))
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.HAS_SDK", True)

        text, record = await query_with_usage("p", MagicMock(), "pathway_enrichment")

        assert text == "done"
        assert record is not None
        assert record.mcp_tool_calls == 2
        assert record.available_tools == ["Bash", "mcp__kestrel__one_hop_query", "mcp__kestrel__get_nodes"]

    async def test_zero_mcp_tool_calls(self, monkeypatch):
        """A run with no tool-use blocks reports mcp_tool_calls == 0 (the degraded signal)."""
        events = [
            _make_text_event("I cannot use those tools"),
            _make_result_message({"input_tokens": 1, "output_tokens": 1,
                                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}),
        ]
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.query", lambda **kw: _mock_query_gen(events))
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.HAS_SDK", True)

        _, record = await query_with_usage("p", MagicMock(), "pathway_enrichment")

        assert record is not None
        assert record.mcp_tool_calls == 0

    async def test_available_tools_none_without_init_event(self, monkeypatch):
        """When no SystemMessage init event is present, available_tools degrades to None."""
        events = [
            _make_tool_use_event("mcp__kestrel__one_hop_query"),
            _make_result_message({"input_tokens": 1, "output_tokens": 1,
                                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}),
        ]
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.query", lambda **kw: _mock_query_gen(events))
        monkeypatch.setattr("kestrel_backend.graph.sdk_utils.HAS_SDK", True)

        _, record = await query_with_usage("p", MagicMock(), "pathway_enrichment")

        assert record is not None
        assert record.mcp_tool_calls == 1
        assert record.available_tools is None


_KESTREL_EXPECTED = ["mcp__kestrel__one_hop_query", "mcp__kestrel__get_nodes"]


class TestClassifyMcpDegradation:
    """Unit 2 (issue #44): structural MCP-degradation classifier."""

    def test_zero_calls_with_phrase_high_confidence(self):
        v = classify_mcp_degradation(
            _KESTREL_EXPECTED, mcp_tool_calls=0,
            result_text="The Kestrel MCP tools are not available in my current tool set.",
        )
        assert v.degraded is True
        assert v.confidence == "high"  # phrase corroborates the structural signal

    def test_zero_calls_no_phrase_structural(self):
        v = classify_mcp_degradation(
            _KESTREL_EXPECTED, mcp_tool_calls=0, result_text="Here are some shared pathways.",
        )
        assert v.degraded is True

    def test_tools_used_not_degraded(self):
        # Healthy run: at least one mcp call, even with no shared neighbors found.
        v = classify_mcp_degradation(_KESTREL_EXPECTED, mcp_tool_calls=3, result_text="")
        assert v.degraded is False

    def test_empty_expected_never_degraded(self):
        # Data-in-prompt nodes (allowed_tools=[]) must never be flagged, even with the phrase.
        v = classify_mcp_degradation(
            [], mcp_tool_calls=0, result_text="not available in my current tool set",
        )
        assert v.degraded is False

    def test_phrase_alone_does_not_trigger(self):
        # Tools were used but the text contains the phrase → not degraded (phrase is corroborator only).
        v = classify_mcp_degradation(
            _KESTREL_EXPECTED, mcp_tool_calls=2,
            result_text="those tools are not available for everything",
        )
        assert v.degraded is False

    def test_definitive_missing_from_init(self):
        # init tool list lacks an expected tool → definitive degraded, even if a call happened.
        v = classify_mcp_degradation(
            _KESTREL_EXPECTED, mcp_tool_calls=1, result_text="",
            available_tools=["mcp__kestrel__one_hop_query", "Bash"],  # get_nodes missing
        )
        assert v.degraded is True
        assert v.confidence == "definitive"

    def test_all_expected_present_and_used_not_degraded(self):
        v = classify_mcp_degradation(
            _KESTREL_EXPECTED, mcp_tool_calls=2, result_text="",
            available_tools=["mcp__kestrel__one_hop_query", "mcp__kestrel__get_nodes"],
        )
        assert v.degraded is False

    def test_tools_registered_zero_calls_no_phrase_not_degraded(self):
        # init list confirms all expected tools present; zero calls + no phrase → not degraded
        # (the model may legitimately skip tools — avoids a false positive for non-mandating nodes)
        v = classify_mcp_degradation(
            _KESTREL_EXPECTED, mcp_tool_calls=0, result_text="No shared neighbors found.",
            available_tools=["mcp__kestrel__one_hop_query", "mcp__kestrel__get_nodes"],
        )
        assert v.degraded is False

    def test_tools_registered_zero_calls_with_phrase_still_degraded(self):
        # the phrase corroborator still trips even when the init list confirms tools present
        v = classify_mcp_degradation(
            _KESTREL_EXPECTED, mcp_tool_calls=0,
            result_text="those tools are not available in my current tool set",
            available_tools=["mcp__kestrel__one_hop_query", "mcp__kestrel__get_nodes"],
        )
        assert v.degraded is True
