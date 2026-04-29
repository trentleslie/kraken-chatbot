"""Tests for shared SDK utilities module."""

from kestrel_backend.graph.sdk_utils import (
    HAS_SDK,
    KESTREL_ARGS,
    KESTREL_COMMAND,
    chunk,
    create_agent_options,
    get_kestrel_mcp_config,
)


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
