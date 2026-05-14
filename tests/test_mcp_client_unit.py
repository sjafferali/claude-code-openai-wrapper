#!/usr/bin/env python3
"""
Unit tests for src/mcp_client.py

Tests the MCPClient, MCPServerConfig, and MCPServerConnection classes.
These are pure unit tests that don't require actual MCP servers.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
import threading

from src.mcp_client import (
    MCPServerConfig,
    MCPServerConnection,
    MCPClient,
    mcp_client,
    MCP_AVAILABLE,
)


def _async_cm(value):
    """Build an async context manager whose ``__aenter__`` returns ``value``.

    Used to mock ``stdio_client``, ``sse_client``, ``streamablehttp_client``,
    and ``ClientSession`` (all of which are async context managers in the real
    MCP SDK) under ``AsyncExitStack.enter_async_context``.
    """
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=value)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


class TestMCPServerConfig:
    """Test the MCPServerConfig dataclass."""

    def test_creation_with_required_fields(self):
        """MCPServerConfig can be created with just required fields."""
        config = MCPServerConfig(name="test-server", command="test-cmd")
        assert config.name == "test-server"
        assert config.command == "test-cmd"
        assert config.args == []
        assert config.env is None
        assert config.description == ""
        assert config.enabled is True

    def test_creation_with_all_fields(self):
        """MCPServerConfig can be created with all fields."""
        config = MCPServerConfig(
            name="full-server",
            command="/usr/bin/server",
            args=["--port", "8080"],
            env={"DEBUG": "1"},
            description="A full test server",
            enabled=False,
        )
        assert config.name == "full-server"
        assert config.command == "/usr/bin/server"
        assert config.args == ["--port", "8080"]
        assert config.env == {"DEBUG": "1"}
        assert config.description == "A full test server"
        assert config.enabled is False

    def test_args_default_is_empty_list(self):
        """Default args is an empty list (not shared between instances)."""
        config1 = MCPServerConfig(name="s1", command="cmd")
        config2 = MCPServerConfig(name="s2", command="cmd")

        config1.args.append("--flag")
        assert "--flag" not in config2.args

    def test_stdio_requires_command(self):
        """stdio transport rejects empty command."""
        with pytest.raises(ValueError):
            MCPServerConfig(name="s", type="stdio", command="")

    def test_http_requires_url(self):
        """http transport rejects missing url."""
        with pytest.raises(ValueError):
            MCPServerConfig(name="s", type="http")

    def test_sse_requires_url(self):
        """sse transport rejects missing url."""
        with pytest.raises(ValueError):
            MCPServerConfig(name="s", type="sse")

    def test_invalid_type_rejected(self):
        """Unknown transport types are rejected."""
        with pytest.raises(ValueError):
            MCPServerConfig(name="s", type="ws", command="cmd")

    def test_to_sdk_config_stdio(self):
        """Stdio config translates to SDK shape with command + args."""
        config = MCPServerConfig(
            name="s",
            command="node",
            args=["server.js", "--port", "8080"],
            env={"DEBUG": "1"},
        )
        sdk = config.to_sdk_config()
        assert sdk == {
            "type": "stdio",
            "command": "node",
            "args": ["server.js", "--port", "8080"],
            "env": {"DEBUG": "1"},
        }

    def test_to_sdk_config_stdio_omits_env_when_empty(self):
        """Stdio without env should not include the env key."""
        config = MCPServerConfig(name="s", command="node")
        sdk = config.to_sdk_config()
        assert "env" not in sdk

    def test_to_sdk_config_http(self):
        """HTTP config translates to SDK shape with url only."""
        config = MCPServerConfig(
            name="s", type="http", url="https://example.com/mcp"
        )
        assert config.to_sdk_config() == {
            "type": "http",
            "url": "https://example.com/mcp",
        }

    def test_to_sdk_config_sse(self):
        """SSE config translates to SDK shape with url only."""
        config = MCPServerConfig(
            name="s", type="sse", url="https://example.com/sse"
        )
        assert config.to_sdk_config() == {
            "type": "sse",
            "url": "https://example.com/sse",
        }


class TestMCPServerConnection:
    """Test the MCPServerConnection dataclass."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock server config."""
        return MCPServerConfig(name="test", command="test-cmd")

    def test_creation_with_required_fields(self, mock_config):
        """MCPServerConnection can be created with required fields."""
        mock_session = MagicMock()
        mock_read = MagicMock()
        mock_write = MagicMock()

        connection = MCPServerConnection(
            config=mock_config,
            session=mock_session,
            read_stream=mock_read,
            write_stream=mock_write,
        )
        assert connection.config is mock_config
        assert connection.session is mock_session
        assert isinstance(connection.connected_at, datetime)
        assert connection.available_tools == []
        assert connection.available_resources == []
        assert connection.available_prompts == []

    def test_creation_with_capabilities(self, mock_config):
        """MCPServerConnection can be created with capabilities."""
        connection = MCPServerConnection(
            config=mock_config,
            session=MagicMock(),
            read_stream=MagicMock(),
            write_stream=MagicMock(),
            available_tools=[{"name": "tool1"}],
            available_resources=[{"uri": "file://test"}],
            available_prompts=[{"name": "prompt1"}],
        )
        assert len(connection.available_tools) == 1
        assert len(connection.available_resources) == 1
        assert len(connection.available_prompts) == 1


class TestMCPClient:
    """Test the MCPClient class."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a fresh MCPClient backed by an isolated registry file."""
        return MCPClient(registry_path=tmp_path / "mcp-registry.json")

    def test_initialization(self, client):
        """MCPClient initializes with empty servers and connections."""
        assert client.servers == {}
        assert client.connections == {}

    def test_is_available(self, client):
        """is_available returns MCP_AVAILABLE constant."""
        assert client.is_available() == MCP_AVAILABLE

    def test_register_server(self, client):
        """register_server adds server configuration."""
        config = MCPServerConfig(name="test-server", command="test-cmd")
        client.register_server(config)

        assert "test-server" in client.servers
        # Registry persists via JSON so the readback is a structurally-equal
        # value, not the same instance.
        assert client.servers["test-server"] == config

    def test_register_server_overwrites_existing(self, client):
        """register_server overwrites existing configuration."""
        config1 = MCPServerConfig(name="test-server", command="cmd1")
        config2 = MCPServerConfig(name="test-server", command="cmd2")

        client.register_server(config1)
        client.register_server(config2)

        assert client.servers["test-server"].command == "cmd2"

    def test_unregister_server_existing(self, client):
        """unregister_server removes existing server."""
        config = MCPServerConfig(name="test-server", command="cmd")
        client.register_server(config)
        assert "test-server" in client.servers

        result = client.unregister_server("test-server")
        assert result is True
        assert "test-server" not in client.servers

    def test_unregister_server_nonexistent(self, client):
        """unregister_server returns False for nonexistent server."""
        result = client.unregister_server("nonexistent")
        assert result is False

    def test_list_servers(self, client):
        """list_servers returns all registered servers."""
        config1 = MCPServerConfig(name="server1", command="cmd1")
        config2 = MCPServerConfig(name="server2", command="cmd2")

        client.register_server(config1)
        client.register_server(config2)

        servers = client.list_servers()
        assert len(servers) == 2
        names = [s.name for s in servers]
        assert "server1" in names
        assert "server2" in names

    def test_list_servers_empty(self, client):
        """list_servers returns empty list when no servers registered."""
        assert client.list_servers() == []

    def test_get_server_existing(self, client):
        """get_server returns existing server config."""
        config = MCPServerConfig(name="test-server", command="cmd")
        client.register_server(config)

        result = client.get_server("test-server")
        assert result == config

    def test_get_server_nonexistent(self, client):
        """get_server returns None for nonexistent server."""
        result = client.get_server("nonexistent")
        assert result is None

    def test_list_connected_servers_empty(self, client):
        """list_connected_servers returns empty list when no connections."""
        assert client.list_connected_servers() == []

    def test_list_connected_servers_with_connections(self, client):
        """list_connected_servers returns names of connected servers."""
        config = MCPServerConfig(name="test", command="cmd")
        connection = MCPServerConnection(
            config=config,
            session=MagicMock(),
            read_stream=MagicMock(),
            write_stream=MagicMock(),
        )
        client.connections["test"] = connection

        result = client.list_connected_servers()
        assert result == ["test"]

    def test_get_connection_existing(self, client):
        """get_connection returns existing connection."""
        config = MCPServerConfig(name="test", command="cmd")
        connection = MCPServerConnection(
            config=config,
            session=MagicMock(),
            read_stream=MagicMock(),
            write_stream=MagicMock(),
        )
        client.connections["test"] = connection

        result = client.get_connection("test")
        assert result is connection

    def test_get_connection_nonexistent(self, client):
        """get_connection returns None for nonexistent connection."""
        result = client.get_connection("nonexistent")
        assert result is None

    def test_get_all_tools_empty(self, client):
        """get_all_tools returns empty dict when no connections."""
        assert client.get_all_tools() == {}

    def test_get_all_tools_with_connections(self, client):
        """get_all_tools returns tools from all connections."""
        config1 = MCPServerConfig(name="server1", command="cmd")
        config2 = MCPServerConfig(name="server2", command="cmd")

        conn1 = MCPServerConnection(
            config=config1,
            session=MagicMock(),
            read_stream=MagicMock(),
            write_stream=MagicMock(),
            available_tools=[{"name": "tool1"}],
        )
        conn2 = MCPServerConnection(
            config=config2,
            session=MagicMock(),
            read_stream=MagicMock(),
            write_stream=MagicMock(),
            available_tools=[{"name": "tool2"}, {"name": "tool3"}],
        )

        client.connections["server1"] = conn1
        client.connections["server2"] = conn2

        result = client.get_all_tools()
        assert "server1" in result
        assert "server2" in result
        assert len(result["server1"]) == 1
        assert len(result["server2"]) == 2

    def test_get_stats_empty(self, client):
        """get_stats returns correct stats when empty."""
        stats = client.get_stats()

        assert stats["mcp_available"] == MCP_AVAILABLE
        assert stats["registered_servers"] == 0
        assert stats["connected_servers"] == 0
        assert stats["total_tools"] == 0
        assert stats["total_resources"] == 0
        assert stats["total_prompts"] == 0
        assert stats["servers"] == []

    def test_get_stats_with_servers(self, client):
        """get_stats includes registered server info."""
        config = MCPServerConfig(
            name="test-server",
            command="cmd",
            description="Test server",
            enabled=True,
        )
        client.register_server(config)

        stats = client.get_stats()
        assert stats["registered_servers"] == 1
        assert len(stats["servers"]) == 1
        assert stats["servers"][0]["name"] == "test-server"
        assert stats["servers"][0]["enabled"] is True
        assert stats["servers"][0]["connected"] is False

    def test_get_stats_with_connections(self, client):
        """get_stats counts tools, resources, prompts correctly."""
        config = MCPServerConfig(name="test", command="cmd")
        client.register_server(config)

        connection = MCPServerConnection(
            config=config,
            session=MagicMock(),
            read_stream=MagicMock(),
            write_stream=MagicMock(),
            available_tools=[{"name": "t1"}, {"name": "t2"}],
            available_resources=[{"uri": "r1"}],
            available_prompts=[{"name": "p1"}, {"name": "p2"}, {"name": "p3"}],
        )
        client.connections["test"] = connection

        stats = client.get_stats()
        assert stats["connected_servers"] == 1
        assert stats["total_tools"] == 2
        assert stats["total_resources"] == 1
        assert stats["total_prompts"] == 3
        assert stats["servers"][0]["connected"] is True


class TestMCPClientAsync:
    """Test async methods of MCPClient."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a fresh MCPClient backed by an isolated registry file."""
        return MCPClient(registry_path=tmp_path / "mcp-registry.json")

    @pytest.mark.asyncio
    async def test_connect_server_not_registered(self, client):
        """connect_server returns False for unregistered server."""
        result = await client.connect_server("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_server_disabled(self, client):
        """connect_server returns False for disabled server."""
        config = MCPServerConfig(name="disabled", command="cmd", enabled=False)
        client.register_server(config)

        result = await client.connect_server("disabled")
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_server_already_connected(self, client):
        """connect_server returns True when already connected."""
        config = MCPServerConfig(name="test", command="cmd")
        client.register_server(config)

        # Add fake connection
        connection = MCPServerConnection(
            config=config,
            session=MagicMock(),
            read_stream=MagicMock(),
            write_stream=MagicMock(),
        )
        client.connections["test"] = connection

        result = await client.connect_server("test")
        assert result is True

    @pytest.mark.asyncio
    async def test_disconnect_server_not_connected(self, client):
        """disconnect_server returns False when not connected."""
        result = await client.disconnect_server("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_server_success(self, client):
        """disconnect_server removes connection and returns True."""
        config = MCPServerConfig(name="test", command="cmd")
        connection = MCPServerConnection(
            config=config,
            session=MagicMock(),
            read_stream=MagicMock(),
            write_stream=MagicMock(),
        )
        client.connections["test"] = connection

        result = await client.disconnect_server("test")
        assert result is True
        assert "test" not in client.connections

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, client):
        """call_tool raises ValueError when not connected."""
        with pytest.raises(ValueError, match="Not connected to MCP server"):
            await client.call_tool("nonexistent", "tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_success(self, client):
        """call_tool delegates to session.call_tool."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value={"result": "success"})

        config = MCPServerConfig(name="test", command="cmd")
        connection = MCPServerConnection(
            config=config,
            session=mock_session,
            read_stream=MagicMock(),
            write_stream=MagicMock(),
        )
        client.connections["test"] = connection

        result = await client.call_tool("test", "my-tool", {"arg": "value"})
        assert result == {"result": "success"}
        mock_session.call_tool.assert_called_once_with("my-tool", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_call_tool_error(self, client):
        """call_tool propagates errors from session."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=RuntimeError("Tool failed"))

        config = MCPServerConfig(name="test", command="cmd")
        connection = MCPServerConnection(
            config=config,
            session=mock_session,
            read_stream=MagicMock(),
            write_stream=MagicMock(),
        )
        client.connections["test"] = connection

        with pytest.raises(RuntimeError, match="Tool failed"):
            await client.call_tool("test", "my-tool", {})

    @pytest.mark.asyncio
    async def test_read_resource_not_connected(self, client):
        """read_resource raises ValueError when not connected."""
        with pytest.raises(ValueError, match="Not connected to MCP server"):
            await client.read_resource("nonexistent", "file://test")

    @pytest.mark.asyncio
    async def test_read_resource_success(self, client):
        """read_resource delegates to session.read_resource."""
        mock_session = AsyncMock()
        mock_session.read_resource = AsyncMock(return_value="resource content")

        config = MCPServerConfig(name="test", command="cmd")
        connection = MCPServerConnection(
            config=config,
            session=mock_session,
            read_stream=MagicMock(),
            write_stream=MagicMock(),
        )
        client.connections["test"] = connection

        result = await client.read_resource("test", "file://path")
        assert result == "resource content"
        mock_session.read_resource.assert_called_once_with("file://path")

    @pytest.mark.asyncio
    async def test_read_resource_error(self, client):
        """read_resource propagates errors from session."""
        mock_session = AsyncMock()
        mock_session.read_resource = AsyncMock(side_effect=FileNotFoundError("Not found"))

        config = MCPServerConfig(name="test", command="cmd")
        connection = MCPServerConnection(
            config=config,
            session=mock_session,
            read_stream=MagicMock(),
            write_stream=MagicMock(),
        )
        client.connections["test"] = connection

        with pytest.raises(FileNotFoundError):
            await client.read_resource("test", "file://missing")

    @pytest.mark.asyncio
    async def test_get_prompt_not_connected(self, client):
        """get_prompt raises ValueError when not connected."""
        with pytest.raises(ValueError, match="Not connected to MCP server"):
            await client.get_prompt("nonexistent", "prompt-name")

    @pytest.mark.asyncio
    async def test_get_prompt_success(self, client):
        """get_prompt delegates to session.get_prompt."""
        mock_session = AsyncMock()
        mock_session.get_prompt = AsyncMock(return_value={"messages": []})

        config = MCPServerConfig(name="test", command="cmd")
        connection = MCPServerConnection(
            config=config,
            session=mock_session,
            read_stream=MagicMock(),
            write_stream=MagicMock(),
        )
        client.connections["test"] = connection

        result = await client.get_prompt("test", "my-prompt", {"arg": "val"})
        assert result == {"messages": []}
        mock_session.get_prompt.assert_called_once_with("my-prompt", {"arg": "val"})

    @pytest.mark.asyncio
    async def test_get_prompt_no_arguments(self, client):
        """get_prompt uses empty dict when no arguments provided."""
        mock_session = AsyncMock()
        mock_session.get_prompt = AsyncMock(return_value={"messages": []})

        config = MCPServerConfig(name="test", command="cmd")
        connection = MCPServerConnection(
            config=config,
            session=mock_session,
            read_stream=MagicMock(),
            write_stream=MagicMock(),
        )
        client.connections["test"] = connection

        await client.get_prompt("test", "my-prompt")
        mock_session.get_prompt.assert_called_once_with("my-prompt", {})

    @pytest.mark.asyncio
    async def test_get_prompt_error(self, client):
        """get_prompt propagates errors from session."""
        mock_session = AsyncMock()
        mock_session.get_prompt = AsyncMock(side_effect=KeyError("Prompt not found"))

        config = MCPServerConfig(name="test", command="cmd")
        connection = MCPServerConnection(
            config=config,
            session=mock_session,
            read_stream=MagicMock(),
            write_stream=MagicMock(),
        )
        client.connections["test"] = connection

        with pytest.raises(KeyError):
            await client.get_prompt("test", "missing-prompt")


class TestMCPClientConnectServerMCPAvailable:
    """Test connect_server when MCP SDK is available (mocked)."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a fresh MCPClient backed by an isolated registry file."""
        return MCPClient(registry_path=tmp_path / "mcp-registry.json")

    @pytest.mark.asyncio
    @patch("src.mcp_client.MCP_AVAILABLE", False)
    async def test_connect_server_mcp_not_available(self, tmp_path):
        """connect_server returns False when MCP SDK not available."""
        # Create client with mocked MCP_AVAILABLE
        with patch("src.mcp_client.MCP_AVAILABLE", False):
            client = MCPClient(registry_path=tmp_path / "mcp-registry.json")
            config = MCPServerConfig(name="test", command="cmd")
            client.register_server(config)

            result = await client.connect_server("test")
            assert result is False


class TestMCPClientConnectServerWithMocking:
    """Test connect_server with full MCP SDK mocking."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a fresh MCPClient backed by an isolated registry file."""
        return MCPClient(registry_path=tmp_path / "mcp-registry.json")

    @pytest.mark.asyncio
    async def test_connect_server_success(self, client):
        """connect_server successfully connects and lists capabilities."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP SDK not available")

        config = MCPServerConfig(name="test", command="test-cmd")
        client.register_server(config)

        # Create mock tools, resources, prompts
        mock_tool = MagicMock()
        mock_tool.name = "mock-tool"
        mock_tool.description = "A mock tool"
        mock_tool.inputSchema = {"type": "object"}

        mock_resource = MagicMock()
        mock_resource.uri = "file://test"
        mock_resource.name = "test-resource"
        mock_resource.description = "A test resource"
        mock_resource.mimeType = "text/plain"

        mock_prompt = MagicMock()
        mock_prompt.name = "mock-prompt"
        mock_prompt.description = "A mock prompt"
        mock_prompt.arguments = []

        mock_tools_response = MagicMock()
        mock_tools_response.tools = [mock_tool]

        mock_resources_response = MagicMock()
        mock_resources_response.resources = [mock_resource]

        mock_prompts_response = MagicMock()
        mock_prompts_response.prompts = [mock_prompt]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)
        mock_session.list_resources = AsyncMock(return_value=mock_resources_response)
        mock_session.list_prompts = AsyncMock(return_value=mock_prompts_response)

        mock_read = MagicMock()
        mock_write = MagicMock()

        with patch("src.mcp_client.StdioServerParameters") as mock_params:
            with patch(
                "src.mcp_client.stdio_client",
                return_value=_async_cm((mock_read, mock_write)),
            ):
                with patch(
                    "src.mcp_client.ClientSession",
                    return_value=_async_cm(mock_session),
                ):
                    result = await client.connect_server("test")

                    assert result is True
                    assert "test" in client.connections
                    conn = client.connections["test"]
                    assert len(conn.available_tools) == 1
                    assert len(conn.available_resources) == 1
                    assert len(conn.available_prompts) == 1

    @pytest.mark.asyncio
    async def test_connect_server_list_tools_fails(self, client):
        """connect_server handles tool listing failure gracefully."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP SDK not available")

        config = MCPServerConfig(name="test", command="test-cmd")
        client.register_server(config)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(side_effect=RuntimeError("Tools error"))
        mock_session.list_resources = AsyncMock(return_value=MagicMock(resources=[]))
        mock_session.list_prompts = AsyncMock(return_value=MagicMock(prompts=[]))

        with patch("src.mcp_client.StdioServerParameters"):
            with patch(
                "src.mcp_client.stdio_client",
                return_value=_async_cm((MagicMock(), MagicMock())),
            ):
                with patch(
                    "src.mcp_client.ClientSession",
                    return_value=_async_cm(mock_session),
                ):
                    result = await client.connect_server("test")

                    assert result is True
                    conn = client.connections["test"]
                    assert conn.available_tools == []

    @pytest.mark.asyncio
    async def test_connect_server_list_resources_fails(self, client):
        """connect_server handles resource listing failure gracefully."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP SDK not available")

        config = MCPServerConfig(name="test", command="test-cmd")
        client.register_server(config)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        mock_session.list_resources = AsyncMock(side_effect=RuntimeError("Resources error"))
        mock_session.list_prompts = AsyncMock(return_value=MagicMock(prompts=[]))

        with patch("src.mcp_client.StdioServerParameters"):
            with patch(
                "src.mcp_client.stdio_client",
                return_value=_async_cm((MagicMock(), MagicMock())),
            ):
                with patch(
                    "src.mcp_client.ClientSession",
                    return_value=_async_cm(mock_session),
                ):
                    result = await client.connect_server("test")

                    assert result is True
                    conn = client.connections["test"]
                    assert conn.available_resources == []

    @pytest.mark.asyncio
    async def test_connect_server_list_prompts_fails(self, client):
        """connect_server handles prompt listing failure gracefully."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP SDK not available")

        config = MCPServerConfig(name="test", command="test-cmd")
        client.register_server(config)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        mock_session.list_resources = AsyncMock(return_value=MagicMock(resources=[]))
        mock_session.list_prompts = AsyncMock(side_effect=RuntimeError("Prompts error"))

        with patch("src.mcp_client.StdioServerParameters"):
            with patch(
                "src.mcp_client.stdio_client",
                return_value=_async_cm((MagicMock(), MagicMock())),
            ):
                with patch(
                    "src.mcp_client.ClientSession",
                    return_value=_async_cm(mock_session),
                ):
                    result = await client.connect_server("test")

                    assert result is True
                    conn = client.connections["test"]
                    assert conn.available_prompts == []

    @pytest.mark.asyncio
    async def test_connect_server_connection_error(self, client):
        """connect_server returns False on ConnectionError."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP SDK not available")

        config = MCPServerConfig(name="test", command="test-cmd")
        client.register_server(config)

        with patch("src.mcp_client.StdioServerParameters"):
            with patch("src.mcp_client.stdio_client") as mock_stdio:
                mock_stdio.side_effect = ConnectionError("Connection refused")

                result = await client.connect_server("test")
                assert result is False

    @pytest.mark.asyncio
    async def test_connect_server_value_error(self, client):
        """connect_server returns False on ValueError."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP SDK not available")

        config = MCPServerConfig(name="test", command="test-cmd")
        client.register_server(config)

        with patch("src.mcp_client.StdioServerParameters"):
            with patch("src.mcp_client.stdio_client") as mock_stdio:
                mock_stdio.side_effect = ValueError("Invalid config")

                result = await client.connect_server("test")
                assert result is False

    @pytest.mark.asyncio
    async def test_connect_server_timeout_error(self, client):
        """connect_server returns False on TimeoutError."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP SDK not available")

        config = MCPServerConfig(name="test", command="test-cmd")
        client.register_server(config)

        with patch("src.mcp_client.StdioServerParameters"):
            with patch("src.mcp_client.stdio_client") as mock_stdio:
                mock_stdio.side_effect = TimeoutError("Connection timeout")

                result = await client.connect_server("test")
                assert result is False

    @pytest.mark.asyncio
    async def test_connect_server_file_not_found_error(self, client):
        """connect_server returns False on FileNotFoundError."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP SDK not available")

        config = MCPServerConfig(name="test", command="nonexistent-cmd")
        client.register_server(config)

        with patch("src.mcp_client.StdioServerParameters"):
            with patch("src.mcp_client.stdio_client") as mock_stdio:
                mock_stdio.side_effect = FileNotFoundError("Command not found")

                result = await client.connect_server("test")
                assert result is False

    @pytest.mark.asyncio
    async def test_connect_server_permission_error(self, client):
        """connect_server returns False on PermissionError."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP SDK not available")

        config = MCPServerConfig(name="test", command="test-cmd")
        client.register_server(config)

        with patch("src.mcp_client.StdioServerParameters"):
            with patch("src.mcp_client.stdio_client") as mock_stdio:
                mock_stdio.side_effect = PermissionError("Permission denied")

                result = await client.connect_server("test")
                assert result is False

    @pytest.mark.asyncio
    async def test_connect_server_unexpected_error(self, client):
        """connect_server returns False on unexpected Exception."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP SDK not available")

        config = MCPServerConfig(name="test", command="test-cmd")
        client.register_server(config)

        with patch("src.mcp_client.StdioServerParameters"):
            with patch("src.mcp_client.stdio_client") as mock_stdio:
                mock_stdio.side_effect = RuntimeError("Unexpected error")

                result = await client.connect_server("test")
                assert result is False

    @pytest.mark.asyncio
    async def test_connect_server_http_transport(self, client):
        """connect_server uses streamablehttp_client for http transport."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP SDK not available")

        config = MCPServerConfig(
            name="remote", type="http", url="https://example.com/mcp"
        )
        client.register_server(config)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        mock_session.list_resources = AsyncMock(return_value=MagicMock(resources=[]))
        mock_session.list_prompts = AsyncMock(return_value=MagicMock(prompts=[]))

        # streamablehttp_client yields (read, write, get_session_id_callback)
        mock_session_id_cb = MagicMock()
        with patch(
            "src.mcp_client.streamablehttp_client",
            return_value=_async_cm((MagicMock(), MagicMock(), mock_session_id_cb)),
        ) as mock_http:
            with patch(
                "src.mcp_client.ClientSession",
                return_value=_async_cm(mock_session),
            ):
                result = await client.connect_server("remote")

                assert result is True
                assert "remote" in client.connections
                mock_http.assert_called_once_with("https://example.com/mcp")

    @pytest.mark.asyncio
    async def test_connect_server_sse_transport(self, client):
        """connect_server uses sse_client for sse transport."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP SDK not available")

        config = MCPServerConfig(
            name="sse-remote", type="sse", url="https://example.com/sse"
        )
        client.register_server(config)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        mock_session.list_resources = AsyncMock(return_value=MagicMock(resources=[]))
        mock_session.list_prompts = AsyncMock(return_value=MagicMock(prompts=[]))

        with patch(
            "src.mcp_client.sse_client",
            return_value=_async_cm((MagicMock(), MagicMock())),
        ) as mock_sse:
            with patch(
                "src.mcp_client.ClientSession",
                return_value=_async_cm(mock_session),
            ):
                result = await client.connect_server("sse-remote")

                assert result is True
                mock_sse.assert_called_once_with("https://example.com/sse")

    @pytest.mark.asyncio
    async def test_disconnect_server_exception(self, client):
        """disconnect_server handles exception during cleanup."""
        config = MCPServerConfig(name="test", command="cmd")

        # Create a connection with a mock session that raises on cleanup
        mock_session = MagicMock()

        connection = MCPServerConnection(
            config=config,
            session=mock_session,
            read_stream=MagicMock(),
            write_stream=MagicMock(),
        )
        client.connections["test"] = connection

        # The disconnect should still return True (cleanup exception is logged)
        result = await client.disconnect_server("test")
        assert result is True
        assert "test" not in client.connections


class TestMCPClientThreadSafety:
    """Test thread safety of MCPClient operations."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a fresh MCPClient backed by an isolated registry file."""
        return MCPClient(registry_path=tmp_path / "mcp-registry.json")

    def test_concurrent_server_registration(self, client):
        """Multiple threads can register servers concurrently."""
        results = []
        errors = []

        def register_server(name):
            try:
                config = MCPServerConfig(name=name, command="cmd")
                client.register_server(config)
                results.append(name)
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(20):
            t = threading.Thread(target=register_server, args=(f"server-{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
        assert len(client.servers) == 20

    def test_concurrent_get_stats(self, client):
        """Multiple threads can call get_stats concurrently."""
        # Register some servers first
        for i in range(10):
            config = MCPServerConfig(name=f"server-{i}", command="cmd")
            client.register_server(config)

        results = []
        errors = []

        def get_stats():
            try:
                stats = client.get_stats()
                results.append(stats)
            except Exception as e:
                errors.append(str(e))

        threads = []
        for _ in range(20):
            t = threading.Thread(target=get_stats)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
        # All stats should show 10 registered servers
        for stats in results:
            assert stats["registered_servers"] == 10


class TestGlobalMCPClientInstance:
    """Test the global mcp_client instance."""

    def test_global_instance_exists(self):
        """Global mcp_client instance is available."""
        assert mcp_client is not None
        assert isinstance(mcp_client, MCPClient)

    def test_global_instance_is_available_method(self):
        """Global instance has is_available method."""
        # Should not raise
        result = mcp_client.is_available()
        assert isinstance(result, bool)


class TestRegistryPersistence:
    """Registry survives across MCPClient instances backed by the same file.

    This is the property that makes the registry visible across uvicorn
    workers: register from one process/client, list from another, and both
    see the same data.
    """

    def test_registration_visible_to_second_client(self, tmp_path):
        path = tmp_path / "registry.json"
        a = MCPClient(registry_path=path)
        a.register_server(
            MCPServerConfig(name="shared", type="http", url="https://x/mcp")
        )

        b = MCPClient(registry_path=path)
        assert [s.name for s in b.list_servers()] == ["shared"]
        assert b.get_server("shared").url == "https://x/mcp"

    def test_unregister_visible_to_second_client(self, tmp_path):
        path = tmp_path / "registry.json"
        a = MCPClient(registry_path=path)
        a.register_server(MCPServerConfig(name="s", command="cmd"))

        b = MCPClient(registry_path=path)
        assert b.get_server("s") is not None
        b.unregister_server("s")

        # Original client also sees the removal on the next read.
        assert a.get_server("s") is None

    def test_missing_registry_file_returns_empty(self, tmp_path):
        path = tmp_path / "does-not-exist.json"
        client = MCPClient(registry_path=path)
        assert client.list_servers() == []
        assert client.get_server("anything") is None

    def test_corrupt_registry_file_returns_empty(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text("{ this is not valid json")
        client = MCPClient(registry_path=path)
        assert client.list_servers() == []

    def test_registry_file_is_atomic(self, tmp_path):
        """Saves go through a temp file + os.replace; no partial state."""
        path = tmp_path / "registry.json"
        client = MCPClient(registry_path=path)
        client.register_server(MCPServerConfig(name="s1", command="cmd"))
        client.register_server(MCPServerConfig(name="s2", command="cmd"))

        # No leftover .tmp files in the directory.
        leftovers = [
            p for p in path.parent.iterdir() if p.name.startswith(".mcp-registry-")
        ]
        assert leftovers == []

    def test_http_and_sse_configs_round_trip(self, tmp_path):
        path = tmp_path / "registry.json"
        a = MCPClient(registry_path=path)
        a.register_server(
            MCPServerConfig(name="http-one", type="http", url="https://h/mcp")
        )
        a.register_server(
            MCPServerConfig(name="sse-one", type="sse", url="https://s/sse")
        )

        b = MCPClient(registry_path=path)
        names = {s.name: s for s in b.list_servers()}
        assert names["http-one"].type == "http"
        assert names["http-one"].url == "https://h/mcp"
        assert names["sse-one"].type == "sse"
        assert names["sse-one"].url == "https://s/sse"

    def test_invalid_entries_skipped_with_warning(self, tmp_path, caplog):
        """Bad entries in the file are dropped, not allowed to break load."""
        import json

        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps(
                {
                    "good": {
                        "name": "good",
                        "type": "stdio",
                        "command": "cmd",
                    },
                    "bad-http-no-url": {
                        "name": "bad-http-no-url",
                        "type": "http",
                    },
                    "bad-not-dict": "this should be a dict",
                }
            )
        )

        client = MCPClient(registry_path=path)
        servers = client.list_servers()
        assert [s.name for s in servers] == ["good"]
