"""
MCP (Model Context Protocol) client for connecting to external MCP servers.

Provides functionality to discover, connect to, and interact with MCP servers
that expose tools, resources, and prompts.
"""

import json
import logging
import os
import tempfile
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

try:
    import fcntl  # type: ignore[import-not-found]

    _HAS_FCNTL = True
except ImportError:  # Windows
    _HAS_FCNTL = False
    fcntl = None  # type: ignore[assignment]

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None

try:
    from mcp.client.sse import sse_client

    SSE_CLIENT_AVAILABLE = True
except ImportError:
    SSE_CLIENT_AVAILABLE = False
    sse_client = None

try:
    from mcp.client.streamable_http import streamablehttp_client

    HTTP_CLIENT_AVAILABLE = True
except ImportError:
    HTTP_CLIENT_AVAILABLE = False
    streamablehttp_client = None

logger = logging.getLogger(__name__)

MCP_TRANSPORT_TYPES = ("stdio", "http", "sse")


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server.

    Supports three transports:
    - stdio: subprocess command (``command`` + ``args`` + ``env``)
    - http:  remote HTTP MCP endpoint (``url``)
    - sse:   remote SSE MCP endpoint (``url``)
    """

    name: str
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    description: str = ""
    enabled: bool = True
    type: str = "stdio"
    url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type not in MCP_TRANSPORT_TYPES:
            raise ValueError(
                f"Invalid MCP server type '{self.type}'. "
                f"Must be one of: {MCP_TRANSPORT_TYPES}"
            )
        if self.type == "stdio":
            if not self.command:
                raise ValueError(
                    f"MCP server '{self.name}': stdio transport requires a 'command'"
                )
        else:
            if not self.url:
                raise ValueError(
                    f"MCP server '{self.name}': {self.type} transport requires a 'url'"
                )

    def to_sdk_config(self) -> Dict[str, Any]:
        """Render this server config in the shape ClaudeAgentOptions.mcp_servers expects.

        The Agent SDK accepts a dict per server with a ``type`` discriminator and
        transport-specific fields. Returned dicts can be assembled into a
        ``{server_name: sdk_config}`` map and assigned to
        ``ClaudeAgentOptions.mcp_servers``.
        """
        if self.type == "stdio":
            sdk: Dict[str, Any] = {
                "type": "stdio",
                "command": self.command,
                "args": list(self.args),
            }
            if self.env:
                sdk["env"] = dict(self.env)
            return sdk
        return {"type": self.type, "url": self.url}


@dataclass
class MCPServerConnection:
    """Represents an active connection to an MCP server.

    Owns an ``AsyncExitStack`` so the transport context manager and the
    ``ClientSession`` are closed atomically on disconnect, regardless of
    transport.
    """

    config: MCPServerConfig
    session: Any  # ClientSession
    read_stream: Any
    write_stream: Any
    exit_stack: Optional[AsyncExitStack] = None
    connected_at: datetime = field(default_factory=datetime.utcnow)
    available_tools: List[Dict[str, Any]] = field(default_factory=list)
    available_resources: List[Dict[str, Any]] = field(default_factory=list)
    available_prompts: List[Dict[str, Any]] = field(default_factory=list)


DEFAULT_REGISTRY_PATH = Path(
    os.environ.get(
        "WRAPPER_MCP_REGISTRY_PATH", "/tmp/claude-wrapper-mcp-servers.json"
    )
)


def _config_to_payload(cfg: MCPServerConfig) -> Dict[str, Any]:
    return {
        "name": cfg.name,
        "type": cfg.type,
        "command": cfg.command,
        "args": list(cfg.args),
        "env": dict(cfg.env) if cfg.env else None,
        "url": cfg.url,
        "description": cfg.description,
        "enabled": cfg.enabled,
    }


def _payload_to_config(payload: Dict[str, Any]) -> Optional[MCPServerConfig]:
    """Build a config from a registry-file payload, skipping invalid entries."""
    try:
        return MCPServerConfig(
            name=payload["name"],
            type=payload.get("type", "stdio"),
            command=payload.get("command", ""),
            args=list(payload.get("args") or []),
            env=payload.get("env"),
            url=payload.get("url"),
            description=payload.get("description", ""),
            enabled=payload.get("enabled", True),
        )
    except (TypeError, ValueError, KeyError) as exc:
        logger.warning(
            f"Skipping invalid MCP registry entry "
            f"{payload.get('name', '<unnamed>')!r}: {exc}"
        )
        return None


class _RegistryStore:
    """JSON-file backing store for the MCP server registry.

    Workers run as separate processes, so an in-memory dict in ``MCPClient``
    would be invisible to siblings. Persisting to a shared file (read fresh
    on every operation, written atomically) lets every worker see the same
    registry. Connections (open subprocess/HTTP transports + ``ClientSession``)
    remain per-process and are deliberately *not* persisted.

    Cross-process write safety is enforced with ``fcntl.flock`` when available
    (Linux/macOS). Atomic visibility is enforced by writing to a sibling temp
    file then ``os.replace``.
    """

    def __init__(self, path: Path):
        self.path = path

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, MCPServerConfig]:
        if not self.path.exists():
            return {}
        try:
            text = self.path.read_text()
        except OSError as exc:
            logger.warning(f"Failed to read MCP registry {self.path}: {exc}")
            return {}
        if not text.strip():
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(f"MCP registry {self.path} is not valid JSON: {exc}")
            return {}
        if not isinstance(data, dict):
            logger.warning(f"MCP registry {self.path} is not a JSON object; ignoring")
            return {}

        servers: Dict[str, MCPServerConfig] = {}
        for name, entry in data.items():
            if not isinstance(entry, dict):
                logger.warning(f"Skipping non-dict MCP registry entry {name!r}")
                continue
            entry.setdefault("name", name)
            cfg = _payload_to_config(entry)
            if cfg is not None:
                servers[cfg.name] = cfg
        return servers

    def save(self, servers: Dict[str, MCPServerConfig]) -> None:
        """Atomically replace the registry file with the given mapping."""
        self._ensure_parent()
        payload = {name: _config_to_payload(cfg) for name, cfg in servers.items()}
        body = json.dumps(payload, indent=2, sort_keys=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".mcp-registry-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def mutate(self, fn) -> Any:  # type: ignore[no-untyped-def]
        """Run ``fn(servers_dict)`` under an exclusive cross-process lock.

        The callback may mutate the dict in place; the result is saved back
        atomically. Returns whatever the callback returns.
        """
        self._ensure_parent()
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with open(lock_path, "w") as lock_fp:
            if _HAS_FCNTL:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                servers = self.load()
                result = fn(servers)
                self.save(servers)
                return result
            finally:
                if _HAS_FCNTL:
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


class MCPClient:
    """Client for managing connections to external MCP servers.

    The registry of server *configurations* is persisted to a JSON file so
    all uvicorn workers see the same set of registered servers (in-memory
    state alone would be per-process and invisible across the worker pool).
    Active *connections* — open subprocesses or HTTP/SSE transports plus
    their ``ClientSession`` — remain per-process and are NOT persisted.

    Path defaults to ``/tmp/claude-wrapper-mcp-servers.json`` and can be
    overridden via the ``WRAPPER_MCP_REGISTRY_PATH`` env var or by passing
    ``registry_path=...`` to the constructor (used by tests).
    """

    def __init__(self, registry_path: Optional[Path] = None):
        if not MCP_AVAILABLE:
            logger.warning("MCP SDK not available. MCP functionality will be disabled.")

        self._store = _RegistryStore(registry_path or DEFAULT_REGISTRY_PATH)
        self.connections: Dict[str, MCPServerConnection] = {}
        self.lock = Lock()

    @property
    def servers(self) -> Dict[str, MCPServerConfig]:
        """Snapshot of the currently registered servers (read fresh from disk).

        Exposed as a property so existing call sites that do
        ``mcp_client.servers`` keep working, while always reflecting writes
        from sibling workers.
        """
        return self._store.load()

    def is_available(self) -> bool:
        """Check if MCP SDK is available."""
        return MCP_AVAILABLE

    def register_server(self, config: MCPServerConfig) -> None:
        """Register an MCP server configuration."""

        def _do(servers: Dict[str, MCPServerConfig]) -> None:
            if config.name in servers:
                logger.warning(
                    f"Overwriting existing MCP server configuration: {config.name}"
                )
            servers[config.name] = config

        self._store.mutate(_do)
        logger.info(f"Registered MCP server: {config.name}")

    def unregister_server(self, name: str) -> bool:
        """Unregister an MCP server."""

        def _do(servers: Dict[str, MCPServerConfig]) -> bool:
            if name in servers:
                del servers[name]
                return True
            return False

        removed: bool = self._store.mutate(_do)
        if removed:
            logger.info(f"Unregistered MCP server: {name}")
        return removed

    def list_servers(self) -> List[MCPServerConfig]:
        """List all registered MCP servers."""
        return list(self._store.load().values())

    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get a specific server configuration."""
        return self._store.load().get(name)

    async def connect_server(self, name: str) -> bool:
        """
        Connect to an MCP server.

        Returns True if connection successful, False otherwise.
        """
        if not MCP_AVAILABLE:
            logger.error("Cannot connect to MCP server: MCP SDK not available")
            return False

        config = self.get_server(name)
        if not config:
            logger.error(f"MCP server not found: {name}")
            return False

        if not config.enabled:
            logger.warning(f"MCP server is disabled: {name}")
            return False

        # Check if already connected
        if name in self.connections:
            logger.info(f"Already connected to MCP server: {name}")
            return True

        exit_stack = AsyncExitStack()
        try:
            if config.type == "stdio":
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env=config.env,
                )
                read, write = await exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
            elif config.type == "http":
                if not HTTP_CLIENT_AVAILABLE:
                    await exit_stack.aclose()
                    logger.error(
                        f"MCP server '{name}': HTTP transport requested but "
                        f"mcp.client.streamable_http is not available. "
                        f"Install a newer version of the mcp package."
                    )
                    return False
                streams = await exit_stack.enter_async_context(
                    streamablehttp_client(config.url)
                )
                # streamablehttp_client yields (read, write, get_session_id_callback)
                read, write = streams[0], streams[1]
            elif config.type == "sse":
                if not SSE_CLIENT_AVAILABLE:
                    await exit_stack.aclose()
                    logger.error(
                        f"MCP server '{name}': SSE transport requested but "
                        f"mcp.client.sse is not available. "
                        f"Install a newer version of the mcp package."
                    )
                    return False
                read, write = await exit_stack.enter_async_context(
                    sse_client(config.url)
                )
            else:
                await exit_stack.aclose()
                logger.error(f"MCP server '{name}': unknown transport '{config.type}'")
                return False

            session = await exit_stack.enter_async_context(ClientSession(read, write))

            # Initialize session
            await session.initialize()

            # List available capabilities
            available_tools = []
            available_resources = []
            available_prompts = []

            try:
                # List tools
                tools_response = await session.list_tools()
                if tools_response and hasattr(tools_response, "tools"):
                    available_tools = [
                        {
                            "name": tool.name,
                            "description": getattr(tool, "description", ""),
                            "input_schema": getattr(tool, "inputSchema", {}),
                        }
                        for tool in tools_response.tools
                    ]
            except Exception as e:
                logger.warning(f"Could not list tools from {name}: {e}")

            try:
                # List resources
                resources_response = await session.list_resources()
                if resources_response and hasattr(resources_response, "resources"):
                    available_resources = [
                        {
                            "uri": resource.uri,
                            "name": getattr(resource, "name", ""),
                            "description": getattr(resource, "description", ""),
                            "mimeType": getattr(resource, "mimeType", None),
                        }
                        for resource in resources_response.resources
                    ]
            except Exception as e:
                logger.warning(f"Could not list resources from {name}: {e}")

            try:
                # List prompts
                prompts_response = await session.list_prompts()
                if prompts_response and hasattr(prompts_response, "prompts"):
                    available_prompts = [
                        {
                            "name": prompt.name,
                            "description": getattr(prompt, "description", ""),
                            "arguments": getattr(prompt, "arguments", []),
                        }
                        for prompt in prompts_response.prompts
                    ]
            except Exception as e:
                logger.warning(f"Could not list prompts from {name}: {e}")

            # Store connection
            connection = MCPServerConnection(
                config=config,
                session=session,
                read_stream=read,
                write_stream=write,
                exit_stack=exit_stack,
                available_tools=available_tools,
                available_resources=available_resources,
                available_prompts=available_prompts,
            )

            with self.lock:
                self.connections[name] = connection

            logger.info(
                f"Connected to MCP server '{name}' ({config.type}): "
                f"{len(available_tools)} tools, "
                f"{len(available_resources)} resources, "
                f"{len(available_prompts)} prompts"
            )

            return True

        except ConnectionError as e:
            await exit_stack.aclose()
            logger.error(f"Connection failed for MCP server '{name}': {e}")
            return False
        except ValueError as e:
            await exit_stack.aclose()
            logger.error(f"Invalid configuration for MCP server '{name}': {e}")
            return False
        except TimeoutError as e:
            await exit_stack.aclose()
            logger.error(f"Connection timeout for MCP server '{name}': {e}")
            return False
        except FileNotFoundError as e:
            await exit_stack.aclose()
            logger.error(f"Command not found for MCP server '{name}': {e}")
            return False
        except PermissionError as e:
            await exit_stack.aclose()
            logger.error(f"Permission denied for MCP server '{name}': {e}")
            return False
        except Exception as e:
            await exit_stack.aclose()
            logger.exception(f"Unexpected error connecting to MCP server '{name}': {e}")
            return False

    async def disconnect_server(self, name: str) -> bool:
        """Disconnect from an MCP server.

        Closes the underlying ``AsyncExitStack`` so the transport and
        ``ClientSession`` are released cleanly.
        """
        with self.lock:
            if name not in self.connections:
                logger.warning(f"Not connected to MCP server: {name}")
                return False

            connection = self.connections[name]
            del self.connections[name]

        try:
            if connection.exit_stack is not None:
                await connection.exit_stack.aclose()
            logger.info(f"Disconnected from MCP server: {name}")
            return True
        except Exception as e:
            logger.exception(f"Unexpected error disconnecting from MCP server '{name}': {e}")
            return False

    def list_connected_servers(self) -> List[str]:
        """List names of currently connected servers."""
        with self.lock:
            return list(self.connections.keys())

    def get_connection(self, name: str) -> Optional[MCPServerConnection]:
        """Get an active server connection."""
        with self.lock:
            return self.connections.get(name)

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on an MCP server.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool response
        """
        connection = self.get_connection(server_name)
        if not connection:
            raise ValueError(f"Not connected to MCP server: {server_name}")

        try:
            response = await connection.session.call_tool(tool_name, arguments)
            return response
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}' on server '{server_name}': {e}")
            raise

    async def read_resource(self, server_name: str, uri: str) -> Any:
        """
        Read a resource from an MCP server.

        Args:
            server_name: Name of the MCP server
            uri: Resource URI

        Returns:
            Resource content
        """
        connection = self.get_connection(server_name)
        if not connection:
            raise ValueError(f"Not connected to MCP server: {server_name}")

        try:
            response = await connection.session.read_resource(uri)
            return response
        except Exception as e:
            logger.error(f"Error reading resource '{uri}' from server '{server_name}': {e}")
            raise

    async def get_prompt(
        self, server_name: str, prompt_name: str, arguments: Dict[str, Any] = None
    ) -> Any:
        """
        Get a prompt from an MCP server.

        Args:
            server_name: Name of the MCP server
            prompt_name: Name of the prompt
            arguments: Prompt arguments

        Returns:
            Prompt content
        """
        connection = self.get_connection(server_name)
        if not connection:
            raise ValueError(f"Not connected to MCP server: {server_name}")

        try:
            response = await connection.session.get_prompt(prompt_name, arguments or {})
            return response
        except Exception as e:
            logger.error(f"Error getting prompt '{prompt_name}' from server '{server_name}': {e}")
            raise

    def get_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available tools from all connected MCP servers.

        Returns dict mapping server name to list of tools.
        """
        with self.lock:
            return {
                name: connection.available_tools for name, connection in self.connections.items()
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about MCP connections."""
        # Read the persisted registry once; the in-process connections dict
        # stays under the thread lock.
        registered = self._store.load()
        with self.lock:
            total_tools = sum(len(conn.available_tools) for conn in self.connections.values())
            total_resources = sum(
                len(conn.available_resources) for conn in self.connections.values()
            )
            total_prompts = sum(len(conn.available_prompts) for conn in self.connections.values())

            return {
                "mcp_available": MCP_AVAILABLE,
                "registered_servers": len(registered),
                "connected_servers": len(self.connections),
                "total_tools": total_tools,
                "total_resources": total_resources,
                "total_prompts": total_prompts,
                "servers": [
                    {
                        "name": name,
                        "enabled": config.enabled,
                        "connected": name in self.connections,
                        "description": config.description,
                    }
                    for name, config in registered.items()
                ],
            }


# Global MCP client instance
mcp_client = MCPClient()
