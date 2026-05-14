"""Tests for _build_claude_options behavior around MCP server attachment."""

from unittest.mock import MagicMock

import pytest

from src.main import (
    _build_claude_options,
    DEFAULT_ALLOWED_TOOLS,
    CLAUDE_TOOLS,
    DEFAULT_MAX_TURNS_NO_TOOLS,
)
from src.mcp_client import MCPServerConfig


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    fake = MagicMock()
    fake.servers = {}
    fake.get_server = lambda name: fake.servers.get(name)
    monkeypatch.setattr("src.main.mcp_client", fake)
    return fake


def _make_request(*, enable_tools=False, max_tokens=None):
    """Build a minimal ChatCompletionRequest-shape mock for _build_claude_options."""
    req = MagicMock()
    req.to_claude_options.return_value = {}
    req.enable_tools = enable_tools
    req.max_tokens = max_tokens
    return req


# ---------------------------------------------------------------------------
# Backwards-compatibility: no MCP servers attached
# ---------------------------------------------------------------------------


def test_no_mcp_enable_tools_false_keeps_no_tools_defaults():
    """Without MCP and enable_tools=False: disallowed_tools + capped max_turns (unchanged)."""
    req = _make_request(enable_tools=False)
    opts = _build_claude_options(req)
    assert opts["disallowed_tools"] == CLAUDE_TOOLS
    assert opts["max_turns"] == DEFAULT_MAX_TURNS_NO_TOOLS
    assert "permission_mode" not in opts


def test_tool_enabled_paths_force_bypass_permissions():
    """Any tool-enabled branch (enable_tools=True or MCP attached) forces
    permission_mode=bypassPermissions, overriding any X-Claude-Permission-Mode
    header. Without tools, the header is preserved."""
    req = _make_request(enable_tools=True)
    opts = _build_claude_options(
        req, claude_headers={"permission_mode": "acceptEdits"}
    )
    assert opts["allowed_tools"] == DEFAULT_ALLOWED_TOOLS
    assert opts["permission_mode"] == "bypassPermissions"


# ---------------------------------------------------------------------------
# MCP attached: auto-allow tools
# ---------------------------------------------------------------------------


def test_mcp_auto_allows_wildcard_when_no_header(clean_registry):
    """Attaching an MCP server with no allowed_tools header auto-allows mcp__server__*."""
    clean_registry.servers["weather-api"] = MCPServerConfig(
        name="weather-api", type="http", url="https://example.com/mcp"
    )
    req = _make_request(enable_tools=False)
    opts = _build_claude_options(
        req, claude_headers={"mcp_server_names": ["weather-api"]}
    )
    assert "mcp__weather-api__*" in opts["allowed_tools"]
    assert opts["permission_mode"] == "bypassPermissions"


def test_mcp_multiple_servers_each_get_wildcard(clean_registry):
    """Each attached MCP server gets its own auto-allowed wildcard."""
    clean_registry.servers["weather-api"] = MCPServerConfig(
        name="weather-api", type="http", url="https://example.com/mcp"
    )
    clean_registry.servers["internal-tools"] = MCPServerConfig(
        name="internal-tools", command="node", args=["s.js"]
    )
    req = _make_request(enable_tools=False)
    opts = _build_claude_options(
        req,
        claude_headers={"mcp_server_names": ["weather-api", "internal-tools"]},
    )
    assert "mcp__weather-api__*" in opts["allowed_tools"]
    assert "mcp__internal-tools__*" in opts["allowed_tools"]


def test_mcp_respects_user_scoped_allowed_tools(clean_registry):
    """When user names specific tools for a server, wrapper does NOT also add the wildcard."""
    clean_registry.servers["weather-api"] = MCPServerConfig(
        name="weather-api", type="http", url="https://example.com/mcp"
    )
    req = _make_request(enable_tools=False)
    opts = _build_claude_options(
        req,
        claude_headers={
            "mcp_server_names": ["weather-api"],
            "allowed_tools": ["mcp__weather-api__get_current"],
        },
    )
    # User scoped explicitly — wrapper must not broaden it
    assert "mcp__weather-api__*" not in opts["allowed_tools"]
    assert "mcp__weather-api__get_current" in opts["allowed_tools"]


def test_mcp_adds_wildcard_for_servers_user_didnt_mention(clean_registry):
    """User scopes server A explicitly; wrapper still auto-allows server B."""
    clean_registry.servers["weather-api"] = MCPServerConfig(
        name="weather-api", type="http", url="https://example.com/mcp"
    )
    clean_registry.servers["internal-tools"] = MCPServerConfig(
        name="internal-tools", command="node", args=["s.js"]
    )
    req = _make_request(enable_tools=False)
    opts = _build_claude_options(
        req,
        claude_headers={
            "mcp_server_names": ["weather-api", "internal-tools"],
            "allowed_tools": ["mcp__weather-api__get_current"],
        },
    )
    # weather-api: user scoped — no wildcard
    assert "mcp__weather-api__*" not in opts["allowed_tools"]
    # internal-tools: user did not scope — wildcard added
    assert "mcp__internal-tools__*" in opts["allowed_tools"]


def test_mcp_composes_with_enable_tools(clean_registry):
    """enable_tools=True + MCP: both built-in defaults and MCP wildcards are present."""
    clean_registry.servers["weather-api"] = MCPServerConfig(
        name="weather-api", type="http", url="https://example.com/mcp"
    )
    req = _make_request(enable_tools=True)
    opts = _build_claude_options(
        req, claude_headers={"mcp_server_names": ["weather-api"]}
    )
    # All built-in defaults should be present
    for builtin in DEFAULT_ALLOWED_TOOLS:
        assert builtin in opts["allowed_tools"]
    assert "mcp__weather-api__*" in opts["allowed_tools"]


def test_mcp_forces_bypass_permissions_over_header(clean_registry):
    """MCP branch overwrites X-Claude-Permission-Mode, same as enable_tools=True."""
    clean_registry.servers["weather-api"] = MCPServerConfig(
        name="weather-api", type="http", url="https://example.com/mcp"
    )
    req = _make_request(enable_tools=False)
    opts = _build_claude_options(
        req,
        claude_headers={
            "mcp_server_names": ["weather-api"],
            "permission_mode": "acceptEdits",
        },
    )
    assert opts["permission_mode"] == "bypassPermissions"


def test_mcp_does_not_set_disallowed_tools(clean_registry):
    """When MCP is attached, the no-tools defaults must not apply."""
    clean_registry.servers["weather-api"] = MCPServerConfig(
        name="weather-api", type="http", url="https://example.com/mcp"
    )
    req = _make_request(enable_tools=False)
    opts = _build_claude_options(
        req, claude_headers={"mcp_server_names": ["weather-api"]}
    )
    assert "disallowed_tools" not in opts
    # max_turns must not be forced to the no-tools default
    assert opts.get("max_turns", 10) != DEFAULT_MAX_TURNS_NO_TOOLS
