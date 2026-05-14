"""Tests for `_resolve_mcp_servers` — registry-to-SDK-config resolution."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.main import _resolve_mcp_servers
from src.mcp_client import MCPServerConfig


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    """Replace the global mcp_client registry with a fresh mock for each test."""
    fake = MagicMock()
    fake.servers = {}

    def get_server(name):
        return fake.servers.get(name)

    fake.get_server = get_server
    monkeypatch.setattr("src.main.mcp_client", fake)
    return fake


def test_empty_returns_empty_dict():
    """No names → empty dict (caller can keep behavior unchanged)."""
    assert _resolve_mcp_servers(None) == {}
    assert _resolve_mcp_servers([]) == {}


def test_resolves_single_stdio_server(clean_registry):
    clean_registry.servers["lookup"] = MCPServerConfig(
        name="lookup",
        command="node",
        args=["server.js"],
        env={"X": "1"},
    )
    result = _resolve_mcp_servers(["lookup"])
    assert result == {
        "lookup": {
            "type": "stdio",
            "command": "node",
            "args": ["server.js"],
            "env": {"X": "1"},
        }
    }


def test_resolves_http_server(clean_registry):
    clean_registry.servers["remote"] = MCPServerConfig(
        name="remote",
        type="http",
        url="https://example.com/mcp",
    )
    result = _resolve_mcp_servers(["remote"])
    assert result == {
        "remote": {"type": "http", "url": "https://example.com/mcp"}
    }


def test_resolves_multiple(clean_registry):
    clean_registry.servers["a"] = MCPServerConfig(name="a", command="cmd1")
    clean_registry.servers["b"] = MCPServerConfig(
        name="b", type="http", url="https://b.example/mcp"
    )
    result = _resolve_mcp_servers(["a", "b"])
    assert set(result.keys()) == {"a", "b"}
    assert result["a"]["type"] == "stdio"
    assert result["b"]["type"] == "http"


def test_unknown_name_raises_400(clean_registry):
    with pytest.raises(HTTPException) as exc_info:
        _resolve_mcp_servers(["does-not-exist"])
    assert exc_info.value.status_code == 400
    assert "does-not-exist" in exc_info.value.detail


def test_disabled_server_raises_400(clean_registry):
    clean_registry.servers["off"] = MCPServerConfig(
        name="off", command="cmd", enabled=False
    )
    with pytest.raises(HTTPException) as exc_info:
        _resolve_mcp_servers(["off"])
    assert exc_info.value.status_code == 400
    assert "disabled" in exc_info.value.detail.lower()


def test_partial_failure_aborts(clean_registry):
    """If one name is unknown, the whole call fails — we don't silently drop tools."""
    clean_registry.servers["a"] = MCPServerConfig(name="a", command="cmd")
    with pytest.raises(HTTPException):
        _resolve_mcp_servers(["a", "unknown"])
