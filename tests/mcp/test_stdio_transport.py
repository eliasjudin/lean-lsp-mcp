"""Tests for stdio transport mode."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from lean_lsp_mcp.app_surface import AppSurfaceConfig, build_app_home_result
from lean_lsp_mcp.auth import AuthConfig, AuthMode
from lean_lsp_mcp.models import AppHomeResult


# ---------------------------------------------------------------------------
# CLI helpers (mirrors test_cli_bind_precedence.py)
# ---------------------------------------------------------------------------

class _FakeMCP:
    def __init__(self, *, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.settings = SimpleNamespace(host=host, port=port)
        self.transports: list[str] = []

    def run(self, *, transport: str) -> None:
        self.transports.append(transport)


def _cli_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "transport": "streamable-http",
        "host": None,
        "port": None,
        "workspace_root": None,
        "profile": None,
        "auth_mode": None,
        "loogle_local": False,
        "loogle_cache_dir": None,
        "repl": False,
        "repl_timeout": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _run_main(
    monkeypatch,
    *,
    fake_mcp: _FakeMCP,
    args: argparse.Namespace,
) -> None:
    fake_server = types.ModuleType("lean_lsp_mcp.server")
    fake_server.mcp = fake_mcp
    monkeypatch.setitem(sys.modules, "lean_lsp_mcp.server", fake_server)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _self: args)

    cli_module = importlib.import_module("lean_lsp_mcp.__init__")
    assert cli_module.main() == 0


def _run_server_import_subprocess(
    *,
    code: str,
    env_overrides: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"

    env = os.environ.copy()
    pythonpath_entries = [str(src_path)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env.setdefault("LEAN_LSP_TEST_MODE", "1")
    env.update(env_overrides)

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


# ---------------------------------------------------------------------------
# CLI: stdio is accepted as a transport choice
# ---------------------------------------------------------------------------

def test_stdio_transport_accepted(monkeypatch) -> None:
    fake_mcp = _FakeMCP()
    _run_main(monkeypatch, fake_mcp=fake_mcp, args=_cli_args(transport="stdio"))

    assert fake_mcp.transports == ["stdio"]
    assert os.environ.get("LEAN_TRANSPORT") == "stdio"


def test_stdio_transport_skips_host_port_override(monkeypatch) -> None:
    fake_mcp = _FakeMCP(host="0.0.0.0", port=9999)
    _run_main(
        monkeypatch,
        fake_mcp=fake_mcp,
        args=_cli_args(transport="stdio", host="1.2.3.4", port=5555),
    )

    # host/port should NOT be overridden on mcp.settings in stdio mode
    assert fake_mcp.settings.host == "0.0.0.0"
    assert fake_mcp.settings.port == 9999
    assert fake_mcp.transports == ["stdio"]


def test_stdio_warns_on_http_only_flags(monkeypatch, caplog) -> None:
    fake_mcp = _FakeMCP()
    with caplog.at_level(logging.WARNING, logger="lean_lsp_mcp"):
        _run_main(
            monkeypatch,
            fake_mcp=fake_mcp,
            args=_cli_args(
                transport="stdio", host="0.0.0.0", port=9999, auth_mode="bearer"
            ),
        )

    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("--host" in w for w in warnings), "Expected warning about --host"
    assert any("--port" in w for w in warnings), "Expected warning about --port"
    assert any("--auth-mode" in w for w in warnings), "Expected warning about --auth-mode"


def test_stdio_no_warn_on_auth_none(monkeypatch, caplog) -> None:
    fake_mcp = _FakeMCP()
    with caplog.at_level(logging.WARNING, logger="lean_lsp_mcp"):
        _run_main(
            monkeypatch,
            fake_mcp=fake_mcp,
            args=_cli_args(transport="stdio", auth_mode="none"),
        )

    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("--auth-mode" in w for w in warnings), (
        "Should not warn about --auth-mode=none"
    )


def test_stdio_import_skips_auth_initialization() -> None:
    proc = _run_server_import_subprocess(
        code="import lean_lsp_mcp.server",
        env_overrides={
            "LEAN_TRANSPORT": "stdio",
            "LEAN_AUTH_MODE": "bearer",
            "LEAN_LSP_MCP_TOKEN": "",
        },
    )

    assert proc.returncode == 0, proc.stderr


def test_stdio_import_ignores_invalid_bind_port() -> None:
    proc = _run_server_import_subprocess(
        code="import lean_lsp_mcp.server",
        env_overrides={
            "LEAN_TRANSPORT": "stdio",
            "LEAN_BIND_PORT": "notanint",
        },
    )

    assert proc.returncode == 0, proc.stderr


def test_http_import_rejects_invalid_bind_port() -> None:
    proc = _run_server_import_subprocess(
        code="import lean_lsp_mcp.server",
        env_overrides={
            "LEAN_TRANSPORT": "streamable-http",
            "LEAN_AUTH_MODE": "none",
            "LEAN_BIND_PORT": "notanint",
        },
    )

    assert proc.returncode != 0
    assert "Invalid LEAN_BIND_PORT value 'notanint'" in proc.stderr


def test_stdio_mixed_auth_checker_is_callable_in_write_profile() -> None:
    proc = _run_server_import_subprocess(
        code=(
            "import asyncio; "
            "import lean_lsp_mcp.server as s; "
            "import sys; "
            "result = asyncio.run(s._mixed_auth_checker(None, 'build')); "
            "sys.exit(0 if callable(s._mixed_auth_checker) and result is None else 3)"
        ),
        env_overrides={
            "LEAN_TRANSPORT": "stdio",
            "LEAN_AUTH_MODE": "bearer",
            "LEAN_SERVER_PROFILE": "write",
            "LEAN_LSP_MCP_TOKEN": "",
        },
    )

    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# App surface metadata in stdio mode
# ---------------------------------------------------------------------------

def test_app_home_result_stdio_transport() -> None:
    config = AppSurfaceConfig()
    auth = AuthConfig(
        mode=AuthMode.OAUTH_AND_BEARER,
        issuer_url=None,
        resource_server_url=None,
        required_scopes=[],
        bearer_token=None,
    )

    result = build_app_home_result(
        app_config=config,
        profile=__import__("lean_lsp_mcp.profiles", fromlist=["ServerProfile"]).ServerProfile.WRITE,
        auth_config=auth,
        workspace_root=Path("/tmp/test-project"),
        read_tool_names=["goal", "diagnostics"],
        write_tool_names=["build"],
        transport="stdio",
    )

    assert isinstance(result, AppHomeResult)
    assert result.transport == "stdio"
    assert result.auth_mode == "none", "stdio should force auth_mode to 'none'"
    assert result.transport_paths is None, "stdio should have no transport_paths"
    assert result.tool_groups.read == ["goal", "diagnostics"]
    assert result.tool_groups.write == ["build"]


def test_app_home_result_http_transport() -> None:
    config = AppSurfaceConfig()
    auth = AuthConfig(
        mode=AuthMode.BEARER,
        issuer_url=None,
        resource_server_url=None,
        required_scopes=[],
        bearer_token="test-token",
    )

    result = build_app_home_result(
        app_config=config,
        profile=__import__("lean_lsp_mcp.profiles", fromlist=["ServerProfile"]).ServerProfile.READ,
        auth_config=auth,
        workspace_root=Path("/tmp/test-project"),
        read_tool_names=["goal"],
        write_tool_names=["build"],
        transport="streamable-http",
    )

    assert result.transport == "streamable-http"
    assert result.auth_mode == "bearer", "HTTP should use actual auth mode"
    assert result.transport_paths is not None
    assert result.transport_paths.streamable_http == "/mcp"
    assert result.transport_paths.sse == "/sse"
