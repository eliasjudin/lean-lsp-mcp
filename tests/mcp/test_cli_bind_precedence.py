from __future__ import annotations

import argparse
import importlib
import os
import sys
import types
from types import SimpleNamespace


class _FakeMCP:
    def __init__(self, *, host: str, port: int) -> None:
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


def test_main_does_not_overwrite_bind_env_when_flags_omitted(monkeypatch) -> None:
    monkeypatch.setenv("LEAN_BIND_HOST", "0.0.0.0")
    monkeypatch.setenv("LEAN_BIND_PORT", "9123")
    fake_mcp = _FakeMCP(host="0.0.0.0", port=9123)

    _run_main(monkeypatch, fake_mcp=fake_mcp, args=_cli_args(host=None, port=None))

    assert os.environ["LEAN_BIND_HOST"] == "0.0.0.0"
    assert os.environ["LEAN_BIND_PORT"] == "9123"
    assert fake_mcp.settings.host == "0.0.0.0"
    assert fake_mcp.settings.port == 9123
    assert fake_mcp.transports == ["streamable-http"]


def test_main_cli_bind_flags_override_env(monkeypatch) -> None:
    monkeypatch.setenv("LEAN_BIND_HOST", "0.0.0.0")
    monkeypatch.setenv("LEAN_BIND_PORT", "9123")
    fake_mcp = _FakeMCP(host="0.0.0.0", port=9123)

    _run_main(
        monkeypatch,
        fake_mcp=fake_mcp,
        args=_cli_args(host="127.0.0.1", port=7777),
    )

    assert os.environ["LEAN_BIND_HOST"] == "127.0.0.1"
    assert os.environ["LEAN_BIND_PORT"] == "7777"
    assert fake_mcp.settings.host == "127.0.0.1"
    assert fake_mcp.settings.port == 7777


def test_main_keeps_defaults_and_env_unset_without_cli(monkeypatch) -> None:
    monkeypatch.delenv("LEAN_BIND_HOST", raising=False)
    monkeypatch.delenv("LEAN_BIND_PORT", raising=False)
    fake_mcp = _FakeMCP(host="127.0.0.1", port=8000)

    _run_main(monkeypatch, fake_mcp=fake_mcp, args=_cli_args(host=None, port=None))

    assert os.environ.get("LEAN_BIND_HOST") is None
    assert os.environ.get("LEAN_BIND_PORT") is None
    assert fake_mcp.settings.host == "127.0.0.1"
    assert fake_mcp.settings.port == 8000
