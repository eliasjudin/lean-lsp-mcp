from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _install_package_stub() -> None:
    if "lean_lsp_mcp" in sys.modules:
        return
    package = types.ModuleType("lean_lsp_mcp")
    package.__path__ = [str(SRC / "lean_lsp_mcp")]
    sys.modules["lean_lsp_mcp"] = package


def ensure_mcp_stub() -> None:
    if "mcp.server.auth.provider" in sys.modules:
        return

    provider_pkg = types.ModuleType("mcp.server.auth.provider")

    class TokenVerifier:  # pragma: no cover - testing stub
        async def verify_token(self, token: str):
            raise NotImplementedError

    class AccessToken:  # pragma: no cover - testing stub
        def __init__(self, token: str, client_id: str, scopes):
            self.token = token
            self.client_id = client_id
            self.scopes = scopes

    provider_pkg.TokenVerifier = TokenVerifier
    provider_pkg.AccessToken = AccessToken

    sys.modules.setdefault("mcp", types.ModuleType("mcp"))
    sys.modules.setdefault("mcp.server", types.ModuleType("mcp.server"))
    sys.modules.setdefault("mcp.server.auth", types.ModuleType("mcp.server.auth"))
    sys.modules["mcp.server.auth.provider"] = provider_pkg


def load_module(module_name: str):
    """Load a module from the source tree without executing package __init__."""

    _install_package_stub()
    module_path = SRC / (module_name.replace(".", "/") + ".py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


__all__ = ["load_module", "ensure_mcp_stub"]
