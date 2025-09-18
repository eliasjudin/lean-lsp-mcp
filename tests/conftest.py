from __future__ import annotations

import sys
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Provide a lightweight package stub so submodule imports do not require
# running the project's __init__ (which depends on external MCP packages).
lean_package = types.ModuleType("lean_lsp_mcp")
lean_package.__path__ = [str(SRC / "lean_lsp_mcp")]
sys.modules.setdefault("lean_lsp_mcp", lean_package)

# Minimal stubs for the MCP package hierarchy used in pure utility modules.
if "mcp" not in sys.modules:
    mcp_pkg = types.ModuleType("mcp")
    server_pkg = types.ModuleType("mcp.server")
    auth_pkg = types.ModuleType("mcp.server.auth")
    provider_pkg = types.ModuleType("mcp.server.auth.provider")

    class TokenVerifier:  # pragma: no cover - simple testing stub
        async def verify_token(self, token: str):
            raise NotImplementedError

    class AccessToken:  # pragma: no cover - simple testing stub
        def __init__(self, token: str, client_id: str, scopes):
            self.token = token
            self.client_id = client_id
            self.scopes = scopes

    provider_pkg.TokenVerifier = TokenVerifier
    provider_pkg.AccessToken = AccessToken

    sys.modules["mcp"] = mcp_pkg
    sys.modules["mcp.server"] = server_pkg
    sys.modules["mcp.server.auth"] = auth_pkg
    sys.modules["mcp.server.auth.provider"] = provider_pkg
