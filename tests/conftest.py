from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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

    mcp_pkg = sys.modules.setdefault("mcp", types.ModuleType("mcp"))
    server_pkg = sys.modules.setdefault("mcp.server", types.ModuleType("mcp.server"))
    auth_pkg = sys.modules.setdefault("mcp.server.auth", types.ModuleType("mcp.server.auth"))
    sys.modules["mcp.server.auth.provider"] = provider_pkg

    fastmcp_pkg = types.ModuleType("mcp.server.fastmcp")

    class Context:  # pragma: no cover - testing stub
        pass

    class FastMCP:  # pragma: no cover - testing stub
        def __init__(self, **_kwargs):
            self.settings = types.SimpleNamespace(host="", port=0)

        def tool(self, *_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

        def run(self, *_, **__):
            raise RuntimeError("FastMCP.run stub should not be called in tests")

    fastmcp_pkg.Context = Context
    fastmcp_pkg.FastMCP = FastMCP

    auth_settings_pkg = types.ModuleType("mcp.server.auth.settings")

    class AuthSettings:  # pragma: no cover - testing stub
        def __init__(self, **kwargs):
            self.options = kwargs

    auth_settings_pkg.AuthSettings = AuthSettings

    utilities_pkg = types.ModuleType("mcp.server.fastmcp.utilities")
    logging_pkg = types.ModuleType("mcp.server.fastmcp.utilities.logging")

    def get_logger(_name: str):  # pragma: no cover - testing stub
        class DummyLogger:
            def __getattr__(self, _attr):
                def noop(*_args, **_kwargs):
                    return None

                return noop

        return DummyLogger()

    logging_pkg.get_logger = get_logger

    utilities_pkg.logging = logging_pkg
    fastmcp_pkg.utilities = utilities_pkg

    sys.modules["mcp.server.fastmcp"] = fastmcp_pkg
    sys.modules["mcp.server.fastmcp.utilities"] = utilities_pkg
    sys.modules["mcp.server.fastmcp.utilities.logging"] = logging_pkg
    sys.modules["mcp.server.auth.settings"] = auth_settings_pkg

    # Keep attributes accessible via parent packages
    server_pkg.fastmcp = fastmcp_pkg
    auth_pkg.settings = auth_settings_pkg

    # Minimal leanclient shim
    leanclient_pkg = types.ModuleType("leanclient")

    class LeanLSPClient:  # pragma: no cover - testing stub
        project_path = ""

        def __init__(self, *_, **__):
            pass

        def close(self):
            pass

    class DocumentContentChange:  # pragma: no cover - testing stub
        def __init__(self, *_args, **_kwargs):
            pass

    leanclient_pkg.LeanLSPClient = LeanLSPClient
    leanclient_pkg.DocumentContentChange = DocumentContentChange

    sys.modules.setdefault("leanclient", leanclient_pkg)



def load_from_src(module: str):
    """Load a module from the src tree without executing package __init__."""

    ensure_mcp_stub()
    path = SRC / (module.replace(".", "/") + ".py")
    spec = importlib.util.spec_from_file_location(module, path)
    module_obj = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module_obj)
    return module_obj


__all__ = ["ensure_mcp_stub", "load_from_src"]
