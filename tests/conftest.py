from __future__ import annotations

import asyncio
import os
import socket
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncContextManager

import pytest

from tests.helpers.mcp_client import (
    MCPClient,
    connect_sse_client,
    connect_streamable_http_client,
)
from tests.helpers.test_project import ensure_test_project


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def test_project_path(repo_root: Path) -> Path:
    try:
        return ensure_test_project(repo_root)
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _server_environment(
    repo_root: Path,
    workspace_root: Path,
    *,
    profile: str = "write",
    auth_mode: str = "none",
    token: str | None = None,
) -> dict[str, str]:
    pythonpath_entries = [str(repo_root / "src")]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)

    env: dict[str, str] = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath_entries),
            "LEAN_LOG_LEVEL": os.environ.get("LEAN_LOG_LEVEL", "ERROR"),
            "LEAN_LSP_TEST_MODE": "1",
            "LEAN_AUTH_MODE": auth_mode,
            "LEAN_SERVER_PROFILE": profile,
            "LEAN_WORKSPACE_ROOT": str(workspace_root),
        }
    )

    if token:
        env["LEAN_LSP_MCP_TOKEN"] = token

    # Optional OIDC settings for auth-mode tests
    for key in (
        "LEAN_OAUTH_ISSUER_URL",
        "LEAN_OAUTH_RESOURCE_SERVER_URL",
        "LEAN_OAUTH_REQUIRED_SCOPES",
    ):
        value = os.environ.get(key)
        if value:
            env[key] = value
    if auth_mode in {"oauth", "oauth_and_bearer", "mixed"}:
        env.setdefault("LEAN_OAUTH_ISSUER_URL", "https://issuer.example.invalid")

    return env


@asynccontextmanager
async def _spawn_server(
    *,
    repo_root: Path,
    env: dict[str, str],
    transport: str,
) -> AsyncIterator[str]:
    host = "127.0.0.1"
    port = _pick_free_port()
    proc_env = dict(env)
    if proc_env.get("LEAN_AUTH_MODE") in {"oauth", "oauth_and_bearer", "mixed"}:
        proc_env.setdefault("LEAN_OAUTH_RESOURCE_SERVER_URL", f"http://{host}:{port}")
    cmd = [
        sys.executable,
        "-m",
        "lean_lsp_mcp",
        "--transport",
        transport,
        "--host",
        host,
        "--port",
        str(port),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(repo_root),
        env=proc_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    endpoint = (
        f"http://{host}:{port}/mcp"
        if transport == "streamable-http"
        else f"http://{host}:{port}/sse"
    )

    try:
        ready = False
        for _ in range(80):
            if proc.returncode is not None:
                break
            try:
                reader, writer = await asyncio.open_connection(host, port)
                writer.close()
                await writer.wait_closed()
                ready = True
                break
            except Exception:
                await asyncio.sleep(0.1)

        if not ready:
            stderr = ""
            if proc.stderr is not None:
                try:
                    stderr = (
                        await asyncio.wait_for(proc.stderr.read(), timeout=0.2)
                    ).decode("utf-8", errors="replace")
                except Exception:
                    stderr = ""
            raise RuntimeError(
                f"Server failed to start for transport={transport}. stderr={stderr}"
            )

        yield endpoint
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()


@pytest.fixture
def remote_client_factory(
    repo_root: Path, test_project_path: Path
) -> Callable[..., AsyncContextManager[MCPClient]]:
    def factory(
        *,
        transport: str = "streamable-http",
        profile: str = "write",
        auth_mode: str = "none",
        token: str | None = None,
        client_token: str | None = None,
    ) -> AsyncContextManager[MCPClient]:
        env = _server_environment(
            repo_root,
            test_project_path,
            profile=profile,
            auth_mode=auth_mode,
            token=token,
        )

        @asynccontextmanager
        async def _ctx() -> AsyncIterator[MCPClient]:
            async with _spawn_server(
                repo_root=repo_root, env=env, transport=transport
            ) as endpoint:
                effective_client_token = (
                    client_token if client_token is not None else token
                )
                headers = (
                    {"Authorization": f"Bearer {effective_client_token}"}
                    if effective_client_token
                    else None
                )
                if transport == "sse":
                    async with connect_sse_client(endpoint, headers=headers) as client:
                        yield client
                else:
                    async with connect_streamable_http_client(
                        endpoint, headers=headers
                    ) as client:
                        yield client

        return _ctx()

    return factory


@pytest.fixture
def mcp_client_factory(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> Callable[[], AsyncContextManager[MCPClient]]:
    def factory() -> AsyncContextManager[MCPClient]:
        return remote_client_factory(transport="streamable-http", profile="write")

    return factory


@pytest.fixture
def sse_client_factory(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> Callable[[], AsyncContextManager[MCPClient]]:
    def factory() -> AsyncContextManager[MCPClient]:
        return remote_client_factory(transport="sse", profile="write")

    return factory


@pytest.fixture
def read_client_factory(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> Callable[[], AsyncContextManager[MCPClient]]:
    def factory() -> AsyncContextManager[MCPClient]:
        return remote_client_factory(transport="streamable-http", profile="read")

    return factory


@pytest.fixture
def bearer_client_factory(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> Callable[..., AsyncContextManager[MCPClient]]:
    def factory(token: str | None = None) -> AsyncContextManager[MCPClient]:
        return remote_client_factory(
            transport="streamable-http",
            profile="write",
            auth_mode="bearer",
            token=token or "test-token",
        )

    return factory


@pytest.fixture
def oauth_and_bearer_client_factory(
    remote_client_factory: Callable[..., AsyncContextManager[MCPClient]],
) -> Callable[..., AsyncContextManager[MCPClient]]:
    def factory(token: str | None = None) -> AsyncContextManager[MCPClient]:
        # For test mode we validate bearer fallback inside oauth_and_bearer mode.
        os.environ["LEAN_OAUTH_ISSUER_URL"] = os.environ.get(
            "LEAN_OAUTH_ISSUER_URL", "https://issuer.example.invalid"
        )
        os.environ["LEAN_OAUTH_RESOURCE_SERVER_URL"] = os.environ.get(
            "LEAN_OAUTH_RESOURCE_SERVER_URL", "https://resource.example.invalid"
        )
        return remote_client_factory(
            transport="streamable-http",
            profile="write",
            auth_mode="oauth_and_bearer",
            token=token or "test-token",
        )

    return factory
