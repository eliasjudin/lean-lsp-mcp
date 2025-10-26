from __future__ import annotations

import functools
import inspect
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from threading import Lock
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, cast

from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger
from pydantic import AnyHttpUrl

from lean_lsp_mcp.client_utils import startup_client
from lean_lsp_mcp.instructions import INSTRUCTIONS
from lean_lsp_mcp.leanclient_provider import is_leanclient_available
from lean_lsp_mcp.tool_spec import TOOL_SPEC_VERSION
from lean_lsp_mcp.utils import OptionalTokenVerifier, log_event

# Import sanitize_exception - must be after client_session is defined to avoid circular import
# Actual import is deferred to avoid circular dependency

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for typing only
    from leanclient import DocumentContentChange, LeanLSPClient
else:
    DocumentContentChange = Any
    LeanLSPClient = Any

from importlib.metadata import PackageNotFoundError, version

SERVER_VERSION: str | None

try:  # pragma: no cover - metadata lookup may fail in tests
    SERVER_VERSION = version("lean-lsp-mcp")
except PackageNotFoundError:  # pragma: no cover - local dev fallback
    SERVER_VERSION = None

logger = get_logger(__name__)

TOOL_SPEC_RESOURCE_URI = f"tool-spec://lean_lsp_mcp/{TOOL_SPEC_VERSION}.json"


@dataclass
class AppContext:
    lean_project_path: Optional[str]
    client: Any
    file_content_hashes: Dict[str, str]
    rate_limit: Dict[str, list[int]]
    rate_limit_lock: Lock
    project_cache: Dict[str, str]
    client_lock: Lock
    file_cache_lock: Lock
    project_cache_lock: Lock


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    try:
        raw_project_path = os.environ.get("LEAN_PROJECT_PATH", "").strip()
        lean_project_path: str | None
        if not raw_project_path:
            lean_project_path = None
        else:
            lean_project_path = os.path.abspath(raw_project_path)

        context = AppContext(
            lean_project_path=lean_project_path,
            client=None,
            file_content_hashes={},
            rate_limit={
                "leansearch": [],
                "loogle": [],
                "lean_state_search": [],
                "hammer_premise": [],
            },
            rate_limit_lock=Lock(),
            project_cache={},
            client_lock=Lock(),
            file_cache_lock=Lock(),
            project_cache_lock=Lock(),
        )
        if context.lean_project_path:
            try:
                dummy_ctx = SimpleNamespace(
                    request_context=SimpleNamespace(lifespan_context=context)
                )
                log_event(
                    logger,
                    level=20,
                    message="Prewarming Lean client",
                    ctx=dummy_ctx,
                    project_path=context.lean_project_path,
                )
                startup_client(cast(Context[Any, Any, Any], dummy_ctx))
            except Exception as exc:  # pragma: no cover - prewarm best effort
                # Lazy import to avoid circular dependency
                from .common import sanitize_exception
                sanitized_error = sanitize_exception(exc, fallback_reason="prewarming Lean client")
                log_event(
                    logger,
                    level=30,
                    message="Lean client prewarm failed",
                    ctx=dummy_ctx,
                    project_path=context.lean_project_path,
                    error=sanitized_error,
                )
        yield context
    finally:
        log_event(
            logger,
            level=20,
            message="Closing Lean LSP client",
            ctx=None,
            client_present=bool(context.client),
        )
        if context.client:
            context.client.close()


mcp_kwargs: Dict[str, Any] = dict(
    name="lean_lsp_mcp",
    instructions=INSTRUCTIONS,
    lifespan=app_lifespan,
)
if is_leanclient_available():
    mcp_kwargs["dependencies"] = ["leanclient"]


auth_token = os.environ.get("LEAN_LSP_MCP_TOKEN")
if auth_token:
    issuer_url = cast(AnyHttpUrl, "http://localhost/dummy-issuer")
    resource_url = cast(AnyHttpUrl, "http://localhost/dummy-resource")
    mcp_kwargs["auth"] = AuthSettings(
        issuer_url=issuer_url,
        resource_server_url=resource_url,
    )
    mcp_kwargs["token_verifier"] = OptionalTokenVerifier(auth_token)

mcp = FastMCP(**mcp_kwargs)


def _strip_meta_from_result(result: Any) -> Any:
    if isinstance(result, dict):
        result.pop("_meta", None)
    return result


def _disable_meta_emission(server: FastMCP) -> None:
    os.environ.setdefault("FASTMCP_DISABLE_META", "1")

    settings = getattr(server, "settings", None)
    if settings is not None:
        for attr_name in dir(settings):
            if "meta" not in attr_name.lower():
                continue
            value = getattr(settings, attr_name)
            try:
                if isinstance(value, bool):
                    setattr(settings, attr_name, False)
                elif isinstance(value, dict):
                    setattr(settings, attr_name, {})
                else:
                    setattr(settings, attr_name, None)
            except (AttributeError, TypeError):
                continue

    def _wrap_callable(parent: Any, name: str, func: Any) -> None:
        if getattr(func, "__lean_strip_meta__", False):
            return
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await func(*args, **kwargs)
                return _strip_meta_from_result(result)

            setattr(async_wrapper, "__lean_strip_meta__", True)
            try:
                setattr(parent, name, async_wrapper)
            except (AttributeError, TypeError):
                pass
        elif callable(func):
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = func(*args, **kwargs)
                return _strip_meta_from_result(result)

            setattr(wrapper, "__lean_strip_meta__", True)
            try:
                setattr(parent, name, wrapper)
            except (AttributeError, TypeError):
                pass

    for attr_name in dir(server):
        lower = attr_name.lower()
        if "meta" in lower or ("result" in lower and "formatter" not in lower):
            attr = getattr(server, attr_name)
            if callable(attr):
                _wrap_callable(server, attr_name, attr)

    for attr_name in (
        "result_transformers",
        "_result_transformers",
        "response_transformers",
        "response_filters",
    ):
        transforms = getattr(server, attr_name, None)
        if isinstance(transforms, list):
            transforms.append(_strip_meta_from_result)
        elif transforms is None:
            try:
                setattr(server, attr_name, [_strip_meta_from_result])
            except (AttributeError, TypeError):
                pass

    for attr_name in (
        "_build_result_meta",
        "build_result_meta",
        "_build_meta_block",
        "build_meta_block",
    ):
        if hasattr(server, attr_name):
            try:
                setattr(server, attr_name, lambda *args, **kwargs: None)
            except (AttributeError, TypeError):
                pass

    if hasattr(server, "server_version"):
        try:
            setattr(server, "server_version", SERVER_VERSION)
        except (AttributeError, TypeError):
            pass


_disable_meta_emission(mcp)


@contextmanager
def client_session(ctx: Context) -> Iterator[Any]:
    lifespan = ctx.request_context.lifespan_context
    lock = getattr(lifespan, "client_lock", None)
    if lock is None:
        yield lifespan.client
        return
    lock.acquire()
    try:
        yield lifespan.client
    finally:
        lock.release()


__all__ = [
    "AppContext",
    "TOOL_SPEC_RESOURCE_URI",
    "client_session",
    "mcp",
]
