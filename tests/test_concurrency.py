from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace

from conftest import load_from_src

server = load_from_src("lean_lsp_mcp.server")


class DummyClient:
    def __init__(self) -> None:
        self.calls = 0
        self._active = 0
        self.max_active = 0

    def get_goal(self, *_args, **_kwargs):
        self._active += 1
        try:
            self.max_active = max(self.max_active, self._active)
            time.sleep(0.01)
            self.calls += 1
            return {"rendered": "goal"}
        finally:
            self._active -= 1


class DummyCtx:
    def __init__(self) -> None:
        class Life:
            def __init__(self) -> None:
                self.client = DummyClient()
                self.client_lock = Lock()

        class RequestContext:
            def __init__(self) -> None:
                self.lifespan_context = Life()

        self.request_context = RequestContext()


def test_client_session_serializes_calls():
    ctx = DummyCtx()

    def worker():
        with server.client_session(ctx) as client:
            assert client is ctx.request_context.lifespan_context.client
            return client.get_goal("f", 0, 0)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda _: worker(), range(5)))

    assert all(result["rendered"] == "goal" for result in results)
    client = ctx.request_context.lifespan_context.client
    assert client.calls == 5
    assert client.max_active == 1


def _make_ctx() -> SimpleNamespace:
    lifespan = SimpleNamespace(rate_limit={})
    request_context = SimpleNamespace(lifespan_context=lifespan)
    return SimpleNamespace(request_context=request_context)


def test_rate_limited_allows_positional_ctx(monkeypatch):
    ctx = _make_ctx()

    @server.rate_limited("demo", max_requests=2, per_seconds=60)
    def tool(ctx, value):
        return value * 2

    assert tool(ctx, 3) == 6
    assert tool(ctx=ctx, value=4) == 8
    assert tool.__doc__.startswith("Limit: 2req/60s.")
