from __future__ import annotations

from types import SimpleNamespace

import pytest

from conftest import load_from_src

server = load_from_src("lean_lsp_mcp.server")


def _make_ctx(project_root: str | None = None) -> SimpleNamespace:
    lifespan = SimpleNamespace(
        lean_project_path=project_root,
        rate_limit={},
    )
    request_context = SimpleNamespace(lifespan_context=lifespan)
    return SimpleNamespace(request_context=request_context)


def test_tool_error_captures_payload_message() -> None:
    payload = {"structured": {"message": "boom"}}
    err = server.ToolError(payload)

    assert err.payload is payload
    assert str(err) == "boom"


def test_tool_error_defaults_to_generic_message() -> None:
    payload = {"structured": {}}
    err = server.ToolError(payload)

    assert str(err) == "Tool error"


@pytest.mark.parametrize(
    "code,expected_fragment",
    [
        (server.ERROR_CLIENT_NOT_READY, "Run `lean_build`"),
        (server.ERROR_BAD_REQUEST, "Call `lean_tool_spec`"),
        (server.ERROR_RATE_LIMIT, "Wait a few seconds"),
        (server.ERROR_NO_GOAL, "Ensure the requested location"),
        (server.ERROR_NOT_GOAL_POSITION, "Move the cursor"),
    ],
)
def test_derive_error_hints_per_code(code: str | None, expected_fragment: str) -> None:
    hints = server._derive_error_hints(
        code=code,
        category=None,
        details={},
        ctx=_make_ctx(project_root=None),
    )

    assert any(expected_fragment in hint for hint in hints)
    assert any("Set `LEAN_PROJECT_PATH`" in hint for hint in hints)


def test_derive_error_hints_with_project_root_skips_project_hint() -> None:
    hints = server._derive_error_hints(
        code=server.ERROR_CLIENT_NOT_READY,
        category=None,
        details={},
        ctx=_make_ctx(project_root="/tmp/project"),
    )

    assert any("Run `lean_build`" in hint for hint in hints)
    assert all("Set `LEAN_PROJECT_PATH`" not in hint for hint in hints)


def test_derive_error_hints_for_io_failure_looks_at_details() -> None:
    hints = server._derive_error_hints(
        code=server.ERROR_IO_FAILURE,
        category=None,
        details={"output": "fail"},
        ctx=_make_ctx(project_root="/tmp/project"),
    )

    assert any("Inspect the failure details" in hint for hint in hints)
    assert any("Run `lean_build`" in hint for hint in hints)


def test_rate_limited_enforces_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    call_order: list[str] = []

    @server.rate_limited("demo", max_requests=2, per_seconds=60)
    def demo_tool(ctx: SimpleNamespace) -> str:
        call_order.append("ok")
        return "ok"

    times = iter([1000, 1000, 1000])
    monkeypatch.setattr(server.time, "time", lambda: next(times))

    ctx = _make_ctx()

    assert demo_tool(ctx) == "ok"
    assert demo_tool(ctx) == "ok"

    result = demo_tool(ctx)
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["code"] == server.ERROR_RATE_LIMIT
    assert structured["details"] == {"max_requests": 2, "per_seconds": 60}
    assert "Limit: 2req/60s." in demo_tool.__doc__

    # Only two successful calls should be recorded
    assert call_order == ["ok", "ok"]
