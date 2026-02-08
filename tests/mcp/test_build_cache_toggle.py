from __future__ import annotations

from pathlib import Path

import pytest

from lean_lsp_mcp.tools_write import maybe_run_cache_prefetch, should_fetch_build_cache


class _FakeProc:
    def __init__(self, return_code: int):
        self._return_code = return_code

    async def wait(self) -> int:
        return self._return_code


class _RecorderLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args) -> None:
        if args:
            self.warnings.append(message % args)
        else:
            self.warnings.append(message)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
    ],
)
def test_should_fetch_build_cache_parses_flags(
    raw: str | None,
    expected: bool,
) -> None:
    env = {} if raw is None else {"LEAN_BUILD_FETCH_CACHE": raw}
    assert should_fetch_build_cache(env) is expected


@pytest.mark.asyncio
async def test_maybe_run_cache_prefetch_skips_when_disabled(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple, dict]] = []
    logger = _RecorderLogger()

    async def report_progress(*, progress: int, total: int, message: str) -> None:
        calls.append(("progress", (progress, total, message), {}))

    async def create_process_exec(*args, **kwargs):
        calls.append(("exec", args, kwargs))
        return _FakeProc(0)

    await maybe_run_cache_prefetch(
        lean_project_path_obj=tmp_path,
        report_progress=report_progress,
        logger=logger,
        create_process_exec=create_process_exec,
        env={},
    )

    assert [name for name, *_ in calls] == []
    assert logger.warnings == []


@pytest.mark.asyncio
async def test_maybe_run_cache_prefetch_runs_when_enabled(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple, dict]] = []
    logger = _RecorderLogger()

    async def report_progress(*, progress: int, total: int, message: str) -> None:
        calls.append(("progress", (progress, total, message), {}))

    async def create_process_exec(*args, **kwargs):
        calls.append(("exec", args, kwargs))
        return _FakeProc(0)

    await maybe_run_cache_prefetch(
        lean_project_path_obj=tmp_path,
        report_progress=report_progress,
        logger=logger,
        create_process_exec=create_process_exec,
        env={"LEAN_BUILD_FETCH_CACHE": "true"},
    )

    assert calls[0][0] == "progress"
    assert calls[1][0] == "exec"
    assert calls[1][1][:4] == ("lake", "exe", "cache", "get")
    assert calls[1][2]["cwd"] == tmp_path
    assert logger.warnings == []


@pytest.mark.asyncio
async def test_maybe_run_cache_prefetch_warns_on_nonzero_exit(tmp_path: Path) -> None:
    logger = _RecorderLogger()

    async def report_progress(*, progress: int, total: int, message: str) -> None:
        return None

    async def create_process_exec(*args, **kwargs):
        return _FakeProc(7)

    await maybe_run_cache_prefetch(
        lean_project_path_obj=tmp_path,
        report_progress=report_progress,
        logger=logger,
        create_process_exec=create_process_exec,
        env={"LEAN_BUILD_FETCH_CACHE": "1"},
    )

    assert logger.warnings == ["`lake exe cache get` failed with return code 7"]
