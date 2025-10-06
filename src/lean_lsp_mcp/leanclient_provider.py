from __future__ import annotations

"""Helpers for working with the optional `leanclient` dependency."""

from typing import Any, Tuple


class LeanclientNotInstalledError(RuntimeError):
    """Raised when the `leanclient` package is required but not available."""

    def __init__(self, message: str | None = None) -> None:
        if message is None:
            message = (
                "`leanclient` is not installed. Install it with `pip install leanclient` "
                "to enable Lean LSP integration."
            )
        super().__init__(message)


try:  # pragma: no cover - import guarded for missing optional dependency
    from leanclient import DocumentContentChange as _DocumentContentChange
    from leanclient import LeanLSPClient as _LeanLSPClient
except ModuleNotFoundError as exc:  # pragma: no cover - runtime fallback when missing
    _LEANCLIENT_IMPORT_ERROR: ModuleNotFoundError | None = exc
    _LeanLSPClient = None  # type: ignore[assignment]
    _DocumentContentChange = None  # type: ignore[assignment]
else:  # pragma: no cover - exercised in environments with leanclient installed
    _LEANCLIENT_IMPORT_ERROR = None


def is_leanclient_available() -> bool:
    """Return ``True`` when the `leanclient` package imported successfully."""

    return _LEANCLIENT_IMPORT_ERROR is None


def ensure_leanclient_available() -> Tuple[type[Any], type[Any]]:
    """Return the `leanclient` classes, or raise if the dependency is missing."""

    if _LEANCLIENT_IMPORT_ERROR is not None or _LeanLSPClient is None or _DocumentContentChange is None:
        raise LeanclientNotInstalledError() from _LEANCLIENT_IMPORT_ERROR
    return _LeanLSPClient, _DocumentContentChange


__all__ = [
    "LeanclientNotInstalledError",
    "ensure_leanclient_available",
    "is_leanclient_available",
]
