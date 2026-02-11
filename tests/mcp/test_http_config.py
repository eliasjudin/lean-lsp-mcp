from __future__ import annotations

import logging

import pytest

from lean_lsp_mcp.http_config import (
    DEFAULT_CORS_ALLOW_ORIGIN_REGEX,
    DEFAULT_CORS_ALLOW_ORIGINS,
    CORSConfig,
    load_cors_config,
    warn_on_wildcard_cors_for_remote_bind,
)


def _clear_cors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LEAN_CORS_ALLOW_ORIGINS",
        "LEAN_CORS_ALLOW_ORIGIN_REGEX",
        "LEAN_CORS_ALLOW_METHODS",
        "LEAN_CORS_ALLOW_HEADERS",
        "LEAN_CORS_EXPOSE_HEADERS",
        "LEAN_CORS_ALLOW_CREDENTIALS",
        "LEAN_CORS_MAX_AGE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_load_cors_config_defaults_to_localhost_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cors_env(monkeypatch)

    config = load_cors_config()

    assert config.allow_origins == DEFAULT_CORS_ALLOW_ORIGINS
    assert config.allow_origin_regex == DEFAULT_CORS_ALLOW_ORIGIN_REGEX


def test_load_cors_config_keeps_explicit_wildcard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cors_env(monkeypatch)
    monkeypatch.setenv("LEAN_CORS_ALLOW_ORIGINS", "*")

    config = load_cors_config()

    assert config.allow_origins == ["*"]
    assert config.allow_origin_regex is None


def test_load_cors_config_rejects_credentials_with_wildcard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cors_env(monkeypatch)
    monkeypatch.setenv("LEAN_CORS_ALLOW_ORIGINS", "*")
    monkeypatch.setenv("LEAN_CORS_ALLOW_CREDENTIALS", "true")

    with pytest.raises(ValueError, match="wildcard"):
        load_cors_config()


def test_warns_on_wildcard_cors_for_remote_bind(caplog: pytest.LogCaptureFixture) -> None:
    config = CORSConfig(
        allow_origins=["*"],
        allow_origin_regex=None,
        allow_methods=["GET"],
        allow_headers=["content-type"],
        expose_headers=[],
        allow_credentials=False,
        max_age=600,
    )

    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("lean_lsp_mcp.test.http_config")
    warn_on_wildcard_cors_for_remote_bind("0.0.0.0", config, logger=logger)

    assert "CORS allow_origins includes wildcard '*'" in caplog.text


def test_no_warning_on_wildcard_cors_for_local_bind(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = CORSConfig(
        allow_origins=["*"],
        allow_origin_regex=None,
        allow_methods=["GET"],
        allow_headers=["content-type"],
        expose_headers=[],
        allow_credentials=False,
        max_age=600,
    )

    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("lean_lsp_mcp.test.http_config")
    warn_on_wildcard_cors_for_remote_bind("127.0.0.1", config, logger=logger)

    assert "CORS allow_origins includes wildcard '*'" not in caplog.text
