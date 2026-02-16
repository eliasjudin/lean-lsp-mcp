from __future__ import annotations

import os
from dataclasses import dataclass

from mcp.server.transport_security import TransportSecuritySettings

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

_LOCAL_BIND_HOSTS = {"127.0.0.1", "localhost", "::1"}
_WILDCARD_BIND_HOSTS = {"0.0.0.0", "::"}

DEFAULT_TRANSPORT_ALLOWED_HOSTS = [
    "127.0.0.1:*",
    "localhost:*",
    "[::1]:*",
]
DEFAULT_TRANSPORT_ALLOWED_ORIGINS = [
    "http://127.0.0.1:*",
    "http://localhost:*",
    "http://[::1]:*",
    "https://chatgpt.com",
    "https://chat.openai.com",
]
DEFAULT_CORS_ALLOW_ORIGINS = [
    "http://127.0.0.1",
    "http://localhost",
    "http://[::1]",
]
DEFAULT_CORS_ALLOW_ORIGIN_REGEX = r"^https?://(127\.0\.0\.1|localhost|\[::1\])(?::\d+)?$"
DEFAULT_CORS_ALLOW_METHODS = ["GET", "POST", "DELETE", "OPTIONS"]
DEFAULT_CORS_ALLOW_HEADERS = ["content-type", "mcp-session-id", "authorization"]
DEFAULT_CORS_EXPOSE_HEADERS = ["Mcp-Session-Id", "mcp-session-id"]


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid {name} value '{raw}'. Expected one of: true/false/1/0/yes/no/on/off."
    )


def _env_non_negative_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name} value '{raw}'. Expected an integer.") from exc
    if value < 0:
        raise ValueError(
            f"Invalid {name} value '{raw}'. Expected a non-negative integer."
        )
    return value


def bind_host_from_env() -> str:
    host = os.environ.get("LEAN_BIND_HOST", "127.0.0.1").strip()
    return host or "127.0.0.1"


def bind_port_from_env() -> int:
    raw = os.environ.get("LEAN_BIND_PORT", "8000").strip() or "8000"
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid LEAN_BIND_PORT value '{raw}'. Expected an integer."
        ) from exc
    if port <= 0:
        raise ValueError(
            f"Invalid LEAN_BIND_PORT value '{raw}'. Expected a positive integer."
        )
    return port


def _default_allowed_hosts(bind_host: str) -> list[str]:
    normalized = bind_host.strip().lower()
    if normalized in _LOCAL_BIND_HOSTS:
        return list(DEFAULT_TRANSPORT_ALLOWED_HOSTS)
    if normalized in _WILDCARD_BIND_HOSTS:
        return []
    return [f"{bind_host}:*"]


def _dns_rebinding_enabled(allowed_hosts: list[str]) -> bool:
    raw = os.environ.get("LEAN_TRANSPORT_DNS_REBIND_PROTECTION", "auto").strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    if raw in {"", "auto"}:
        return bool(allowed_hosts)
    raise ValueError(
        "Invalid LEAN_TRANSPORT_DNS_REBIND_PROTECTION value. Expected true/false/auto."
    )


def build_transport_security(
    bind_host: str,
    *,
    logger,
) -> TransportSecuritySettings:
    allowed_hosts = _split_csv(os.environ.get("LEAN_TRANSPORT_ALLOWED_HOSTS"))
    if not allowed_hosts:
        allowed_hosts = _default_allowed_hosts(bind_host)

    allowed_origins = _split_csv(os.environ.get("LEAN_TRANSPORT_ALLOWED_ORIGINS"))
    if not allowed_origins:
        allowed_origins = list(DEFAULT_TRANSPORT_ALLOWED_ORIGINS)

    enabled = _dns_rebinding_enabled(allowed_hosts)
    if enabled and not allowed_hosts:
        logger.warning(
            "DNS rebinding protection requested but no allowed hosts are configured. "
            "Set LEAN_TRANSPORT_ALLOWED_HOSTS or bind to a specific host. Disabling "
            "DNS rebinding protection for this process."
        )
        enabled = False

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=enabled,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


def _is_local_bind_host(bind_host: str) -> bool:
    return bind_host.strip().lower() in _LOCAL_BIND_HOSTS


@dataclass(frozen=True)
class CORSConfig:
    allow_origins: list[str]
    allow_origin_regex: str | None
    allow_methods: list[str]
    allow_headers: list[str]
    expose_headers: list[str]
    allow_credentials: bool
    max_age: int


def load_cors_config() -> CORSConfig:
    allow_origins = _split_csv(os.environ.get("LEAN_CORS_ALLOW_ORIGINS"))
    allow_origin_regex = (
        os.environ.get("LEAN_CORS_ALLOW_ORIGIN_REGEX", "").strip() or None
    )
    if not allow_origins and allow_origin_regex is None:
        allow_origins = list(DEFAULT_CORS_ALLOW_ORIGINS)
        allow_origin_regex = DEFAULT_CORS_ALLOW_ORIGIN_REGEX

    allow_methods = _split_csv(os.environ.get("LEAN_CORS_ALLOW_METHODS"))
    if not allow_methods:
        allow_methods = list(DEFAULT_CORS_ALLOW_METHODS)

    allow_headers = _split_csv(os.environ.get("LEAN_CORS_ALLOW_HEADERS"))
    if not allow_headers:
        allow_headers = list(DEFAULT_CORS_ALLOW_HEADERS)

    expose_headers = _split_csv(os.environ.get("LEAN_CORS_EXPOSE_HEADERS"))
    if not expose_headers:
        expose_headers = list(DEFAULT_CORS_EXPOSE_HEADERS)

    allow_credentials = _env_bool("LEAN_CORS_ALLOW_CREDENTIALS", default=False)
    max_age = _env_non_negative_int("LEAN_CORS_MAX_AGE", default=600)

    if allow_credentials and "*" in allow_origins:
        raise ValueError(
            "LEAN_CORS_ALLOW_CREDENTIALS=true requires explicit LEAN_CORS_ALLOW_ORIGINS "
            "(wildcard '*' is not allowed with credentials)."
        )

    return CORSConfig(
        allow_origins=allow_origins,
        allow_origin_regex=allow_origin_regex,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        expose_headers=expose_headers,
        allow_credentials=allow_credentials,
        max_age=max_age,
    )


def warn_on_wildcard_cors_for_remote_bind(
    bind_host: str,
    cors_config: CORSConfig,
    *,
    logger,
) -> None:
    if "*" not in cors_config.allow_origins:
        return
    if _is_local_bind_host(bind_host):
        return
    logger.warning(
        "CORS allow_origins includes wildcard '*' while binding to %s. "
        "This permits any website origin to call the server. "
        "Set LEAN_CORS_ALLOW_ORIGINS to an explicit allowlist for remote binds.",
        bind_host,
    )
