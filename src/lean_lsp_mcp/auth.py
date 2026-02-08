from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import jwt
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

from lean_lsp_mcp.http_client import HttpRequestError, request_json_sync


class AuthMode(str, Enum):
    NONE = "none"
    OAUTH = "oauth"
    BEARER = "bearer"
    OAUTH_AND_BEARER = "oauth_and_bearer"
    MIXED = "mixed"


@dataclass(frozen=True)
class AuthConfig:
    mode: AuthMode
    issuer_url: str | None
    resource_server_url: str | None
    required_scopes: list[str]
    bearer_token: str | None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def load_auth_config() -> AuthConfig:
    allow_none = _env_flag("LEAN_ALLOW_NO_AUTH") or _env_flag("LEAN_LSP_TEST_MODE")
    default_mode = "none" if allow_none else "oauth_and_bearer"
    raw_mode = os.environ.get("LEAN_AUTH_MODE", default_mode).strip().lower()
    if raw_mode not in {m.value for m in AuthMode}:
        raise ValueError(
            "Invalid LEAN_AUTH_MODE. Expected one of: "
            "none, oauth, bearer, oauth_and_bearer, mixed"
        )

    mode = AuthMode(raw_mode)
    if mode == AuthMode.NONE and not allow_none:
        raise ValueError(
            "LEAN_AUTH_MODE=none is only allowed in local/test mode. "
            "Set LEAN_ALLOW_NO_AUTH=true explicitly for unsecured local runs."
        )

    issuer = os.environ.get("LEAN_OAUTH_ISSUER_URL", "").strip() or None
    resource = os.environ.get("LEAN_OAUTH_RESOURCE_SERVER_URL", "").strip() or None
    scopes_raw = os.environ.get("LEAN_OAUTH_REQUIRED_SCOPES", "").strip()
    scopes = [s for s in (x.strip() for x in scopes_raw.split(",")) if s]
    bearer = os.environ.get("LEAN_LSP_MCP_TOKEN", "").strip() or None

    if mode in {AuthMode.OAUTH, AuthMode.OAUTH_AND_BEARER, AuthMode.MIXED} and (
        not issuer or not resource
    ):
        raise ValueError(
            "LEAN_OAUTH_ISSUER_URL and LEAN_OAUTH_RESOURCE_SERVER_URL are required for oauth modes."
        )

    if mode in {AuthMode.BEARER, AuthMode.OAUTH_AND_BEARER} and not bearer:
        raise ValueError("LEAN_LSP_MCP_TOKEN is required for bearer modes.")

    return AuthConfig(
        mode=mode,
        issuer_url=issuer,
        resource_server_url=resource,
        required_scopes=scopes,
        bearer_token=bearer,
    )


class OIDCJWTVerifier:
    """Minimal OIDC JWT verifier backed by discovery + JWKS."""

    def __init__(
        self,
        *,
        issuer_url: str,
        resource_server_url: str,
        required_scopes: list[str],
        timeout: float = 10.0,
    ):
        self.issuer_url = issuer_url.rstrip("/")
        self.resource_server_url = resource_server_url
        self.required_scopes = required_scopes
        self.timeout = timeout
        self._jwks: dict[str, Any] | None = None
        self._jwks_fetched_at = 0.0
        self._jwks_ttl = 300.0
        self._lock = asyncio.Lock()

    async def verify(self, token: str) -> AccessToken | None:
        try:
            payload = await asyncio.to_thread(self._decode_token, token)
        except Exception:  # noqa: BLE001
            return None

        scopes = self._extract_scopes(payload)
        if self.required_scopes and not set(self.required_scopes).issubset(set(scopes)):
            return None

        client_id = str(
            payload.get("azp")
            or payload.get("client_id")
            or payload.get("sub")
            or "oauth-client"
        )
        exp = payload.get("exp")
        exp_int = int(exp) if isinstance(exp, (int, float)) else None

        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            expires_at=exp_int,
            resource=self.resource_server_url,
        )

    def _decode_token(self, token: str) -> dict[str, Any]:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        algorithm = header.get("alg", "RS256")

        jwk = self._select_jwk(kid)
        key_obj = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))

        options = {"verify_aud": bool(self.resource_server_url)}
        payload = jwt.decode(
            token,
            key=key_obj,
            algorithms=[algorithm],
            issuer=self.issuer_url,
            audience=self.resource_server_url if self.resource_server_url else None,
            options=options,
        )
        if not isinstance(payload, dict):
            raise ValueError("Invalid token payload type")
        return payload

    def _extract_scopes(self, payload: dict[str, Any]) -> list[str]:
        scopes: list[str] = []
        scope = payload.get("scope")
        if isinstance(scope, str):
            scopes.extend([s for s in scope.split() if s])

        scp = payload.get("scp")
        if isinstance(scp, str):
            scopes.extend([s for s in scp.split() if s])
        elif isinstance(scp, list):
            scopes.extend([s for s in scp if isinstance(s, str)])

        return sorted(set(scopes))

    def _select_jwk(self, kid: str | None) -> dict[str, Any]:
        jwks = self._get_jwks_sync()
        keys = jwks.get("keys", []) if isinstance(jwks, dict) else []
        if not isinstance(keys, list):
            raise ValueError("Malformed JWKS payload")

        if kid:
            for key in keys:
                if isinstance(key, dict) and key.get("kid") == kid:
                    return key

        for key in keys:
            if isinstance(key, dict) and key.get("kty") == "RSA":
                return key

        raise ValueError("No usable JWK key found")

    def _get_jwks_sync(self) -> dict[str, Any]:
        now = time.time()
        if self._jwks is not None and (now - self._jwks_fetched_at) < self._jwks_ttl:
            return self._jwks

        # Best effort lock-free refresh for sync call path (safe under duplicate fetches).
        discovery_url = self._discovery_url()
        metadata = self._fetch_json_sync(discovery_url)
        jwks_uri = metadata.get("jwks_uri")
        if not isinstance(jwks_uri, str) or not jwks_uri:
            raise ValueError("OIDC discovery document missing jwks_uri")

        jwks = self._fetch_json_sync(jwks_uri)
        if not isinstance(jwks, dict):
            raise ValueError("Invalid JWKS payload")

        self._jwks = jwks
        self._jwks_fetched_at = now
        return jwks

    def _discovery_url(self) -> str:
        if self.issuer_url.endswith("/.well-known/openid-configuration"):
            return self.issuer_url
        return f"{self.issuer_url}/.well-known/openid-configuration"

    def _fetch_json_sync(self, url: str) -> dict[str, Any]:
        try:
            payload = request_json_sync("GET", url, timeout=self.timeout)
        except HttpRequestError as exc:
            raise ValueError(str(exc)) from exc
        if not isinstance(payload, dict):
            raise ValueError("Invalid JSON payload type")
        return payload


class CompositeTokenVerifier(TokenVerifier):
    """Token verifier supporting OIDC JWT validation and static bearer fallback."""

    def __init__(self, config: AuthConfig):
        self.config = config
        self._oidc = None
        if config.mode in {AuthMode.OAUTH, AuthMode.OAUTH_AND_BEARER, AuthMode.MIXED}:
            assert config.issuer_url is not None
            assert config.resource_server_url is not None
            self._oidc = OIDCJWTVerifier(
                issuer_url=config.issuer_url,
                resource_server_url=config.resource_server_url,
                required_scopes=config.required_scopes,
            )

    async def verify_token(self, token: str | None) -> AccessToken | None:
        if not token:
            return None

        if self._oidc is not None:
            access = await self._oidc.verify(token)
            if access is not None:
                return access

        if self.config.mode in {
            AuthMode.BEARER,
            AuthMode.OAUTH_AND_BEARER,
            AuthMode.MIXED,
        }:
            expected = self.config.bearer_token
            if expected and secrets.compare_digest(token, expected):
                return AccessToken(
                    token=token,
                    client_id="lean-lsp-mcp-bearer",
                    scopes=self.config.required_scopes,
                    resource=self.config.resource_server_url,
                )

        return None


def auth_settings_and_verifier() -> tuple[
    AuthConfig,
    AuthSettings | None,
    CompositeTokenVerifier | None,
]:
    config = load_auth_config()
    if config.mode == AuthMode.NONE:
        return config, None, None

    verifier = CompositeTokenVerifier(config)
    if config.mode == AuthMode.MIXED:
        # Mixed auth: initialize/list_tools stay unauthenticated. Tool handlers enforce auth.
        return config, None, verifier

    # FastMCP requires AuthSettings whenever token_verifier is set.
    issuer = config.issuer_url or "http://localhost/dummy-issuer"
    resource = config.resource_server_url or "http://localhost/dummy-resource"

    auth = AuthSettings(
        issuer_url=issuer,
        resource_server_url=resource,
        required_scopes=config.required_scopes or None,
    )
    return config, auth, verifier


def bearer_token_from_header(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def oauth_resource_metadata_url(config: AuthConfig) -> str | None:
    if not config.resource_server_url:
        return None
    return (
        f"{config.resource_server_url.rstrip('/')}/.well-known/oauth-protected-resource"
    )
