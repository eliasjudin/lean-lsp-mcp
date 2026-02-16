from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping

import certifi
import httpx


USER_AGENT = "lean-lsp-mcp/1.0"


class HttpRequestError(RuntimeError):
    """Raised when an HTTP request fails after retries."""


def _merge_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    return merged


async def request_json(
    method: str,
    url: str,
    *,
    timeout: float = 10.0,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    json_data: Any = None,
    retries: int = 2,
    backoff_seconds: float = 0.75,
) -> Any:
    """Send an async HTTP request and return decoded JSON with retry support."""
    merged_headers = _merge_headers(headers)
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                verify=certifi.where(),
                headers=merged_headers,
            ) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                )
                response.raise_for_status()
                return response.json()
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt == retries:
                break
            await asyncio.sleep(backoff_seconds * (2**attempt))
        except ValueError as exc:
            raise HttpRequestError(f"Invalid JSON response from {url}: {exc}") from exc

    raise HttpRequestError(f"HTTP request failed after retries: {last_exc}")


def request_json_sync(
    method: str,
    url: str,
    *,
    timeout: float = 10.0,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    json_data: Any = None,
    retries: int = 2,
    backoff_seconds: float = 0.75,
) -> Any:
    """Send a sync HTTP request and return decoded JSON with retry support."""
    merged_headers = _merge_headers(headers)
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            with httpx.Client(
                timeout=timeout,
                verify=certifi.where(),
                headers=merged_headers,
            ) as client:
                response = client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                )
                response.raise_for_status()
                return response.json()
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(backoff_seconds * (2**attempt))
        except ValueError as exc:
            raise HttpRequestError(f"Invalid JSON response from {url}: {exc}") from exc

    raise HttpRequestError(f"HTTP request failed after retries: {last_exc}")
