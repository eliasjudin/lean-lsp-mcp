"""URL validation utilities for external API endpoints."""

from __future__ import annotations

import urllib.parse


def validate_api_url(url: str, env_name: str) -> str:
    """Validate and normalize API URLs from environment variables.
    
    Args:
        url: The URL to validate
        env_name: Name of the environment variable (for error messages)
        
    Returns:
        Normalized URL with trailing slash removed
        
    Raises:
        ValueError: If URL is invalid or uses unsupported scheme
    """
    if not url or not url.strip():
        raise ValueError(f"{env_name} must not be empty")
    
    url = url.strip()
    
    # Parse the URL
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        # Don't include the full URL in error to avoid exposing potential secrets
        raise ValueError(f"{env_name} contains an invalid URL format") from exc
    
    # Validate scheme
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"{env_name} must use http or https scheme, got: {parsed.scheme}"
        )
    
    # If no scheme provided, assume https
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urllib.parse.urlparse(url)
    
    # Validate netloc (hostname)
    if not parsed.netloc:
        raise ValueError(f"{env_name} must include a valid hostname")
    
    # Return normalized URL
    return url.rstrip("/")


__all__ = ["validate_api_url"]
