from __future__ import annotations

import os
from enum import Enum


class ServerProfile(str, Enum):
    READ = "read"
    WRITE = "write"


def get_server_profile() -> ServerProfile:
    raw = os.environ.get("LEAN_SERVER_PROFILE", "read").strip().lower()
    if raw not in {ServerProfile.READ.value, ServerProfile.WRITE.value}:
        raise ValueError(
            "Invalid LEAN_SERVER_PROFILE. Expected 'read' or 'write', "
            f"got '{raw or '<empty>'}'."
        )
    return ServerProfile(raw)


def write_tools_enabled(profile: ServerProfile) -> bool:
    return profile == ServerProfile.WRITE


def external_tools_enabled(tool_name: str) -> bool:
    """Return whether an external web-backed tool is enabled.

    Defaults to enabled. Set LEAN_ENABLE_<TOOLNAME>=false to disable.
    Example: LEAN_ENABLE_LEANSEARCH=false
    """
    key = f"LEAN_ENABLE_{tool_name.upper()}"
    raw = os.environ.get(key, "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}
