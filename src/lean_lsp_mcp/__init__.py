import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="Lean LSP MCP Server")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["streamable-http", "sse"],
        default="streamable-http",
        help="Transport method for the server. Default is 'streamable-http'.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Host port")
    parser.add_argument(
        "--workspace-root",
        type=str,
        help="Workspace Lean project root (equivalent to LEAN_WORKSPACE_ROOT)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["read", "write"],
        help="Server profile (equivalent to LEAN_SERVER_PROFILE)",
    )
    parser.add_argument(
        "--auth-mode",
        type=str,
        choices=["none", "oauth", "bearer", "oauth_and_bearer", "mixed"],
        help="Auth mode (equivalent to LEAN_AUTH_MODE)",
    )
    parser.add_argument(
        "--loogle-local",
        action="store_true",
        help="Enable local loogle (auto-installs on first run, ~5-10 min).",
    )
    parser.add_argument(
        "--loogle-cache-dir",
        type=str,
        help="Override loogle cache location (default: ~/.cache/lean-lsp-mcp/loogle)",
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Enable fast REPL-based multi-attempt. Requires Lean REPL.",
    )
    parser.add_argument(
        "--repl-timeout",
        type=int,
        help="REPL command timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    if args.workspace_root:
        os.environ["LEAN_WORKSPACE_ROOT"] = args.workspace_root
    if args.profile:
        os.environ["LEAN_SERVER_PROFILE"] = args.profile
    if args.auth_mode:
        os.environ["LEAN_AUTH_MODE"] = args.auth_mode
    os.environ["LEAN_BIND_HOST"] = args.host
    os.environ["LEAN_BIND_PORT"] = str(args.port)
    if args.loogle_local:
        os.environ["LEAN_LOOGLE_LOCAL"] = "true"
    if args.loogle_cache_dir:
        os.environ["LEAN_LOOGLE_CACHE_DIR"] = args.loogle_cache_dir
    if args.repl:
        os.environ["LEAN_REPL"] = "true"
    if args.repl_timeout:
        os.environ["LEAN_REPL_TIMEOUT"] = str(args.repl_timeout)

    # Import after env overrides so server initialization sees final config.
    from lean_lsp_mcp.server import mcp

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport=args.transport)
    return 0
