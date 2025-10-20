import argparse


def main():
    from lean_lsp_mcp.server import mcp

    parser = argparse.ArgumentParser(description="lean_lsp_mcp server")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse", "http", "streamable-http", "streamable_http"],
        default="stdio",
        help=(
            "Transport method for the server. Accepts 'stdio', 'sse', 'http', "
            "or 'streamable-http' (with 'streamable_http' alias). Default is 'stdio'."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for transport",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Host port for transport",
    )
    args = parser.parse_args()
    mcp.settings.host = args.host
    mcp.settings.port = args.port

    # Normalize transport aliases for FastMCP
    transport = args.transport
    if transport in {"http", "streamable_http"}:
        transport = "streamable-http"

    mcp.run(transport=transport)
