from __future__ import annotations

from conftest import load_from_src

search_tools = load_from_src("lean_lsp_mcp.server_components.search_tools")


def test_build_service_url_handles_nested_paths():
    build_url = search_tools._build_service_url

    assert build_url("https://example.com", "api/search") == "https://example.com/api/search"
    assert build_url("https://example.com/api", "api/search") == "https://example.com/api/search"
    assert build_url("https://example.com/custom", "api/search") == "https://example.com/custom/api/search"
    assert build_url("https://example.com/custom/", "api/search") == "https://example.com/custom/api/search"
    assert build_url("https://example.com/custom", "/api/search") == "https://example.com/custom/api/search"


def test_build_service_url_handles_retrieve_endpoint():
    build_url = search_tools._build_service_url

    assert build_url("http://leanpremise.net/api", "retrieve") == "http://leanpremise.net/api/retrieve"
