# Changelog

## 0.8.0 - 2025-10-02

### Breaking Changes
- Replaced the legacy `ok_response` / `error_response` JSON envelope with MCP-native `CallToolResult` helpers.
- Structured diagnostics now use 0-based LSP ranges and include severity labels + numeric codes.

### Added
- Uniform metadata for all tools (`request_id`, `duration_ms`, server version when available).
- Structured summaries for run-code snippets, multi-attempts, and external search responses.

### Documentation
- Clarified tool instructions and README examples to show the new MCP envelope and diagnostic schema.
