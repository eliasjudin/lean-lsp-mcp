#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <new_version>"
  echo "Example: $0 0.20.1"
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

new_version="$1"
if [[ ! "$new_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z]+)*$ ]]; then
  die "invalid version '$new_version' (expected semver, e.g. 0.20.1)"
fi

require_cmd git
require_cmd uv

if command -v python3 >/dev/null 2>&1; then
  python_cmd="python3"
elif command -v python >/dev/null 2>&1; then
  python_cmd="python"
else
  die "missing required command: python3 (or python)"
fi

if [[ -n "$(git status --porcelain)" ]]; then
  die "working tree is not clean; commit/stash changes before releasing"
fi

echo "Preparing release $new_version"
read -r -p "Proceed with version bump, checks, commit, tag, and push? (y/N) " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Release cancelled."
  exit 1
fi

echo "Updating version in pyproject.toml"
"$python_cmd" - "$new_version" <<'PY'
import re
import sys
from pathlib import Path

version = sys.argv[1]
path = Path("pyproject.toml")
text = path.read_text(encoding="utf-8")
updated, count = re.subn(
    r'(?m)^version\s*=\s*"[^\"]+"$',
    f'version = "{version}"',
    text,
    count=1,
)
if count != 1:
    raise SystemExit("Could not locate project version line in pyproject.toml")
path.write_text(updated, encoding="utf-8")
PY

echo "Running pre-release checks"
uv sync --extra dev
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest tests/mcp/ -v
uv build

git add pyproject.toml
if [[ -f CHANGELOG.md ]]; then
  git add CHANGELOG.md
fi

git commit -m "Release $new_version"
git tag -a "v$new_version" -m "lean-lsp-mcp $new_version"

git push
git push --tags

cat <<EOF
Release $new_version committed and tagged.
Build artifacts verified with uv build.

Publishing is intentionally manual. Review artifacts, then run:
  uv publish
EOF
