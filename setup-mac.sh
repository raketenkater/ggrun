#!/usr/bin/env bash
# macOS one-command setup for a self-contained llm-server install home.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/scripts/setup-home.sh" mac "$@"
