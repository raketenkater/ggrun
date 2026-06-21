#!/usr/bin/env bash
# Linux one-command setup for a self-contained ggrun install home.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/scripts/setup-home.sh" linux "$@"
