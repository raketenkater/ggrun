#!/usr/bin/env bash
# The pytest list CI runs. The catalog refresh runs the same list before it
# proposes a change, so both stay in step.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 -m pytest "$@" \
  tests/test_downloader_quant.py tests/test_download_integrity.py \
  tests/test_update_recommendations.py tests/test_mimo_catalog.py \
  tests/test_measure_bandwidth.py tests/test_install_contracts.py \
  tests/test_setup_terminal.py tests/test_release_relocation.py \
  tests/test_gpu_e2e.py tests/test_agent_workload.py \
  tests/test_verify_recommended_download.py tests/test_catalog_repo_guard.py \
  tests/test_verify_installed_serving.py tests/test_synthetic_gguf.py
