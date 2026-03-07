#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-core}"

case "${MODE}" in
  core)
    echo "[quality][core] ruff check (core scope)"
    uv run --extra dev ruff check main.py agent/domain agent/tools agent/application/contracts.py
    echo "[quality][core] mypy (core scope)"
    uv run --extra dev mypy --config-file pyproject.toml \
      main.py agent/domain agent/tools agent/application/contracts.py
    ;;
  full)
    echo "[quality][full] ruff check (full repo)"
    uv run --extra dev ruff check .
    echo "[quality][full] mypy (full agent package)"
    uv run --extra dev mypy --config-file pyproject.toml main.py agent
    ;;
  *)
    echo "Usage: bash scripts/quality_gate.sh [core|full]"
    exit 2
    ;;
esac
