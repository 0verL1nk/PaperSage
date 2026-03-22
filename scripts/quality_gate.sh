#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-core}"

case "${MODE}" in
  core)
    echo "[quality][core] ruff check (core scope)"
    uv run --extra dev ruff check main.py agent/domain agent/tools agent/application/contracts.py
    echo "[quality][core] ty (core scope)"
    uvx ty check main.py agent/domain agent/tools agent/application/contracts.py
    ;;
  full)
    echo "[quality][full] ruff check (full repo)"
    uv run --extra dev ruff check .
    echo "[quality][full] ty (full agent package)"
    uvx ty check main.py agent
    ;;
  unused)
    echo "[quality][unused] unused imports and variables"
    uv run --extra dev python scripts/python_cleanup.py check
    echo "[quality][unused] suspected dead code report"
    uv run --extra dev python scripts/python_cleanup.py deadcode
    ;;
  *)
    echo "Usage: bash scripts/quality_gate.sh [core|full|unused]"
    exit 2
    ;;
esac
