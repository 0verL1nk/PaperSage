#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-core}"
UV="${UV_BIN:-}"
UVX="${UVX_BIN:-}"
# ty is pre-1.0 and ships new diagnostics in patch releases; pinning keeps
# unrelated PRs from turning red when upstream tightens a rule.
TY_SPEC="ty@0.0.112"

if [[ -z "${UV}" ]]; then
  UV="$(command -v uv || true)"
  if [[ -z "${UV}" && "$(uname -s)" =~ ^(MINGW|MSYS|CYGWIN) ]]; then
    UV="$(command -v uv.exe || true)"
  fi
fi
if [[ -z "${UVX}" ]]; then
  UVX="$(command -v uvx || true)"
  if [[ -z "${UVX}" && "$(uname -s)" =~ ^(MINGW|MSYS|CYGWIN) ]]; then
    UVX="$(command -v uvx.exe || true)"
  fi
fi
if [[ -z "${UV}" || -z "${UVX}" ]]; then
  echo "uv and uvx must be available on PATH (or set UV_BIN and UVX_BIN)." >&2
  exit 127
fi

case "${MODE}" in
  core)
    echo "[quality][core] repository development rules"
    "${UV}" run --extra dev python scripts/repository_guard.py --check
    echo "[quality][core] ruff check (core scope)"
    "${UV}" run --extra dev ruff check api agent/domain agent/tools agent/application/contracts.py
    echo "[quality][core] ty (core scope)"
    "${UVX}" "${TY_SPEC}" check api agent/domain agent/tools agent/application/contracts.py
    ;;
  full)
    echo "[quality][full] ruff check (full repo)"
    "${UV}" run --extra dev ruff check .
    echo "[quality][full] ty (full agent package)"
    "${UVX}" "${TY_SPEC}" check api agent
    ;;
  unused)
    echo "[quality][unused] unused imports and variables"
    "${UV}" run --extra dev python scripts/python_cleanup.py check
    echo "[quality][unused] suspected dead code report"
    "${UV}" run --extra dev python scripts/python_cleanup.py deadcode
    ;;
  *)
    echo "Usage: bash scripts/quality_gate.sh [core|full|unused]"
    exit 2
    ;;
esac
