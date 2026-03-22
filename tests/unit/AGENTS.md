# tests/unit/ AGENTS.md

**Generated:** 2026-03-22

## OVERVIEW
Unit tests for agent, ui, utils, and pages modules. 75 test files, 8120 lines.

## STRUCTURE
```
tests/unit/
├── __init__.py
├── agent/                   # (empty dir, reserved)
└── test_*.py                # Flat naming — 73+ test files at root level
```

## CONVENTIONS

1. **Test discovery**: Explicit path required — `uv run --extra dev python -m pytest tests/unit/`
   - NOT standard `pytest tests/` (root-level tests would conflict)
2. **Naming**: `test_*.py` files inside `tests/unit/` subdirectories
3. **Fixtures**: Shared fixtures in `tests/unit/conftest.py`
4. **Isolation**: Each test file should be self-contained; avoid cross-file fixture dependencies
5. **No live API calls**: Unit tests must mock LLM, DB, and external services

## QUALITY GATE
```bash
# Core gate (blocking CI)
bash scripts/quality_gate.sh core

# Full unit coverage
uv run --extra dev python -m pytest tests/unit -q

# Specific module
uv run --extra dev python -m pytest tests/unit/test_agent/test_turn_engine.py -q
```

## ANTI-PATTERNS

1. Tests calling live API keys — use mocking
2. Cross-test state pollution — each test must clean up
3. Tests in `tests/unit/` that need integration fixtures — move to `tests/integration/`
4. Skipping tests with `@pytest.mark.skip` without issue tracking
