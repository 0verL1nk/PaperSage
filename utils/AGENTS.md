# utils/ AGENTS.md

**Generated:** 2026-03-22

## OVERVIEW
Legacy utilities and compat layer. Being phased out — new logic goes elsewhere.

## STRUCTURE
```
utils/
├── __init__.py
├── utils.py              # ⚠️ TECH DEBT: ~3000 lines, single file
├── page_helpers.py       # Streamlit page helpers
├── task_queue.py         # Redis + RQ worker queue
├── tasks.py              # ⚠️ sys.path.insert hack — TO BE FIXED
├── schemas.py            # Shared schemas
├── compare_parser.py      # Method comparison parser
└── logs/
    └── __init__.py
```

## CRITICAL ISSUES

### 1. utils/utils.py — Giant File (~3000 lines)
Violates AGENTS.md: single file >800 lines is forbidden.

**Contents (mixed responsibilities — should be split)**:
- DB migrations
- LLM API calls
- Page state management
- Document processing
- Logging setup

**Rule**: `utils/utils.py` 只减不增. New capabilities MUST go into new modules under appropriate layers.

### 2. sys.path.insert Hack (utils/tasks.py lines 13, 32)
```python
sys.path.insert(0, os.path.dirname(...))  # FORBIDDEN
```
Violates AGENTS.md rule #4. Standardize imports instead.

## CONVENTIONS

- No new business logic here.
- Page helpers are UI-layer only — don't add agent logic.
- Worker tasks run in separate process; any cross-import issues must be solved via package structure, not `sys.path`.
- Redis/RQ for async task queue; tasks must be self-contained (no Streamlit session state).

## ANTI-PATTERNS

1. Adding new functions to `utils/utils.py` — go to the appropriate layer instead.
2. Using `sys.path.insert` for imports — use proper package/directory structure.
3. Silent exception handling (`except Exception: pass`) — forbidden here like everywhere else.
4. Mixing UI code with business logic in page_helpers.

## TESTING
```bash
uv run --extra dev python -m pytest tests/unit/ -q
```
No dedicated unit tests under `utils/` — tests live in `tests/unit/`.
