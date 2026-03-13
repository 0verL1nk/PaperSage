.PHONY: run lint quality-core quality-full quality-unused \
	cleanup-check cleanup-fix cleanup-deadcode cleanup-whitelist

run:
	uv run --extra dev paper-sage

lint:
	uv run --extra dev python scripts/python_cleanup.py fix-safe
	uv run --extra dev ruff check .
	uv run --extra dev python scripts/python_cleanup.py check

quality-core:
	bash scripts/quality_gate.sh core

quality-full:
	bash scripts/quality_gate.sh full

quality-unused:
	bash scripts/quality_gate.sh unused

cleanup-check:
	uv run --extra dev python scripts/python_cleanup.py check

cleanup-fix:
	uv run --extra dev python scripts/python_cleanup.py fix-safe

cleanup-deadcode:
	uv run --extra dev python scripts/python_cleanup.py deadcode

cleanup-whitelist:
	uv run --extra dev python scripts/python_cleanup.py deadcode --make-whitelist
