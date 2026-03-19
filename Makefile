EVAL_FIXTURE ?= tests/evals/fixtures/agent_task_eval_set_v1.jsonl
EVAL_ENV_FILE ?= /home/ling/LLM_App_Final/.env
EVAL_CASE_ID ?=
EVAL_LIMIT ?= 1
EVAL_OUTPUT ?=
AGENT_LLM_REQUEST_TIMEOUT ?=
JUDGE_MODEL ?=
JUDGE_BASE_URL ?=

LIVE_SMOKE_CASE_ARG := $(if $(strip $(EVAL_CASE_ID)),--case-id $(EVAL_CASE_ID),)
LIVE_SMOKE_OUTPUT_ARG := $(if $(strip $(EVAL_OUTPUT)),--output $(EVAL_OUTPUT),)
LIVE_SMOKE_TIMEOUT_ENV := $(if $(strip $(AGENT_LLM_REQUEST_TIMEOUT)),AGENT_LLM_REQUEST_TIMEOUT=$(AGENT_LLM_REQUEST_TIMEOUT),)
BASELINE_JUDGE_MODEL_ARG := $(if $(strip $(JUDGE_MODEL)),--judge-model $(JUDGE_MODEL),)
BASELINE_JUDGE_BASE_URL_ARG := $(if $(strip $(JUDGE_BASE_URL)),--judge-base-url $(JUDGE_BASE_URL),)

.PHONY: lint quality-core quality-full quality-unused \
	cleanup-check cleanup-fix cleanup-deadcode cleanup-whitelist \
	eval-baseline eval-baseline-judge eval-live-smoke eval-live-smoke-no-judge

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

eval-baseline:
	uv run --extra dev python tests/evals/run_agent_task_completion_baseline.py --fixture $(EVAL_FIXTURE)

eval-baseline-judge:
	uv run --extra dev python tests/evals/run_agent_task_completion_baseline.py --fixture $(EVAL_FIXTURE) $(BASELINE_JUDGE_MODEL_ARG) $(BASELINE_JUDGE_BASE_URL_ARG)

eval-live-smoke:
	$(LIVE_SMOKE_TIMEOUT_ENV) uv run --extra dev python tests/evals/run_agent_task_completion_live_smoke.py --fixture $(EVAL_FIXTURE) --env-file $(EVAL_ENV_FILE) --limit $(EVAL_LIMIT) $(LIVE_SMOKE_CASE_ARG) $(LIVE_SMOKE_OUTPUT_ARG)

eval-live-smoke-no-judge:
	$(LIVE_SMOKE_TIMEOUT_ENV) uv run --extra dev python tests/evals/run_agent_task_completion_live_smoke.py --fixture $(EVAL_FIXTURE) --env-file $(EVAL_ENV_FILE) --limit $(EVAL_LIMIT) --disable-judge $(LIVE_SMOKE_CASE_ARG) $(LIVE_SMOKE_OUTPUT_ARG)
