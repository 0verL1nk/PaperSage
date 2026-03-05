import json
import os
from pathlib import Path

from agent.a2a.coordinator import WORKFLOW_PLAN_ACT, WORKFLOW_PLAN_ACT_REPLAN, WORKFLOW_REACT
from agent.a2a.router import auto_select_workflow_mode


FIXTURE_PATH = Path("tests/evals/fixtures/multi_agent_eval_set_v1.jsonl")
VALID_WORKFLOWS = {WORKFLOW_REACT, WORKFLOW_PLAN_ACT, WORKFLOW_PLAN_ACT_REPLAN}


def _load_fixture() -> list[dict]:
    rows = []
    for raw_line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def test_phase0_fixture_has_enough_cases():
    rows = _load_fixture()
    assert len(rows) >= 20


def test_phase0_fixture_fields_are_valid():
    rows = _load_fixture()
    for row in rows:
        assert isinstance(row.get("id"), str) and row["id"]
        assert isinstance(row.get("category"), str) and row["category"]
        assert isinstance(row.get("prompt"), str) and row["prompt"].strip()
        assert row.get("expected_workflow") in VALID_WORKFLOWS


def test_phase0_fixture_matches_current_heuristic_router():
    rows = _load_fixture()
    strict_mode = os.getenv("STRICT_ROUTER_FIXTURE", "").strip().lower() in {"1", "true", "yes"}
    for row in rows:
        mode, _reason = auto_select_workflow_mode(row["prompt"])
        if strict_mode:
            assert mode == row["expected_workflow"], row["id"]
        else:
            assert mode in VALID_WORKFLOWS, row["id"]
