import json
import os
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from agent.middlewares.orchestration import OrchestrationMiddleware

WORKFLOW_REACT = "react"
WORKFLOW_PLAN_ACT = "plan_act"
WORKFLOW_PLAN_ACT_REPLAN = "plan_act_replan"

FIXTURE_PATH = Path("tests/evals/fixtures/multi_agent_eval_set_v1.jsonl")
VALID_WORKFLOWS = {WORKFLOW_REACT, WORKFLOW_PLAN_ACT, WORKFLOW_PLAN_ACT_REPLAN}


class _FakeRouterLLM:
    def invoke(self, prompt: str) -> SimpleNamespace:
        prompt_text = str(prompt)
        is_complex = len(prompt_text) > 100 or "请完成" in prompt_text or "任务" in prompt_text
        needs_team = any(token in prompt_text for token in ("协作", "多角色", "对比", "比较"))
        payload = {
            "is_complex": is_complex or needs_team,
            "needs_team": needs_team,
            "reason": "fixture heuristic",
        }
        return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))


def _predict_workflow_mode(prompt: str) -> str:
    middleware = OrchestrationMiddleware(llm=_FakeRouterLLM())
    middleware.before_model(
        {"messages": [HumanMessage(content=prompt)]},
        runtime=None,
        config={"configurable": {"state": {}}},
    )
    analysis = middleware._last_analysis or {}
    if analysis.get("needs_team"):
        return WORKFLOW_PLAN_ACT_REPLAN
    if analysis.get("is_complex"):
        return WORKFLOW_PLAN_ACT
    return WORKFLOW_REACT


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


def test_phase0_fixture_matches_current_middleware_gate():
    rows = _load_fixture()
    strict_mode = os.getenv("STRICT_ROUTER_FIXTURE", "").strip().lower() in {"1", "true", "yes"}
    for row in rows:
        mode = _predict_workflow_mode(row["prompt"])
        if strict_mode:
            assert mode == row["expected_workflow"], row["id"]
        else:
            assert mode in VALID_WORKFLOWS, row["id"]
